"""

.. include:: ../include/links.rst
"""

import multiprocessing as mp

import numpy as np
from scipy import stats, optimize

try:
    from tqdm import tqdm
except:
    tqdm = None

try:
    import pyfftw
except:
    pyfftw = None

import dynesty

from .beam import smear, ConvolveFFTW
from .geometry import projected_polar
from ..data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from ..data.util import trim_shape, unpack, cinv
from ..data.fitargs import FitArgs
from ..models.higher_order import bisym_model

import warnings
warnings.simplefilter('ignore', RuntimeWarning)

def smoothing(array, weight=1):
    """
    A penalty function for encouraging smooth arrays. 
    
    For each bin, it computes the average of the bins to the left and right and
    computes the chi squared of the bin with that average. It repeats the
    values at the left and right edges, so they are effectively smoothed with
    themselves.

    Args:
        array (`numpy.ndarray`_):
            Array to be analyzed for smoothness.
        weight (:obj:`float`, optional):
            Normalization factor for resulting chi squared value

    Returns:
        :obj:`float`: Chi squared value that serves as a measurement for how
        smooth the array is, normalized by the weight.
    """

    edgearray = np.array([array[0], *array, array[-1]]) #bin edges
    avgs = (edgearray[:-2] + edgearray[2:])/2 #average of surrounding bins
    chisq = (avgs - array)**2 / np.abs(array) #chi sq of each bin to averages
    chisq[~np.isfinite(chisq)] = 0 #catching nans
    return chisq.sum() * weight

def unifprior(key, params, bounds, indx=0, func=lambda x:x):
    '''
    Uniform prior transform for a given key in the params and bounds dictionaries.

    Args:
        key (:obj:`str`):
            Key in params and bounds dictionaries.
        params (:obj:`dict`):
            Dictionary of untransformed fit parameters. Assumes the format
            produced :func:`nirvana.fitting.unpack`.
        params (:obj:`dict`):
            Dictionary of uniform prior bounds on fit parameters. Assumes the
            format produced :func:`nirvana.fitting.unpack`.
        indx (:obj:`int`, optional):
            If the parameter is an array, what index of the array to start at.
        func (:obj:`function`, optional):
            Function used to map input values to bounds. Defaults to uniform.
    
    Returns:
        :obj:`float` or `numpy.ndarray`_ of transformed fit parameters.

    '''
    if bounds[key].ndim > 1:
        return (func(bounds[key][:,1]) - func(bounds[key][:,0])) * params[key][indx:] + func(bounds[key][:,0])
    else:
        return (func(bounds[key][1]) - func(bounds[key][0])) * params[key] + func(bounds[key][0])

def ptform(params, args):
    '''
    Prior transform for :class:`dynesty.NestedSampler` fit. 
    
    Defines the prior volume for the supplied set of parameters. Uses uniform
    priors by default but can switch to truncated normal if specified.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  

    Returns:
        :obj:`tuple`: Tuple of parameter values transformed into the prior
        volume.
    '''

    #unpack params and bounds into dicts
    paramdict = unpack(params, args)
    bounddict = unpack(args.bounds, args, bound=True)

    #uniform priors defined by bounds
    #uniform prior on sin(inc)
    #incfunc = lambda i: np.cos(np.radians(i))
    #incp = np.degrees(np.arccos(unifprior('inc', paramdict, bounddict,func=incfunc)))
    pap = unifprior('pa', paramdict, bounddict)
    incp = stats.norm.ppf(paramdict['inc'], *bounddict['inc'])
    #pap = stats.norm.ppf(paramdict['pa'], *bounddict['pa'])
    pabp = unifprior('pab', paramdict, bounddict)
    vsysp = unifprior('vsys', paramdict, bounddict)

    #continuous prior to correlate bins
    if args.weight == -1:
        vtp  = np.array(paramdict['vt'])
        v2tp = np.array(paramdict['v2t'])
        v2rp = np.array(paramdict['v2r'])
        vs = [vtp, v2tp, v2rp]
        if args.disp:
            sigp = np.array(paramdict['sig'])
            vs += [sigp]

        #step outwards from center bin to make priors correlated
        for vi in vs:
            mid = len(vi)//2
            vi[mid] = 400 * vi[mid]
            for i in range(mid-1, -1+args.fixcent, -1):
                vi[i] = stats.norm.ppf(vi[i], vi[i+1], 50)
            for i in range(mid+1, len(vi)):
                vi[i] = stats.norm.ppf(vi[i], vi[i-1], 50)

    #uncorrelated bins with unif priors
    else:
        vtp  = unifprior('vt',  paramdict, bounddict, int(args.fixcent))
        v2tp = unifprior('v2t', paramdict, bounddict, int(args.fixcent))
        v2rp = unifprior('v2r', paramdict, bounddict, int(args.fixcent))
        if args.disp: 
            sigp = unifprior('sig', paramdict, bounddict)

    #reassemble params array
    repack = [incp, pap, pabp, vsysp]

    #do centers if desired
    if args.nglobs == 6: 
        xcp = unifprior('xc', paramdict, bounddict)
        ycp = unifprior('yc', paramdict, bounddict)
        repack += [xcp,ycp]

    #do scatter terms with logunif
    if args.scatter:
        velscp = unifprior('vel_scatter', paramdict, bounddict, func=lambda x:10**x)
        sigscp = unifprior('sig_scatter', paramdict, bounddict, func=lambda x:10**x)

    #repack all the velocities
    repack += [*vtp, *v2tp, *v2rp]
    if args.disp: repack += [*sigp]
    if args.scatter: repack += [velscp, sigscp]
    return repack

def loglike(params, args):
    '''
    Log likelihood for :class:`dynesty.NestedSampler` fit. 
    
    Makes a model based on current parameters and computes a chi squared with
    original data.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''

    #unpack params into dict
    paramdict = unpack(params, args)

    #make velocity and dispersion models
    velmodel, sigmodel = bisym_model(args, paramdict)

    #compute chi squared value with error if possible
    llike = (velmodel - args.kin.vel)**2

    #inflate ivar with noise floor
    if args.kin.vel_ivar is not None: 
        if args.scatter: 
            vel_ivar = 1/(1/args.kin.vel_ivar + paramdict['vel_scatter']**2)
        else:
            vel_ivar = 1/(1/args.kin.vel_ivar + args.noise_floor**2)
        llike = llike * vel_ivar 
    llike = -.5 * np.ma.sum(llike + np.log(2*np.pi * vel_ivar))

    #add in penalty for non smooth rotation curves
    if args.weight != -1:
        if args.scatter: velweight = args.weight / paramdict['vel_scatter']
        else: velweight = args.weight
        llike = llike - smoothing(paramdict['vt'],  velweight) \
                      - smoothing(paramdict['v2t'], velweight) \
                      - smoothing(paramdict['v2r'], velweight)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq
        sigdata = np.sqrt(args.kin.sig_phys2)
        sigdataivar = np.sqrt(args.kin.sig_phys2_ivar) if args.kin.sig_phys2_ivar is not None else np.ones_like(sigdata)
        siglike = (sigmodel - sigdata)**2

        #inflate ivar with noisefloor
        if sigdataivar is not None: 
            if args.scatter: 
                sigdataivar = 1/(1/args.kin.sig_ivar + paramdict['sig_scatter']**2)
            else:
                sigdataivar = 1/(1/sigdataivar + args.noise_floor**2)
            siglike = siglike * sigdataivar - .5 * np.log(2*np.pi * sigdataivar)

        llike -= .5*np.ma.sum(siglike)

        #smooth profile
        if args.weight != -1:
            if args.scatter: sigweight = args.weight / paramdict['sig_scatter']
            else: sigweight = args.weight
            llike -= smoothing(paramdict['sig'], sigweight*.1)

    #apply a penalty to llike if 2nd order terms are too large
    if hasattr(args, 'penalty') and args.penalty:
        if args.scatter: penalty = args.penalty / paramdict['vel_scatter']
        else: penalty = args.penalty
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        llike -= penalty * (v2tm - vtm)/vtm
        llike -= penalty * (v2rm - vtm)/vtm

    return llike

def covarlike(params, args):
    '''
    Log likelihood function utilizing the full covariance matrix of the data.

    Performs the same function as :func:`loglike` but uses the covariance
    matrix for all of the spaxels rather than just the errors for each
    individual spaxel. It takes the exact same arguments and outputs the same
    things too, so it should be able to be switched in and out.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''
    #unpack, generate models and resids
    paramdict = unpack(params, args)
    velmodel, sigmodel = bisym_model(args, paramdict)
    velresid = (velmodel - args.kin.vel)[~args.kin.vel_mask]
    sigresid = (sigmodel - args.kin.sig)[~args.kin.sig_mask]

    #calculate loglikes for velocity and dispersion
    vellike = -.5 * velresid.T.dot(args.velcovinv.dot(velresid)) + args.velcoeff
    if sigmodel is not None:
        siglike = -.5 * sigresid.T.dot(args.sigcovinv.dot(sigresid)) + args.sigcoeff
    else: siglike = 0

    #smoothing penalties
    if args.weight and args.weight != -1:
        weightlike = - smoothing(paramdict['vt'],  args.weight) \
                     - smoothing(paramdict['v2t'], args.weight) \
                     - smoothing(paramdict['v2r'], args.weight)
        if siglike: 
            weightlike -= smoothing(paramdict['sig'], args.weight*.1)
    else: weightlike = 0

    #second order penalties
    if hasattr(args, 'penalty') and args.penalty:
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        penlike = - (args.penalty * (v2tm - vtm)/vtm) \
                  - (args.penalty * (v2rm - vtm)/vtm)
    else: penlike = 0 

    return vellike + siglike + weightlike + penlike 

def fit(plate, ifu, galmeta = None, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-11', nbins=None,
        cores=10, maxr=None, cen=True, weight=10, smearing=True, points=500,
        stellar=False, root=None, verbose=False, disp=True, 
        fixcent=True, remotedir=None, floor=5, penalty=100,
        mock=None, covar=False, scatter=False, maxbins=10):
    '''
    Main function for fitting a MaNGA galaxy with a nonaxisymmetric model.

    Gets velocity data for the MaNGA galaxy with the given plateifu and fits it
    according to the supplied arguments. Will fit a nonaxisymmetric model based
    on models from Leung (2018) and Spekkens & Sellwood (2007) to describe
    bisymmetric features as well as possible. Uses `dynesty` to explore
    parameter space to find best fit values.

    Args:
        plate (:obj:`int`):
            MaNGA plate number for desired galaxy.
        ifu (:obj:`int`):
            MaNGA IFU design number for desired galaxy.
        daptype (:obj:`str`, optional):
            DAP type included in filenames.
        dr (:obj:`str`, optional):
            Name of MaNGA data release in file paths.
        nbins (:obj:`int`, optional):
            Number of radial bins to use. Will be calculated automatically if
            not specified.
        cores (:obj:`int`, optional):
            Number of threads to use for parallel fitting.
        maxr (:obj:`float`, optional):
            Maximum radius to make bin edges extend to. Will be calculated
            automatically if not specified.
        cen (:obj:`bool`, optional):
            Flag for whether or not to fit the position of the center.
        weight (:obj:`float`, optional):
            How much weight to assign to the smoothness penalty of the rotation
            curves. 
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to fits.
        points (:obj:`int`, optional):
            Number of live points for :class:`dynesty.NestedSampler` to use.
        stellar (:obj:`bool`, optional):
            Flag to fit stellar velocity information instead of gas.
        root (:obj:`str`, optional):
            Direct path to maps and cube files, circumventing `dr`.
        verbose (:obj:`bool`, optional):
            Flag to give verbose output from :class:`dynesty.NestedSampler`.
        disp (:obj:`bool`, optional):
            Flag for whether to fit the velocity dispersion profile as well.
            2010. Not currently functional
        fixcent (:obj:`bool`, optional):
            Flag for whether to fix the center velocity bin at 0.
        remotedir (:obj:`str`, optional):
            If a directory is given, it will download data from sas into that
            base directory rather than looking for it locally
        floor (:obj:`float`, optional):
            Intrinsic scatter to add to velocity and dispersion errors in
            quadrature in order to inflate errors to a more realistic level.
        penalty (:obj:`float`, optional):
            Penalty to impose in log likelihood if 2nd order velocity profiles
            have too high of a mean value. Forces model to fit dominant
            rotation with 1st order profile
        mock (:obj:`tuple`, optional):
            A tuple of the `params` and `args` objects output by
            :func:`nirvana.plotting.fileprep` to fit instead of real data. Can
            be used to fit a galaxy with known parameters for testing purposes.
        covar (:obj:`bool`, optional):
            Whether to use the (currently nonfunctional) covariance likelihood
            rather than the normal one
        scatter (:obj:`bool`, optional):
            Whether to include intrinsic scatter as a fit parameter. Currently
            not working well.
        maxbins (:obj:`int`, optional):
            Maximum number of radial bins to allow. Overridden by ``nbins`` if
            it's larger.

    Returns:
        :class:`dynesty.NestedSampler`: Sampler from `dynesty` containing
        information from the fit.    
        :class:`~nirvana.data.fitargs.FitArgs`: Object with all of the relevant
        data for the galaxy as well as the parameters used for the fit.
    '''

    #set number of global parameters 
    #inc, pa, pab, vsys by default, xc and yc optionally
    nglobs = 6 if cen else 4

    #set up mock galaxy data with real residuals if desired
    if mock is not None:
        args, params, residnum = mock
        args.kin.vel, args.kin.sig = bisym_model(args, params)

        #add in real residuals from fit
        if residnum:
            try:
                residlib = np.load('residlib.dict', allow_pickle=True)
                vel2d = args.kin.remap('vel')
                resid = trim_shape(residlib[residnum], vel2d)
                newvel = vel2d + resid
                args.kin.vel = args.kin.bin(newvel)
                args.kin.remask(resid.mask)
            except:
                raise ValueError('Could not apply residual correctly. Check that residlib.dict is in the appropriate place')


    #get info on galaxy and define bins and starting guess
    else:
        if stellar:
            kin = MaNGAStellarKinematics.from_plateifu(plate, ifu,
                    daptype=daptype, dr=dr, cube_path=root, image_path=root,
                    maps_path=root, remotedir=remotedir, covar=covar,
                    positive_definite=True)
        else:
            kin = MaNGAGasKinematics.from_plateifu(plate, ifu, line='Ha-6564',
                    daptype=daptype, dr=dr,  cube_path=root, image_path=root,
                    maps_path=root, remotedir=remotedir, covar=covar,
                    positive_definite=True)

        #set basic fit parameters for galaxy
        veltype = 'Stars' if stellar else 'Gas'
        args = FitArgs(kin, veltype, nglobs, weight, disp, fixcent, floor, penalty,
                points, smearing, maxr, scatter)

    #get galaxy metadata
    if galmeta is not None: 
        if mock is None: args.kin.phot_inc = galmeta.guess_inclination()
        args.kin.reff = galmeta.reff

    #clip bad regions of the data
    args.clip()

    #set bins manually if nbins is specified
    if nbins is not None: 
        if nbins > maxbins: maxbins = nbins
        args.setedges(nbins, nbin=True, maxr=maxr)

    #set bins automatically based off of FWHM and photometric inc
    else: 
        inc = args.getguess(galmeta=galmeta)[1] if args.kin.phot_inc is None else args.kin.phot_inc
        args.setedges(inc, maxr=maxr)

        #keep number of bins under specified limit
        if len(args.edges) > maxbins + 1 + args.fixcent:
            args.setedges(maxbins, nbin=True, maxr=maxr)

    #discard if number of bins is too small
    if len(args.edges) - fixcent < 3:
        raise ValueError('Galaxy unsuitable: too few radial bins')

    #set up fftw for speeding up convolutions
    if pyfftw is not None: args.conv = ConvolveFFTW(args.kin.spatial_shape)
    else: args.conv = None

    #starting positions for all parameters based on a quick fit
    #not used in dynesty
    theta0 = args.getguess(galmeta=galmeta)
    ndim = len(theta0)

    #clip and invert covariance matrices
    if args.kin.vel_covar is not None and covar: 
        #goodvelcovar = args.kin.vel_covar[np.ix_(goodvel, goodvel)]
        goodvelcovar = np.diag(1/args.kin.vel_ivar)[np.ix_(goodvel, goodvel)]# + 1e-10
        args.velcovinv = cinv(goodvelcovar)
        sign, logdet = np.linalg.slogdet(goodvelcovar)#.todense())
        if sign != 1:
            raise ValueError('Determinant of velocity covariance is not positive')
        args.velcoeff = -.5 * (np.log(2 * np.pi) * goodvel.sum() + logdet)

        if args.kin.sig_phys2_covar is not None:
            goodsig = ~args.kin.sig_mask
            #goodsigcovar = args.kin.sig_covar[np.ix_(goodsig, goodsig)]
            goodsigcovar = np.diag(1/args.kin.sig_ivar)[np.ix_(goodsig, goodsig)]# + 1e-10
            args.sigcovinv = cinv(goodsigcovar)
            sign, logdet = np.linalg.slogdet(goodsigcovar)#.todense())
            if sign != 1:
                raise ValueError('Determinant of dispersion covariance is not positive')
            args.sigcoeff = -.5 * (np.log(2 * np.pi) * goodsig.sum() + logdet)

        else: args.sigcovinv = None

        if not np.isfinite(args.velcovinv).all():
            raise Exception('nans in velcovinv')
        if not np.isfinite(args.sigcovinv).all():
            raise Exception('nans in sigcovinv')
        if not np.isfinite(args.velcoeff):
            raise Exception('nans in velcoeff')
        if not np.isfinite(args.sigcoeff):
            raise Exception('nans in sigcoeff')

    else: args.velcovinv, args.sigcovinv = (None, None)


    #adjust dimensions according to fit params
    nbin = len(args.edges) - args.fixcent
    if disp: ndim += nbin + args.fixcent
    if scatter: ndim += 2
    args.setnbins(nbin)
    print(f'{nbin + args.fixcent} radial bins, {ndim} parameters')
    
    #prior bounds and asymmetry defined based off of guess
    if galmeta is not None: 
        args.setphotpa(galmeta)
        args.setbounds(incpad=3, incgauss=True)#, papad=10, pagauss=True)
    else: args.setbounds(incpad=3, incgauss=True)
    args.getasym()

    #open up multiprocessing pool if needed
    if cores > 1:
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    #dynesty sampler with periodic pa and pab
    if not covar: sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=points,
            periodic=[1,2], pool=pool,
            ptform_args = [args], logl_args = [args], verbose=verbose)
    else: sampler = dynesty.NestedSampler(covarlike, ptform, ndim, nlive=points,
            periodic=[1,2], pool=pool,
            ptform_args = [args], logl_args = [args], verbose=verbose)
    sampler.run_nested()

    if pool is not None: pool.close()

    return sampler, args
