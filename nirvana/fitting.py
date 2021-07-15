"""

.. include:: ../include/links.rst
"""

import sys
import argparse
import multiprocessing as mp

import numpy as np
from scipy import stats, optimize

import matplotlib.pyplot as plt

from astropy.io import fits

try:
    from tqdm import tqdm
except:
    tqdm = None

import dynesty
try:
    from ultranest import ReactiveNestedSampler, stepsampler
except:
    ReactiveNestedSampler = None
    stepsampler = None

from .models.beam import smear, ConvolveFFTW
from .data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from .data.kinematics import Kinematics
from .data.fitargs import FitArgs
from .data.util import trim_shape

from .models.geometry import projected_polar

def bisym_model(args, paramdict, plot=False, relative_pab=False):
    '''
    Evaluate a bisymmetric velocity field model for given parameters.

    The model for this is a second order nonaxisymmetric model taken from
    Leung (2018) who in turn took it from Spekkens & Sellwood (2007). It
    evaluates the specified models at the desired coordinates.

    Args:
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        paramdict (:obj:`dict`): 
            Dictionary of galaxy parameters that are being fit. Assumes the
            format produced :func:`nirvana.fitting.unpack`.
        plot (:obj:`bool`, optional): 
            Flag to return resulting models as 2D arrays instead of 1D for 
            plotting purposes.
        relative_pab (:obj:`bool`, optional):
            Whether to define the second order position angle relative to the
            first order position angle (better for fitting) or absolutely
            (better for output).

    Returns:
        :obj:`tuple`: Tuple of two objects that are the model velocity field and
        the model velocity dispersion (if `args.disp = True`, otherwise second
        object is `None`). Arrays are 1D unless specified otherwise and should
        be rebinned to match the data.

    '''

    #convert angles to polar and normalize radial coorinate
    inc, pa, pab = np.radians([paramdict['inc'], paramdict['pa'], paramdict['pab']])
    if not relative_pab: pab = (pab - pa) % (2*np.pi)
    r, th = projected_polar(args.grid_x-paramdict['xc'], args.grid_y-paramdict['yc'], pa, inc)

    #interpolate the velocity arrays over full coordinates
    if len(args.edges) != len(paramdict['vt']):
        raise ValueError(f"Bin edge and velocity arrays are not the same shape: {len(args.edges)} and {len(paramdict['vt'])}")
    vtvals  = np.interp(r, args.edges, paramdict['vt'])
    v2tvals = np.interp(r, args.edges, paramdict['v2t'])
    v2rvals = np.interp(r, args.edges, paramdict['v2r'])

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    velmodel = paramdict['vsys'] + np.sin(inc) * (vtvals * np.cos(th) \
             - v2tvals * np.cos(2 * (th - pab)) * np.cos(th) \
             - v2rvals * np.sin(2 * (th - pab)) * np.sin(th))


    #define dispersion and surface brightness if desired
    if args.disp: 
        sigmodel = np.interp(r, args.edges, paramdict['sig'])
        sb = args.remap('sb', masked=False)
    else: 
        sigmodel = None
        sb = None

    #apply beam smearing if beam is given
    try: conv
    except: conv = None
    if args.beam_fft is not None:
        if hasattr(args, 'smearing') and not args.smearing: pass
        else: sbmodel, velmodel, sigmodel = smear(velmodel, args.beam_fft, sb=sb, 
                sig=sigmodel, beam_fft=True, cnvfftw=conv, verbose=False)

    #remasking after convolution
    if args.vel_mask is not None: velmodel = np.ma.array(velmodel, mask=args.remap('vel_mask'))
    if args.sig_mask is not None: sigmodel = np.ma.array(sigmodel, mask=args.remap('sig_mask'))

    #rebin data
    binvel = np.ma.MaskedArray(args.bin(velmodel), mask=args.vel_mask)
    if sigmodel is not None: binsig = np.ma.MaskedArray(args.bin(sigmodel), mask=args.sig_mask)
    else: binsig = None

    #return a 2D array for plotting reasons
    if plot:
        velremap = args.remap(binvel, masked=True)
        if sigmodel is not None: 
            sigremap = args.remap(binsig, masked=True)
            return velremap, sigremap
        return velremap

    return binvel, binsig

def unpack(params, args, jump=None, bound=False, relative_pab=False):
    """
    Utility function to carry around a bunch of values in the Bayesian fit.

    Takes all of the parameters that are being fit and turns them from a long
    and poorly organized tuple into an easily accessible dictionary that allows
    for much easier access to the values.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        jump (:obj:`int`, optional):
            How many indices to jump between different velocity components (i.e.
            how many bins there are). If not given, it will just determine this
            from `args.edges`.
        relative_pab (:obj:`bool`, optional):
            Whether to define the second order position angle relative to the
            first order position angle (better for fitting) or absolutely
            (better for output).

    Returns:
        :obj:`dict`: Dictionary with keys for inclination `inc`, first order
        position angle `pa`, second order position angle `pab`, systemic
        velocity `vsys`, x and y center coordinates `xc` and `yc`,
        `numpy.ndarray`_ of first order tangential velocities `vt`,
        `numpy.ndarray`_ objects of second order tangential and radial
        velocities `v2t` and `v2r`, and `numpy.ndarray`_ of velocity
        dispersions `sig`. Arrays have lengths that are the same as the
        number of bins (determined automatically or from `jump`). All angles
        are in degrees and all velocities must be in consistent units.
    """
    paramdict = {}

    #global parameters with and without center
    paramdict['xc'], paramdict['yc'] = [0,0]
    if args.nglobs == 4:
        paramdict['inc'],paramdict['pa'],paramdict['pab'],paramdict['vsys'] = params[:args.nglobs]
    elif args.nglobs == 6:
        paramdict['inc'],paramdict['pa'],paramdict['pab'],paramdict['vsys'],paramdict['xc'],paramdict['yc'] = params[:args.nglobs]

    #adjust pab if necessary
    if not relative_pab:
        paramdict['pab'] = (paramdict['pab'] + paramdict['pa']) % 360

    #figure out what indices to get velocities from
    start = args.nglobs
    if jump is None: jump = len(args.edges) - args.fixcent

    #velocities
    paramdict['vt']  = params[start:start + jump]
    paramdict['v2t'] = params[start + jump:start + 2*jump]
    paramdict['v2r'] = params[start + 2*jump:start + 3*jump]

    #add in 0 center bin
    if args.fixcent and not bound:
        paramdict['vt']  = np.insert(paramdict['vt'],  0, 0)
        paramdict['v2t'] = np.insert(paramdict['v2t'], 0, 0)
        paramdict['v2r'] = np.insert(paramdict['v2r'], 0, 0)

    #get sigma values and fill in center bin if necessary
    if args.disp: 
        sigjump = jump + args.fixcent
        end = start + 3*jump + sigjump
        paramdict['sig'] = params[start + 3*jump:end]
    else: end = start + 3*jump

    return paramdict

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

def trunc(q, mean, std, left, right):
    """
    Wrapper function for the ``ppf`` method of the `scipy.stats.truncnorm`_
    function. This makes defining edges easier.
    
    Args:
        q (:obj:`float`):
            Desired quantile.
        mean (:obj:`float`):
            Mean of distribution
        std (:obj:`float`):
            Standard deviation of distribution.
        left (:obj:`float`):
            Left bound of truncation.
        right (:obj:`float`):
            Right bound of truncation.

    Returns:
        :obj:`float`: Value of the distribution at the desired quantile
    """
    a,b = (left-mean)/std, (right-mean)/std #transform to z values
    return stats.truncnorm.ppf(q,a,b,mean,std)

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
    
    Returns:
        :obj:`float` or `numpy.ndarray`_ of transformed fit parameters.

    '''
    if bounds[key].ndim > 1:
        return (func(bounds[key][:,1]) - func(bounds[key][:,0])) * params[key][indx:] + func(bounds[key][:,0])
    else:
        return (func(bounds[key][1]) - func(bounds[key][0])) * params[key] + func(bounds[key][0])

def ptform(params, args, gaussprior=False):
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
        gaussprior (:obj:`bool`, optional):
            Flag to use the (experimental) truncated normal priors.

    Returns:
        :obj:`tuple`: Tuple of parameter values transformed into the prior
        volume.
    '''

    #unpack params and bounds into dicts
    paramdict = unpack(params, args)
    bounddict = unpack(args.bounds, args, bound=True)

    #attempt at smarter posteriors, currently super slow though
    #truncated gaussian prior around guess values
    if gaussprior and args.guess is not None:
        guessdict = unpack(args.guess,args)
        incp  = trunc(paramdict['inc'],guessdict['incg'],2,guessdict['incg']-5,guessdict['incg']+5)
        pap   = trunc(paramdict['pa'],guessdict['pag'],10,0,360)
        pabp  = 180 * paramdict['pab']
        vsysp = trunc(paramdict['vsys'],guessdict['vsysg'],1,guessdict['vsysg']-5,guessdict['vsysg']+5)
        vtp  = trunc(paramdict['vt'],guessdict['vtg'],50,0,400)
        v2tp = trunc(paramdict['v2t'],guessdict['v2tg'],50,0,200)
        v2rp = trunc(paramdict['v2r'],guessdict['v2rg'],50,0,200)

    #uniform priors defined by bounds
    else:
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
        if gaussprior:
            xcp = stats.norm.ppf(paramdict['xc'], guessdict['xc'], 5)
            ycp = stats.norm.ppf(paramdict['yc'], guessdict['yc'], 5)
        else:
            xcp = unifprior('xc', paramdict, bounddict)
            ycp = unifprior('yc', paramdict, bounddict)
        repack += [xcp,ycp]

    #repack all the velocities
    repack += [*vtp, *v2tp, *v2rp]
    if args.disp: repack += [*sigp]
    return repack

def loglike(params, args, squared=False):
    '''
    Log likelihood for :class:`dynesty.NestedSampler` fit. 
    
    Makes a model based on current parameters and computes a chi squared with
    tht
    original data.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        squared (:obj:`bool`, optional):
            Whether to compute the chi squared against the square of the
            dispersion profile or not. 

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''

    #unpack params into dict
    paramdict = unpack(params, args)

    #make velocity and dispersion models
    velmodel, sigmodel = bisym_model(args, paramdict)

    #mask border if necessary
    if args.bordermask is not None:
        velmodel = np.ma.array(velmodel, mask=args.bordermask)
        if sigmodel is not None:
            sigmodel = np.ma.array(sigmodel, mask=args.bordermask)

    #compute chi squared value with error if possible
    llike = (velmodel - args.vel)**2

    #inflate ivar with noise floor
    if args.vel_ivar is not None: 
        vel_ivar = 1/(1/args.vel_ivar + args.noise_floor**2)
        llike = llike * vel_ivar - .5 * np.log(2*np.pi * vel_ivar)
    llike = -.5 * np.ma.sum(llike)

    #add in penalty for non smooth rotation curves
    if args.weight != -1:
        llike = llike - smoothing(paramdict['vt'],  args.weight) \
                      - smoothing(paramdict['v2t'], args.weight) \
                      - smoothing(paramdict['v2r'], args.weight)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq with squared sigma or not
        if squared:
            sigdata = args.sig_phys2
            sigdataivar = args.sig_phys2_ivar if args.sig_phys2_ivar is not None else np.ones_like(sigdata)
            siglike = (sigmodel**2 - sigdata)**2

        #calculate chisq with unsquared data
        else:
            sigdata = np.sqrt(args.sig_phys2)
            sigdataivar = np.sqrt(args.sig_phys2_ivar) if args.sig_phys2_ivar is not None else np.ones_like(sigdata)
            siglike = (sigmodel - sigdata)**2

        #inflate ivar with noisefloor
        if sigdataivar is not None: 
            sigdataivar = 1/(1/sigdataivar + args.noise_floor**2)
            siglike = siglike * sigdataivar - .5 * np.log(2*np.pi * sigdataivar)
        llike -= .5*np.ma.sum(siglike)

        #smooth profile
        if args.weight != -1:
            llike -= smoothing(paramdict['sig'], args.weight*.1)

    #apply a penalty to llike if 2nd order terms are too large
    if hasattr(args, 'penalty') and args.penalty:
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        llike -= args.penalty * (v2tm - vtm)/vtm
        llike -= args.penalty * (v2rm - vtm)/vtm

    return llike

def fit(plate, ifu, galmeta = None, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-11', nbins=None,
        cores=10, maxr=None, cen=True, weight=10, smearing=True, points=500,
        stellar=False, root=None, verbose=False, disp=True, 
        fixcent=True, method='dynesty', remotedir=None, floor=5, penalty=100,
        mock=None):
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
        method (:obj:`str`, optional):
            Which fitting method to use. Defaults to `'dynesty'` but can also
            be `'ultranest'` or `'lsq'`.
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

    Returns:
        :class:`dynesty.NestedSampler`: Sampler from `dynesty` containing
        information from the fit.    
        :class:`~nirvana.data.fitargs.FitArgs`: Object with all of the relevant
        data for the galaxy as well as the parameters used for the fit.
    '''
    # Check if ultranest can be used
    if method == 'ultranest' and stepsampler is None:
        raise ImportError('Could not import ultranest.  Cannot use ultranest sampler!')

    if mock is not None:
        args, params, residnum = mock
        args.vel, args.sig = bisym_model(args, params)
        if residnum:
            try:
                residlib = np.load('residlib.dict', allow_pickle=True)
                vel2d = args.remap('vel')
                resid = trim_shape(residlib[residnum], vel2d)
                newvel = vel2d + resid
                args.vel = args.bin(newvel)
                args.remask(resid.mask)
            except:
                raise ValueError('Could not apply residual correctly. Check that residlib.dict is in the appropriate place')


    #get info on galaxy and define bins and starting guess
    else:
        if stellar:
            args = MaNGAStellarKinematics.from_plateifu(plate, ifu, daptype=daptype, dr=dr,
                                                        cube_path=root,
                                                        image_path=root, maps_path=root, 
                                                        remotedir=remotedir)
        else:
            args = MaNGAGasKinematics.from_plateifu(plate, ifu, line='Ha-6564', daptype=daptype,
                                                    dr=dr,  cube_path=root,
                                                    image_path=root, maps_path=root, 
                                                    remotedir=remotedir)

    #set basic fit parameters for galaxy
    args.setnglobs(6) if cen else args.setnglobs(4)
    args.setweight(weight)
    args.setdisp(disp)
    args.setfixcent(fixcent)
    args.setnoisefloor(floor)
    args.setpenalty(penalty)
    args.npoints = points
    args.smearing = smearing

    #set bin edges
    if galmeta is not None: 
        if mock is None: args.phot_inc = galmeta.guess_inclination()
        args.reff = galmeta.reff

    inc = args.getguess(galmeta=galmeta)[1] if args.phot_inc is None else args.phot_inc
    if nbins is not None: args.setedges(nbins, nbin=True, maxr=maxr)
    else: args.setedges(inc, maxr=maxr)

    #discard if number of bins is too small
    if len(args.edges) - fixcent < 3:
        raise ValueError('Galaxy unsuitable: too few radial bins')

    #define a variable for speeding up convolutions
    #has to be a global because multiprocessing can't pickle cython
    global conv
    conv = ConvolveFFTW(args.spatial_shape)

    #starting positions for all parameters based on a quick fit
    #not used in dynesty
    args.clip()
    theta0 = args.getguess(galmeta=galmeta)
    ndim = len(theta0)

    #adjust dimensions according to fit params
    nbin = len(args.edges) - args.fixcent
    if disp: ndim += nbin + args.fixcent
    args.setnbins(nbin)
    print(f'{nbin + args.fixcent} radial bins, {ndim} parameters')
    
    #prior bounds and asymmetry defined based off of guess
    #args.setbounds()
    args.setbounds(incpad=3, incgauss=True)
    args.getasym()

    #open up multiprocessing pool if needed
    if cores > 1 and method == 'dynesty':
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    #experimental support for ultranest
    if method == 'ultranest':
        #define names of parameters
        names = ['inc', 'pa', 'pab', 'vsys', 'xc', 'yc']
        for i in range(args.fixcent, nbin+1): names += ['vt'  + str(i)]
        for i in range(args.fixcent, nbin+1): names += ['v2t' + str(i)]
        for i in range(args.fixcent, nbin+1): names += ['v2r' + str(i)]
        for i in range(nbin+1): names += ['sig' + str(i)]

        #wraparound parameters for pa and pab
        wrap = np.zeros(len(names), dtype=bool)
        wrap[1:3] = True

        #define likelihood and prior with arguments
        ulike = lambda params: loglike(params, args)
        uprior = lambda params: ptform(params, args)

        #ultranest step sampler
        sampler = ReactiveNestedSampler(names, ulike, uprior, wrapped_params=wrap, 
                log_dir=f'/data/manga/digiorgio/nirvana/ultranest/{plate}/{ifu}')
        sampler.stepsampler = stepsampler.RegionSliceSampler(nsteps = 2*len(names))
        sampler.run()

    elif method == 'lsq':
        #minfunc = lambda x: loglike(x, args)
        def minfunc(params):
            velmodel, sigmodel = bisym_model(args, unpack(params, args))
            velchisq = (velmodel - args.vel)**2 * args.vel_ivar
            sigchisq = (sigmodel - args.sig)**2 * args.sig_ivar
            return velchisq + sigchisq

        lsqguess = np.append(args.guess, [np.median(args.sig)] * (args.nbins + args.fixcent))
        sampler = optimize.least_squares(minfunc, x0=lsqguess, method='trf',
                  bounds=(args.bounds[:,0], args.bounds[:,1]), verbose=2, diff_step=[.01] * len(lsqguess))
        args.guess = lsqguess

    elif method == 'dynesty':
        #dynesty sampler with periodic pa and pab
        sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=points,
                periodic=[1,2], pool=pool,
                ptform_args = [args], logl_args = [args], verbose=verbose)
        sampler.run_nested()

        if pool is not None: pool.close()

    else:
        raise ValueError('Choose a valid fitting method: dynesty, ultranest, or lsq')

    return sampler, args
