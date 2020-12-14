#!/usr/bin/env python

import sys
import argparse
import multiprocessing as mp

from IPython import embed

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

from astropy.io import fits

try:
    from tqdm import tqdm
except:
    tqdm = None

import dynesty

from .models.beam import smear, ConvolveFFTW
from .data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from .data.kinematics import Kinematics
from .data.fitargs import FitArgs

from .models.geometry import projected_polar

def bisym_model(args, paramdict, plot=False):
    '''
    Evaluate a bisymmetric velocity field model for given parameters.

    The model for this is a second order nonaxisymmetric model taken from
    Leung (2018) who in turn took it from Spekkens & Sellwood (2007). It
    evaluates the specified models at the desired coordinates.

    Args:
        args (:class:`nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        paramdict (:obj:`dict`): Dictionary of galaxy parameters that are
            being fit. Assumes the format produced 
            :func:`nirvana.fitting.unpack`.
        plot (:obj:`bool`, optional): 
            Flag to return resulting models as 2D arrays instead of 1D for 
            plotting purposes.

    Returns:
        :obj:`tuple`: Tuple of two objects that are the model velocity field and
        the model velocity dispersion (if `args.disp = True`, otherwise second
        object is `None`). Arrays are 1D unless specified otherwise and should
        be rebinned to match the data.

    '''

    #convert angles to polar and normalize radial coorinate
    inc, pa, pab = np.radians([paramdict['inc'], paramdict['pa'], paramdict['pab']])
    #pa = np.radians(183)
    r, th = projected_polar(args.grid_x-paramdict['xc'], args.grid_y-paramdict['yc'], pa, inc)
    r /= args.reff

    #if hasattr(args, 'maxr') and args.maxr != None:
    #    r = np.ma.array(r, mask = r > args.maxr)
    #    th = np.ma.array(th, mask = r > args.maxr)

    #insert a fixed central bin if it is being ignored by fit
    if args.fixcent and paramdict['vts'][0] != 0:
        vts  = np.insert(paramdict['vts'],  0, 0)
        v2ts = np.insert(paramdict['v2ts'], 0, 0)
        v2rs = np.insert(paramdict['v2rs'], 0, 0)
    else:
        vts  = paramdict['vts']
        v2ts = paramdict['v2ts']
        v2rs = paramdict['v2rs']

    #interpolate velocity values for all r 
    bincents = (args.edges[:-1] + args.edges[1:])/2
    vtvals  = np.interp(r, bincents, vts)
    v2tvals = np.interp(r, bincents, v2ts)
    v2rvals = np.interp(r, bincents, v2rs)

    #define dispersion and surface brightness if desired
    if args.disp: 
        sigmodel = np.interp(r, bincents, paramdict['sig'])
        sb = args.remap('sb')
    else: 
        sigmodel = None
        sb = None

    try: conv
    except: conv = None

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    velmodel = paramdict['vsys'] + np.sin(inc) * (vtvals * np.cos(th) \
             - v2tvals * np.cos(2 * (th - pab)) * np.cos(th) \
             - v2rvals * np.sin(2 * (th - pab)) * np.sin(th))

    #apply beam smearing if beam is given
    if args.beam_fft is not None:
        sbmodel, velmodel, sigmodel = smear(velmodel, args.beam_fft, sb=sb, sig=sigmodel, beam_fft=True, cnvfftw=conv)

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

def unpack(params, args, jump=None):
    '''
    Utility function to carry around a bunch of values in the Bayesian fit.

    Takes all of the parameters that are being fit and turns them from a long
    and poorly organized tuple into an easily accessible dictionary that allows
    for much easier access to the values.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        jump (:obj:`int`, optional):
            How many indices to jump between different velocity components (i.e.
            how many bins there are). If not given, it will just determine this
            from `args.edges`.

    Returns:
        :obj:`dict`: Dictionary with keys for inclination `inc`, first order
        position angle `pa`, second order position angle `pab`, systemic
        velocity `vsys`, x and y center coordinates `xc` and `yc`, `np.ndarray`_
        of first order tangential velocities `vts`, `np.ndarray`_ objects of
        second order tangential and radial velocities `v2ts` and `v2rs`, and
        `np.ndarray`_ of velocity dispersions `sig`. Arrays have lengths that
        are the same as the number of bins (determined automatically or from
        `jump`). All angles are in degrees and all velocities must be in
        consistent units. 
    '''

    paramdict = {}
    #global parameters with and without center
    paramdict['xc'], paramdict['yc'] = [0,0]
    if args.nglobs == 4:
        paramdict['inc'],paramdict['pa'],paramdict['pab'],paramdict['vsys'] = params[:args.nglobs]
    elif args.nglobs == 6:
        paramdict['inc'],paramdict['pa'],paramdict['pab'],paramdict['vsys'],paramdict['xc'],paramdict['yc'] = params[:args.nglobs]

    #figure out what indices to get velocities from
    start = args.nglobs
    if jump is None: 
        jump = len(args.edges)-1
        if args.fixcent: jump -= 1

    #velocities
    paramdict['vts']  = params[start:start + jump]
    paramdict['v2ts'] = params[start + jump:start + 2*jump]
    paramdict['v2rs'] = params[start + 2*jump:start + 3*jump]

    #get sigma values and fill in center bin if necessary
    if args.disp: 
        if args.fixcent: sigjump = jump+1
        else: sigjump = jump
        end = start + 3*jump + sigjump
        paramdict['sig'] = params[start + 3*jump:end]
    else: end = start + 3*jump

    if hasattr(args, 'mix') and args.mix:
        paramdict['Q'] = params[end]
        paramdict['M'] = params[end+1]
        paramdict['lnV'] = params[end+2]

    return paramdict

def smoothing(array, weight=1):
    '''
    A penalty function for encouraging smooth arrays. 
    
    For each bin, it computes the average of the bins to the left and right and
    computes the chi squared of the bin with that average. It assumes 0 at the
    left edge and repeats the final value at the right edge. 

    Args:
        array (`np.ndarray`_):
            Array to be analyzed for smoothness.
        weight (:obj:`float`, optional):
            Normalization factor for resulting chi squared value

    Returns:
        :obj:`float`: Chi squared value that serves as a measurement for how
        smooth the array is, normalized by the weight.
    '''

    edgearray = np.array([0, *array,array[-1]]) #bin edges
    avgs = (edgearray[:-2] + edgearray[2:])/2 #average of each bin
    chisq = (avgs - array)**2 / np.abs(array) #chi sq of each bin to averages
    chisq[~np.isfinite(chisq)] = 0 #catching nans
    return chisq.sum() * weight

def trunc(q, mean, std, left, right):
    '''
    Wrapper function for the :func:`scipy.stats.truncnorm.ppf: function to make
    defining edges easier. 
    
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
    '''

    a,b = (left-mean)/std, (right-mean)/std #transform to z values
    return stats.truncnorm.ppf(q,a,b,mean,std)

def dynprior(params, args, gaussprior=False):
    '''
    Prior transform for :class:`dynesty.NestedSampler` fit. 
    
    Defines the prior volume for the supplied set of parameters. Uses uniform
    priors by default but can switch to truncated normal if specified.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        gaussprior (:obj:`bool`, optional):
            Flag to use the (experimental) truncated normal priors.

    Returns:
        :obj:`tuple`: Tuple of parameter values transformed into the prior
        volume.
    '''

    paramdict = unpack(params,args)

    #attempt at smarter posteriors, currently super slow though
    #truncated gaussian prior around guess values
    if gaussprior and args.guess is not None:
        guessdict = unpack(args.guess,args)
        incp  = trunc(paramdict['inc'],guessdict['incg'],2,guessdict['incg']-5,guessdict['incg']+5)
        pap   = trunc(paramdict['pa'],guessdict['pag'],10,0,360)
        pabp  = 180 * paramdict['pab']
        vsysp = trunc(paramdict['vsys'],guessdict['vsysg'],1,guessdict['vsysg']-5,guessdict['vsysg']+5)
        vtsp  = trunc(paramdict['vts'],guessdict['vtsg'],50,0,400)
        v2tsp = trunc(paramdict['v2ts'],guessdict['v2tsg'],50,0,200)
        v2rsp = trunc(paramdict['v2rs'],guessdict['v2rsg'],50,0,200)

    else:
        #uniform transformations to cover full angular range
        incp = 85 * paramdict['inc']
        pap = 360 * paramdict['pa']
        pabp = 180 * paramdict['pab']

        #uniform guesses for reasonable values for velocities
        vsysp = (2*paramdict['vsys'] - 1) * 100
        vtsp = 400 * paramdict['vts']
        v2tsp = 200 * paramdict['v2ts']
        v2rsp = 200 * paramdict['v2rs']
        if args.disp: sigp = 300 * paramdict['sig']
        if args.mix:
            Qp = paramdict['Q']
            Mp = (2*paramdict['M'] - 1) * 1000
            lnVp = (2*paramdict['lnV'] - 1) * 20

    #reassemble params array
    repack = [incp,pap,pabp,vsysp]

    #do centers if desired
    if args.nglobs == 6: 
        if gaussprior:
            xcp = stats.norm.ppf(xc,xcg,5)
            ycp = stats.norm.ppf(yc,ycg,5)
        else:
            xc = (2*xc - 1) * 20
            yc = (2*yc - 1) * 20
        repack += [xcp,ycp]

    #repack all the velocities
    repack += [*vtsp, *v2tsp, *v2rsp]
    if args.disp: repack += [*sigp]
    if args.mix:  repack += [Qp, Mp, lnVp]
    return repack

def loglike(params, args, squared=False):
    '''
    Log likelihood for :class:`dynesty.NestedSampler` fit. 
    
    Makes a model based on current parameters and computes a chi squared with
    the original data.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        squared (:obj:`bool`, optional):
            Whether to compute the chi squared against the square of the
            dispersion profile or not. 

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''
    if args.mix: return mixlike(params, args)

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
    if args.vel_ivar is not None: llike *= args.vel_ivar
    llike = -.5*np.ma.sum(llike)

    #add in penalty for non smooth rotation curves
    llike = llike - smoothing(paramdict['vts'],  args.weight) \
                  - smoothing(paramdict['v2ts'], args.weight) \
                  - smoothing(paramdict['v2rs'], args.weight)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq with squared sigma or not
        if squared:
            sigdata = args.sig_phys2
            sigdataivar = args.sig_phys2_ivar
            siglike = (sigmodel**2 - sigdata)**2
        else:
            sigdata = np.sqrt(args.sig_phys2)
            sigdataivar = np.sqrt(args.sig_phys2_ivar)
            siglike = (sigmodel - sigdata)**2

        if sigdataivar is not None: siglike *= sigdataivar
        llike = llike - .5*np.ma.sum(siglike)
        llike = llike - smoothing(paramdict['sig'], args.weight)

    return llike

def mixlike(params, args):
    paramdict = unpack(params, args)

    #make velocity and dispersion models
    velmodel, sigmodel = bisym_model(args, paramdict)

    #mask border if necessary
    if args.bordermask is not None:
        velmodel = np.ma.array(velmodel, mask=args.bordermask)
        if sigmodel is not None:
            sigmodel = np.ma.array(sigmodel, mask=args.bordermask)

    #compute chi squared value with error if possible
    goodvar = 1/args.vel_ivar if args.vel_ivar is not None else np.ones_like(args.vel)
    badvar = np.exp(paramdict['lnV']) + 1/goodvar
    goodlike = -.5 * ((velmodel - args.vel)**2 / goodvar + np.log(goodvar)) 
    badlike = -.5 * ((velmodel - args.vel)**2 / badvar + np.log(badvar)) 
    goodlike += np.log(paramdict['Q'])
    badlike  += np.log(1 - paramdict['Q'])
    llike = np.logaddexp(goodlike, badlike)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq with squared sigma or not
        sigdata = np.sqrt(args.sig_phys2)
        goodsigvar = np.sqrt(args.sig_phys2_ivar) if args.sig_phys2_ivar is not None else np.ones_like(sigdata)
        badsigvar = np.exp(paramdict['lnV']) + 1/goodsigvar
        goodsiglike = -.5 * ((sigmodel - sigdata)**2 / goodsigvar + np.log(goodsigvar))
        badsiglike = -.5 * ((sigmodel - sigdata)**2 / badsigvar + np.log(badvar))
        goodsiglike += np.log(paramdict['Q'])
        badsiglike  += np.log(1 - paramdict['Q'])
        siglike = np.logaddexp(goodsiglike, badsiglike)
        llike = np.logaddexp(llike, siglike)

    llike = np.sum(llike)
    #add in penalty for non smooth rotation curves
    llike = llike - smoothing(paramdict['vts'],  args.weight) \
                  - smoothing(paramdict['v2ts'], args.weight) \
                  - smoothing(paramdict['v2rs'], args.weight)
    if sigmodel is not None: 
        llike = llike - smoothing(paramdict['sig'], args.weight)

    return llike

def fit(plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10', nbins=None,
        cores=10, maxr=None, cen=False, weight=10, smearing=True, points=500,
        stellar=False, root=None, verbose=False, fixcent=True, disp=True, mix=True):
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
        fixcent (:obj:`bool:, optional):
            Flag to require the center bin of the velocity field to be 0.
        disp (:obj:`bool`, optional):
            Flag for whether to fit the velocity dispersion profile as well.

    Returns:
        :class:`dynesty.NestedSampler`: Sampler from `dynesty` containing
        information from the fit.    
    '''

    #mock galaxy using stored values
    if plate == 0:
        mock = np.load('mockparams.npy', allow_pickle=True)[ifu]
        print('Using mock:', mock['name'])
        params = [mock['inc'], mock['pa'], mock['pab'], mock['vsys'], mock['vts'], mock['v2ts'], mock['v2rs'], mock['sig']]
        args = Kinematics.mock(56,*params)
        cnvfftw = ConvolveFFTW(args.spatial_shape)
        smeared = smear(args.remap('vel'), args.beam_fft, beam_fft=True, sig=args.remap('sig'), sb=args.remap('sb'), cnvfftw=cnvfftw)
        args.sb  = args.bin(smeared[0])
        args.vel = args.bin(smeared[1])
        args.sig = args.bin(smeared[2])
        args.fwhm  = 2.44

    #get info on galaxy and define bins and starting guess
    else:
        if stellar:
            args = MaNGAStellarKinematics.from_plateifu(plate, ifu, daptype=daptype, dr=dr,
                                                        ignore_psf=not smearing, cube_path=root,
                                                        maps_path=root)
        else:
            args = MaNGAGasKinematics.from_plateifu(plate, ifu, line='Ha-6564', daptype=daptype,
                                                    dr=dr, ignore_psf=not smearing, cube_path=root,
                                                    maps_path=root)

    #set basic parameters for galaxy
    args.setnglobs(6) if cen else args.setnglobs(4)
    args.setfixcent(fixcent)
    args.setweight(weight)
    args.setdisp(disp)
    args.setmix(mix)

    #do a quick fit to get inclination to set bin edges
    if nbins is not None: args.setedges(nbins, nbin=True, maxr=maxr)
    else:
        vmax, inc, pa, h, vsys = args.getguess()
        args.setedges(inc, maxr=maxr)

    #define a variable for speeding up convolutions
    #has to be a global because multiprocessing can't pickle cython
    global conv
    conv = ConvolveFFTW(args.spatial_shape)

    #starting positions for all parameters based on a quick fit
    args.clip()
    theta0 = args.getguess()
    ndim = len(theta0)

    #adjust dimensions accordingly
    if fixcent: ndim -= 3
    if disp: ndim += len(args.edges)-1
    if mix: ndim += 3
    print(f'{len(args.edges)-1} radial bins')
    
    #open up multiprocessing pool if needed
    if cores > 1:
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    #dynesty sampler with periodic pa and pab
    sampler = dynesty.NestedSampler(loglike, dynprior, ndim , pool=pool,
            periodic=[1,2], nlive=points,
            ptform_args = [args], logl_args = [args], verbose=verbose)
    sampler.run_nested()

    if pool is not None: pool.close()
    return sampler
