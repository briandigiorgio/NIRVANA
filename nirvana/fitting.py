#!/usr/bin/env python

'''
TODO:
fit dispersion
fit stars
binning scheme
fake data
spekkens code pitfalls
'''

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

def bisym_model(args,paramdict,plot=False):
    '''
    Evaluate a nonaxisymmetric velocity field model taken from Leung
    (2018)/Spekkens & Sellwood (2007) at given x and y coordinates according to
    supplied bin edges in radial coordinate er by interpolating velocity values
    between edges. Needs tangential velocity vt, 2nd order tangential and
    radial velocities v2t and v2r, inclination inc and position angle pa in
    deg, bar position angle pab in deg, systemic velocity vsys, and x and y
    offsets xc and yc. Returns evaluated model in same shape as x and y. 
    '''

    #convert angles to polar and normalize radial coorinate
    inc,pa,pab = np.radians([paramdict['inc'],paramdict['pa'],paramdict['pab']])
    r, th = projected_polar(args.grid_x-paramdict['xc'],args.grid_y-paramdict['yc'],pa,inc)
    r /= args.reff

    if args.fixcent:
        vts  = np.insert(paramdict['vts'],  0, 0)
        v2ts = np.insert(paramdict['v2ts'], 0, 0)
        v2rs = np.insert(paramdict['v2rs'], 0, 0)
    else:
        vts  = paramdict['vts']
        v2ts = paramdict['v2ts']
        v2rs = paramdict['v2rs']

    #interpolate velocity values for all r 
    bincents = (args.edges[:-1] + args.edges[1:])/2
    vtvals  = np.interp(r,bincents,vts)
    v2tvals = np.interp(r,bincents,v2ts)
    v2rvals = np.interp(r,bincents,v2rs)

    if args.disp: 
        sigmodel = np.interp(r,bincents,paramdict['sig'])
        sb = args.remap('sb')
    else: 
        sigmodel = None
        sb = None

    try: conv
    except: conv = None

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    velmodel = paramdict['vsys']+ np.sin(inc) * (vtvals*np.cos(th) - v2tvals*np.cos(2*(th-pab))*np.cos(th)- v2rvals*np.sin(2*(th-pab))*np.sin(th))
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
    Now a dictionary!
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
        paramdict['sig'] = params[start + 3*jump:start + 3*jump + sigjump]

    return paramdict

def smoothing(array, weight):
    '''
    A penalty function for encouraging smooth rotation curves. Computes a
    rolling average of the curve by taking the average of the values of the
    bins on either side of a given bin (but not the bin itself) and then
    computes a chisq between that array of averages and the current rotation
    curve. The weight is how much to scale up the final result to adjust how
    important smoothness is compared to fitting the data. It uses 0 at the left
    edge and repeats the final value at the right edge.  
    '''

    edgearray = np.array([0,*array,array[-1]])
    avgs = (edgearray[:-2] + edgearray[2:])/2
    chisq = (avgs - array)**2 / np.abs(array)
    chisq[~np.isfinite(chisq)] = 0
    return chisq.sum() * weight

def trunc(q,mean,std,left,right):
    '''
    Helper function for the truncated normal distribution. Returns the value
    at quantile q for a truncated normal distribution with specified mean,
    std, and left/right bounds.
    '''

    a,b = (left-mean)/std, (right-mean)/std
    return stats.truncnorm.ppf(q,a,b,mean,std)

def dynprior(params,args,gaussprior=False):
    '''
    Prior transform for dynesty fit. Takes in standard params and args and
    defines a prior volume for all of the relevant fit parameters. At this
    point, all of the prior transformations are uniform and pretty
    unintelligent. Returns parameter prior transformations.  
    '''

    paramdict = unpack(params,args)

    #attempt at smarter posteriors, currently super slow though
    #truncated gaussian prior around guess values
    if gaussprior and args.guess is not None:
        guessdict = unpack(args.guess,args)
        incp  = trunc(paramdict['inc'],guessdict['incg'],2,guessdict['incg']-5,guessdict['incg']+5)
        pap   = trunc(paramdict['pa'],guessdict['pag'],10,0,360)
        #pabp  = trunc(pab,pabg,45,0,180)
        pabp = 180 * paramdict['pab']
        vsysp = trunc(paramdict['vsys'],guessdict['vsysg'],1,guessdict['vsysg']-5,guessdict['vsysg']+5)
        vtsp  = trunc(paramdict['vts'],guessdict['vtsg'],50,0,400)
        v2tsp = trunc(paramdict['v2ts'],guessdict['v2tsg'],50,0,200)
        v2rsp = trunc(paramdict['v2rs'],guessdict['v2rsg'],50,0,200)

    else:
        #uniform transformations to cover full angular range
        incp = 90 * paramdict['inc']
        pap = 360 * paramdict['pa']
        pabp = 180 * paramdict['pab']

        #uniform guesses for reasonable values for velocities
        vsysp = (2*paramdict['vsys']- 1) * 20
        vtsp = 400 * paramdict['vts']
        v2tsp = 200 * paramdict['v2ts']
        v2rsp = 200 * paramdict['v2rs']
        if args.disp: sigp = 200 * paramdict['sig']

    #reassemble params array
    repack = [incp,pap,pabp,vsysp]

    if args.nglobs == 6: 
        #xc = (2*xc - 1) * 20
        #yc = (2*yc - 1) * 20
        xcp = stats.norm.ppf(xc,xcg,5)
        ycp = stats.norm.ppf(yc,ycg,5)
        repack += [xcp,ycp]

    repack += [*vtsp,*v2tsp,*v2rsp]
    if args.disp: repack += [*sigp]
    return repack

def loglike(params, args):
    '''
    Log likelihood function for dynesty. Makes a model of the velocity
    field with current parameter vales and performs a chi squared on it across
    the whole vf weighted by ivar to get a log likelihood value. 
    '''
    paramdict = unpack(params,args)

    #make vf model and perform chisq
    velmodel, sigmodel = bisym_model(args,paramdict)

    #mask border if necessary
    if args.bordermask is not None:
        velmodel = np.ma.array(velmodel, mask=args.bordermask)
        if sigmodel is not None:
            sigmodel = np.ma.array(sigmodel, mask=args.bordermask)

    llike = (velmodel - args.vel)**2
    if args.vel_ivar is not None: llike *= args.vel_ivar
    llike = -.5*np.ma.sum(llike)

    #smoothing of rotation curves
    llike = llike - smoothing(paramdict['vts'],args.weight) - smoothing(paramdict['v2ts'],args.weight) - smoothing(paramdict['v2rs'],args.weight)

    #add in sigma model if applicable
    if sigmodel is not None:
        siglike = (sigmodel - args.sig)**2
        if args.sig_ivar is not None: siglike *= args.sig_ivar
        llike = llike - .5*np.ma.sum(siglike)
        llike = llike - smoothing(paramdict['sig'],args.weight)

    return llike

def logpost(params, args):
    '''
    Log posterior for emcee/ptemcee fit. Really just gets prior and likelihood
    values, nothing fancy. 
    '''

    lprior = logprior(params,args)
    if not np.isfinite(lprior):
        return -np.inf
    llike = loglike(params, args)
    return lprior + llike

def fit(plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10', nbins=None, cores=10, maxr=None, cen=False, weight=10, smearing=True, points=500, stellar=False, root=None, verbose=False, fixcent=True, disp=True):
    '''
    Main function for velocity field fitter. Takes a given plate and ifu and
    fits a nonaxisymmetric velocity field with nbins number of radial bins
    based on Spekkens & Sellwood (2007) and Leung (2018). Will run normal MCMC
    with emcee with walkers number of walkers and steps number of steps by
    default, but if number of walker temperatures ntemps is given, it will use
    ptemcee instead for parallel tempered MCMC. If dyn is set to true, dynesty
    will be used instead (this is the best choice). Can specify number of cores
    with cores, can fit center of velocity field if cen is set to true, and can
    specify starting walker positions with start. Returns a sampler from the
    chosen package.  
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

    #do a quick fit to get inclination to set bin edges
    if nbins is not None: args.setedges(nbins, nbin=True, maxr=maxr)
    else:
        vmax, inc, pa, h, vsys = args.getguess()
        args.setedges(inc, maxr=maxr)

    #define a variable for speeding up convolutions
    #has to be a global because multiprocessing can't pickle cython
    global conv
    conv = ConvolveFFTW(args.spatial_shape)

    #starting positions for all parameters based on quick fit
    theta0 = args.getguess()
    ndim = len(theta0)

    #adjust dimensions accordingly
    if fixcent: ndim -= 3
    if disp: ndim += len(args.edges)-1
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
    sampler.run_nested()#maxiter=1000)

    if pool is not None: pool.close()
    return sampler
