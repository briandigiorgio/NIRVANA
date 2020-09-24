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

try:
    import emcee
except:
    emcee = None

try:
    import ptemcee
except:
    ptemcee = None

import dynesty

from .models.beam import smear
from .data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from .data.kinematics import Kinematics
from .data.fitargs import FitArgs

from .models.geometry import projected_polar

def barmodel(args,inc,pa,pab,vsys,vts,v2ts,v2rs,xc=0,yc=0,plot=False):
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
    inc,pa,pab = np.radians([inc,pa,pab])
    r, th = projected_polar(args.grid_x-xc,args.grid_y-yc,pa,inc)
    r /= args.reff

    if args.fixcent:
        vts  = np.insert(vts,  0, 0)
        v2ts = np.insert(v2ts, 0, 0)
        v2rs = np.insert(v2rs, 0, 0)
        

    #interpolate velocity values for all r 
    bincents = (args.edges[:-1] + args.edges[1:])/2
    vtvals  = np.interp(r,bincents,vts)
    v2tvals = np.interp(r,bincents,v2ts)
    v2rvals = np.interp(r,bincents,v2rs)

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    model = vsys + np.sin(inc) * (vtvals*np.cos(th) - v2tvals*np.cos(2*(th-pab))*np.cos(th)- v2rvals*np.sin(2*(th-pab))*np.sin(th))
    if args.beam_fft is not None:
        model = smear(model, args.beam_fft, beam_fft=True)[1]
    if plot:
        return args.remap_data(np.ma.MaskedArray(args.bin(model), mask=args.vel_mask), masked=True)
    return np.ma.MaskedArray(args.bin(model), mask=args.vel_mask)

def unpack(params, nglobs):
    '''
    Utility function to carry around a bunch of values in the Bayesian fit.
    Should probably be a class.
    '''

    #global parameters with and without center
    xc, yc = [0,0]
    if nglobs == 4: inc,pa,pab,vsys = params[:nglobs]
    elif nglobs == 6: inc,pa,pab,vsys,xc,yc = params[:nglobs]

    #velocities
    vts  = params[nglobs::3]
    v2ts = params[nglobs+1::3]
    v2rs = params[nglobs+2::3]
    return inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs

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

    inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args.nglobs)

    #attempt at smarter posteriors, currently super slow though
    #truncated gaussian prior around guess values
    if gaussprior and args.guess is not None:
        incg,pag,pabg,vsysg,xcg,ycg,vtsg,v2tsg,v2rsg = unpack(args.guess,args.nglobs)
        incp  = trunc(inc,incg,2,incg-5,incg+5)
        pap   = trunc(pa,pag,10,0,360)
        #pabp  = trunc(pab,pabg,45,0,180)
        pabp = 180 * pab
        vsysp = trunc(vsys,vsysg,1,vsysg-5,vsysg+5)
        vtsp  = trunc(vts,vtsg,50,0,400)
        v2tsp = trunc(v2ts,v2tsg,50,0,200)
        v2rsp = trunc(v2rs,v2rsg,50,0,200)

    else:
        #uniform transformations to cover full angular range
        incp = 90 * inc
        pap = 360 * pa
        pabp = 180 * pab

        #uniform guesses for reasonable values for velocities
        vsysp = (2*vsys - 1) * 20
        vtsp = 400 * vts
        v2tsp = 200 * v2ts
        v2rsp = 200 * v2rs

    repack = [incp,pap,pabp,vsysp]

    if args.nglobs == 6: 
        #xc = (2*xc - 1) * 20
        #yc = (2*yc - 1) * 20
        xcp = stats.norm.ppf(xc,xcg,5)
        ycp = stats.norm.ppf(yc,ycg,5)
        repack += [xcp,ycp]

    #reassemble params array
    for i in range(len(vts)): repack += [vtsp[i],v2tsp[i],v2rsp[i]]
    return repack

def logprior(params, args):
    '''
    Log prior function for emcee/ptemcee. Pretty simple uniform priors on
    everything to cover their full range of reasonable values. 
    '''

    inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args.nglobs)
    #uniform priors on everything with guesses for velocities
    if inc < 0 or inc > 90 or abs(xc) > 20 or abs(yc) > 20 or abs(vsys) > 50 or (vts > 400).any() or (v2ts > 400).any() or (v2rs > 400).any():# or (vts < 0).any():
        return -np.inf
    return 0
    
def loglike(params, args):
    '''
    Log likelihood function for emcee/ptemcee. Makes a model of the velocity
    field with current parameter vales and performs a chi squared on it across
    the whole vf weighted by ivar to get a log likelihood value. 
    '''
    inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args.nglobs)

    #make vf model and perform chisq
    vfmodel = barmodel(args,inc,pa,pab,vsys,vts,v2ts,v2rs,xc,yc)
    # vfmodel is masked
    llike = (vfmodel - args.vel)**2
    if args.vel_ivar is not None:
        llike *= args.vel_ivar
    llike = -.5*np.ma.sum(llike) #chisq
    #llike -= args.weight * (smoothing(vts) - smoothing(v2ts) - smoothing(v2rs))
    llike = llike - smoothing(vts,args.weight) - smoothing(v2ts,args.weight) - smoothing(v2rs,args.weight)
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

def barfit(plate, ifu, daptype='HYB10-MILESHC-MASTARHC', dr='MPL-9', nbins=10, cores=20,
           walkers=100, steps=1000, maxr=1.5, ntemps=None, cen=False, start=False, dyn=True,
           weight=10, smearing=True, points=500, stellar=False, root=None, verbose=False,
           fixcent=True):
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

    #mock galaxy using Andrew's values for 8078-12703
    if plate == 0 and ifu == 0 :
        mockparams = np.load('mockparams.npy', allow_pickle=True)
        args = Kinematics.mock(55,*mockparams)

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

    args.setnglobs(6) if cen else args.setnglobs(4)
    args.setfixcent(fixcent)
    args.setedges(nbins, maxr)
    args.setweight(weight)
    theta0 = args.getguess()
    ndim = len(theta0)
    if fixcent: ndim -= 3

    #open up multiprocessing pool if needed
    if cores > 1 and not ntemps:
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    #choose the appropriate sampler and starting positions based on user input
    if dyn:
        #dynesty sampler with periodic pa and pab
        sampler = dynesty.NestedSampler(loglike, dynprior, ndim , pool=pool,
                periodic=[1,2], nlive=points,# queue_size=cores, 
                ptform_args = [args], logl_args = [args], verbose=verbose)
        sampler.run_nested()
    
    else:
        if ntemps:
            if ptemcee is None:
                raise ImportError('ptemcee is not available.  Run pip install ptemcee and rerun.')
            #parallel tempered MCMC using somewhat outdated ptemcee
            pos = 1e-4*np.random.randn(ntemps,walkers,len(theta0)) + theta0
            sampler = ptemcee.Sampler(walkers, len(theta0), loglike, logprior, loglargs=[args], logpargs=[args], threads=cores, ntemps=ntemps)

        else:
            if emcee is None:
                raise ImportError('emcee is not available.  Run pip install emcee and rerun.')
            #normal MCMC with emcee
            if type(start) != bool:
                #set starting positions if given
                #useful for switching samplers in the middle of a run
                if start.ndim == 3: pos = start[:,-1,:]
                elif start.ndim == 4: pos = start[0,:,-1,:]
                elif start.shape == (walkers,len(theta0)): pos == start
                else: raise Exception('Bad starting position input')

            #set starting positions as normal
            else: pos = np.array([theta0 + 1e-4*np.random.randn(len(theta0)) for i in range(walkers)])
            sampler = emcee.EnsembleSampler(walkers, len(theta0), logpost, args=[args], pool=pool)

        #run MCMC
        _iter = sampler.sample(pos, iterations=steps)
        if tqdm is not None:
            _iter = tqdm(_iter, total=steps, leave=True, dynamic_ncols=True)
        for i, result in enumerate(_iter):
            pass

    if cores > 1 and not ntemps: pool.close()
    return sampler
