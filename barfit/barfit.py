#!/usr/bin/env python

'''
TODO:
fit dispersion
fit stars
binning scheme
'''

import numpy as np
import matplotlib.pyplot as plt
import emcee, sys, ptemcee, corner, dynesty, pickle, argparse
from astropy.io import fits
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import leastsq
from scipy.signal import convolve
from beam_smearing import apply_beam_smearing as smear
from scipy import stats
from galaxy import Galaxy

def polar(x,y,i,pa): 
    '''
    Transform x,y coordinates from Cartesian to polar coordinates rotated at
    angle pa and inclination i. Returns radial coordinate r and aziumuthal
    coordinate pa. All angles in radians.  
    '''

    yd = (x*np.cos(pa) + y*np.sin(pa))
    xd = (y*np.cos(pa) - x*np.sin(pa))/np.cos(i) 
    r = np.sqrt(xd**2 + yd**2) 
    th = (np.pi/2 - np.arctan2(yd,xd)) % (np.pi*2)
    return r, th 

def rotcurveeval(x,y,vmax,inc,pa,h,vsys=0,xc=0,yc=0):
    '''
    Evaluate a simple tanh rotation curve with asymtote vmax, inclination inc
    in degrees, position angle pa in degrees, rotation scale h, systematic
    velocity vsys, and x and y offsets xc and yc. Returns array in same shape
    as input x andy.
    '''

    inc, pa = np.radians([inc,pa])
    r,th = polar(x-xc,y-yc,inc,pa)
    model = -vmax * np.tanh(r/h) * np.cos(th) * np.sin(inc) + vsys
    return model

def getguess(vf,x=None,y=None,ivar=None,er=None,edges=None):
    '''
    Generate a set of guess parameters for a given velocity field vf with ivar
    by fitting it with a simple rotation curve using least squares and
    sampling the resulting cuve to generate values in bins specified by edges.
    Requires x,y, and elliptical radius coordinates er as well. Returns an
    array in format [inc,pa,pab,vsys] + [vt,v2t,v2r]*(number of bins). Inc and
    pa in degrees. Assumes pab = pa.
    '''

    if isinstance(vf, Galaxy):
        vf,x,y,ivar,er,edges = (vf.vf, vf.x, vf.y, vf.ivar, vf.er, vf.edges)
    elif x is None or y is None or ivar is None or er is None or edges is None:
        raise ValueError('Must give a Galaxy object or all other parameters.')

    #define a minimization function and feed it to simple leastsquares
    minfunc = lambda params,vf,x,y,e: np.array((vf - \
            rotcurveeval(x,y,*params))/e).flatten()
    vmax,inc,pa,h,vsys = leastsq(minfunc, (200,45,180,3,0), 
            args = (vf,x,y,ivar**-.5))[0]

    #check and fix signs if galaxy was fit upside down
    if np.product(np.sign([vmax,inc,h])) < 0: pa += 180
    pa %= 360
    vmax,inc,h = np.abs([vmax,inc,h])

    #generate model of vf and start assembling array of guess values
    model = rotcurveeval(x,y,vmax,inc,pa,h,vsys)
    guess = [inc,pa,pa,vsys,0,0,0]

    #iterate through bins and get vt value for each bin, dummy value for others
    vts = np.zeros(len(edges)-1)
    v2ts = np.array([10] * len(edges-1))
    v2rs = np.array([10] * len(edges-1))
    for i in range(1,len(edges)-1):
        cut = (er > edges[i]) * (er < edges[i+1])
        vts[i] = np.max(model[cut])
        guess += [vts[i], v2ts[i], v2rs[i]]
    
    #clean and return
    guess = np.array(guess)
    guess[np.isnan(guess)] = 100
    return guess

def barmodel(args,inc,pa,pab,vsys,vts,v2ts,v2rs,xc=0,yc=0):
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
    r, th = polar(args.x-xc,args.y-yc,inc,pa)
    r /= (r.max()/args.er.max())

    #interpolate velocity values for all r 
    bincents = (args.edges[:-1] + args.edges[1:])/2
    vtvals = np.interp(r,bincents,vts)
    v2tvals = np.interp(r,bincents,v2ts)
    v2rvals = np.interp(r,bincents,v2rs)

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    model = vsys + np.sin(inc) * (vtvals*np.cos(th) - v2tvals*np.cos(2*(th-pab))*np.cos(th)- v2rvals*np.sin(2*(th-pab))*np.sin(th))
    if args.psf is not None:
        #shape = [int(np.sqrt(len(psf)))]*2
        #for a in psf: a.reshape(shape)
        psf, flux, sigma, mask = psf
        model = smear(model, args.psf, args.flux, args.sigma, mask=args.mask)[1]
    return model

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

class FitArgs(Galaxy):
    '''
    Extension of Galaxy class to carry around a few more useful variables when
    performing dynesty fit. 
    '''

    def setnglobs(self, nglobs):
        '''
        Set number of global variables in fit.
        '''

        self.nglobs = nglobs

    def setweight(self, weight):
        '''
        Set weight to assign to smoothness of rotation curves in fit.
        '''

        self.weight = weight

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

def dynprior(params,args):
    '''
    Prior transform for dynesty fit. Takes in standard params and args and
    defines a prior volume for all of the relevant fit parameters. At this
    point, all of the prior transformations are uniform and pretty
    unintelligent. Returns parameter prior transformations.  
    '''

    inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args.nglobs)

    #if args.guess is not None:
    #    incg,pag,pabg,vsysg,xcg,ycg,vtsg,v2tsg,v2rsg = unpack(args.guess,args.nglobs)
    #    incp = stats.norm.ppf(inc,incg,10)
    #    pap = stats.norm.ppf(pa,pag,20)
    #    pabp = stats.norm.ppf(pab,pabg,45)

    #uniform transformations to cover full angular range
    incp = 90 * inc
    pap = 360 * pa
    pabp = 360 * pab

    #uniform guesses for reasonable values for velocities
    vsysp = (2*vsys - 1) * 50
    vtsp = 400 * vts
    v2tsp = 200 * v2ts
    v2rsp = 200 * v2rs

    if args.nglobs == 6: 
        xc = (2*xc - 1) * 20
        yc = (2*yc - 1) * 20

    #reassemble params array
    repack = [incp,pap,pabp,vsysp]
    if args.nglobs == 6: repack += [xc,yc]
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
    llike = -.5*np.sum((vfmodel - args.vf)**2 * args.ivar) #chisq
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

def barfit(plate,ifu, nbins=10, cores=20, walkers=100, steps=1000, maxr=1.5,
        ntemps=None, cen=False,start=False,dyn=True,weight=10,smearing=True):
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

    #get info on galaxy and define bins and starting guess
    args = FitArgs(plate,ifu)
    args.setnglobs(6) if cen else args.setnglobs(4)
    args.makeedges(nbins, maxr)
    args.setweight(weight)
    theta0 = args.getguess()

    #open up multiprocessing pool if needed
    if cores > 1 and not ntemps: pool = mp.Pool(cores)
    else: pool = None

    #choose the appropriate sampler and starting positions based on user input
    if dyn:
        #dynesty sampler with periodic pa and pab, reflective inc
        sampler = dynesty.NestedSampler(loglike,dynprior,len(theta0),pool=pool,
                queue_size=cores, periodic = [1,2], reflective = [0], 
                ptform_args = [args], logl_args = [args])
        sampler.run_nested(print_func=None)
    
    else:
        if ntemps:
            #parallel tempered MCMC using somewhat outdated ptemcee
            pos = 1e-4*np.random.randn(ntemps,walkers,len(theta0)) + theta0
            sampler = ptemcee.Sampler(walkers, len(theta0), loglike, logprior, loglargs=[args], logpargs=[args], threads=cores, ntemps=ntemps)

        else:
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
        for i, result in enumerate(tqdm(sampler.sample(pos, iterations=steps),
                total=steps, leave=True, dynamic_ncols=True)):
            pass

    if cores > 1 and not ntemps: pool.close()
    return sampler

