#!/usr/bin/env python

'''
TODO:
smooth profiles
fit dispersion
beam smearing/psf
fit stars
binning scheme
'''

import numpy as np
import matplotlib.pyplot as plt
import emcee, sys, ptemcee, corner, dynesty, pickle
from astropy.io import fits
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import leastsq, least_squares
from scipy.signal import convolve

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

def getvfinfo(plate, ifu, path = None, ivar = True, xy = True, disp = True, ell=True, stellar = False, spx = True, flat=False):
	'''
	Get relevant Ha info for a MaNGA galaxy. Assumes graymalkin file system.
	Includes flags to return ivar, SPX XY coords, dispersion, and elliptical
	coordinates and return in that order. Also has flags to return stellar
	instead of Ha, use HYB10 binning instead of SPX, and flatten returned
	arrays.
	'''

	#set desired extension
	if spx:
		bintype = 'SPX'
	else:
		bintype = 'HYB10'

	#set correct path to files and load, defaults to graymalkin
	if path == None:
		path = '/data/manga/spectro/analysis/MPL-9/%s-MILESHC-MASTARHC'%bintype
	f = fits.open('%s/%s/%s/manga-%s-%s-MAPS-%s-MILESHC-MASTARHC.fits.gz'
			% (path,plate,ifu,plate,ifu,bintype))

	#get stellar vf, flux, mask
	if stellar:
		vf = f['STELLAR_VEL'].data
		flux = f['SPX_MFLUX'].data
		m = f['STELLAR_VEL_MASK'].data != 0

		#get dispersion and error, mask spaxels with unrealistically low error
		if disp:
			sigma = f['STELLAR_SIGMA'].data
			sigmaivar = f['STELLAR_SIGMA_IVAR'].data
			inst = f['STELLAR_SIGMACORR'].data[0]
			m += (sigmaivar > inst)
			sigma = np.ma.array(sigma, mask = m)
			sigmaivar = np.ma.array(sigmaivar, mask = m)

		#mask arrays, add to return list
		vf = np.ma.array(vf, mask = m)
		flux = np.ma.array(flux, mask = m)
		ret = [vf, flux]

		#get velocity ivar
		if ivar:
			ivar = f['STELLAR_VEL_IVAR'].data
			ivar = np.ma.array(ivar, mask = m)
			ret += [ivar]

		#get dispersion info
		if disp: ret += [sigma,sigmaivar]

	#same for halpha
	else:
		vf = f['EMLINE_GVEL'].data[23]
		flux = f['EMLINE_GFLUX'].data[23]
		m = f['EMLINE_GVEL_MASK'].data[23] != 0

		if disp:
			sigma = f['EMLINE_GSIGMA'].data[23]
			sigmaivar = f['EMLINE_GSIGMA_IVAR'].data[23]
			inst = f['EMLINE_INSTSIGMA'].data[23]
			m += (sigmaivar > inst)
			sigma = np.ma.array(sigma, mask = m)
			sigmaivar = np.ma.array(sigmaivar, mask = m)
			
		vf = np.ma.array(vf, mask = m)
		flux = np.ma.array(flux, mask = m)
		ret = [vf, flux]

		if ivar:
			ivar = f['EMLINE_GVEL_IVAR'].data[23]
			ivar = np.ma.array(ivar, mask = m)
			ret += [ivar]
		if disp: ret += [sigma,sigmaivar]

	#on sky xy coordinates for spaxels
	if xy:
		x,y = [f['SPX_SKYCOO'].data[1], f['SPX_SKYCOO'].data[0]]
		x = np.ma.array(x, mask = m)
		y = np.ma.array(y, mask = m)
		ret += [x,y]
	
	#elliptical polar coordinates
	if ell:
		er,eth = [f['SPX_ELLCOO'].data[1], f['SPX_ELLCOO'].data[3]]
		er = np.ma.array(er, mask = m)
		eth = np.ma.array(eth, mask = m)
		ret += [er,eth]

	f.close()

	#flatten output arrays (good for emcee, bad for imshow)
	if flat:
		flat = []
		for r in ret:
			flat += [r.flatten()]
		return flat

	#return everything in order of [vf,flux,ivar,sigma,sigmaivar,x,y,er,eth,fwhm]
	return ret

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

def getguess(vf,x,y,ivar,er,edges):
	'''
	Generate a set of guess parameters for a given velocity field vf with ivar
	by fitting it with a simple rotation curve using least squares and
	sampling the resulting cuve to generate values in bins specified by edges.
	Requires x,y, and elliptical radius coordinates er as well. Returns an
	array in format [inc,pa,h,vsys] + [vt,v2t,v2r]*(number of bins). Inc and
	pa in degrees. 
	'''

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
	guess = [inc,pa,h,vsys,0,0,0]

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

def bareval(x,y,vt,v2t,v2r,inc,pa,pab,vsys,xc=0,yc=0):
	'''
	Evaluate a nonaxisymmetric velocity field model taken from Leung
	(2018)/Spekkens & Sellwood (2007) at given x and y coordinates. Needs
	tangential velocity vt, 2nd order tangential and radial velocities v2t and
	v2r, inclination inc and position angle pa in deg, bar position angle pab
	in deg, systemic velocity vsys, and x and y offsets xc and yc. Returns
	evaluated model in same shape as x and y.

	Deprecated
	'''

	#convert to polar
	inc, pa, pab = np.radians([inc,pa,pab])
	r, th = polar(x-xc,y-yc,inc,pa)

	#spekkens and sellwood 2nd order (from andrew's thesis)
	return vsys + np.sin(inc) * (vt*np.cos(th) - v2t*np.cos(2*(th-pab))*np.cos(th)- v2r*np.sin(2*(th-pab))*np.sin(th))

def barmodel(x,y,er,eth,edges,inc,pa,pab,vsys,vts,v2ts,v2rs,xc=0,yc=0):
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
	r, th = polar(x-xc,y-yc,inc,pa)
	r /= (r.max()/er.max())

	#interpolate velocity values for all r 
	bincents = (edges[:-1] + edges[1:])/2
	vtvals = np.interp(r,bincents,vts)
	v2tvals = np.interp(r,bincents,v2ts)
	v2rvals = np.interp(r,bincents,v2rs)

	#spekkens and sellwood 2nd order vf model (from andrew's thesis)
	model = vsys + np.sin(inc) * (vtvals*np.cos(th) - v2tvals*np.cos(2*(th-pab))*np.cos(th)- v2rvals*np.sin(2*(th-pab))*np.sin(th))
	return model

	#model = np.zeros_like(x)
	#for i in range(len(edges)-1):
	#	cut = (er > edges[i]) * (er < edges[i+1])
	#	model[cut] = bareval(x[cut],y[cut],vts[i],v2ts[i],v2rs[i],inc,pa,pab,vsys,xc,yc)
	#return model

def unpack(params, args):
	'''
	Utility function to carry around a bunch of values in the Bayesian fit.
	Should probably be a class.
	'''

	vf, ivar, edges, er, eth, x, y, nglobs = args #galaxy data
	xc, yc = [0,0]

	#global parameters with and without center
	if nglobs == 4: inc,pa,pab,vsys = params[:nglobs]
	elif nglobs == 6: inc,pa,pab,vsys,xc,yc = params[:nglobs]

	#velocities
	vts  = params[nglobs::3]
	v2ts = params[nglobs+1::3]
	v2rs = params[nglobs+2::3]
	return vf,ivar,edges,er,eth,x,y,nglobs,inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs

def dynprior(params,args):
	'''
	Prior transform for dynesty fit. Takes in standard params and args and
	defines a prior volume for all of the relevant fit parameters. At this
	point, all of the prior transformations are uniform and pretty
	unintelligent. Returns parameter prior transformations.  
	'''
	vf,ivar,edges,er,eth,x,y,nglobs,inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args)

	#uniform transformations to cover full angular range
	incp = 90*inc
	pap = 360 * pa
	pabp = 360 * pab

	#uniform guesses for reasonable values for velocities
	vsysp = (2*vsys - 1) * 50
	vtsp = 400 * vts
	v2tsp = 200 * v2ts
	v2rsp = 200 * v2rs

	#reassemble params array
	repack = [incp,pap,pabp,vsysp]
	for i in range(len(vts)): repack += [vtsp[i],v2tsp[i],v2rsp[i]]
	return repack

def logprior(params, args):
	'''
	Log prior function for emcee/ptemcee. Pretty simple uniform priors on
	everything to cover their full range of reasonable values. 
	'''

	vf,ivar,edges,er,eth,x,y,nglobs,inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args)
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

	vf,ivar,edges,er,eth,x,y,nglobs,inc,pa,pab,vsys,xc,yc,vts,v2ts,v2rs = unpack(params,args)

	#make vf model and perform chisq
	vfmodel = barmodel(x,y,er,eth,edges,inc,pa,pab,vsys,vts,v2ts,v2rs,xc,yc)
	llike = -.5*np.sum((vfmodel - vf)**2 * ivar) #chisq
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

def barfit(plate,ifu, nbins=10, cores=20, walkers=100, steps=1000, 
		ntemps=None, cen=False,start=False,dyn=False):
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

	#get info on galaxy and define bins, combine into easy to handle list
	vf,flux,ivar,sigma,sigmaivar,x,y,er,eth = getvfinfo(plate,ifu,flat=True)
	edges = np.linspace(0,1.5,nbins+1)
	args = [vf, ivar, edges, er, eth, x, y]

	#set number of global parameters
	if cen: nglobs = 6
	else: nglobs = 4
	args += [nglobs]

	#get and print starting guesses for parameters
	theta0 = getguess(vf,x,y,ivar,er,edges)
	print(theta0)

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
				if start.ndim == 3:	pos = start[:,-1,:]
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

def cornerplot(sampler, burn=-1000, **args):
	'''
	Make a cornerplot with an emcee/ptemcee sampler. Will only look at samples
	after step burn. Takes args for corner.corner.
	'''

	if type(sampler) == np.ndarray: chain = sampler 
	else: chain = sampler.chain
	if chain.ndim == 4:
		chains = chain[:,:,burn:,:].reshape(-1,chain.shape[-1])
	elif chain.ndim == 3:
		chains = chain[:, burn:, :].reshape(-1,chain.shape[-1])
	corner.corner(chains,**args)

def chainvis(sampler, titles = ['$inc$ (deg)',r'$\phi$ (deg)',r'$\phi_b$ (deg)',r'$v_{sys}$ (km/s)'], alpha=.1, nplots=None):
	'''
	Look at all the chains of an emcee sampler in one plot to check
	convergence. Can specify number of variables to plot with nplots if there
	are too many to see well. Can set alpha of individual chains with alpha and
	titles of plots with titles.
	'''

	#get array of chains
	if type(sampler) == np.ndarray:
		chain = sampler
	else:
		chain = sampler.chain

	#reshape chain array so it works and get appropriate dimensions
	if chain.ndim == 2: chain = chain[:,:,np.newaxis]
	elif chain.ndim == 4: chain = chain.reshape(-1,chain.shape[2],chain.shape[3])
	nwalk,nstep,nvar = chain.shape
	if nplots: nvar = nplots
	if len(titles) < nvar: #failsafe if not enough titles
		titles = np.arange(nvar)

	#make a plot for each variable
	plt.figure(figsize=(4*nvar,4))
	for var in range(nvar):
		plt.subplot(1,nvar,var+1)
		plt.plot(chain[:,:,var].T, 'k-', lw = .2, alpha = alpha, rasterized=True)
		plt.xlabel(titles[var])
	plt.tight_layout()
	plt.show()

def mcmeds(sampler, burn = -1000):
	'''
	Return medians for each variable in an emcee sampler after step burn.
	'''
		
	return np.median(sampler.chain[:,burn:,:], axis = (0,1))

def dmeds(samp):
	'''
	Get median values for each variable in a dynesty sampler.
	'''

	#get samples and weights
	if type(samp)==dynesty.results.Results: res = samp
	else: res = samp.results
	samps = res.samples
	weights = np.exp(res.logwt - res.logz[-1])
	meds = np.zeros(samps.shape[1])

	#iterate through and get 50th percentile of values
	for i in range(samps.shape[1]):
		meds[i] = dynesty.utils.quantile(samps[:,i],[.5],weights)[0]
	return meds

def dcorner(samp,**args):
	'''
	Make a cornerplot of a dynesty sampler. Takes args for
	dynesty.plotting.cornerplot.
	'''

	if type(samp) == dynesty.results.Results:
		dynesty.plotting.cornerplot(samp, **args)
	else:
		dynesty.plotting.cornerplot(samp.results, **args)

def checkbins(plate,ifu,nbins):
	'''
	Make a plot to see whether the number of spaxels in each bin make sense for
	a given number of bins for a MaNGA galaxy with given plate ifu. 
	'''

	vf,flux,ivar,sigma,sigmaivar,x,y,er,eth = getvfinfo(plate,ifu)
	edges = np.linspace(0,1.5,nbins+1)[:-1]

	if nbins%2: nbins += 1
	plt.figure(figsize=(12,6))
	nrow = nbins//5
	ncol = nbins//nrow
	for i in range(len(edges)-1):
		plt.subplot(nrow,ncol,i+1)
		cut = (er>edges[i])*(er<edges[i+1])
		plt.imshow(np.ma.array(vf, mask=~cut), cmap='RdBu')

def dprofs(samp, edges=False, ax=None, **args):
	'''
	Turn a dynesty sampler output by barfit into a set of radial velocity
	profiles. Can plot if edges are given and will plot on a given axis ax if
	supplied. Takes args for plt.plot.  
	'''

	#get and unpack median values for params
	meds = dmeds(samp)
	inc, pa, pab, vsys = meds[:4]
	vts = meds[4::3]
	v2ts = meds[5::3]
	v2rs = meds[6::3]

	#plot profiles if edges are given
	if type(edges) != bool: 
		if not ax: f,ax = plt.subplots()
		ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
		[ax.plot(edges[:-1], p, label=ls[i], **args) for i,p in enumerate([vts,v2ts,v2rs])]
		plt.xlabel(r'$R_e$')
		plt.ylabel(r'$v$ (km/s)')
		plt.legend()

	return inc, pa, pab, vsys, vts, v2ts, v2rs

def summaryplot(f,nbins):
	'''
	Make a summary plot for a given dynesty file with MaNGA velocity field, the
	model that dynesty fit, the residuals of the fit, and the velocity
	profiles.  
	'''

	#get chains, edges, parameter values, vf info, model
	chains = pickle.load(open('dyn8078-10','rb'))
	edges = np.linspace(0,1.5,nbins+1)
	inc,pa,pab,vsys,vts,v2ts,v2rs = dprofs(chains)
	vf,flux,ivar,sigma,sigmaivar,x,y,er,eth = getvfinfo(8078,12703)
	m = barmodel(x,y,er,eth,edges,inc,pa,pab,vsys,vts,v2ts,v2rs)
	plt.figure(figsize = (8,8))

	#MaNGA Ha velocity field
	plt.subplot(221)
	plt.title(r'H$\alpha$ Data')
	plt.imshow(vf,cmap='jet',origin='lower')
	plt.colorbar(label='km/s')

	#VF model from dynesty fit
	plt.subplot(222)
	plt.title('Model')
	plt.imshow(m,'jet',origin='lower') 
	plt.colorbar(label='km/s')

	#Residuals from fit
	plt.subplot(223)
	plt.title('Residuals')
	plt.imshow(vf-m,'jet',origin='lower')
	plt.colorbar(label='km/s')

	#Radial velocity profiles
	plt.subplot(224)
	dprofs(chains,edges,plt.gca())
	plt.tight_layout()

def psfvf(vf, psf):
	vfc = signal.convolve(vf,psf,mode='same')
	vfc = vf[buf:-buf,buf:-buf]
	return svf
