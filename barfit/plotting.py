"""
Plotting for barfit results.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import dynesty
import corner
import pickle

from .barfit import barmodel, unpack
from .data.manga import MaNGAStellarKinematics, MaNGAGasKinematics
from .data.kinematics import Kinematics

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

def chainvis(sampler, titles=None, alpha=0.1, nplots=None):
    '''
    Look at all the chains of an emcee sampler in one plot to check
    convergence. Can specify number of variables to plot with nplots if there
    are too many to see well. Can set alpha of individual chains with alpha and
    titles of plots with titles.
    '''
    if titles is None:
        titles = ['$inc$ (deg)', r'$\phi$ (deg)', r'$\phi_b$ (deg)', r'$v_{sys}$ (km/s)']

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

def dmeds(samp,stds=False):
    '''
    Get median values for each variable in a dynesty sampler.
    '''

    #get samples and weights
    if type(samp) == str: res = pickle.load(open(samp,'rb'))
    elif type(samp)==dynesty.results.Results: res = samp
    else: res = samp.results
    samps = res.samples
    weights = np.exp(res.logwt - res.logz[-1])
    meds = np.zeros(samps.shape[1])

    #iterate through and get 50th percentile of values
    for i in range(samps.shape[1]):
        meds[i] = dynesty.utils.quantile(samps[:,i],[.5],weights)[0]

    #pull out 1 sigma values on either side of the mean as well if desired
    if stds:
        lstd = np.zeros(samps.shape[1])
        ustd = np.zeros(samps.shape[1])
        for i in range(samps.shape[1]):
            lstd[i] = dynesty.utils.quantile(samps[:,i],[.5-.6826/2],weights)[0]
            ustd[i] = dynesty.utils.quantile(samps[:,i],[.5+.6826/2],weights)[0]
        return meds, lstd, ustd

    return meds

def dcorner(f,**args):
    '''
    Make a cornerplot of a dynesty sampler. Takes args for
    dynesty.plotting.cornerplot.
    '''

    if type(f) == str: res = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: res = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: res = f.results

    dynesty.plotting.cornerplot(res, **args)

def checkbins(plate,ifu,nbins):
    '''
    Make a plot to see whether the number of spaxels in each bin make sense for
    a given number of bins for a MaNGA galaxy with given plate ifu. 
    '''

    vf,flux,m,ivar,sigma,sigmaivar,x,y,er,eth = getvfinfo(plate,ifu,psf=False)
    edges = np.linspace(0,1.5,nbins+1)[:-1]

    if nbins%2: nbins += 1
    plt.figure(figsize=(12,6))
    nrow = nbins//5
    ncol = nbins//nrow
    for i in range(len(edges)-1):
        plt.subplot(nrow,ncol,i+1)
        cut = (er>edges[i])*(er<edges[i+1])
        plt.imshow(np.ma.array(vf, mask=~cut), cmap='RdBu')

def dprofs(samp, args, plot=None, stds=False, **kwargs):
    '''
    Turn a dynesty sampler output by barfit into a set of radial velocity
    profiles. Can plot if edges are given and will plot on a given axis ax if
    supplied. Takes args for plt.plot.  
    '''

    #get and unpack median values for params
    meds = dmeds(samp, stds)
    if stds: meds, lstd, ustd = meds
    paramdict = unpack(meds, args)

    #insert 0 for center bin if necessary
    if args.fixcent:
        vts  = np.insert(paramdict['vts'],  0, 0)
        v2ts = np.insert(paramdict['v2ts'], 0, 0)
        v2rs = np.insert(paramdict['v2rs'], 0, 0)
    else:
        vts  = paramdict['vts']
        v2ts = paramdict['v2ts']
        v2rs = paramdict['v2rs']

    #get standard deviations
    if stds:
        start = args.nglobs
        jump = len(args.edges)-1
        if args.fixcent: jump -= 1
        paramdict['vtl']  = lstd[start:start + jump]
        paramdict['v2tl'] = lstd[start + jump:start + 2*jump]
        paramdict['v2rl'] = lstd[start + 2*jump:start + 3*jump]
        paramdict['vtu']  = ustd[start:start + jump]
        paramdict['v2tu'] = ustd[start + jump:start + 2*jump]
        paramdict['v2ru'] = ustd[start + 2*jump:start + 3*jump]
        if args.disp: 
            if args.fixcent: sigjump = jump+1
            else: sigjump = jump
            paramdict['sigl'] = lstd[start + 3*jump:start + 3*jump + sigjump]
            paramdict['sigu'] = ustd[start + 3*jump:start + 3*jump + sigjump]

        #add in central bin if necessary
        if args.fixcent:
            paramdict['vtl']  = np.insert(paramdict['vtl'],  0, 0)
            paramdict['v2tl'] = np.insert(paramdict['v2tl'], 0, 0)
            paramdict['v2rl'] = np.insert(paramdict['v2rl'], 0, 0)
            paramdict['vtu']  = np.insert(paramdict['vtu'],  0, 0)
            paramdict['v2tu'] = np.insert(paramdict['v2tu'], 0, 0)
            paramdict['v2ru'] = np.insert(paramdict['v2ru'], 0, 0)

    #plot profiles if edges are given
    if plot is not None: 
        if not isinstance(plot, matplotlib.axes._subplots.Axes): f,plot = plt.subplots()
        ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
        [plot.plot(args.edges[:-1], p, label=ls[i], **kwargs) for i,p in enumerate([vts,v2ts,v2rs])]
        if stds: 
            [plot.fill_between(args.edges[:-1], p[0], p[1], alpha=.5) 
                    for i,p in enumerate([[paramdict['vtl'],paramdict['vtu']],[paramdict['v2tl'],paramdict['v2tu']],[paramdict['v2rl'],paramdict['v2ru']]])]
        plt.xlabel(r'$R_e$')
        plt.ylabel(r'$v$ (km/s)')
        plt.legend()

    return paramdict

def summaryplot(f,nbins,plate,ifu,smearing=True,stellar=False,fixcent=False):
    '''
    Make a summary plot for a given dynesty file with MaNGA velocity field, the
    model that dynesty fit, the residuals of the fit, and the velocity
    profiles.  
    '''

    #get chains, edges, parameter values, vf info, model
    if type(f) == str: chains = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: chains = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results


    #mock galaxy using Andrew's values for 8078-12703
    if plate == 0 and ifu == 0 :
        mockparams = dprofs(pickle.load(open('mock.out','rb')))
        gal = Kinematics.mock(55,mockparams['inc'],mockparams['pa'],mockparams['pab'], mockparams['vsys'], mockparams['vts'], mockparams['v2ts'], mockparams['v2rs'])

    else:
        if stellar:
            gal = MaNGAStellarKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing)
        else:
            gal = MaNGAGasKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing)

    gal.setedges(nbins,1.5)
    gal.setfixcent(fixcent)
    gal.setdisp(True)
    gal.setnglobs(4)

    resdict = dprofs(chains, gal, stds=True)
    velmodel, sigmodel = barmodel(gal,resdict,plot=True)

    [gal.remap(a) for a in ['vel','sig']]

    plt.figure(figsize = (12,12))

    plt.subplot(331)
    ax = plt.gca()
    plt.axis('off')
    plt.title(f'{plate}-{ifu}',size=20)
    plt.text(.1, .8, r'$i$: %0.1f$^\circ$'%resdict['inc'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .6, r'$\phi$: %0.1f$^\circ$'%resdict['pa'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .4, r'$\phi_b$: %0.1f$^\circ$'%resdict['pab'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .2, r'$v_{{sys}}$: %0.1f km/s'%resdict['vsys'], 
            transform=ax.transAxes, size=20)

    #Radial velocity profiles
    plt.subplot(332)
    dprofs(chains, gal, plt.gca(), stds=True)
    plt.ylim(bottom=0)
    plt.title('Velocity Profiles')

    #dispersion profile
    plt.subplot(333)
    plt.plot(gal.edges[:-1], resdict['sig'])
    plt.fill_between(gal.edges[:-1], resdict['sigl'], resdict['sigu'], alpha=.5)
    plt.ylim(bottom=0)
    plt.title('Velocity Dispersion Profile')

    #MaNGA Ha velocity field
    plt.subplot(334)
    plt.title(r'H$\alpha$ Velocity Data')
    plt.imshow(gal.vel_r,cmap='jet',origin='lower')
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    #VF model from dynesty fit
    plt.subplot(335)
    plt.title('Velocity Model')
    plt.imshow(velmodel,'jet',origin='lower',vmin=gal.vel_r.min(),vmax=gal.vel_r.max()) 
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    #Residuals from fit
    plt.subplot(336)
    plt.title('Velocity Residuals')
    resid = gal.vel_r-velmodel
    vmax = min(np.abs(gal.vel_r-velmodel).max(),50)
    plt.imshow(gal.vel_r-velmodel,'jet',origin='lower',vmin=-vmax,vmax=vmax)
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    #MaNGA Ha velocity disp
    plt.subplot(337)
    plt.title(r'H$\alpha$ Velocity Dispersion Data')
    plt.imshow(gal.sig_r,cmap='jet',origin='lower')
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    #disp model from dynesty fit
    plt.subplot(338)
    plt.title('Velocity Dispersion Model')
    plt.imshow(sigmodel,'jet',origin='lower',vmin=gal.sig_r.min(),vmax=gal.sig_r.max()) 
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    #Residuals from disp fit
    plt.subplot(339)
    plt.title('Dispersion Residuals')
    resid = gal.sig_r-sigmodel
    vmax = min(np.abs(gal.sig_r-sigmodel).max(),50)
    plt.imshow(gal.sig_r-sigmodel,'jet',origin='lower',vmin=-vmax,vmax=vmax)
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    plt.colorbar(label='km/s')

    plt.tight_layout()
    return dprofs(chains, gal)
