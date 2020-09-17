"""
Plotting for barfit results.
"""

import numpy as np
from matplotlib import pyplot as plt

import dynesty
import corner
import pickle

from .barfit import barmodel
from .beam_smearing import apply_beam_smearing
from .data.manga import MaNGAStellarKinematics, MaNGAGasKinematics

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

def summaryplot(f,nbins,plate,ifu,smearing=True):
    '''
    Make a summary plot for a given dynesty file with MaNGA velocity field, the
    model that dynesty fit, the residuals of the fit, and the velocity
    profiles.  
    '''

    #get chains, edges, parameter values, vf info, model
    if type(f) == str: chains = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: chains = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results
    inc,pa,pab,vsys,vts,v2ts,v2rs = dprofs(chains)
    gal = MaNGAGasKinematics.from_plateifu(plate,ifu,dr='MPL-9',daptype='HYB10-MILESHC-MASTARHC')
    gal.setedges(nbins,1.5)
    model = barmodel(gal,inc,pa,pab,vsys,vts,v2ts,v2rs,plot=True)
    vf = gal.remap('vel')
    if smearing: 
        model = apply_beam_smearing(model, gal.psf, gal.remap('sb'), gal.remap('sig'), mask=gal.remap('vel_mask'))[1]
    plt.figure(figsize = (8,8))
    plt.suptitle(f'{plate}-{ifu}')

    #MaNGA Ha velocity field
    plt.subplot(221)
    plt.title(r'H$\alpha$ Data')
    plt.imshow(vf,cmap='jet',origin='lower')
    plt.colorbar(label='km/s')

    #VF model from dynesty fit
    plt.subplot(222)
    plt.title('Model')
    plt.imshow(model,'jet',origin='lower',vmin=vf.min(),vmax=vf.max()) 
    plt.colorbar(label='km/s')

    #Residuals from fit
    plt.subplot(223)
    plt.title('Residuals')
    resid = vf-model
    vmax = min(np.abs(vf-model).max(),50)
    plt.imshow(vf-model,'jet',origin='lower',vmin=-vmax,vmax=vmax)
    plt.colorbar(label='km/s')

    #Radial velocity profiles
    plt.subplot(224)
    dprofs(chains,gal.edges,plt.gca())
    plt.ylim(bottom=0)
    plt.tight_layout()
