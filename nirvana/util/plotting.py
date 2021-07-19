"""
Plotting for nirvana outputs.

.. include:: ../include/links.rst
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import re
import os
import traceback
import multiprocessing as mp
from functools import partial

import dynesty
import corner
import pickle
from glob import glob
from tqdm import tqdm
from astropy.io import fits

from ..models.higher_order import bisym_model
from ..models.beam import smear, ConvolveFFTW
from ..models.geometry import projected_polar
from ..data.manga import MaNGAStellarKinematics, MaNGAGasKinematics
from ..data.kinematics import Kinematics
from ..data.util import unpack
from .fits_prep import fileprep, dynmeds, profs


def summaryplot(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, cen=True,
                fixcent=True, save=False, clobber=False, remotedir=None, gal=None, relative_pab=False):
    """
    Make a summary plot for a `nirvana` output file with MaNGA velocity
    field.

    Shows the values for the global parameters of the galaxy, the rotation
    curves (with 1-sigma lower and upper bounds) for the different velocity
    components, then comparisons of the MaNGA data, the model, and the
    residuals for the rotational velocity and the velocity dispersion.

    Args:
        f (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            `.fits` file, sampler, results, `.nirv` file of dumped results
            from :func:`~nirvana.fitting.fit`. If this is in the regular
            format from the automatic outfile generator in
            :func:`~nirvana.scripts.nirvana.main` then it will fill in most
            of the rest of the parameters by itself.
        plate (:obj:`int`, optional):
            MaNGA plate number for desired galaxy. Can be auto filled by `f`.
        ifu (:obj:`int`, optional):
            MaNGA IFU number for desired galaxy. Can be auto filled by `f`.
        smearing (:obj:`bool`, optional):
            Whether or not to apply beam smearing to models. Can be auto
            filled by `f`.
        stellar (:obj:`bool`, optional):
            Whether or not to use stellar velocity data instead of gas. Can
            be auto filled by `f`.
        maxr (:obj:`float`, optional):
            Maximum radius to make edges go out to in units of effective
            radii. Can be auto filled by `f`.
        cen (:obj:`bool`, optional):
            Whether the position of the center was fit. Can be auto filled by
            `f`.
        fixcent (:obj:`bool`, optional):
            Whether the center velocity bin was held at 0 in the fit. Can be
            auto filled by `f`.
        save (:obj:`bool`, optional):
            Flag for whether to save the plot. Will save as a pdf in the same
            directory as `f` is in but inside a folder called `plots`.
        clobber (:obj:`bool`, optional):
            Flag to overwrite plot file if it already exists. Only matters if
            `save=True`
        remotedir (:obj:`str`, optional):
            Directory to load MaNGA data files from, or save them if they are
            not found and are remotely downloaded.
    """

    #check if plot file already exists
    if save and not clobber:
        path = f[:f.rfind('/')+1]
        fname = f[f.rfind('/')+1:-5]
        if os.path.isfile(f'{path}/plots/{fname}.pdf'):
            raise ValueError('Plot file already exists')

    #unpack input file into useful objects
    args, resdict = fileprep(f, plate, ifu, smearing, stellar, maxr, cen, fixcent, remotedir=remotedir, gal=gal)

    #generate velocity models
    velmodel, sigmodel = bisym_model(args,resdict,plot=True,relative_pab=relative_pab)
    vel_r = args.kin.remap('vel')
    sig_r = np.sqrt(args.kin.remap('sig_phys2')) if hasattr(args.kin, 'sig_phys2') else args.kin.remap('sig')

    if args.kin.vel_ivar is None: args.kin.vel_ivar = np.ones_like(args.kin.vel)
    if args.kin.sig_ivar is None: args.kin.sig_ivar = np.ones_like(args.kin.sig)

    #calculate number of variables
    if 'velmask' in resdict:
        fill = len(resdict['velmask'])
        fixcent = resdict['vt'][0] == 0
        lenmeds = 6 + 3*(fill - resdict['velmask'].sum() - fixcent) + (fill - resdict['sigmask'].sum())
    else: lenmeds = len(resdict['vt'])
    nvar = len(args.kin.vel) + len(args.kin.sig) - lenmeds

    #calculate reduced chisq for vel and sig
    rchisqv = np.sum((vel_r - velmodel)**2 * args.kin.remap('vel_ivar')) / nvar
    rchisqs = np.sum((sig_r - sigmodel)**2 * args.kin.remap('sig_ivar')) / nvar

    #print global parameters on figure
    fig = plt.figure(figsize = (12,9))
    plt.subplot(3,4,1)
    ax = plt.gca()
    infobox(ax, resdict, args, cen, relative_pab)

    #image
    plt.subplot(3,4,2)
    if args.kin.image is not None: plt.imshow(args.kin.image)
    else: plt.text(.5,.5, 'No image found', horizontalalignment='center',
            transform=plt.gca().transAxes, size=14)

    plt.axis('off')

    #Radial velocity profiles
    plt.subplot(3,4,3)
    ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
    for i,v in enumerate(['vt', 'v2t', 'v2r']):
        plt.plot(args.edges, resdict[v], label=ls[i]) 

    errors = [[resdict['vtl'], resdict['vtu']], [resdict['v2tl'], resdict['v2tu']], [resdict['v2rl'], resdict['v2ru']]]
    for i,p in enumerate(errors):
        plt.fill_between(args.edges, p[0], p[1], alpha=.5) 

    plt.ylim(bottom=0)
    plt.legend(loc=2)
    plt.xlabel('Radius (arcsec)')
    plt.ylabel(r'$v$ (km/s)')
    plt.title('Velocity Profiles')

    #dispersion profile
    plt.subplot(3,4,4)
    plt.plot(args.edges, resdict['sig'])
    plt.fill_between(args.edges, resdict['sigl'], resdict['sigu'], alpha=.5)
    plt.ylim(bottom=0)
    plt.title('Velocity Dispersion Profile')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel(r'$v$ (km/s)')

    #MaNGA Ha velocity field
    plt.subplot(3,4,5)
    plt.title(f"{resdict['type']} Velocity Data")
    vmax = min(np.max(np.abs(vel_r)), 300)
    plt.imshow(vel_r, cmap='jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Vel model from dynesty fit
    plt.subplot(3,4,6)
    plt.title('Velocity Model')
    plt.imshow(velmodel,'jet', origin='lower', vmin=-vmax, vmax=vmax) 
    plt.tick_params(left=False, bottom=False,labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(label='km/s', cax=cax)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Residuals from vel fit
    plt.subplot(3,4,7)
    plt.title('Velocity Residuals')
    resid = vel_r - velmodel
    vmax = min(np.abs(vel_r-velmodel).max(), 50)
    plt.imshow(vel_r-velmodel, 'jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(label='km/s', cax=cax)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Chisq from vel fit
    plt.subplot(3,4,8)
    plt.title('Velocity Chi Squared')
    velchisq = (vel_r - velmodel)**2 * args.kin.remap('vel_ivar')
    plt.imshow(velchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    #MaNGA Ha velocity disp
    plt.subplot(3,4,9)
    plt.title(f"{resdict['type']} Dispersion Data")
    vmax = min(np.max(sig_r), 200)
    plt.imshow(sig_r, cmap='jet', origin='lower', vmax=vmax, vmin=0)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=0)

    #disp model from dynesty fit
    plt.subplot(3,4,10)
    plt.title('Dispersion Model')
    plt.imshow(sigmodel, 'jet', origin='lower', vmin=0, vmax=vmax) 
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=0)

    #Residuals from disp fit
    plt.subplot(3,4,11)
    plt.title('Dispersion Residuals')
    resid = sig_r - sigmodel
    vmax = min(np.abs(sig_r - sigmodel).max(), 50)
    plt.imshow(sig_r-sigmodel, 'jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Chisq from sig fit
    plt.subplot(3,4,12)
    plt.title('Dispersion Chi Squared')
    sigchisq = (sig_r - sigmodel)**2 * args.kin.remap('sig_ivar')
    plt.imshow(sigchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    plt.tight_layout()

    if save:
        path = f[:f.rfind('/')+1]
        fname = f[f.rfind('/')+1:-5]
        plt.savefig(f'{path}plots/{fname}.pdf', format='pdf')
        plt.close()

    return fig

def separate_components(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, cen=True,
        fixcent=True, save=False, clobber=False, remotedir=None, gal=None, relative_pab=False, cmap='RdBu'):
    """
    Make a plot `nirvana` output file with the different velocity components
    searated.

    Plot the first order velocity component and the two second order velocity
    components next to each other along with the full model, data, image, and
    global parameters

    The created plot contains the global parameters of the galaxy, the image
    of the galaxy, and the data of the velocity field on the first row. The
    second row is the full model followed by the different components broken
    out with + and = signs between.

    Args:
        f (:class:`dynesty.NestedSampler`, :obj:`str`, :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`, optional):
            MaNGA plate number for desired galaxy. Must be specified if
            `auto=False`.
        ifu (:obj:`int`, optional):
            MaNGA IFU design number for desired galaxy. Must be specified if
            `auto=False`.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of
            gas.
        cen (:obj:`bool`, optional):
            Flag for whether the position of the center was fit.
        
    """

    args, resdict = fileprep(f, plate, ifu, smearing, stellar, maxr, cen, fixcent, remotedir=remotedir, gal=gal)
    z = np.zeros(len(resdict['vt']))
    vtdict, v2tdict, v2rdict = [resdict.copy(), resdict.copy(), resdict.copy()]
    vtdict['v2t'] = z
    vtdict['v2r'] = z
    v2tdict['vt'] = z
    v2tdict['v2r'] = z
    v2rdict['vt'] = z
    v2rdict['v2t'] = z
    if maxr is not None:
        r,th = projected_polar(args.kin.x, args.kin.y, *np.radians((resdict['pa'], resdict['inc'])))
        rmask = r > maxr
        args.kin.vel_mask |= rmask
        args.kin.sig_mask |= rmask

    velmodel, sigmodel = bisym_model(args, resdict, plot=True)
    vtmodel,  sigmodel = bisym_model(args, vtdict,  plot=True)
    v2tmodel, sigmodel = bisym_model(args, v2tdict, plot=True)
    v2rmodel, sigmodel = bisym_model(args, v2rdict, plot=True)
    vel_r = args.kin.remap('vel')

    #must set all masked areas to 0 or else vmax calculations barf
    for v in [vel_r, velmodel, vtmodel, v2tmodel, v2rmodel]:
        v.data[v.mask] = 0
        v -= resdict['vsys'] #recenter at 0

    v2model = v2tmodel + v2rmodel
    v2model.data[v2model.mask] = 0

    velresid = vel_r - velmodel
    vtresid  = vel_r - v2tmodel - v2rmodel
    v2tresid = vel_r - vtmodel - v2rmodel
    v2rresid = vel_r - vtmodel - v2tmodel
    v2resid  = vel_r - vtmodel

    datavmax = min(np.max(np.abs([vel_r, velmodel])), 300)
    velvmax = min(np.max(np.abs(velresid)), 300)
    vtvmax = min(np.max(np.abs([vtmodel, vtresid])), 300)
    v2tvmax = min(np.max(np.abs([v2tmodel, v2tresid])), 300)
    v2rvmax = min(np.max(np.abs([v2rmodel, v2rresid])), 300)
    v2vmax = min(np.max(np.abs([v2model, v2resid])), 300)

    plt.figure(figsize = (15,9))

    plt.subplot(3,5,1)
    ax = plt.gca()
    infobox(ax, resdict, args)

    #image
    plt.subplot(3,5,2)
    plt.imshow(args.kin.image)
    plt.axis('off')

    #MaNGA Ha velocity field
    plt.subplot(3,5,3)
    plt.title(r'Velocity Data')
    plt.imshow(vel_r, cmap='RdBu', origin='lower', vmin=-datavmax, vmax=datavmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    #Radial velocity profiles
    plt.subplot(3,5,4)
    ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
    for i,v in enumerate(['vt', 'v2t', 'v2r']):
        plt.plot(args.edges, resdict[v], label=ls[i]) 

    errors = [[resdict['vtl'], resdict['vtu']], [resdict['v2tl'], resdict['v2tu']], [resdict['v2rl'], resdict['v2ru']]]
    for i,p in enumerate(errors):
        plt.fill_between(args.edges, p[0], p[1], alpha=.5) 
    plt.ylim(bottom=0)
    plt.legend(loc=2)
    plt.xlabel('Radius (arcsec)', labelpad=-1)
    plt.ylabel(r'$v$ (km/s)')
    plt.title('Velocity Profiles')
    plt.gca().tick_params(direction='in')

    plt.subplot(3,5,6)
    plt.imshow(velmodel, cmap = 'RdBu', origin='lower', vmin=-datavmax, vmax=datavmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #plt.text(1.15,.5,'=', transform=plt.gca().transAxes, size=30)
    plt.title(r'Model', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,7)
    plt.imshow(vtmodel, cmap = 'RdBu', origin='lower', vmin=-vtvmax, vmax=vtvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #plt.text(1.15,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.title(r'$V_t$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,8)
    plt.imshow(v2tmodel, cmap = 'RdBu', origin='lower', vmin=-v2tvmax, vmax=v2tvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    #plt.text(1.15,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.title(r'$V_{2t}$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,9)
    plt.imshow(v2rmodel, cmap = 'RdBu', origin='lower', vmin=-v2rvmax, vmax=v2rvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'$V_{2r}$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,10)
    plt.imshow(v2model, cmap = 'RdBu', origin='lower', vmin=-v2vmax, vmax=v2vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'$V_{2t} + V_{2r}$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,11)
    plt.imshow(velresid, cmap='RdBu', origin='lower', vmin=-velvmax, vmax=velvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'Data $-$ Model', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,12)
    plt.imshow(vtresid, cmap='RdBu', origin='lower', vmin=-vtvmax, vmax=vtvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'Data$- (V_{2t} + V_{2r})$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,13)
    plt.imshow(v2tresid, cmap='RdBu', origin='lower', vmin=-v2tvmax, vmax=v2tvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'Data$- (V_t + V_{2r})$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,14)
    plt.imshow(v2rresid, cmap='RdBu', origin='lower', vmin=-v2rvmax, vmax=v2rvmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'Data$- (V_t + V_{2t})$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.subplot(3,5,15)
    plt.imshow(v2resid, cmap='RdBu', origin='lower', vmin=-v2vmax, vmax=v2vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title(r'Data$- V_t$', fontsize=16)
    cax = mal(plt.gca()).append_axes('bottom', size='5%', pad=0)
    cb = plt.colorbar(cax=cax, orientation='horizontal')
    cax.tick_params(direction='in')
    cb.set_label('km/s', labelpad=-2)

    plt.tight_layout(pad=-.025)

    if save:
        path = f[:f.rfind('/')+1]
        fname = f[f.rfind('/')+1:-5]
        plt.savefig(f'{path}plots/sepcomp_{fname}.pdf', format='pdf')
        plt.close()

def sinewave(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, cen=True): 
    """
    Compare the `nirvana` fit to the data azimuthally in radial bins.

    Breaks down the data into radial bins and plots the velocity data points
    in each bin azimuthally, overplotting the fit for each bin. These are
    separated out into a Joy Division style plot.

    The plot provides the data points in each radial bin plotted azimuthally,
    color coded and separated on an arbitrary y axis. The curve the fit
    generated is plotted on top.

    Args:
        f (:class:`dynesty.NestedSampler`, :obj:`str`, :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`, optional):
            MaNGA plate number for desired galaxy. Must be specified if
            `auto=False`.
        ifu (:obj:`int`, optional):
            MaNGA IFU design number for desired galaxy. Must be specified if
            `auto=False`.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of
            gas.
        cen (:obj:`bool`, optional):
            Flag for whether the position of the center was fit.
    """

    #prep the data, parameters, and coordinates
    args, resdict, chains, meds = fileprep(f, plate, ifu, smearing, stellar, maxr, cen, fixcent)
    inc, pa, pab = np.radians([resdict['inc'], resdict['pa'], resdict['pab']])
    r,th = projected_polar(args.kin.x, args.kin.y, pa, inc)

    plt.figure(figsize=(4,len(args.edges)*.75))
    c = plt.cm.jet(np.linspace(0,1,len(args.edges)-1))
    plt.title(f"{resdict['plate']}-{resdict['ifu']} {resdict['type']}")

    #for each radial bin, plot data points and model
    for i in range(len(args.edges)-1):
        cut = (r > args.edges[i]) * (r < args.edges[i+1])
        sort = np.argsort(th[cut])
        thcs = th[cut][sort]
        plt.plot(np.degrees(thcs), args.kin.vel[cut][sort]+100*i, '.', c=c[i])
        
        #generate model from fit parameters
        velmodel = resdict['vsys'] + np.sin(inc) * (resdict['vt'][i] * np.cos(thcs) \
                 - resdict['v2t'][i] * np.cos(2 * (thcs - pab)) * np.cos(thcs) \
                 - resdict['v2r'][i] * np.sin(2 * (thcs - pab)) * np.sin(thcs))
        plt.plot(np.degrees(thcs), velmodel+100*i, 'k--')
        plt.tick_params(left=False, labelleft=False)
        plt.xlabel('Azimuth (deg)')
        plt.tight_layout()

def safeplot(f, func='sum', **kwargs):
    '''
    Call :func:`~nirvana.plotting.summaryplot` in a safe way.

    Really should be a decorator but I couldn't figure it out.

    Args:
        f (:obj:`str`):
            Name of the `.fits` file you want to plot.
        kwargs (optional):
            Arguments for `~nirvana.plotting.summaryplot`.
    '''
    if func not in ['sum', 'sep']: raise ValueError('Please provide a valid plotting function: sum or sep')
    try:
        if func == 'sum': summaryplot(f, save=True, **kwargs)
        elif func == 'sep': separate_components(f, save=True, **kwargs)
    except Exception:
        print(f, 'failed')
        print(traceback.format_exc())

def plotdir(directory='/data/manga/digiorgio/nirvana/', fname='*-*_*.nirv', cores=20, func='sum', **kwargs):
    '''
    Make summaryplots of an entire directory of output files.

    Will try to look for automatically named nirvana output files unless told
    otherwise. 
    
    CAUTION: If you use too many cores and don't call `plt.ioff()` before this,
    this function may crash the desktop environment of your operating system
    because it tries to open too many windows at once.

    Args:
        directory (:obj:`str`, optional):
            Directory to look for files in
        fname (:obj:`str`, optional):
            Filename format for files you want plotted with appropriate
            wildcards. Defaults to standard nirvana output format
        cores (:obj:`int`, optional):
            Number of cores to use for multiprocessing. CAUTION: If you use too
            many cores and don't call `plt.ioff()` before this, this function
            may crash the desktop environment of your operating system because
            it tries to open too many windows at once.
        kwargs (optional):
            Arguments for `~nirvana.plotting.summaryplot`.
    '''

    plt.ioff() #turn off plot displaying (don't know if this works in a script)
    fs = glob(directory + fname)
    if len(fs) == 0: raise FileNotFoundError('No files found')
    else: print(len(fs), 'files found')
    with mp.Pool(cores) as p:
        p.map(partial(safeplot, func=func), fs)

def infobox(plot, resdict, args, cen=True, relative_pab=False):
    #generate velocity models
    velmodel, sigmodel = bisym_model(args,resdict,plot=True,relative_pab=relative_pab)
    vel_r = args.kin.remap('vel')
    sig_r = np.sqrt(args.kin.remap('sig_phys2')) if hasattr(args, 'sig_phys2') else args.kin.remap('sig')

    #calculate number of variables
    if 'velmask' in resdict:
        fill = len(resdict['velmask'])
        fixcent = resdict['vt'][0] == 0
        lenmeds = 6 + 3*(fill - resdict['velmask'].sum() - fixcent) + (fill - resdict['sigmask'].sum())
    else: lenmeds = len(resdict['vt'])
    nvar = len(args.kin.vel) + len(args.kin.sig) - lenmeds

    #calculate reduced chisq for vel and sig
    rchisqv = np.sum((vel_r - velmodel)**2 * args.kin.remap('vel_ivar')) / nvar
    rchisqs = np.sum((sig_r - sigmodel)**2 * args.kin.remap('sig_ivar')) / nvar

    #print global parameters on figure
    plot.axis('off')
    ny = 6 + 2*cen
    fontsize = 14 - 2*cen
    ys = np.linspace(1 - .01*fontsize, 0, ny)

    plot.set_title(f"{resdict['plate']}-{resdict['ifu']} {resdict['type']}",size=18)
    plot.text(.1, ys[0], r'$i$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg. (phot: %0.1f$^\circ$)'
            %(resdict['inc'], resdict['incu'] - resdict['inc'], 
            resdict['inc'] - resdict['incl'], args.kin.phot_inc),
            transform=plot.transAxes, size=fontsize)
    plot.text(.1, ys[1], r'$\phi$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg.'
            %(resdict['pa'], resdict['pau'] - resdict['pa'], 
            resdict['pa'] - resdict['pal']), transform=plot.transAxes, size=fontsize)
    plot.text(.1, ys[2], r'$\phi_b$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg.'
            %(resdict['pab'], resdict['pabu'] - resdict['pab'], 
            resdict['pab'] - resdict['pabl']), transform=plot.transAxes, size=fontsize)
    plot.text(.1, ys[3], r'$v_{{sys}}$: %0.1f$^{+%0.1f}_{-%0.1f}$ km/s'
            %(resdict['vsys'], resdict['vsysu'] - resdict['vsys'], 
            resdict['vsys'] - resdict['vsysl']),transform=plot.transAxes, size=fontsize)
    plot.text(.1, ys[4], r'$\chi_v^2$: %0.1f,   $\chi_s^2$: %0.1f' % (rchisqv, rchisqs), 
            transform=plot.transAxes, size=fontsize)
    plot.text(.1, ys[5], 'Asymmetry: %0.3f' % args.arc,
            transform=plot.transAxes, size=fontsize)
    if cen: 
        plot.text(.1, ys[6], r'$x_c: %0.1f$" $ ^{+%0.1f}_{-%0.1f}$' %  
                    (resdict['xc'], abs(resdict['xcu'] - resdict['xc']),
                    abs(resdict['xcl'] - resdict['xc'])), 
                    transform=plot.transAxes, size=fontsize)
        plot.text(.1, ys[7], r'$y_c: %0.1f$" $ ^{+%0.1f}_{-%0.1f}$' %  
                    (resdict['yc'], abs(resdict['ycu'] - resdict['yc']), 
                    abs(resdict['ycl'] - resdict['yc'])),
                    transform=plot.transAxes, size=fontsize)
