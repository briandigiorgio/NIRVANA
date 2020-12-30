"""
Plotting for nirvana outputs.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal

import dynesty
import dynesty.plotting
import corner
import pickle

from .fitting import bisym_model, unpack
from .data.manga import MaNGAStellarKinematics, MaNGAGasKinematics
from .data.kinematics import Kinematics
from .models.beam import smear, ConvolveFFTW
from .models.geometry import projected_polar

def dynmeds(samp, stds=False):
    '''
    Get median values for each variable's posterior in a
    :class:`dynesty.NestedSampler` sampler. 

    Args:
        samp (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        stds (:obj:`bool`, optional):
            Flag for whether or not to return standard deviations of the
            posteriors as well.

    Returns:
        `np.ndarray`_: Median values of all of the parameters in the `dynesty`
        sampler.

        if `stds == True`, it will instead return a tuple of three `np.ndarray`_
        objects. The first is the median values, the second is the lower 1 sigma
        bound for all of the posteriors, and the third is the upper 1 sigma
        bound.
    '''

    #get samples and weights
    if type(samp) == str: res = pickle.load(open(samp,'rb'))
    elif type(samp)==dynesty.results.Results: res = samp
    else: res = samp.results
    samps = res.samples
    weights = np.exp(res.logwt - res.logz[-1])

    #iterate through and get 50th percentile of values
    meds = np.zeros(samps.shape[1])
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

def dcorner(f, **args):
    '''
    Make a cornerplot of a :class:`dynesty.NestedSampler` sampler.

    Wrapper function for :func:`dynesty.plotting.cornerplot`.

    Args:
        f (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        **args:
            Arguments for :func:`dynesty.plotting.cornerplot`.
        
    Returns:
        Plot: A cornerplot of the parameters from the sampler.
    '''

    #load results appropriately
    if type(f) == str: res = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: res = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: res = f.results

    dynesty.plotting.cornerplot(res, **args)

def profs(samp, args, plot=None, stds=False, jump=None, **kwargs):
    '''
    Turn a sampler output by `nirvana` into a set of rotation curves.
    
    Args:
        samp (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        args (:class:`nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        plot (:class:`matplotlib.axes._subplots.Axes`, optional):
            Axis to plot the rotation curves on. If not specified, it will not
            try to plot anything.
        stds (:obj:`bool`, optional):
            Flag for whether to fetch the standard deviations as well.
        jump (:obj:`int`, optional):
            Number of radial bins in the sampler. Will be calculated
            automatically if not specified.
        **kwargs:
            args for :func:`plt.plot`.

    Returns:
        :obj:`dict`: Dictionary with all of the median values of the posteriors
        in the sampler. Has keys for inclination `inc`, first order position
        angle `pa`, second order position angle `pab`, systemic velocity `vsys`,
        x and y center coordinates `xc` and `yc`, `np.ndarray`_ of first order
        tangential velocities `vt`, `np.ndarray`_ objects of second order
        tangential and radial velocities `v2t` and `v2r`, and `np.ndarray`_ of
        velocity dispersions `sig`. If `stds == True` it will also contain keys
        for the 1 sigma lower bounds of the velocity parameters `vtl`, `v2tl`,
        `v2rl`, and `sigl` as well as their 1 sigma upper bounds `vtu`, `v2tu`,
        `v2ru`, and `sigu`. Arrays have lengths that are the same as the number
        of bins (determined automatically or from `jump`). All angles are in
        degrees and all velocities must be in consistent units. 

        if `plot == True`, it will also display a plot of the profiles.
    '''

    #get and unpack median values for params
    meds = dynmeds(samp, stds)
    if stds: meds, lstd, ustd = meds
    paramdict = unpack(meds, args, jump=jump)

    #get standard deviations and put them into the dictionary
    if stds:
        start = args.nglobs
        jump = len(args.edges)
        vs = ['vt', 'v2t', 'v2r']
        for i,v in enumerate(vs):
            for b in ['l','u']:
                exec(f'paramdict["{v}{b}"] = {b}std[start + {i}*jump : start + {i+1}*jump]')

        #dispersion stds
        if args.disp: 
            sigjump = jump + 1
            paramdict['sigl'] = lstd[start + 3*jump:start + 3*jump + sigjump]
            paramdict['sigu'] = ustd[start + 3*jump:start + 3*jump + sigjump]

    #plot profiles if desired
    if plot is not None: 
        if not isinstance(plot, matplotlib.axes._subplots.Axes): f,plot = plt.subplots()
        ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
        [plot.plot(args.edges, p, label=ls[i], **kwargs) 
                for i,p in enumerate([paramdict['vt'], paramdict['v2t'], paramdict['v2r']])]

        #add in lower and upper bounds
        if stds: 
            [plot.fill_between(args.edges, p[0], p[1], alpha=.5) 
                for i,p in enumerate([[paramdict['vtl'], paramdict['vtu']],
                                      [paramdict['v2tl'], paramdict['v2tu']],
                                      [paramdict['v2rl'], paramdict['v2ru']]])]

        plt.xlabel(r'$R_e$')
        plt.ylabel(r'$v$ (km/s)')
        plt.legend(loc=2)

    return paramdict

def summaryplot(f, plate, ifu, smearing=True, stellar=False, maxr=None, mix=False):
    '''
    Make a summary plot for a `nirvana` output file with MaNGA velocity field.

    Shows the values for the global parameters of the galaxy, the rotation
    curves (with 1 sigma lower and upper bounds) for the different velocity
    components, then comparisons of the MaNGA data, the model, and the residuals
    for the rotational velocity and the velocity dispersion. 

    Args:
        f (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`):
            MaNGA plate number for desired galaxy.
        ifu (:obj:`int`):
            MaNGA IFU design number for desired galaxy.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of gas.
        mix (:obj:`bool`, optional):
            Flag for whether or not the fit is a Bayesian mixture. [NOT WORKING]
        
    Returns:
        :obj:`dict`: Dictionary with all of the median values of the posteriors
        in the sampler. Has keys for inclination `inc`, first order position
        angle `pa`, second order position angle `pab`, systemic velocity `vsys`,
        x and y center coordinates `xc` and `yc`, `np.ndarray`_ of first order
        tangential velocities `vt`, `np.ndarray`_ objects of second order
        tangential and radial velocities `v2t` and `v2r`, and `np.ndarray`_ of
        velocity dispersions `sig`. If `stds == True` it will also contain keys
        for the 1 sigma lower bounds of the velocity parameters `vtl`, `v2tl`,
        `v2rl`, and `sigl` as well as their 1 sigma upper bounds `vtu`, `v2tu`,
        `v2ru`, and `sigu`. Arrays have lengths that are the same as the number
        of bins (determined automatically or from `jump`). All angles are in
        degrees and all velocities must be in consistent units. 

        Plot: The values for the global parameters of the galaxy, the rotation
        curves (with 1 sigma lower and upper bounds) for the different velocity
        components, then comparisons of the MaNGA data, the model, and the
        residuals for the rotational velocity and the velocity dispersion. 
    '''

    #get sampler in right format
    if type(f) == str: chains = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: chains = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results

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

    #load in MaNGA data
    else:
        if stellar:
            args = MaNGAStellarKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing)
        else:
            args = MaNGAGasKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing)

    #set relevant parameters for galaxy
    args.setdisp(True)
    args.setnglobs(4)
    args.setmix(mix)
    args.clip()

    vel_r = args.remap('vel')
    sig_r = args.remap('sig') if args.sig_phys2 is None else np.sqrt(np.abs(args.remap('sig_phys2')))

    #get appropriate number of edges  by looking at length of meds
    meds = dynmeds(chains)
    nbins = (len(meds) - args.nglobs - 3*args.mix)/4
    if not nbins.is_integer(): 
        raise ValueError('Dynesty output array has a bad shape.')
    else: nbins = int(nbins)
    args.setedges(nbins-1, nbin=True, maxr=maxr)
    args.clip()

    #recalculate model that was fit
    resdict = profs(chains, args, stds=True)
    #if maxr is not None:
    #    r,th = projected_polar(args.x, args.y, resdict['pa'], resdict['inc'])
    #    r /= args.reff
    #    rmask = r > maxr
    #    args.vel_mask |= rmask
    #    args.sig_mask |= rmask
    velmodel, sigmodel = bisym_model(args,resdict,plot=True)


    #mask border if necessary
    if args.bordermask is not None:
        velmodel = np.ma.array(velmodel, mask=args.bordermask)
        vel_r = np.ma.array(vel_r, mask=args.bordermask)
        if sigmodel is not None:
            sigmodel = np.ma.array(sigmodel, mask=args.bordermask)
            sig_r = np.ma.array(sig_r, mask=args.bordermask)

    if args.vel_ivar is None: args.vel_ivar = np.ones_like(args.vel)
    if args.sig_ivar is None: args.sig_ivar = np.ones_like(args.sig)

    nvar = len(args.vel) + len(args.sig) - len(meds)
    rchisq = np.sum((vel_r - velmodel)**2 * args.remap('vel_ivar')) / nvar

    #print global parameters on figure
    plt.figure(figsize = (12,9))
    plt.subplot(3,4,1)
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
    plt.text(.1, 0, r'$\chi_r^2$: %0.1f'%rchisq, 
            transform=ax.transAxes, size=20)

    #image
    if args.image is not None:
        plt.subplot(3,4,2)
        plt.imshow(args.image)
        plt.axis('off')

    #Radial velocity profiles
    plt.subplot(3,4,3)
    profs(chains, args, plt.gca(), stds=True)
    plt.ylim(bottom=0)
    plt.title('Velocity Profiles')

    #dispersion profile
    plt.subplot(3,4,4)
    plt.plot(args.edges, resdict['sig'])
    plt.fill_between(args.edges, resdict['sigl'], resdict['sigu'], alpha=.5)
    plt.ylim(bottom=0)
    plt.title('Velocity Dispersion Profile')
    plt.xlabel(r'$R_e$')
    plt.ylabel(r'$v$ (km/s)')

    #MaNGA Ha velocity field
    plt.subplot(3,4,5)
    if not stellar: plt.title(r'H$\alpha$ Velocity Data')
    else: plt.title(r'Stellar Velocity Data')
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
    velchisq = (vel_r - velmodel)**2 * args.remap('vel_ivar')
    plt.imshow(velchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    #MaNGA Ha velocity disp
    plt.subplot(3,4,9)
    if not stellar: plt.title(r'H$\alpha$ Dispersion Data')
    else: plt.title(r'Stellar Dispersion Data')
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
    sigchisq = (sig_r - sigmodel)**2 * args.remap('sig_ivar')
    plt.imshow(sigchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    plt.tight_layout()

    if args.mix:
        dynesty.plotting.cornerplot(chains, dims = [-3, -2, -1])

    return profs(chains, args)

def separate_components(f, plate, ifu, maxr=1.5):
    if type(f) == str: chains = pickle.load(open(f,'rb'))
    elif type(f) == np.ndarray: chains = f
    elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results
    args = MaNGAGasKinematics.from_plateifu(plate,ifu)#, ignore_psf=not smearing)
    args.setnglobs(4)
    vel_r = args.remap('vel')
    sig_r = args.remap('sig') if args.sig_phys2 is None else np.sqrt(np.abs(args.remap('sig_phys2')))

    #get appropriate number of edges  by looking at length of meds
    nbins = (len(dynmeds(chains)) - args.nglobs)/4
    if not nbins.is_integer(): raise ValueError('Dynesty output array has a bad shape.')
    else: nbins = int(nbins)
    args.setedges(nbins, nbin=True, maxr=maxr)

    #recalculate model that was fit
    resdict = profs(chains, args, stds=True)
    z = np.zeros(len(resdict['vt']))
    vtdict, v2tdict, v2rdict = [resdict.copy(), resdict.copy(), resdict.copy()]
    vtdict['v2t'] = z
    vtdict['v2r'] = z
    v2tdict['vt'] = z
    v2tdict['v2r'] = z
    v2rdict['vt'] = z
    v2rdict['v2t'] = z
    if maxr is not None:
        r,th = projected_polar(args.x, args.y, resdict['pa'], resdict['inc'])
        r /= args.reff
        rmask = r > maxr
        args.vel_mask |= rmask
        args.sig_mask |= rmask

    velmodel, sigmodel = bisym_model(args, resdict, plot=True)
    vtmodel,  sigmodel = bisym_model(args, vtdict,  plot=True)
    v2tmodel, sigmodel = bisym_model(args, v2tdict, plot=True)
    v2rmodel, sigmodel = bisym_model(args, v2rdict, plot=True)
    vel_r = args.remap('vel')

    plt.figure(figsize = (12,6))
    vmax = min(max(np.max(np.abs(velmodel)), np.max(np.abs(vtmodel)), np.max(np.abs(v2tmodel)), np.max(np.abs(v2rmodel)), np.max(np.abs(vel_r))), 300)

    plt.subplot(241)
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

    #image
    plt.subplot(242)
    plt.imshow(args.image)
    plt.axis('off')

    #MaNGA Ha velocity field
    plt.subplot(243)
    plt.title(r'H$\alpha$ Velocity Data')
    vmax = min(np.max(np.abs(vel_r)), 300)
    plt.imshow(vel_r, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    plt.subplot(245)
    plt.imshow(velmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'=', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V$', fontsize=16)

    plt.subplot(246)
    plt.imshow(vtmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V_t$', fontsize=16)

    plt.subplot(247)
    plt.imshow(v2tmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V_{2t}$', fontsize=16)

    plt.subplot(248)
    plt.imshow(v2rmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.xlabel(r'$V_{2r}$', fontsize=16)

    plt.tight_layout(h_pad=2)
