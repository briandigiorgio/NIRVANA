"""
Script that runs the an axisymmetric, least-squares fit for MaNGA data.
"""
import os
import argparse

from IPython import embed

import numpy as np

from matplotlib import pyplot, rc, patches, ticker, colors
from nirvana.models.geometry import projected_polar
from nirvana import data
from nirvana import plotting
from nirvana.models.oned import HyperbolicTangent, Exponential, ExpBase, Const, PolyEx
from nirvana.models.axisym import AxisymmetricDisk


def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plate', default=None, type=int, 
                        help='MaNGA plate identifier (e.g., 8138)')
    parser.add_argument('ifu', default=None, type=int, 
                        help='MaNGA ifu identifier (e.g., 12704)')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-10', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the directory to the MaNGA galaxy data (see also '
                             '--redux and --analysis), and it will be ignored if the root '
                             'directory is set directly (using --root).')
    parser.add_argument('--redux', default=None, type=str,
                        help='Top-level directory with the MaNGA DRP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_REDUX.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.')
    parser.add_argument('--root', default=None, type=str,
                        help='Path with *all* fits files required for the fit.  This includes ' \
                             'the DRPall file, the DRP LOGCUBE file, and the DAP MAPS file.  ' \
                             'The LOGCUBE file is only required if the beam-smearing is ' \
                             'included in the fit.')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Run fit in verbose mode.')
    parser.add_argument('--odir', type=str, default=os.getcwd(), help='Directory for output files')
    parser.add_argument('--nodisp', dest='disp', default=True, action='store_false',
                        help='Only fit the velocity field (ignore velocity dispersion)')
    parser.add_argument('--nopsf', dest='smear', default=True, action='store_false',
                        help='Ignore the map PSF (i.e., ignore beam-smearing)')
    parser.add_argument('--covar', default=False, action='store_true',
                        help='Include the nominal covariance in the fit')
    parser.add_argument('--fix_cen', default=False, action='store_true',
                        help='Fix the dynamical center coordinate to the galaxy center')
    parser.add_argument('--fix_inc', default=False, action='store_true',
                        help='Fix the inclination to the guess inclination based on the '
                             'photometric ellipticity')
    parser.add_argument('-t', '--tracer', default='Gas', type=str,
                        help='The tracer to fit; must be either Gas or Stars.')
    parser.add_argument('--rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
    parser.add_argument('--dc', default='Exponential', type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.')

    # TODO: Other options:
    #   - Fit with least-squares vs. dynesty
    #   - Type of rotation curve
    #   - Type of dispersion profile
    #   - Include the surface-brightness weighting

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):

    #---------------------------------------------------------------------------
    # Setup
    #  - Check the input
    if args.tracer not in ['Gas', 'Stars']:
        raise ValueError('Tracer to fit must be either Gas or Stars.')
    #  - Check that the output directory exists, and if not create it
    if not os.path.isdir(args.odir):
        os.makedirs(args.odir)

    #---------------------------------------------------------------------------
    # Read the data to fit
    if args.tracer == 'Gas':
        kin = data.manga.MaNGAGasKinematics.from_plateifu(args.plate, args.ifu,
                                                          daptype=args.daptype, dr=args.dr,
                                                          redux_path=args.redux,
                                                          cube_path=args.root,
                                                          image_path=args.root,
                                                          analysis_path=args.analysis,
                                                          maps_path=args.root,
                                                          ignore_psf=not args.smear,
                                                          covar=args.covar,
                                                          positive_definite=True)
    elif args.tracer == 'Stars':
        kin = data.manga.MaNGAStellarKinematics.from_plateifu(args.plate, args.ifu,
                                                              daptype=args.daptype, dr=args.dr,
                                                              redux_path=args.redux,
                                                              cube_path=args.root,
                                                              image_path=args.root,
                                                              analysis_path=args.analysis,
                                                              maps_path=args.root,
                                                              ignore_psf=not args.smear,
                                                              covar=args.covar,
                                                              positive_definite=True)
    else:
        # NOTE: Should never get here given the check above.
        raise ValueError(f'Unknown tracer: {args.tracer}')

    # Setup the metadata
    galmeta = data.manga.MaNGAGlobalPar(args.plate, args.ifu, redux_path=args.redux, dr=args.dr,
                                        drpall_path=args.root)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Get the guess parameters and the model parameterizations
    #   - Geometry
    pa, vproj = galmeta.guess_kinematic_pa(kin.grid_x, kin.grid_y, kin.remap('vel'),
                                           return_vproj=True)
    p0 = np.array([0., 0., pa, galmeta.guess_inclination(), 0.])

    #   - Rotation Curve
    rc = None
    if args.rc == 'HyperbolicTangent':
        # TODO: Maybe want to make the guess hrot based on the effective radius...
        p0 = np.append(p0, np.array([vproj, 1.]))
        rc = HyperbolicTangent(lb=np.array([0., 1e-3]), ub=np.array([1000., kin.max_radius()]))
    elif args.rc == 'PolyEx':
        p0 = np.append(p0, np.array([vproj, 1., 0.1]))
        rc = PolyEx(lb=np.array([0., 1e-3, -1.]), ub=np.array([1000., kin.max_radius(), 1.]))
    else:
        raise ValueError(f'Unknown RC parameterization: {args.rc}')

    #   - Dispersion profile
    dc = None
    if args.disp:
        sig0 = galmeta.guess_central_dispersion(kin.grid_x, kin.grid_y, kin.remap('sig'))
        # For disks, 1 Re = 1.7 hr (hr = disk scale length). The dispersion
        # e-folding length is ~2 hr, meaning that I use a guess of 2/1.7 Re for
        # the dispersion e-folding length.
        if args.dc == 'Exponential':
            p0 = np.append(p0, np.array([sig0, 2*galmeta.reff/1.7]))
            dc = Exponential(lb=np.array([0., 1e-3]), ub=np.array([1000., kin.max_radius()]))
        elif args.dc == 'ExpBase':
            p0 = np.append(p0, np.array([sig0, 2*galmeta.reff/1.7, 1.]))
            dc = ExpBase(lb=np.array([0., 1e-3, 0.]), ub=np.array([1000., kin.max_radius(), 100.]))
        elif args.dc == 'Const':
            p0 = np.append(p0, np.array([sig0]))
            dc = Const(lb=np.array([0.]), ub=np.array([1000.]))
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Setup the full velocity-field model
    disk = AxisymmetricDisk(rc=rc, dc=dc)

    # Hardcoded Procedure:
    #   - Reject data with large errors and low S/N
    #       - These are rejected throughout
    #   - Fit with the inclination and center fixed
    #   - Reject very large outliers, sigma_rej=15
    #   - Refit with the inclination and center fixed
    #       - Ignore measured scatter
    #   - Reject large outliers, sigma_rej=10
    #   - Refit with the inclination and center fixed
    #       - Include measured scatter
    #       - Save results, use as starting point for the remainder
    #   - Refit with the inclination fixed
    #       - Include measured scatter
    #       - Save results
    #   - Refit with the center fixed
    #       - Include measured scatter
    #       - Save results
    #   - Refit with everything free
    #       - Include measured scatter
    #       - Save results
    #   - Reject very large outliers, sigma_rej=15
    #   - Refit with everything free
    #       - Include measured scatter
    #   - Reject large outliers, sigma_rej=10
    #   - Refit with everything free
    #       - Include measured scatter
    #       - Save results

    #---------------------------------------------------------------------------
    # Run the first fit using all the data
    fix = np.array([True, True, False, True, False])
    npar = p0.size
    fix = np.append(fix, np.zeros(npar-fix.size, dtype=bool))
    disk.lsq_fit(kin, sb_wgt=True, p0=p0, fix=fix, verbose=2)
    # Show
    axisym_fit_plot(galmeta, kin, disk)

    exit()

    # Get the models
    models = disk.model()
    vmod = models[0] if args.disp else models
    # Rejected based on error-weighted residuals, accounting for intrinsic scatter
    resid = kin.vel - kin.bin(vmod)
    err = 1/np.sqrt(kin.vel_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm)
    vel_sig, vel_rej, vel_gpm = scat.iter_fit(fititer=5, verbose=2)
    scat.show()
    kin.vel_mask = np.logical_not(vel_gpm)
    scatter = np.array([vel_sig])

    if args.disp:
        # Rejected based on error-weighted residuals, accounting for intrinsic scatter
        resid = kin.sig_phys2 - kin.bin(models[1])**2
        err = 1/np.ma.sqrt(kin.sig_phys2_ivar)
        scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.sig_gpm)
        sig_sig, sig_rej, sig_gpm = scat.iter_fit(fititer=5, verbose=2)
        scat.show()
        kin.sig_mask = np.logical_not(sig_gpm)
        scatter = np.array([vel_sig, sig_sig])

    # Refit with new mask, include scatter and covariance
    p0 = disk.par
    disk.lsq_fit(kin, scatter=scatter, sb_wgt=True, p0=disk.par, verbose=2)

    axisym_fit_plot(galmeta, kin, disk)

    exit()

#    models = disk.model()
#    vmod = models[0] if args.disp else models
#    # Reject
#    resid = kin.vel - kin.bin(vmod)
#    scat = data.scatter.IntrinsicScatter(resid, err=err) #, gpm=disk.vel_gpm)
#    sig, rej, gpm = scat.iter_fit(fititer=5, verbose=2)

    print(disk.par)

    axisym_fit_plot(galmeta, kin, disk)

    embed()
    exit()


# TODO: Add keyword for:
#   - Radial sampling for 1D model RCs and dispersion profiles
def axisym_fit_plot(galmeta, kin, disk, ofile=None, par=None, par_err=None):
    """
    Construct the QA plot for the result of fitting an
    :class:`~nirvana.model.axisym.AxisymmetricDisk` model to a galaxy.

    """
    logformatter = ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(
                                            int(np.maximum(-np.log10(y),0)))).format(y))

    rc('font', size=8)

    _par = disk.par if par is None else par
    _par_err = disk.par_err if par_err is None else par_err

    disk.par = _par
    disk.par_err = _par_err

    # Rebuild the 2D maps
    sb_map = kin.remap('sb')
    snr_map = sb_map * np.sqrt(kin.remap('sb_ivar', mask=kin.sb_mask))
    v_map = kin.remap('vel')
    v_err_map = np.ma.power(kin.remap('vel_ivar', mask=kin.vel_mask), -0.5)
    s_map = np.ma.sqrt(kin.remap('sig_phys2'))
    s_err_map = np.ma.power(kin.remap('sig_phys2_ivar', mask=kin.sig_mask), -0.5)/2/s_map

    # Construct the model data, both binned data and maps
    models = disk.model()
    intr_models = disk.model(ignore_beam=True)
    if disk.dc is None:
        vmod = kin.bin(models)
        vmod_map = kin.remap(vmod, mask=kin.vel_mask)
        vmod_intr = kin.bin(intr_models)
        vmod_intr_map = kin.remap(vmod_intr, mask=kin.vel_mask)
        smod = None
        smod_map = None
        smod_intr = None
        smod_intr_map = None
    else:
        vmod = kin.bin(models[0])
        vmod_map = kin.remap(vmod, mask=kin.vel_mask)
        vmod_intr = kin.bin(intr_models[0])
        vmod_intr_map = kin.remap(vmod_intr, mask=kin.vel_mask)
        smod = kin.bin(models[1])
        smod_map = kin.remap(smod, mask=kin.sig_mask)
        smod_intr = kin.bin(intr_models[1])
        smod_intr_map = kin.remap(smod_intr, mask=kin.sig_mask)

    # Get the projected rotational velocity
    #   - Disk-plane coordinates
    r, th = projected_polar(kin.x - disk.par[0], kin.y - disk.par[1], *np.radians(disk.par[2:4]))
    #   - Mask for data along the major axis
    wedge = 30.
    major_gpm = data.util.select_major_axis(r, th, r_range='all', wedge=wedge)
    #   - Projected rotation velocities
    indx = major_gpm & np.logical_not(kin.vel_mask)
    vrot_r = r[indx]
    vrot = (kin.vel[indx] - disk.par[4])/np.cos(th[indx])
    vrot_wgt = kin.vel_ivar[indx]*np.cos(th[indx])**2
    vrot_err = np.sqrt(1/vrot_wgt)
    vrot_mod = (vmod[indx] - disk.par[4])/np.cos(th[indx])

    # Get the binned data and the 1D model profiles
    fwhm = galmeta.psf_fwhm[1]  # Selects r band!
    maxr = np.amax(r)
    modelr = np.arange(0, maxr, 0.1)
    binr = np.arange(fwhm/2, maxr, fwhm)
    binw = np.full(binr.size, fwhm, dtype=float)
    indx = major_gpm & np.logical_not(kin.vel_mask)
    _, vrot_uwmed, vrot_uwmad, _, _, _, _, vrot_ewmean, vrot_ewsdev, vrot_ewerr, vrot_ntot, \
        vrot_nbin, vrot_bin_gpm = data.util.bin_stats(vrot_r, vrot, binr, binw, wgts=vrot_wgt,
                                                      fill_value=0.0) 
    # Construct the binned model RC using the same weights
    _, vrotm_uwmed, vrotm_uwmad, _, _, _, _, vrotm_ewmean, vrotm_ewsdev, vrotm_ewerr, vrotm_ntot, \
        vrotm_nbin, _ = data.util.bin_stats(vrot_r[vrot_bin_gpm], vrot_mod[vrot_bin_gpm], binr,
                                            binw, wgts=vrot_wgt[vrot_bin_gpm], fill_value=0.0) 
    # Finely sampled 1D model rotation curve
    vrot_intr_model = disk.rc.sample(modelr, par=disk.rc_par())

    if smod is not None:
        indx = np.logical_not(kin.sig_mask) & (kin.sig_phys2 > 0)
        sprof_r = r[indx]
        sprof = np.sqrt(kin.sig_phys2[indx])
        sprof_wgt = 4*kin.sig_phys2[indx]*kin.sig_phys2_ivar[indx]
        sprof_err = np.sqrt(1/sprof_wgt)
        _, sprof_uwmed, sprof_uwmad, _, _, _, _, sprof_ewmean, sprof_ewsdev, sprof_ewerr, \
            sprof_ntot, sprof_nbin, sprof_bin_gpm \
                    = data.util.bin_stats(sprof_r, sprof, binr, binw, wgts=sprof_wgt,
                                          fill_value=0.0) 
        # Construct the binned model dispersion profile using the same weights
        _, sprofm_uwmed, sprofm_uwmad, _, _, _, _, sprofm_ewmean, sprofm_ewsdev, sprofm_ewerr, \
            sprofm_ntot, sprofm_nbin, _ \
                    = data.util.bin_stats(r[indx][sprof_bin_gpm], smod[indx][sprof_bin_gpm], binr,
                                          binw, wgts=sprof_wgt[sprof_bin_gpm], fill_value=0.0) 
        # Finely sampled 1D model dispersion profile
        sprof_intr_model = disk.dc.sample(modelr, par=disk.dc_par())

    # Set the extent for the 2D maps
    extent = [np.amax(kin.grid_x), np.amin(kin.grid_x), np.amin(kin.grid_y), np.amax(kin.grid_y)]
    Dx = max(extent[0]-extent[1], extent[3]-extent[2]) # *1.01
    skylim = np.array([ (extent[0]+extent[1] - Dx)/2., 0.0 ])
    skylim[1] = skylim[0] + Dx

    # Create the plot
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2*w,2*h))

    # Patch for the FWHM of the beam to plot on all maps

    #-------------------------------------------------------------------
    # Surface-brightness
    sb_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(sb_map), 0.90, 1.05))
    sb_lim = data.util.atleast_one_decade(sb_lim)
    
    ax = plotting.init_ax(fig, [0.02, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plotting.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sb_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[0], vmax=sb_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\mu$', ha='right', va='center', transform=cax.transAxes)

#    ax.text(0.5, 0.9, r'$I_{{\rm H}\alpha}\ (10^{-17}\ {\rm erg/s/cm}^2{\rm /spaxel})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # S/N
    snr_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(snr_map), 0.90, 1.05))
    snr_lim = data.util.atleast_one_decade(snr_lim)

    ax = plotting.init_ax(fig, [0.02, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plotting.rotate_y_ticks(ax, 90, 'center')
#    ax.set_yticklabels(ax.get_yticks(), rotation=90, va='center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(snr_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=snr_lim[0], vmax=snr_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, 'S/N', ha='right', va='center', transform=cax.transAxes)

#    ax.text(0.5, 0.9, r'$B_{r}\ (10^{-17}\ {\rm erg/s/cm}^2{\rm /spaxel/ang})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Velocity
    vel_lim = data.util.growth_lim(np.ma.append(v_map, vmod_map), 0.90, 1.05,
                                   midpoint=disk.par[4])
#    mult = get_multiple(vel_lim)

    ax = plotting.init_ax(fig, [0.215, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)
#    axt = ax.twiny()
#    axt.set_xlim(skylim[::-1])
#    axt.set_ylim(skylim)

#    ax.text(0.5, 0.9, r'$V_{{\rm obs, H}\alpha}\ ({\rm km/s})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plotting.init_ax(fig, [0.215, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(s_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Model
    ax = plotting.init_ax(fig, [0.410, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(vmod_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)

#    ax.text(0.5, 0.9, r'$V_{{\rm obs, H}\alpha}\ ({\rm km/s})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plotting.init_ax(fig, [0.410, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(smod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)


    #-------------------------------------------------------------------
    # Velocity Model Residuals
    v_resid = v_map - vmod_map
    v_res_lim = data.util.growth_lim(v_resid, 0.80, 1.15, midpoint=0.0)

    ax = plotting.init_ax(fig, [0.605, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=v_res_lim[0], vmax=v_res_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\Delta V$', ha='right', va='center', transform=cax.transAxes)

#    ax.text(0.5, 0.9, r'$V_{{\rm obs, H}\alpha}\ ({\rm km/s})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Velocity Dispersion Residuals
    s_resid = s_map - smod_map
    s_res_lim = data.util.growth_lim(s_resid, 0.80, 1.15, midpoint=0.0)

    ax = plotting.init_ax(fig, [0.605, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(s_resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=s_res_lim[0], vmax=s_res_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal') #, format=logformatter)
    cax.text(-0.05, 0.1, r'$\Delta\sigma$', ha='right', va='center', transform=cax.transAxes)


    #-------------------------------------------------------------------
    # Velocity Model Chi-square
    v_chi = np.ma.divide(np.absolute(v_resid), v_err_map)
    v_chi_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(v_chi), 0.90, 1.15))
    v_chi_lim = data.util.atleast_one_decade(v_chi_lim)

    ax = plotting.init_ax(fig, [0.800, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_chi, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=v_chi_lim[0], vmax=v_chi_lim[1]),
                   zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\chi(V)$', ha='right', va='center', transform=cax.transAxes)

#    ax.text(0.5, 0.9, r'$V_{{\rm obs, H}\alpha}\ ({\rm km/s})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Velocity Dispersion Model Chi-square
    s_chi = np.ma.divide(np.absolute(s_resid), s_err_map)
    s_chi_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(s_chi), 0.90, 1.15))
    s_chi_lim = data.util.atleast_one_decade(s_chi_lim)

    ax = plotting.init_ax(fig, [0.800, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(s_chi, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=s_chi_lim[0], vmax=s_chi_lim[1]),
                   zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, r'$\chi(\sigma)$', ha='right', va='center', transform=cax.transAxes)




    #-------------------------------------------------------------------
    # Intrinsic Velocity Model
    ax = plotting.init_ax(fig, [0.800, 0.305, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.50, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    im = ax.imshow(vmod_intr_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)

    ax.text(0.5, 1.2, 'Intrinsic Model', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)


#    ax.text(0.5, 0.9, r'$V_{{\rm obs, H}\alpha}\ ({\rm km/s})$',
#            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Intrinsic Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plotting.init_ax(fig, [0.800, 0.110, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.10, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    im = ax.imshow(smod_intr_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)


    #-------------------------------------------------------------------
    # Rotation curve
    r_lim = [0.0, np.amax(np.append(binr[vrot_nbin > 5], binr[sprof_nbin > 5]))*1.1]
    rc_lim = [0.0, np.amax(vrot_ewmean[vrot_nbin > 5])*1.1]

    reff_lines = np.arange(galmeta.reff, r_lim[1], galmeta.reff)

    ax = plotting.init_ax(fig, [0.27, 0.27, 0.51, 0.23], facecolor='0.9', top=False, right=False)
    ax.set_xlim(r_lim)
    ax.set_ylim(rc_lim)
    plotting.rotate_y_ticks(ax, 90, 'center')
    if smod is None:
        ax.text(0.5, -0.13, r'$R$ [arcsec]', horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10)
    else:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

    indx = vrot_nbin > 0
    ax.scatter(vrot_r, vrot, marker='.', color='k', s=30, lw=0, alpha=0.6, zorder=1)
    ax.scatter(binr[indx], vrot_ewmean[indx], marker='o', edgecolors='none', s=100, alpha=1.0,
               facecolors='0.5', zorder=3)
    ax.scatter(binr[indx], vrotm_ewmean[indx], edgecolors='C3', marker='o', lw=3, s=100,
               alpha=1.0, facecolors='none', zorder=4)
    ax.errorbar(binr[indx], vrot_ewmean[indx], yerr=vrot_ewsdev[indx], color='0.6', capsize=0,
                linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrot_intr_model, color='C3', zorder=5, lw=0.5)
    for l in reff_lines:
        ax.axvline(x=l, linestyle='--', lw=0.5, zorder=2, color='k')

    axt = plotting.get_twin(ax, 'x')
    axt.set_xlim(np.array(r_lim) * galmeta.kpc_per_arcsec())
    axt.set_ylim(rc_lim)
    ax.text(0.5, 1.14, r'$R$ [$h^{-1}$ kpc]',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10)

    kin_inc = disk.par[3]
    axt = plotting.get_twin(ax, 'y')
    axt.set_xlim(r_lim)
    axt.set_ylim(np.array(rc_lim)/np.sin(np.radians(kin_inc)))
    plotting.rotate_y_ticks(axt, 90, 'center')
    axt.spines['right'].set_color('0.4')
    axt.tick_params(which='both', axis='y', colors='0.4')
    axt.yaxis.label.set_color('0.4')

    ax.add_patch(patches.Rectangle((0.62,0.03), 0.36, 0.19, facecolor='w', lw=0, edgecolor='none',
                                   zorder=5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.97, 0.13, r'$V_{\rm rot}\ \sin i$ [km/s; left axis]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)
    ax.text(0.97, 0.04, r'$V_{\rm rot}$ [km/s; right axis]', ha='right', va='bottom', color='0.4',
            transform=ax.transAxes, fontsize=10, zorder=6)

#    ax.text(0.97, 0.6, r'$V_{{\rm H}\alpha}$', color='r',
#            horizontalalignment='center', verticalalignment='center', rotation='vertical',
#            transform=ax.transAxes, fontsize=16)
#    ax.text(0.97, 0.4, r'$V_\ast$', color='b',
#            horizontalalignment='center', verticalalignment='center', rotation='vertical',
#            transform=ax.transAxes, fontsize=16)
#
#    if reff is not None:
#        axt = ax.twiny()
#        axt.set_xlim(np.array(r_lim)/reff)
#        axt.set_ylim(rc_lim)
#        ax.text(0.5, 1.15, r'$R/R_{\rm eff}$',
#                horizontalalignment='center', verticalalignment='center',
#                transform=ax.transAxes, fontsize=16)

    #-------------------------------------------------------------------
    # Velocity Dispersion profile
    if smod is not None:
        sprof_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(sprof_ewmean[sprof_nbin > 5]),
                                                        0.9, 1.5))
        sprof_lim = data.util.atleast_one_decade(sprof_lim)

        ax = plotting.init_ax(fig, [0.27, 0.04, 0.51, 0.23], facecolor='0.9')
        ax.set_xlim(r_lim)
        ax.set_ylim(sprof_lim)#[10,275])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(logformatter)
        plotting.rotate_y_ticks(ax, 90, 'center')

        indx = sprof_nbin > 0
        ax.scatter(sprof_r, sprof, marker='.', color='k', s=30, lw=0, alpha=0.6, zorder=1)
        ax.scatter(binr[indx], sprof_ewmean[indx], marker='o', edgecolors='none', s=100, alpha=1.0,
                   facecolors='0.5', zorder=3)
        ax.scatter(binr[indx], sprofm_ewmean[indx], edgecolors='C3', marker='o', lw=3, s=100,
                   alpha=1.0, facecolors='none', zorder=4)
        ax.errorbar(binr[indx], sprof_ewmean[indx], yerr=sprof_ewsdev[indx], color='0.6',
                    capsize=0, linestyle='', linewidth=1, alpha=1.0, zorder=2)
        ax.plot(modelr, sprof_intr_model, color='C3', zorder=5, lw=0.5)
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=2, color='k')

        ax.text(0.5, -0.13, r'$R$ [arcsec]', horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10)

        ax.add_patch(patches.Rectangle((0.81,0.86), 0.17, 0.09, facecolor='w', lw=0, edgecolor='none',
                                    zorder=5, alpha=0.7, transform=ax.transAxes))
        ax.text(0.97, 0.87, r'$\sigma_{\rm los}$ [km/s]', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=10, zorder=6)

    #-------------------------------------------------------------------
    # SDSS image
    ax = fig.add_axes([0.01, 0.29, 0.23, 0.23])
    if kin.image is not None:
        ax.imshow(kin.image)
    else:
        ax.text(0.5, 0.5, 'No Image', horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=20)

    ax.text(0.5, 1.05, 'SDSS gri Composite', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=10)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if galmeta.primaryplus:
        sample='Primary+'
    elif galmeta.secondary:
        sample='Secondary'
    elif galmeta.ancillary:
        sample='Ancillary'
    else:
        sample='Filler'

    ax.text(0.00, -0.05, 'MaNGA ID:',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.05, f'{galmeta.mangaid}',
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(0.00, -0.13, 'Observation:',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.13, f'{galmeta.plate}-{galmeta.ifu}',
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(0.00, -0.21, 'Sample:',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.21, f'{sample}',
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Redshift
    ax.text(0.00, -0.29, 'Redshift:',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.29, '{0:.4f}'.format(galmeta.z),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Mag
    ax.text(0.00, -0.37, 'Mag (N,r,i):',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.37, '{0:.1f}/{1:.1f}/{2:.1f}'.format(*galmeta.mag),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)

    # PSF FWHM
    ax.text(0.00, -0.45, 'FWHM (g,r):',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.45, '{0:.2f}, {1:.2f}'.format(*galmeta.psf_fwhm[:2]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)


    # Sersic n
    ax.text(0.00, -0.53, r'Sersic $n$:',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.53, '{0:.2f}'.format(galmeta.sersic_n),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Stellar Mass
    ax.text(0.00, -0.61, r'$\log(\mathcal{M}_\ast/\mathcal{M}_\odot$):',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.61, '{0:.2f}'.format(np.log10(galmeta.mass)),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Phot Inclination
    ax.text(0.00, -0.69, r'$i_{\rm phot}$ [deg]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.69, '{0:.1f}'.format(galmeta.guess_inclination()),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)

    # Fitted center
    ax.text(0.00, -0.77, r'$x_0$ [arcsec]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.77, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[0], disk.par_err[0]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(0.00, -0.85, r'$y_0$ [arcsec]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.85, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[1], disk.par_err[1]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # PA
    ax.text(0.00, -0.93, r'$\phi_0$ [deg]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.93, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[2], disk.par_err[2]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Kinematic Inclination
    ax.text(0.00, -1.01, r'$i_{\rm kin}$ [deg]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.01, r'{0:.1f} $\pm$ {1:.1f}'.format(disk.par[3], disk.par_err[3]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    # Systemic velocity
    ax.text(0.00, -1.09, r'$V_{\rm sys}$ [km/s]',
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.09, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[4], disk.par_err[4]),
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
            fontsize=10)

    # TODO:
    # Room for ~4-6 more...
    # Add errors (if available)
    # Label vertical lines in profile plots as 1, 2 Re, etc
    # Surface brightness units
    # Unit in general
    pyplot.show()
    return 
    exit()

    if pdf:
        ofile = 'vf_summary/{0}_{1}_{2}.pdf'.format(plt, ifu, daptype)
        print('writing: {0}'.format(ofile))
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    else:
        pyplot.show()
    fig.clear()
    pyplot.close(fig)







