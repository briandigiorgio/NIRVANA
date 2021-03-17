"""
Script that runs the axisymmetric, least-squares fit for MaNGA data.
"""
import os
import argparse

from IPython import embed

import numpy as np

from matplotlib import pyplot, rc, patches, ticker, colors

from astropy.io import fits

# For versioning
import sys
import scipy
import astropy
from .. import __version__

from .. import data
from ..util import plot
from ..util.bitmask import BitMask
from ..util import fileio
from ..models.geometry import projected_polar
from ..models.oned import HyperbolicTangent, Exponential, ExpBase, Const, PolyEx
from ..models.axisym import AxisymmetricDisk

# TODO: Setup a logger
# TODO: Need to test different modes.
#   Tested so far:
#       - Fit Gas or stars
#       - Fit with psf and velocity dispersion
#   Need to test:
#       - Fit without psf
#       - Fit with covariance
#       - Fit without velocity dispersion


class AxisymmetricDiskFitBitMask(BitMask):
    """
    Bin-by-bin mask used to track axisymmetric disk fit rejections.
    """
    def __init__(self):
        # TODO: np.array just used for slicing convenience
        mask_def = np.array([['DIDNOTUSE', 'Data not used because it was flagged on input.'],
                             ['REJ_ERR', 'Data rejected because of its large measurement error.'],
                             ['REJ_SNR', 'Data rejected because of its low signal-to-noise.'],
                             ['REJ_UNR', 'Data rejected after first iteration and are so ' \
                                         'discrepant from the other data that we expect the ' \
                                         'measurements are unreliable.'],
                             ['REJ_RESID', 'Data rejected due to iterative rejection process ' \
                                           'of model residuals.']])
        super().__init__(mask_def[:,0], descr=mask_def[:,1])

    @staticmethod
    def base_flags():
        """
        Return the list of "base-level" flags that are *always* ignored,
        regardless of the fit iteration.
        """
        return ['DIDNOTUSE', 'REJ_ERR', 'REJ_SNR', 'REJ_UNR']


class AxisymmetricDiskGlobalBitMask(BitMask):
    """
    Fit-wide quality flag.
    """
    def __init__(self):
        # TODO: np.array just used for slicing convenience
        mask_def = np.array([['LOWINC', 'Fit has an erroneously low inclination']])
        super().__init__(mask_def[:,0], descr=mask_def[:,1])


def disk_fit_reject(kin, disk, disp=None, vel_mask=None, vel_sigma_rej=5, show_vel=False,
                    vel_plot=None, sig_mask=None, sig_sigma_rej=5, show_sig=False, sig_plot=None,
                    rej_flag='REJ_RESID', verbose=False):
    """
    Reject kinematic data based on the error-weighted residuals with respect
    to a disk model.

    The rejection iteration is done using
    :class:`~nirvana.data.scatter.IntrinsicScatter`, independently for the
    velocity and velocity dispersion measurements (if the latter is selected
    and/or available).

    Note that you can both show the QA plots and have them written to a file
    (e.g., ``show_vel`` can be True and ``vel_plot`` can provide a file).
    
    Args:
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data being fit.
        disk (:class:`~nirvana.models.axisym.AxisymmetricDisk`):
            Object that performed the fit and has the best-fitting parameters.
        disp (:obj:`bool`, optional):
            Flag to include the velocity dispersion rejection in the
            iteration. If None, rejection is included if ``kin`` has velocity
            dispersion data and ``disk`` has a disperion parameterization.
        vel_mask (`numpy.ndarray`_):
            Bitmask used to track velocity rejections.
        vel_sigma_rej (:obj:`float`, optional):
            Rejection sigma for the velocity measurements.
        show_vel (:obj:`bool`, optional):
            Show the QA plot for the velocity rejection (see
            :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
        vel_plot (:obj:`str`, optional):
            Write the QA plot for the velocity rejection to this file (see
            :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
        sig_mask (`numpy.ndarray`_):
            Bitmask used to track dispersion rejections.
        sig_sigma_rej (:obj:`float`, optional):
            Rejection sigma for the dispersion measurements.
        show_sig (:obj:`bool`, optional):
            Show the QA plot for the velocity dispersion rejection (see
            :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
        sig_plot (:obj:`str`, optional):
            Write the QA plot for the velocity dispersion rejection to this
            file (see :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
        rej_flag (:obj:`str`, optional):
            Rejection flag giving the reason these data were rejected. Must
            be a valid flag for :class:`AxisymmetricDiskFitBitMask`.
        verbose (:obj:`bool`, optional):
            Verbose scatter fitting output.

    Returns:
        :obj:`tuple`: Returns two pairs of objects, one for each kinematic
        moment. The first object is the vector flagging the data that should
        be rejected and the second is the estimated intrinsic scatter about
        the model. If the dispersion is not included in the rejection, the
        last two objects returned are both None.
    """
    # Instantiate the bitmask.
    # TODO: This is cheap to redo everytime, but could also make it a part of
    # the AxisymmetricDisk class...
    disk_bm = AxisymmetricDiskFitBitMask()
    # Get the models
    models = disk.model()
    _verbose = 2 if verbose else 0

    # Reject based on error-weighted residuals, accounting for intrinsic
    # scatter
    vmod = models[0] if len(models) == 2 else models
    resid = kin.vel - kin.bin(vmod)
    err = 1/np.sqrt(kin.vel_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm, npar=disk.nfree)
    vel_sig, vel_rej, vel_gpm = scat.iter_fit(sigma_rej=vel_sigma_rej, fititer=5, verbose=_verbose)
    # Incorporate into mask
    if vel_mask is not None and np.any(vel_rej):
        vel_mask[vel_rej] = disk_bm.turn_on(vel_mask[vel_rej], rej_flag)
    # Show and/or plot the result, if requested
    if show_vel:
        scat.show()
    if vel_plot is not None:
        scat.show(ofile=vel_plot)

    if disp is None:
        disp = kin.sig is not None and disk.dc is not None
    if not disp:
        # Not rejecting dispersion so we're done
        return vel_rej, vel_sig, None, None

    # Reject based on error-weighted residuals, accounting for intrinsic
    # scatter
    resid = kin.sig_phys2 - kin.bin(models[1])**2
    err = 1/np.ma.sqrt(kin.sig_phys2_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.sig_gpm, npar=disk.nfree)
    sig_sig, sig_rej, sig_gpm = scat.iter_fit(sigma_rej=sig_sigma_rej, fititer=5, verbose=_verbose)
    # Incorporate into mask
    if sig_mask is not None and np.any(sig_rej):
        sig_mask[sig_rej] = disk_bm.turn_on(sig_mask[sig_rej], rej_flag)
    # Show and/or plot the result, if requested
    if show_sig:
        scat.show()
    if sig_plot is not None:
        scat.show(ofile=sig_plot)

    return vel_rej, vel_sig, sig_rej, sig_sig


# TODO: Consolidate this function with the one above
def disk_fit_resid_dist(kin, disk, disp=None, vel_mask=None, show_vel=False, vel_plot=None,
                        sig_mask=None, show_sig=False, sig_plot=None):
    """
    """
    # Get the models
    models = disk.model()

    # Show the error-normalized distributions for the velocity-field residuals
    vmod = models[0] if len(models) == 2 else models
    resid = kin.vel - kin.bin(vmod)
    err = 1/np.sqrt(kin.vel_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm, npar=disk.nfree)
    scat.sig = 0. if disk.scatter is None else disk.scatter[0]
    scat.rej = np.zeros(resid.size, dtype=bool) if vel_mask is None else vel_mask > 0
    # Show and/or plot the result, if requested
    if show_vel:
        scat.show(title='Velocity field residuals')
    if vel_plot is not None:
        scat.show(ofile=vel_plot, title='Velocity field residuals')

    # Decide if we're done
    if disp is None:
        disp = kin.sig is not None and disk.dc is not None
    if not disp:
        # Yep
        return

    # Show the error-normalized distributions for the dispersion residuals
    resid = kin.sig_phys2 - kin.bin(models[1])**2
    err = 1/np.ma.sqrt(kin.sig_phys2_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.sig_gpm, npar=disk.nfree)
    scat.sig = 0. if disk.scatter is None else disk.scatter[1]
    scat.rej = np.zeros(resid.size, dtype=bool) if sig_mask is None else sig_mask > 0
    # Show and/or plot the result, if requested
    if show_sig:
        scat.show(title='Dispersion field residuals')
    if sig_plot is not None:
        scat.show(ofile=sig_plot, title='Dispersion field residuals')


def reset_to_base_flags(kin, vel_mask, sig_mask):
    """
    Reset the masks to only include the "base" flags.
    
    As the best-fit parameters change over the course of a set of rejection
    iterations, the residuals with respect to the model change. This method
    resets the flags back to the base-level rejection (i.e., independent of
    the model), allowing the rejection to be based on the most recent set of
    parameters and potentially recovering good data that was previously
    rejected because of a poor model fit.

    .. warning::
        The objects are *all* modified in place.

    Args:
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data being fit.
        vel_mask (`numpy.ndarray`_):
            Bitmask used to track velocity rejections.
        sig_mask (`numpy.ndarray`_):
            Bitmask used to track dispersion rejections. Can be None.
    """
    # Instantiate the bitmask.
    # TODO: This is cheap to redo everytime, but could also make it a part of
    # the AxisymmetricDisk class...
    disk_bm = AxisymmetricDiskFitBitMask()
    # Turn off the relevant rejection for all pixels
    vel_mask = disk_bm.turn_off(vel_mask, flag='REJ_RESID')
    # Reset the data mask held by the Kinematics object
    kin.vel_mask = disk_bm.flagged(vel_mask, flag=disk_bm.base_flags())
    if sig_mask is not None:
        # Turn off the relevant rejection for all pixels
        sig_mask = disk_bm.turn_off(sig_mask, flag='REJ_RESID')
        # Reset the data mask held by the Kinematics object
        kin.sig_mask = disk_bm.flagged(sig_mask, flag=disk_bm.base_flags())


def initialize_primary_header(galmeta):
    hdr = fits.Header()

    hdr['MANGADR'] = (galmeta.dr, 'MaNGA Data Release')
    hdr['MANGAID'] = (galmeta.mangaid, 'MaNGA ID number')
    hdr['PLATEIFU'] = (f'{galmeta.plate}-{galmeta.ifu}', 'MaNGA observation plate and IFU')

    # Add versioning
    hdr['VERSPY'] = ('.'.join([ str(v) for v in sys.version_info[:3]]), 'Python version')
    hdr['VERSNP'] = (np.__version__, 'Numpy version')
    hdr['VERSSCI'] = (scipy.__version__, 'Scipy version')
    hdr['VERSAST'] = (astropy.__version__, 'Astropy version')
    hdr['VERSNIRV'] = (__version__, 'NIRVANA version')

    return hdr


def add_wcs(hdr, kin):
    if kin.grid_wcs is None:
        return hdr
    return hdr + kin.grid_wcs.to_header()


def finalize_header(hdr, ext, bunit=None, hduclas2='DATA', err=False, qual=False, bm=None,
                    bit_type=None, prepend=True):

    # Don't change the input header
    _hdr = hdr.copy()

    # Add the units
    if bunit is not None:
        _hdr['BUNIT'] = (bunit, 'Unit of pixel value')

    # Add the common HDUCLASS keys
    _hdr['HDUCLASS'] = ('SDSS', 'SDSS format class')
    _hdr['HDUCLAS1'] = ('IMAGE', 'Data format')
    if hduclas2 == 'DATA':
        _hdr['HDUCLAS2'] = 'DATA'
        if err:
            _hdr['ERRDATA'] = (ext+'_IVAR' if prepend else 'IVAR',
                                'Associated inv. variance extension')
        if qual:
            _hdr['QUALDATA'] = (ext+'_MASK' if prepend else 'MASK',
                                'Associated quality extension')
        return _hdr

    if hduclas2 == 'ERROR':
        _hdr['HDUCLAS2'] = 'ERROR'
        _hdr['HDUCLAS3'] = ('INVMSE', 'Value is inverse mean-square error')
        _hdr['SCIDATA'] = (ext, 'Associated data extension')
        if qual:
            _hdr['QUALDATA'] = (ext+'_MASK' if prepend else 'MASK',
                                'Associated quality extension')
        return _hdr

    if hduclas2 == 'QUALITY':
        _hdr['HDUCLAS2'] = 'QUALITY'
        if bit_type is None:
            if bm is None:
                raise ValueError('Must provide the bit type or the bitmask object.')
            else:
                bit_type = bm.minimum_dtype()
        _hdr['HDUCLAS3'] = mask_data_type(bit_type)
        _hdr['SCIDATA'] = (ext, 'Associated data extension')
        if err:
            _hdr['ERRDATA'] = (ext+'_IVAR' if prepend else 'IVAR',
                                'Associated inv. variance extension')
        if bm is not None:
            # Add the bit values
            bm.to_header(_hdr)
        return _hdr
            
    raise ValueError('HDUCLAS2 must be DATA, ERROR, or QUALITY.')


def build_map_header(hdr, author, multichannel=False, maskname=None):
    hdr = hdr.copy()
    hdr = DAPFitsUtil.clean_map_header(hdr, multichannel=multichannel)
    hdr['AUTHOR'] = author
    if maskname is not None:
        hdr['MASKNAME'] = maskname
    return hdr


def mask_data_type(bit_type):
    if bit_type in [np.uint64, np.int64]:
        return ('FLAG64BIT', '64-bit mask')
    if bit_type in [np.uint32, np.int32]:
        return ('FLAG32BIT', '32-bit mask')
    if bit_type in [np.uint16, np.int16]:
        return ('FLAG16BIT', '16-bit mask')
    if bit_type in [np.uint8, np.int8]:
        return ('FLAG8BIT', '8-bit mask')
    if bit_type == np.bool:
        return ('MASKZERO', 'Binary mask; zero values are good/unmasked')
    raise ValueError('Invalid bit_type: {0}!'.format(str(bit_type)))


def _fit_meta_dtype(par_names):
    """
    """
    gp = [(f'G_{n}'.upper(), np.float) for n in par_names]
    bp = [(f'F_{n}'.upper(), np.float) for n in par_names]
    bpe = [(f'E_{n}'.upper(), np.float) for n in par_names]
    
    return [('MANGAID', '<U30'),
            ('PLATE', np.int16),
            ('IFU', np.int16),
            ('OBJRA', np.float),
            ('OBJDEC', np.float),
            ('Z', np.float),
            ('ASEC2KPC', np.float),
            ('REFF', np.float),
            ('SERSICN', np.float),
            ('PA', np.float),
            ('ELL', np.float),
            ('Q0', np.float),
            ('NVEL', np.int),
            ('VSCT', np.float),
            ('VCHI2', np.float),
            ('NSIG', np.int),
            ('SSCT', np.float),
            ('SCHI2', np.float),
            ('CHI2', np.float),
            ('RCHI2', np.float)] + gp + bp + bpe


def axisym_fit_data(galmeta, kin, p0, disk, ofile, vmask, smask, compress=True):
    """
    Construct a fits file with the results.
    """
    # Instantiate the bitmask.
    # TODO: This is cheap to redo everytime, but could also make it a part of
    # the AxisymmetricDisk class...
    disk_bm = AxisymmetricDiskFitBitMask()

    # Rebuild the 2D maps
    #   - Bin ID
    binid = kin.remap('binid', masked=False, fill_value=-1)
    #   - Disk-plane coordinates
    r, th = projected_polar(kin.grid_x - disk.par[0], kin.grid_y - disk.par[1],
                            *np.radians(disk.par[2:4]))
    #   - Surface-brightness (in per spaxel units not per sq. arcsec).
    didnotuse = disk_bm.minimum_dtype()(disk_bm.turn_on(0, flag='DIDNOTUSE'))
    sb = kin.remap('sb', masked=False, fill_value=0.)
    sb_ivar = kin.remap('sb_ivar', masked=False, fill_value=0.)
    _mask = kin.remap('sb_mask', masked=False, fill_value=True)
    sb_mask = disk_bm.init_mask_array(sb.shape)
    sb_mask[_mask] = disk_bm.turn_on(sb_mask[_mask], flag='DIDNOTUSE')
    #   - Velocity
    vel = kin.remap('vel', masked=False, fill_value=0.)
    vel_ivar = kin.remap('vel_ivar', masked=False, fill_value=0.)
    vel_mask = kin.remap(vmask, masked=False, fill_value=didnotuse)
    #   - Corrected velocity dispersion squared
    sigsqr = None if disk.dc is None else kin.remap('sig_phys2', masked=False, fill_value=0.)
    sigsqr_ivar = None if disk.dc is None \
                        else kin.remap('sig_phys2_ivar', masked=False,fill_value=0.)
    sigsqr_mask = None if disk.dc is None or smask is None \
                        else kin.remap(smask, masked=False, fill_value=didnotuse)

    # Best-fit model, both binned data and maps
    # TODO: Don't bin the intrinsic model?
    # TODO: Include the binned radial profiles shown in the output plot?
    models = disk.model()
    intr_models = disk.model(ignore_beam=True)
    if disk.dc is None:
        vel_mod = kin.remap(kin.bin(models), masked=False, fill_value=0.)
        vel_mod_intr = kin.remap(kin.bin(intr_models), masked=False, fill_value=0.)
        sig_mod = None
        sig_mod_intr = None
    else:
        vel_mod = kin.remap(kin.bin(models[0]), masked=False, fill_value=0.)
        vel_mod_intr = kin.remap(kin.bin(intr_models[0]), masked=False, fill_value=0.)
        sig_mod = kin.remap(kin.bin(models[1]), masked=False, fill_value=0.)
        sig_mod_intr = kin.remap(kin.bin(intr_models[1]), masked=False, fill_value=0.)

    # Instantiate and fill the single-row table with the metadata:
    disk_par_names = disk.par_names(short=True)
    metadata = fileio.init_record_array(1, _fit_meta_dtype(disk_par_names))
    metadata['MANGAID'] = galmeta.mangaid
    metadata['PLATE'] = galmeta.plate
    metadata['IFU'] = galmeta.ifu
    metadata['OBJRA'] = galmeta.ra
    metadata['OBJDEC'] = galmeta.dec
    metadata['Z'] = galmeta.z
    metadata['ASEC2KPC'] = galmeta.kpc_per_arcsec()
    metadata['REFF'] = galmeta.reff
    metadata['SERSICN'] = galmeta.sersic_n
    metadata['PA'] = galmeta.pa
    metadata['ELL'] = galmeta.ell
    metadata['Q0'] = galmeta.q0
    
    vfom, sfom = disk._get_fom()(disk.par, sep=True)
    metadata['NVEL'] = np.sum(disk.vel_gpm)
    metadata['VSCT'] = 0.0 if disk.scatter is None else disk.scatter[0]
    metadata['VCHI2'] = np.sum(vfom**2)
    if disk.dc is not None:
        metadata['NSIG'] = np.sum(disk.sig_gpm)
        metadata['SSCT'] = 0.0 if disk.scatter is None else disk.scatter[1]
        metadata['SCHI2'] = np.sum(sfom**2)
    metadata['CHI2'] = metadata['VCHI2'] + metadata['SCHI2']    # SCHI2 is 0 if sigma not fit
    metadata['RCHI2'] = metadata['CHI2'] / (metadata['NVEL'] + metadata['NSIG'] - disk.np)

    for n, gp, p, pe in zip(disk_par_names, p0, disk.par, disk.par_err):
        metadata[f'G_{n}'.upper()] = gp
        metadata[f'F_{n}'.upper()] = p
        metadata[f'E_{n}'.upper()] = pe

    # Build the output fits extension (base) headers
    prihdr = initialize_primary_header(galmeta)
    maphdr = add_wcs(prihdr, kin)
    if kin.beam is None:
        psfhdr = None
    else:
        psfhdr = prihdr.copy()
        psfhdr['PSFNAME'] = (kin.psf_name, 'Original PSF name, if known')

    hdus = [fits.PrimaryHDU(header=prihdr),
            fits.ImageHDU(data=binid, header=finalize_header(maphdr, 'BINID'), name='BINID'),
            fits.ImageHDU(data=r, header=finalize_header(maphdr, 'R'), name='R'),
            fits.ImageHDU(data=th, header=finalize_header(maphdr, 'THETA'), name='THETA'),
            fits.ImageHDU(data=sb,
                          header=finalize_header(maphdr, 'FLUX',
                                                 bunit='1E-17 erg/s/cm^2/ang/spaxel', err=True,
                                                 qual=True),
                          name='FLUX'),
            fits.ImageHDU(data=sb_ivar,
                          header=finalize_header(maphdr, 'FLUX',
                                                 bunit='(1E-17 erg/s/cm^2/ang/spaxel)^{-2}',
                                                 hduclas2='ERROR', qual=True),
                          name='FLUX_IVAR'),
            fits.ImageHDU(data=sb_mask,
                          header=finalize_header(maphdr, 'FLUX', hduclas2='QUALITY', err=True,
                                                 bm=disk_bm),
                          name='FLUX_MASK'),
            fits.ImageHDU(data=vel,
                          header=finalize_header(maphdr, 'VEL', bunit='km/s', err=True, qual=True),
                          name='VEL'),
            fits.ImageHDU(data=vel_ivar,
                          header=finalize_header(maphdr, 'VEL', bunit='(km/s)^{-2}',
                                                 hduclas2='ERROR', qual=True),
                          name='VEL_IVAR'),
            fits.ImageHDU(data=vel_mask,
                          header=finalize_header(maphdr, 'VEL', hduclas2='QUALITY', err=True,
                                                 bm=disk_bm),
                          name='VEL_MASK'),
            fits.ImageHDU(data=vel_mod, header=finalize_header(maphdr, 'VEL_MOD', bunit='km/s'),
                          name='VEL_MOD'),
            fits.ImageHDU(data=vel_mod_intr,
                          header=finalize_header(maphdr, 'VEL_MODI', bunit='km/s'),
                          name='VEL_MODI')]

    if disk.dc is not None:
        hdus += [fits.ImageHDU(data=sigsqr,
                               header=finalize_header(maphdr, 'SIGSQR', bunit='(km/s)^2',
                                                      err=True, qual=True),
                               name='SIGSQR'),
                 fits.ImageHDU(data=sigsqr_ivar,
                          header=finalize_header(maphdr, 'SIGSQR', bunit='(km/s)^{-4}',
                                                 hduclas2='ERROR', qual=True),
                          name='SIGSQR_IVAR'),
            fits.ImageHDU(data=sigsqr_mask,
                          header=finalize_header(maphdr, 'SIGSQR', hduclas2='QUALITY', err=True,
                                                 bm=disk_bm),
                          name='SIGSQR_MASK'),
            fits.ImageHDU(data=sig_mod, header=finalize_header(maphdr, 'SIG_MOD', bunit='km/s'),
                          name='SIG_MOD'),
            fits.ImageHDU(data=sig_mod_intr,
                          header=finalize_header(maphdr, 'SIG_MODI', bunit='km/s'),
                          name='SIG_MODI')]

    if kin.beam is not None:
        hdus += [fits.ImageHDU(data=kin.beam, header=finalize_header(psfhdr, 'PSF'), name='PSF')]

    hdus += [fits.BinTableHDU.from_columns([fits.Column(name=n,
                                                        format=fileio.rec_to_fits_type(metadata[n]),
                                                        array=metadata[n])
                                             for n in metadata.dtype.names], name='FITMETA')]

    if ofile.split('.')[-1] == 'gz':
        _ofile = ofile[:ofile.rfind('.')] 
        compress = True
    else:
        _ofile = ofile

    fits.HDUList(hdus).writeto(_ofile, overwrite=True, checksum=True)
    if compress:
        fileio.compress_file(_ofile, overwrite=True)
        os.remove(_ofile)


# TODO: Add keyword for:
#   - Radial sampling for 1D model RCs and dispersion profiles
def axisym_fit_plot(galmeta, kin, disk, par=None, par_err=None, fix=None, ofile=None):
    """
    Construct the QA plot for the result of fitting an
    :class:`~nirvana.model.axisym.AxisymmetricDisk` model to a galaxy.

    """
    logformatter = plot.get_logformatter()

    # Change the style
    rc('font', size=8)

    _par = disk.par if par is None else par
    _par_err = disk.par_err if par_err is None else par_err
    _fix = np.zeros(disk.np, dtype=bool) if fix is None else fix

    disk.par = _par
    disk.par_err = _par_err

    # Get the fit statistics
    vfom, sfom = disk._get_fom()(disk.par, sep=True)
    nvel = np.sum(disk.vel_gpm)
    vsct = 0.0 if disk.scatter is None else disk.scatter[0]
    vchi2 = np.sum(vfom**2)
    if disk.dc is None:
        nsig = 0
        ssct = 0.
        schi2 = 0.
    else:
        nsig = np.sum(disk.sig_gpm)
        ssct = 0.0 if disk.scatter is None else disk.scatter[1]
        schi2 = np.sum(sfom**2)
    chi2 = vchi2 + schi2
    rchi2 = chi2 / (nvel + nsig - disk.np)

    # Rebuild the 2D maps
    sb_map = kin.remap('sb')
    snr_map = sb_map * np.sqrt(kin.remap('sb_ivar', mask=kin.sb_mask))
    v_map = kin.remap('vel')
    v_err_map = np.ma.power(kin.remap('vel_ivar', mask=kin.vel_mask), -0.5)
    s_map = np.ma.sqrt(kin.remap('sig_phys2', mask=kin.sig_mask))
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

    #-------------------------------------------------------------------
    # Surface-brightness
    sb_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(sb_map), 0.90, 1.05))
    sb_lim = data.util.atleast_one_decade(sb_lim)
    
    ax = plot.init_ax(fig, [0.02, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sb_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[0], vmax=sb_lim[1]), zorder=4)
    # TODO: For some reason, the combination of the use of a masked array and
    # setting the formatter to logformatter leads to weird behavior in the map.
    # Use something like the "pallete" object described here?
    #   https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\mu$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # S/N
    snr_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(snr_map), 0.90, 1.05))
    snr_lim = data.util.atleast_one_decade(snr_lim)

    ax = plot.init_ax(fig, [0.02, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(snr_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=snr_lim[0], vmax=snr_lim[1]), zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, 'S/N', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity
    vel_lim = data.util.growth_lim(np.ma.append(v_map, vmod_map), 0.90, 1.05,
                                   midpoint=disk.par[4])
    ax = plot.init_ax(fig, [0.215, 0.775, 0.19, 0.19])
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

    #-------------------------------------------------------------------
    # Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plot.init_ax(fig, [0.215, 0.580, 0.19, 0.19])
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
    ax = plot.init_ax(fig, [0.410, 0.775, 0.19, 0.19])
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

    #-------------------------------------------------------------------
    # Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plot.init_ax(fig, [0.410, 0.580, 0.19, 0.19])
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

    ax = plot.init_ax(fig, [0.605, 0.775, 0.19, 0.19])
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

    #-------------------------------------------------------------------
    # Velocity Dispersion Residuals
    s_resid = s_map - smod_map
    s_res_lim = data.util.growth_lim(s_resid, 0.80, 1.15, midpoint=0.0)

    ax = plot.init_ax(fig, [0.605, 0.580, 0.19, 0.19])
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

    ax = plot.init_ax(fig, [0.800, 0.775, 0.19, 0.19])
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
    cax.text(-0.02, 1.1, r'$|\Delta V|/\epsilon$', ha='right', va='center',
             transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion Model Chi-square
    s_chi = np.ma.divide(np.absolute(s_resid), s_err_map)
    s_chi_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(s_chi), 0.90, 1.15))
    s_chi_lim = data.util.atleast_one_decade(s_chi_lim)

    ax = plot.init_ax(fig, [0.800, 0.580, 0.19, 0.19])
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
    cax.text(-0.02, 0.4, r'$|\Delta \sigma|/\epsilon$', ha='right', va='center',
             transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Intrinsic Velocity Model
    ax = plot.init_ax(fig, [0.800, 0.305, 0.19, 0.19])
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

    #-------------------------------------------------------------------
    # Intrinsic Velocity Dispersion
    sig_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(np.ma.append(s_map, smod_map)),
                                                  0.80, 1.05))
    sig_lim = data.util.atleast_one_decade(sig_lim)

    ax = plot.init_ax(fig, [0.800, 0.110, 0.19, 0.19])
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

    # Annotate with the intrinsic scatter included
    ax.text(0.00, -0.2, r'V scatter, $\epsilon_v$:', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.00, -0.2, f'{vsct:.1f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(0.00, -0.3, r'$\sigma^2$ scatter, $\epsilon_{\sigma^2}$:', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.00, -0.3, f'{ssct:.1f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)

    #-------------------------------------------------------------------
    # Rotation curve
    r_lim = [0.0, np.amax(np.append(binr[vrot_nbin > 5], binr[sprof_nbin > 5]))*1.1]
    rc_lim = [0.0, np.amax(vrot_ewmean[vrot_nbin > 5])*1.1]

    reff_lines = np.arange(galmeta.reff, r_lim[1], galmeta.reff)

    ax = plot.init_ax(fig, [0.27, 0.27, 0.51, 0.23], facecolor='0.9', top=False, right=False)
    ax.set_xlim(r_lim)
    ax.set_ylim(rc_lim)
    plot.rotate_y_ticks(ax, 90, 'center')
    if smod is None:
        ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
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

    axt = plot.get_twin(ax, 'x')
    axt.set_xlim(np.array(r_lim) * galmeta.kpc_per_arcsec())
    axt.set_ylim(rc_lim)
    ax.text(0.5, 1.14, r'$R$ [$h^{-1}$ kpc]', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)

    kin_inc = disk.par[3]
    axt = plot.get_twin(ax, 'y')
    axt.set_xlim(r_lim)
    axt.set_ylim(np.array(rc_lim)/np.sin(np.radians(kin_inc)))
    plot.rotate_y_ticks(axt, 90, 'center')
    axt.spines['right'].set_color('0.4')
    axt.tick_params(which='both', axis='y', colors='0.4')
    axt.yaxis.label.set_color('0.4')

    ax.add_patch(patches.Rectangle((0.62,0.03), 0.36, 0.19, facecolor='w', lw=0, edgecolor='none',
                                   zorder=5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.97, 0.13, r'$V_{\rm rot}\ \sin i$ [km/s; left axis]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)
    ax.text(0.97, 0.04, r'$V_{\rm rot}$ [km/s; right axis]', ha='right', va='bottom', color='0.4',
            transform=ax.transAxes, fontsize=10, zorder=6)

    #-------------------------------------------------------------------
    # Velocity Dispersion profile
    if smod is not None:
        sprof_lim = np.power(10.0, data.util.growth_lim(np.ma.log10(sprof_ewmean[sprof_nbin > 5]),
                                                        0.9, 1.5))
        sprof_lim = data.util.atleast_one_decade(sprof_lim)

        ax = plot.init_ax(fig, [0.27, 0.04, 0.51, 0.23], facecolor='0.9')
        ax.set_xlim(r_lim)
        ax.set_ylim(sprof_lim)#[10,275])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(logformatter)
        plot.rotate_y_ticks(ax, 90, 'center')

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

        ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)

        ax.add_patch(patches.Rectangle((0.81,0.86), 0.17, 0.09, facecolor='w', lw=0,
                                       edgecolor='none', zorder=5, alpha=0.7,
                                       transform=ax.transAxes))
        ax.text(0.97, 0.87, r'$\sigma_{\rm los}$ [km/s]', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=10, zorder=6)

    #-------------------------------------------------------------------
    # SDSS image
    ax = fig.add_axes([0.01, 0.29, 0.23, 0.23])
    if kin.image is not None:
        ax.imshow(kin.image)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes,
                fontsize=20)

    ax.text(0.5, 1.05, 'SDSS gri Composite', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)
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

    # MaNGA ID
    ax.text(0.00, -0.05, 'MaNGA ID:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.05, f'{galmeta.mangaid}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    # Observation
    ax.text(0.00, -0.13, 'Observation:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.13, f'{galmeta.plate}-{galmeta.ifu}', ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sample selection
    ax.text(0.00, -0.21, 'Sample:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.21, f'{sample}', ha='right', va='center', transform=ax.transAxes, fontsize=10)
    # Redshift
    ax.text(0.00, -0.29, 'Redshift:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.29, '{0:.4f}'.format(galmeta.z), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Mag
    ax.text(0.00, -0.37, 'Mag (N,r,i):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.37, '{0:.1f}/{1:.1f}/{2:.1f}'.format(*galmeta.mag), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # PSF FWHM
    ax.text(0.00, -0.45, 'FWHM (g,r):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.45, '{0:.2f}, {1:.2f}'.format(*galmeta.psf_fwhm[:2]), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sersic n
    ax.text(0.00, -0.53, r'Sersic $n$:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.53, '{0:.2f}'.format(galmeta.sersic_n), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Stellar Mass
    ax.text(0.00, -0.61, r'$\log(\mathcal{M}_\ast/\mathcal{M}_\odot$):', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.61, '{0:.2f}'.format(np.log10(galmeta.mass)), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Phot Inclination
    ax.text(0.00, -0.69, r'$i_{\rm phot}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.69, '{0:.1f}'.format(galmeta.guess_inclination()), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Fitted center
    ax.text(0.00, -0.77, r'$x_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if fix[0] else 'k')
    ax.text(1.01, -0.77, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[0], disk.par_err[0]),
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if fix[0] else 'k')
    ax.text(0.00, -0.85, r'$y_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if fix[1] else 'k')
    ax.text(1.01, -0.85, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[1], disk.par_err[1]),
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if fix[1] else 'k')
    # Position angle
    ax.text(0.00, -0.93, r'$\phi_0$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if fix[2] else 'k')
    ax.text(1.01, -0.93, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[2], disk.par_err[2]),
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if fix[2] else 'k')
    # Kinematic Inclination
    ax.text(0.00, -1.01, r'$i_{\rm kin}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if fix[3] else 'k')
    ax.text(1.01, -1.01, r'{0:.1f} $\pm$ {1:.1f}'.format(disk.par[3], disk.par_err[3]),
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if fix[3] else 'k')
    # Systemic velocity
    ax.text(0.00, -1.09, r'$V_{\rm sys}$ [km/s]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if fix[4] else 'k')
    ax.text(1.01, -1.09, r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[4], disk.par_err[4]),
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if fix[4] else 'k')
    # Reduced chi-square
    ax.text(0.00, -1.17, r'$\chi^2_\nu$', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.17, f'{rchi2:.2f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)

    # TODO:
    # Room for ~4-6 more...
    # Add errors (if available)
    # Label vertical lines in profile plots as 1, 2 Re, etc
    # Surface brightness units
    # Unit in general

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

    # Reset to default style
    pyplot.rcdefaults()


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
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity level.  0=only status output written to terminal; 1=show '
                             'fit result QA plot; 2=full output.')
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
    parser.add_argument('--min_vel_snr', default=None, type=float,
                        help='Minimum S/N to include for velocity measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--min_sig_snr', default=None, type=float,
                        help='Minimum S/N to include for dispersion measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--max_vel_err', default=None, type=float,
                        help='Maximum velocity error to include in fit.')
    parser.add_argument('--max_sig_err', default=None, type=float,
                        help='Maximum velocity dispersion error to include in fit '
                             '(ignored if dispersion not being fit).')
    parser.add_argument('--screen', dest='screen', action='store_true', default=False) 

    # TODO: Other options:
    #   - Fit with least-squares vs. dynesty
    #   - Type of rotation curve
    #   - Type of dispersion profile
    #   - Include the surface-brightness weighting

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    # Running the script behind a screen, so switch the matplotlib backend
    if args.screen:
        pyplot.switch_backend('agg')

    #---------------------------------------------------------------------------
    # Setup
    #  - Check the input
    if args.tracer not in ['Gas', 'Stars']:
        raise ValueError('Tracer to fit must be either Gas or Stars.')
    #  - Check that the output directory exists, and if not create it
    if not os.path.isdir(args.odir):
        os.makedirs(args.odir)
    #  - Set the output root name
    oroot = f'nirvana-manga-axisym-{args.plate}-{args.ifu}-{args.tracer}'
    #  - Verbosity:
    debug = args.verbose > 1

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
    print('Setting up guess parameters and parameterization classes.')
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

    # Report
    print(f'Rotation curve parameterization class: {rc.__class__.__name__}')
    if args.disp:
        print(f'Dispersion profile parameterization class: {dc.__class__.__name__}')
    print('Input guesses:')
    print(f'               Position angle: {pa:.1f}')
    print(f'                  Inclination: {p0[3]:.1f}')
    print(f'     Projected Rotation Speed: {vproj:.1f}')
    if args.disp:
        print(f'  Central Velocity Dispersion: {sig0:.1f}')
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Setup the full velocity-field model
    # Setup the masks
    print('Initializing data masking')
    disk_bm = AxisymmetricDiskFitBitMask()
    vel_mask = disk_bm.init_mask_array(kin.vel.shape)
    vel_mask[kin.vel_mask] = disk_bm.turn_on(vel_mask[kin.vel_mask], 'DIDNOTUSE')
    if args.disp:
        sig_mask = disk_bm.init_mask_array(kin.sig.shape)
        sig_mask[kin.sig_mask] = disk_bm.turn_on(sig_mask[kin.sig_mask], 'DIDNOTUSE')
    else:
        sig_mask = None

    # Reject based on error
    vel_rej, sig_rej = kin.clip_err(max_vel_err=args.max_vel_err, max_sig_err=args.max_sig_err)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements removed because of large errors.')
        vel_mask[vel_rej] = disk_bm.turn_on(vel_mask[vel_rej], 'REJ_ERR')
    if args.disp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements removed because of large errors.')
        sig_mask[sig_rej] = disk_bm.turn_on(sig_mask[sig_rej], 'REJ_ERR')

    # Reject based on S/N
    vel_rej, sig_rej = kin.clip_snr(min_vel_snr=args.min_vel_snr, min_sig_snr=args.min_sig_snr)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements removed because of low S/N.')
        vel_mask[vel_rej] = disk_bm.turn_on(vel_mask[vel_rej], 'REJ_SNR')
    if args.disp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements removed because of low S/N.')
        sig_mask[sig_rej] = disk_bm.turn_on(sig_mask[sig_rej], 'REJ_SNR')

    #---------------------------------------------------------------------------
    # Define the fitting object
    disk = AxisymmetricDisk(rc=rc, dc=dc)

    # Constrain the center to be in the middle third of the map relative to the
    # photometric center. The mean in the calculation is to mitigate that some
    # galaxies can be off center, but the detail here and how well it works
    # hasn't been well tested.
    dx = np.mean([abs(np.amin(kin.x)), abs(np.amax(kin.x))])
    dy = np.mean([abs(np.amin(kin.y)), abs(np.amax(kin.y))])
    lb, ub = disk.par_bounds(base_lb=np.array([-dx/3, -dy/3, -350., 0., -500.]),
                             base_ub=np.array([dx/3, dy/3, 350., 89., 500.]))
    print(f'If free, center constrained within +/- {dx/3:.1f} in X and +/- {dy/3:.1f} in Y.')

    #---------------------------------------------------------------------------
    # Fit iteration 1: Fit all data but fix the inclination and center
    #                x0    y0    pa     inc   vsys    rc+dc parameters
    fix = np.append([True, True, False, True, False], np.zeros(p0.size-5, dtype=bool))
    print('Running fit iteration 1')
    # TODO: sb_wgt is always true throughout. Make this a command-line
    # parameter?
    disk.lsq_fit(kin, sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub, verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 2:
    #   - Reject very large outliers. This is aimed at finding data that is
    #     so descrepant from the model that it's reasonable to expect the
    #     measurements are bogus.
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk_fit_reject(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_sigma_rej=15,
                              show_vel=debug, sig_mask=sig_mask, sig_sigma_rej=15, show_sig=debug,
                              rej_flag='REJ_UNR')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit, again with the inclination and center fixed. However, do not
    #     use the parameters from the previous fit as the starting point, and
    #     ignore the estimated intrinsic scatter.
    print('Running fit iteration 2')
    disk.lsq_fit(kin, sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub, verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 3: 
    #   - Perform a more restricted rejection
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk_fit_reject(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_sigma_rej=10,
                              show_vel=debug, sig_mask=sig_mask, sig_sigma_rej=10, show_sig=debug,
                              rej_flag='REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    print('Running fit iteration 3')
    scatter = np.array([vel_sig, sig_sig])
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, scatter=scatter,
                 verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 4: 
    #   - Recover data from the restricted rejection
    reset_to_base_flags(kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk_fit_reject(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_sigma_rej=10,
                              show_vel=debug, sig_mask=sig_mask, sig_sigma_rej=10, show_sig=debug,
                              rej_flag='REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    print('Running fit iteration 4')
    scatter = np.array([vel_sig, sig_sig])
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, scatter=scatter,
                 verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 5: 
    #   - Recover data from the restricted rejection
    reset_to_base_flags(kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk_fit_reject(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_sigma_rej=10,
                              show_vel=debug, sig_mask=sig_mask, sig_sigma_rej=10, show_sig=debug,
                              rej_flag='REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Now fit as requested by the user, freeing one or both of the
    #     inclination and center. Use the previous fit as the starting point
    #     and include the estimated intrinsic scatter.
    #                    x0     y0     pa     inc    vsys
    base_fix = np.array([False, False, False, False, False])
    if args.fix_cen:
        base_fix[:2] = True
    if args.fix_inc:
        base_fix[3] = True
    fix = np.append(base_fix, np.zeros(p0.size-5, dtype=bool))
    print('Running fit iteration 5')
    scatter = np.array([vel_sig, sig_sig])
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, scatter=scatter,
                 verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 6:
    #   - Recover data from the restricted rejection
    reset_to_base_flags(kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters.
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk_fit_reject(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_sigma_rej=10,
                              show_vel=debug, sig_mask=sig_mask, sig_sigma_rej=10, show_sig=debug,
                              rej_flag='REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Redo previous fit
    print('Running fit iteration 6')
    scatter = np.array([vel_sig, sig_sig])
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, scatter=scatter,
                 verbose=args.verbose)
    # Show
    if args.verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    # Plot the final residuals
    dv_plot = os.path.join(args.odir, f'{oroot}-vdist.png')
    ds_plot = os.path.join(args.odir, f'{oroot}-sdist.png')
    disk_fit_resid_dist(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_plot=dv_plot,
                        sig_mask=sig_mask, sig_plot=ds_plot)

    # Create the final fit plot
    fit_plot = os.path.join(args.odir, f'{oroot}-fit.png')
    axisym_fit_plot(galmeta, kin, disk, fix=fix, ofile=fit_plot)

    # Write the output file
    data_file = os.path.join(args.odir, f'{oroot}.fits.gz')
    axisym_fit_data(galmeta, kin, p0, disk, data_file, vel_mask, sig_mask)


