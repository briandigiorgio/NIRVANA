"""
Module with the derived instances for MaNGA kinematics.
"""
import os

from IPython import embed

import numpy as np
from astropy.io import fits

from .kinematics import Kinematics

# TODO: This is put here to avoid a dependence on mangadap and/or
# marvin.
def channel_dictionary(hdu, ext, prefix='C'):
    """
    Construct a dictionary of the channels in a MAPS file.

    Copied from mangadap.util.fileio.channel_dictionary
    """
    channel_dict = {}
    for k, v in hdu[ext].header.items():
        if k[:len(prefix)] == prefix:
            try:
                i = int(k[len(prefix):])-1
            except ValueError:
                continue
            channel_dict[v] = i
    return channel_dict

# TODO:
#   - What's the minimum MPL that this will work with
#   - Have this use Marvin to read the data
#   - Use the bits directly instead of just testing > 0
class MaNGAGasKinematics(Kinematics):
    """
    Class to read and hold ionized-gas kinematics from MaNGA.

    Args:
        maps_file (:obj:`str`):
            Name of the file with the DAP-produced maps.
        cube_file (:obj:`str`):
            The name of the DRP-produced LOGCUBE file with the
            reconstructed PSF.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
        line (:obj:`str`, optional):
            The name of the emission-line to use for the kinematics.
    """
    def __init__(self, maps_file, cube_file, psf_ext='RPSF', line='Ha-6564'):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError('File does not exist: {0}'.format(maps_file))
        if not os.path.isfile(cube_file):
            raise FileNotFoundError('File does not exist: {0}'.format(cube_file))

        # Read the PSF
        with fits.open(cube_file) as hdu:
            if psf_ext not in hdu:
                raise KeyError('{0} does not include an extension {1}.'.format(cube_file, psf_ext))
            psf = hdu[psf_ext].data

        # Read the kinematic maps
        with fits.open(maps_file) as hdu:
            eml = channel_dictionary(hdu, 'EMLINE_GVEL')
            if line not in eml:
                raise KeyError('{0} does not contain channel {1}.'.format(maps_file, line))
            x = hdu['SPX_SKYCOO'].data[0]
            y = hdu['SPX_SKYCOO'].data[1]
            binid = hdu['BINID'].data[3]
            sb = hdu['EMLINE_GFLUX'].data[eml[line]]
            sb_ivar = hdu['EMLINE_GFLUX_IVAR'].data[eml[line]]
            sb_mask = hdu['EMLINE_GFLUX_MASK'].data[eml[line]] > 0
            vel = hdu['EMLINE_GVEL'].data[eml[line]]
            vel_ivar = hdu['EMLINE_GVEL_IVAR'].data[eml[line]]
            vel_mask = hdu['EMLINE_GVEL_MASK'].data[eml[line]] > 0
            sig = hdu['EMLINE_GSIGMA'].data[eml[line]]
            sig_ivar = hdu['EMLINE_GSIGMA_IVAR'].data[eml[line]]
            sig_mask = hdu['EMLINE_GSIGMA_MASK'].data[eml[line]] > 0
            sig_corr = hdu['EMLINE_INSTSIGMA'].data[eml[line]]

        super(MaNGAGasKinematics, self).__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask, x=x,
                                                 y=y, sb=sb, sb_ivar=sb_ivar, sb_mask=sb_mask,
                                                 sig=sig, sig_ivar=sig_ivar, sig_mask=sig_mask,
                                                 sig_corr=sig_corr, psf=psf, binid=binid)


class MaNGAStellarKinematics(Kinematics):
    """
    Class to read and hold stellar kinematics from MaNGA.

    Args:
        maps_file (:obj:`str`):
            Name of the file with the DAP-produced maps.
        cube_file (:obj:`str`):
            The name of the DRP-produced LOGCUBE file with the
            reconstructed PSF.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
    """
    def __init__(self, maps_file, cube_file, psf_ext='GPSF'):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError('File does not exist: {0}'.format(maps_file))
        if not os.path.isfile(cube_file):
            raise FileNotFoundError('File does not exist: {0}'.format(cube_file))

        # Read the PSF
        with fits.open(cube_file) as hdu:
            if psf_ext not in hdu:
                raise KeyError('{0} does not include an extension {1}.'.format(cube_file, psf_ext))
            psf = hdu[psf_ext].data

        bintype = maps_file.split('.')[0].split('-')[-3]
        coo_ext = 'SPX_SKYCOO' if bintype == 'SPX' else 'BIN_LWSKYCOO'

        # Read the kinematic maps
        with fits.open(maps_file) as hdu:
            x = hdu[coo_ext].data[0]
            y = hdu[coo_ext].data[1]
            binid = hdu['BINID'].data[1]
            sb = hdu['BIN_MFLUX'].data
            sb_ivar = hdu['BIN_MFLUX_IVAR'].data
            sb_mask = np.logical_not((sb > 0) & (sb_ivar > 0))
            vel = hdu['STELLAR_VEL'].data
            vel_ivar = hdu['STELLAR_VEL_IVAR'].data
            vel_mask = hdu['STELLAR_VEL_MASK'].data > 0
            sig = hdu['STELLAR_SIGMA'].data
            sig_ivar = hdu['STELLAR_SIGMA_IVAR'].data
            sig_mask = hdu['STELLAR_SIGMA_MASK'].data > 0
            sig_corr = hdu['STELLAR_SIGMACORR'].data[0]

        super(MaNGAStellarKinematics, self).__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask,
                                                     x=x, y=y, sb=sb, sb_ivar=sb_ivar,
                                                     sb_mask=sb_mask, sig=sig, sig_ivar=sig_ivar,
                                                     sig_mask=sig_mask, sig_corr=sig_corr,
                                                     psf=psf, binid=binid)
        


