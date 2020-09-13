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


def read_manga_psf(cube_file, psf_ext):
    """
    Read the image of the reconstructed datacube point-spread
    function from the MaNGA DRP CUBE file.

    .. warning::

        No check is performed to ensure that the provided extension
        is a valid PSF extension.

    Args:
        cube_file (:obj:`str`):
            The name of the DRP-produced LOGCUBE fits file with the
            reconstructed PSF.
        psf_ext (:obj:`str`):
            The name of the extension with the reconstructed PSF;
            e.g. ``'GPSF'``.

    Returns:
        `numpy.ndarray`_: Array with the reconstructed PSF.

    Raises:
        FileNotFoundError:
            Raised if the provided ``cube_file`` does not exist.
        KeyError:
            Raised if the provided ``psf_ext`` is not a valid fits
            extension.
    """
    if not os.path.isfile(cube_file):
        raise FileNotFoundError('File does not exist: {0}'.format(cube_file))

    # Read the PSF
    # TODO: Switch from print to a logger
    print('Reading {0} ... '.format(cube_file))
    with fits.open(cube_file) as hdu:
        if psf_ext not in hdu:
            raise KeyError('{0} does not include an extension {1}.'.format(
                            cube_file, psf_ext))
        psf = hdu[psf_ext].data
    print('Done')
    return psf


def manga_files_from_plateifu(plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10',
                              redux_path=None, cube_path=None, analysis_path=None, maps_path=None):
    """
    Get the DAP maps and DRP datacube files for a given plate and
    IFU.

    Args:
        plate (:obj:`int`):
            Plate number
        ifu (:obj:`int`):
            IFU design number
        daptype (:obj:`str`, optional):
            The identifier for the method used by the DAP to analyze
            the data.
        dr (:obj:`str`, optional):
            Data release identifier that matches the directory with
            the data.
        redux_path (:obj:`str`, optional):
            The top-level directory with all DRP output. If None,
            this will be set to the ``MANGA_SPECTRO_REDUX``
            environmental variable, if it is defined.
        cube_path (:obj:`str`, optional):
            This provides the *direct* path to the datacube file,
            circumventing the use of ``dr`` and ``redux_path``.
        analysis_path (:obj:`str`, optional):
            The top-level directory with all DAP output. If None,
            this will be set to the ``MANGA_SPECTRO_ANALYSIS``
            environmental variable, if it is defined.
        maps_path (:obj:`str`, optional):
            This provides the *direct* path to the maps file,
            circumventing the use of ``dr`` and ``analysis_path``.

    Returns:
        :obj:`tuple`: The full path to the maps file followed by the
        full path to the data cube file.

    Raises:
        ValueError:
            Raised if the directories to either the maps or cube file
            could not be determined from the input.
        NotADirectoryError:
            Raised if the directory can be defined but does not exist.
    """
    if cube_path is None:
        _redux_path = os.getenv('MANGA_SPECTRO_REDUX') if redux_path is None else redux_path
        if _redux_path is None:
            raise ValueError('Could not define top-level root for DRP output.')
        cube_path = os.path.join(os.path.abspath(_redux_path), dr, str(plate), 'stack')
    if not os.path.isdir(cube_path):
        raise NotADirectoryError('No such directory: {0}'.format(cube_path))

    cube_file = os.path.abspath(os.path.join(cube_path,
                                             'manga-{0}-{1}-LOGCUBE.fits.gz'.format(plate, ifu)))

    if maps_path is None:
        _analysis_path = os.getenv('MANGA_SPECTRO_ANALYSIS') \
                            if analysis_path is None else analysis_path
        if _analysis_path is None:
            raise ValueError('Could not define top-level root for DAP output.')
        maps_path = os.path.join(os.path.abspath(_analysis_path), dr, daptype, str(plate), str(ifu))
    if not os.path.isdir(maps_path):
        raise NotADirectoryError('No such directory: {0}'.format(maps_path))

    maps_file = os.path.abspath(os.path.join(maps_path, 'manga-{0}-{1}-MAPS-{2}.fits.gz'.format(
                                             plate, ifu, daptype)))

    return maps_file, cube_file


class MaNGAKinematics(Kinematics):
    """
    Base class for MaNGA derived classes the provides common functionality.

    This class *should not* be instantiated by itself.
    """

    @classmethod
    def from_plateifu(cls, plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10',
                      redux_path=None, cube_path=None, analysis_path=None, maps_path=None,
                      ignore_psf=False, **kwargs):
        """
        Instantiate the object using the plate and IFU number.

        This uses :func:`manga_files_from_plateifu` to construct the
        names of the MAPS and LOGCUBE files that should be located on
        disk.  See that function for the description of the arguments.

        Args:
            ignore_psf (:obj:`bool`, optional):
                Ignore the point-spread function when collecting the
                data. I.e., this instantiates the object setting
                ``cube_file=None``.
            **kwargs:
                Additional arguments that are passed directly to the
                nominal instantiation method.
        """
        maps_file, cube_file = manga_files_from_plateifu(plate, ifu, daptype=daptype, dr=dr,
                                                         redux_path=redux_path,
                                                         cube_path=cube_path,
                                                         analysis_path=analysis_path,
                                                         maps_path=maps_path)
        if ignore_psf:
            cube_file = None
        return cls(maps_file, cube_file=cube_file, **kwargs)


# TODO:
#   - What's the minimum MPL that this will work with
#   - Have this use Marvin to read the data
#   - Use the bits directly instead of just testing > 0
class MaNGAGasKinematics(MaNGAKinematics):
    """
    Class to read and hold ionized-gas kinematics from MaNGA.

    Args:
        maps_file (:obj:`str`):
            Name of the file with the DAP-produced maps.
        cube_file (:obj:`str`, optional):
            The name of the DRP-produced LOGCUBE file with the
            reconstructed PSF. If None, the PSF image will not be
            used in constructing the base
            :class:`~barfit.data.kinematics.Kinematics` object.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
        line (:obj:`str`, optional):
            The name of the emission-line to use for the kinematics.
    """
    def __init__(self, maps_file, cube_file=None, psf_ext='RPSF', line='Ha-6564'):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError('File does not exist: {0}'.format(maps_file))

        # Get the PSF, if possible
        psf = None if cube_file is None else read_manga_psf(cube_file, psf_ext)

        # Establish whether or not the gas kinematics were determined
        # on a spaxel-by-spaxel basis, which determines which extension
        # to use for the on-sky coordinates for each unique
        # measurement. The binned coordinates are only used if the data
        # is from the `VOR` bin case (which is probably never going to
        # be used with this package, but anyway...)
        bintype = maps_file.split('.')[0].split('-')[-3]
        coo_ext = 'BIN_LWSKYCOO' if 'VOR' in bintype else 'SPX_SKYCOO'

        # Read the kinematic maps
        # TODO: Switch from print to a logger
        print('Reading {0} ... '.format(maps_file))
        with fits.open(maps_file) as hdu:
            eml = channel_dictionary(hdu, 'EMLINE_GVEL')
            if line not in eml:
                raise KeyError('{0} does not contain channel {1}.'.format(maps_file, line))
            x = hdu[coo_ext].data[0]
            y = hdu[coo_ext].data[1]
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
        print('Done')

        super(MaNGAGasKinematics, self).__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask, x=x,
                                                 y=y, sb=sb, sb_ivar=sb_ivar, sb_mask=sb_mask,
                                                 sig=sig, sig_ivar=sig_ivar, sig_mask=sig_mask,
                                                 sig_corr=sig_corr, psf=psf, binid=binid)


class MaNGAStellarKinematics(MaNGAKinematics):
    """
    Class to read and hold stellar kinematics from MaNGA.

    Args:
        maps_file (:obj:`str`):
            Name of the file with the DAP-produced maps.
        cube_file (:obj:`str`, optional):
            The name of the DRP-produced LOGCUBE file with the
            reconstructed PSF. If None, the PSF image will not be
            used in constructing the base
            :class:`~barfit.data.kinematics.Kinematics` object.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
    """
    def __init__(self, maps_file, cube_file, psf_ext='GPSF'):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError('File does not exist: {0}'.format(maps_file))

        # Get the PSF, if possible
        psf = None if cube_file is None else read_manga_psf(cube_file, psf_ext)

        # Establish whether or not the stellar kinematics were
        # determined on a spaxel-by-spaxel basis, which determines
        # which extensions to use for the on-sky coordinates and mean
        # flux for each unique measurement.
        # TODO: Actually, the BIN_* extensions are right in either case
        # for the stellar kinematics, but I'll leave this for now...
        bintype = maps_file.split('.')[0].split('-')[-3]
        coo_ext = 'SPX_SKYCOO' if bintype == 'SPX' else 'BIN_LWSKYCOO'
        flux_ext = 'SPX_MFLUX' if bintype == 'SPX' else 'BIN_MFLUX'

        # Read the kinematic maps
        print('Reading {0} ... '.format(maps_file))
        with fits.open(maps_file) as hdu:
            x = hdu[coo_ext].data[0]
            y = hdu[coo_ext].data[1]
            binid = hdu['BINID'].data[1]
            sb = hdu[flux_ext].data
            sb_ivar = hdu['{0}_IVAR'.format(flux_ext)].data
            sb_mask = np.logical_not((sb > 0) & (sb_ivar > 0))
            vel = hdu['STELLAR_VEL'].data
            vel_ivar = hdu['STELLAR_VEL_IVAR'].data
            vel_mask = hdu['STELLAR_VEL_MASK'].data > 0
            sig = hdu['STELLAR_SIGMA'].data
            sig_ivar = hdu['STELLAR_SIGMA_IVAR'].data
            sig_mask = hdu['STELLAR_SIGMA_MASK'].data > 0
            sig_corr = hdu['STELLAR_SIGMACORR'].data[0]
        print('Done')

        super(MaNGAStellarKinematics, self).__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask,
                                                     x=x, y=y, sb=sb, sb_ivar=sb_ivar,
                                                     sb_mask=sb_mask, sig=sig, sig_ivar=sig_ivar,
                                                     sig_mask=sig_mask, sig_corr=sig_corr,
                                                     psf=psf, binid=binid)



