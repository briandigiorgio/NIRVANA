"""
Module with the derived instances for MaNGA kinematics.

.. |ee|  unicode:: U+00E9
    :trim:

.. |Sersic|  replace:: S |ee| rsic

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
import os
import glob
import warnings

from pkg_resources import resource_filename

from IPython import embed

import numpy as np
from scipy import sparse
import matplotlib.image as img

from astropy.io import fits

from .util import get_map_bin_transformations, impose_positive_definite
from .kinematics import Kinematics
from .meta import GlobalPar
from ..util.bitmask import BitMask


def channel_dictionary(hdu, ext, prefix='C'):
    """
    Construct a dictionary of the channels in a MaNGA MAPS file.

    Copied from mangadap.util.fileio.channel_dictionary

    Args:
        hdu (`astropy.io.fits.HDUList`_):
            The open fits file.
        ext (:obj:`int`, :obj:`str`):
            The name or index number of the fits extension with the
            relevant map channels and the header with the channel
            names.
        prefix (:obj:`str`, optional):
            The key that is used as a prefix for all header keywords
            associated with channel names. The header keyword is the
            combination of this prefix and the 1-indexed channel
            number.

    Returns:
        :obj:`dict`: The dictionary that provides the channel index
        associate with the channel name.
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


def read_manga_psf(cube_file, psf_ext, fwhm=False, quiet=False):
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
        fwhm (:obj:`bool`, optional):
            If true, will return the g band FWHM as well
        quiet (:obj:`bool`, optional):
            Suppress printed output.

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
    if not quiet:
        print('Reading {0} ... '.format(cube_file))
    with fits.open(cube_file) as hdu:
        if psf_ext not in hdu:
            raise KeyError('{0} does not include an extension {1}.'.format(
                            cube_file, psf_ext))
        psf = hdu[psf_ext].data
        if fwhm: fwhm = hdu[0].header['GFWHM']
    if not quiet:
        print('Done')
    if fwhm: return psf, fwhm
    return psf


def manga_files_from_plateifu(plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10',
                              redux_path=None, cube_path=None, image_path=None, analysis_path=None,
                              maps_path=None, check=True):
    """
    Get MaNGA files used by NIRVANA.

    .. warning::

        The method does not check that these files exist.

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
        image_path (:obj:`str`, optional):
            This provides the *direct* path to the image file,
            circumventing the use of ``dr`` and ``redux_path``.
        analysis_path (:obj:`str`, optional):
            The top-level directory with all DAP output. If None,
            this will be set to the ``MANGA_SPECTRO_ANALYSIS``
            environmental variable, if it is defined.
        maps_path (:obj:`str`, optional):
            This provides the *direct* path to the maps file,
            circumventing the use of ``dr`` and ``analysis_path``.
        check (:obj:`bool`, optional):
            Check that the directories with the expected files exist.

    Returns:
        :obj:`tuple`: Full path to three files: (1) the DAP MAPS file, (2)
        the DRP LOGCUBE file, and (3) the SDSS 3-color PNG image. If the
        image path cannot be defined or does not exist, the image file name
        is returned as ``None``.

    Raises:
        ValueError:
            Raised if the directories to either the maps or cube file
            could not be determined from the input.
        NotADirectoryError:
            Raised if the directory where the datacube should exist can be
            defined but does not exist.
    """
    if cube_path is None or image_path is None:
        _redux_path = os.getenv('MANGA_SPECTRO_REDUX') if redux_path is None else redux_path
        if _redux_path is None and cube_path is None:
            raise ValueError('Could not define top-level root for DRP output.')
    if cube_path is None:
        cube_path = os.path.join(os.path.abspath(_redux_path), dr, str(plate), 'stack')
    if image_path is None:
        image_path = None if _redux_path is None \
                        else os.path.join(os.path.abspath(_redux_path), dr, str(plate), 'images')
        if image_path is None:
            warnings.warn('Could not define directory to image PNGs')
        elif not os.path.isdir(image_path):
            warnings.warn('No such directory: {0}'.format(image_path))
            image_path = None
    if check and not os.path.isdir(cube_path):
        raise NotADirectoryError('No such directory: {0}'.format(cube_path))

    cube_file = os.path.abspath(os.path.join(cube_path, f'manga-{plate}-{ifu}-LOGCUBE.fits.gz'))
    image_file = None if image_path is None \
                    else os.path.abspath(os.path.join(image_path, f'{ifu}.png'))

    if maps_path is None:
        _analysis_path = os.getenv('MANGA_SPECTRO_ANALYSIS') \
                            if analysis_path is None else analysis_path
        if _analysis_path is None:
            raise ValueError('Could not define top-level root for DAP output.')
        maps_path = os.path.join(os.path.abspath(_analysis_path), dr, daptype, str(plate), str(ifu))

    if check and not os.path.isdir(maps_path):
        raise NotADirectoryError('No such directory: {0}'.format(maps_path))

    maps_file = os.path.abspath(os.path.join(maps_path,
                                             f'manga-{plate}-{ifu}-MAPS-{daptype}.fits.gz'))

    return maps_file, cube_file, image_file


def read_drpall(plate, ifu, redux_path, dr, ext='elpetro', quiet=False):
    """
    Read the NASA Sloan Atlas data from the DRPall file for the
    appropriate data release.

    Args:
        plate (:obj:`int`):
            Plate of galaxy.
        ifu (:obj:`int`):
            IFU of galaxy.
        redux_path (:obj:`str`, optional):
            The top-level directory with all DRP output. If None,
            this will be set to the ``MANGA_SPECTRO_REDUX``
        dr (:obj:`str`, optional):
            Data release identifier that matches the directory with
            the data.
        ext (:obj:`str`):
            Whether to use the `elpetro` or `sersic` derived values
            in the NSA catalog.
        quiet (:obj:`bool`, optional):
            Suppress printed output.

    Returns:
        :obj:`tuple`: tuple of inclination, position angle, and |Sersic| n
        values. Angles are in degrees.

    Raises:
        FileNotFoundError:
            Raised if the DRPall file can't be found in the specified
            place.
    """
    # Read the drpall file
    if not quiet:
        print('Reading {0} ... '.format(cube_file))
    drpall_file = glob.glob(f'{cubepath}/{dr}/drpall*.fits')[0]
    plateifu = f'{plate}-{ifu}'

    if not drpall_file:
        raise FileNotFoundError('Could not find drpall file')

    with fits.open(drpall_file) as hdu:
        data = hdu[1].data[hdu[1].data['plateifu'] == plateifu]
        inc = np.degrees(np.arccos(data[f'nsa_{ext}_ba']))
        pa = data[f'nsa_{ext}_phi']
        n = data['nsa_sersic_n']

    if not quiet:
        print('Done')
    return inc, pa, n


def sdss_bitmask(bitgroup):
    """
    Return a :class:`~nirvana.util.bitmask.BitMask` instance for the
    specified group of SDSS maskbits.

    Args:
        bitgroup (:obj:`str`):
            The name designating the group of bitmasks to read.

    Returns:
        :class:`~nirvana.util.bitmask.BitMask`: Instance used to parse SDSS
        maskbits.
    """
    sdssMaskbits = os.path.join(resource_filename('nirvana', 'config'), 'sdss', 'sdssMaskbits.par')
    return BitMask.from_par_file(sdssMaskbits, bitgroup)


def parse_manga_targeting_bits(mngtarg1, mngtarg3=None):
    """
    Return boolean flags identifying the target samples selected by the
    provided bits.

    Args:
        mngtarg1 (:obj:`int`, `numpy.ndarray`_):
            One or more targeting bit values from the MANGA_TARGET1 group.
        mngtarg3 (:obj:`int`, `numpy.ndarray`_, optional):
            One or more targeting bit values from the MANGA_TARGET3 group,
            the ancillary targeting bits. Should have the same shape as
            ``mngtarg1``.

    Returns:
        :obj:`tuple`: Four booleans or boolean arrays that identify the
        target as being (1) part of the Primary+ (Primary and/or
        Color-enhanced) sample, (2) the Secondary sample, (3) an ancillary
        target, and/or (4) a filler target.
    """
    bitmask = sdss_bitmask('MANGA_TARGET1')
    primaryplus = bitmask.flagged(mngtarg1, flag=['PRIMARY_v1_2_0', 'COLOR_ENHANCED_v1_2_0'])
    secondary = bitmask.flagged(mngtarg1, flag='SECONDARY_v1_2_0')
    ancillary = (np.zeros(mngtarg1.shape, dtype=bool) 
                    if isinstance(primaryplus, np.ndarray) else False) \
                    if mngtarg3 is None else mngtarg3 > 0
    other = np.logical_not(ancillary) & np.logical_not(primaryplus) & np.logical_not(secondary)

    return primaryplus, secondary, ancillary, other


def manga_map_covar(ivar, binid=None, rho_sig=1.92, rlim=3.2, min_ivar=1e-10, rho_tol=1e-5,
                    positive_definite=True, fill=False):
    r"""
    Construct a nominal covariance matrix for a MaNGA wavelength channel or
    DAP map.

    See https://sdss-mangadap.readthedocs.io/en/latest/aggregation.html

    This method constructs a correlation matrix with correlation coefficients
    defined as

    .. math::

        \rho_{ij} = \exp(-D_{ij}^2 / 2\sigma_{\rho}^2)

    where :math:`D_{ij}` is the distance between two spaxels in the spatial
    dimension of the datacube (in the number spaxels, not arcsec). Any pixels
    with :math:`D_{ij} > R_{\rm lim}` is set to zero, where :math:`R_{\rm
    lim}` is the limiting radius of the kernel used in the datacube
    rectification; this is provided here as ``rlim`` in pixels.

    We found in Westfall et al. (2019, AJ, 158, 231) that this is a
    reasonable approximation for the formal covariance that results from
    Shepard's rectification method.

    There is an unknown relation between the dispersion of the kernel used by
    Shepard's method and the value of :math:`\sigma_{\rho}`. Tests show that
    it is tantalizingly close to :math:`\sigma_{\rho} = \sqrt{2}\sigma`,
    where :math:`\sigma` is in pixels; however, a formal derivation of this
    hasn't been done and is complicated by the focal-plane sampling of the
    row-stacked spectra.

    Within the limits of how the focal-plane sampling changes with
    wavelength, we found the value of :math:`\sigma_{\rho}` varies little
    with wavelength in MaNGA. Here, we assume :math:`\sigma_{\rho}` is fully
    wavelength independent. Preliminary tests also show that the covariance
    in *derived* properties provides in the DAP MAPS files also exhibit this
    same level of covariance. Therefore, you can get an approximate
    covariance matrix for *any* DAP map using this method.

    Args:
        ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
            The inverse variance map. Shape must be 2D and square. Any value
            below ``min_ivar`` is assumed to be masked. Any mask associated
            with masked-array input is also incorporated into the output.
        binid (`numpy.ndarray`_, optional):
            The bin ID associated with each spaxel. Values of -1 indicate
            that the value is not associated with any bin; multiple spaxels
            with the same bin ID are assumed to have the same ``ivar``. If
            None, all spaxels are assumed to be independent and any masking
            is dictated by the value and mask for ``ivar``. Shape must match
            ``ivar``.
        sigma_rho (:obj:`float`, optional):
            The :math:`\sigma_{\rho}` of the Gaussian function used to
            approximate the trend of the correlation coefficient with spaxel
            separation. Default is as determined by Westfall et al. (2019,
            AJ, 158, 231).
        rlim (:obj:`float`, optional):
            The limiting radius of the image reconstruction (Gaussian) kernel
            in pixels. The default is the value used by the MaNGA DRP when
            constructing the MaNGA datacubes.
        min_ivar (:obj:`float`, optional):
            Minimum viable inverse variance. Anything below this value is
            assumed to be masked. If None, anything that is :math:`\leq 0` is
            masked.
        rho_tol (:obj:`float`, optional):
            Any correlation coefficient less than this is assumed
            to be equivalent to (and set to) 0.
        positive_definite (:obj:`bool`, optional):
            Force the covariance matrix to be positive definite. This step is
            expensive given the need to determine the eigenvalues and
            eigenvectors of the covariance matrix; see
            :func:`~nirvana.data.util.impose_positive_definite`. However, this
            step is important if you intend to invert the covariance matrix
            for, e.g., chi-squaare/likelihood calculations. Nominal
            construction of the covariance matrix typically will *not* return
            a positive matrix. If the covariance matrix will be further
            manipulated, it's likely worth doing that manipulation first and
            then ensuring it's positive definite; see
            :class:`nirvana.data.kinematics.Kinematics`.
        fill (:obj:`bool`, optional):
            Largely a convenience that can be used to reshape the returned
            covariance matrix. If True, the returned covariance matrix is
            *always* matched to the shape of the input ``ivar`` matrix.
            Covariance regions associated with masked spaxels, masked either
            due to an inverse variance that is :math:`\leq 0` or because the
            spaxel is not associated with a bin, are set to 0. The boolean
            "good-pixel mask", the first returned object, selects the regions
            ``ivar`` with the covariance calculation (i.e., this array is the
            same regardless of the value of ``fill``). The behavior is such
            that the following should always pass::

                _, covar = manga_map_covar(ivar, fill=True)
                gpm, subcovar = manga_map_covar(ivar, fill=False)
                assert np.array_equal(covar[np.ix_(gpm.ravel(),gpm.ravel())].toarray(),
                                      subcovar.toarray())

    Returns:
        :obj:`tuple`: Two objects are returned: (1) a boolean array with the
        same shape as ``ivar`` selecting the spaxels used when calculating
        the covariance matrix and (2) a `scipy.sparse.csr_matrix`_ with the
        covariance matrix. See the ``fill`` argument for a description of the
        shape of the returned covariance matrix.
    """
    # Get the good-pixel mask
    bpm = np.ma.getmaskarray(ivar).copy() if isinstance(ivar, np.ma.MaskedArray) \
                else np.zeros(ivar.shape, dtype=bool)
    bpm |= np.logical_not(ivar > 0) if min_ivar is None else ivar < min_ivar
    if binid is not None:
        bpm |= binid < 0
    gpm = np.logical_not(bpm)

    # Compute the variance
    var = np.zeros(ivar.shape, dtype=float)
    var[gpm] = 1./ivar[gpm]

    # Get the spaxel coordinates
    nn = np.prod(var.shape)
    i, j = map(lambda x : x.reshape(var.shape)[gpm],
               np.unravel_index(np.arange(nn), shape=var.shape))
    
    # Get the correlation between the valid spaxels
    d = np.square(i[:,None]-i[None,:]) + np.square(j[:,None]-j[None,:])
    rho = np.exp(-d/np.square(rho_sig)/2)
    rho[d > np.square(2*rlim)] = 0.
    if rho_tol is not None:
        # Impose the correlation tolerance
        rho[rho < rho_tol] = 0.
    # Convert to a sparse matrix
    indx = rho > 0
    ngood = np.sum(gpm)
    rho = sparse.csr_matrix((rho[indx], np.where(indx)), shape=(ngood,ngood))

    if binid is not None:
        # Construct the binning matrices
        # TODO: I don't think this takes very long, but we may want to skip
        # this step also if the data isn't binned.
        bin_transform = get_map_bin_transformations(binid=binid)[-1]
        # Check if anything is binned
        is_binned = bin_transform.shape[0] < rho.shape[0]
        # The remainder of this is only needed if the data has been binned
        if is_binned:
            bin_transform = bin_transform[:,gpm.ravel()]
            # Use the transformation matrix to calculate the correlation
            # coefficients between the binned data
            rho = bin_transform.dot(rho.dot(bin_transform.T))
            # Then use an inverse operation to effectively replicate and
            # produce the correlation matrix for the binned map
            inv_transform = bin_transform.T.copy()
            inv_transform[inv_transform > 0] = 1.
            rho = inv_transform.dot(rho.dot(inv_transform.T))
            # Renormalize
            t = 1./np.sqrt(rho.diagonal())
            rho = rho.multiply(np.outer(t,t))

    # Calculate the covariance matrix
    cov = rho.multiply(np.sqrt(np.outer(var[gpm], var[gpm]))).tocsr()

    if positive_definite:
        # Force the matrix to be positive definite
        cov = impose_positive_definite(cov)

    if not np.all(np.isfinite(cov.toarray())):
        raise ValueError('Covariance matrix includes non-finite values.')

    # TODO: Leaving this for the moment to make sure that the construction of
    # the covariance matrix was successful
    assert np.allclose(cov.diagonal(), var[gpm]), \
            'CODING ERROR: Poor construction of the covariance matrix'

    if not fill:
        return gpm, cov

    # Fill out the full covariance matrix so that its size matches the input
    # map size
    _cov = sparse.lil_matrix((ivar.size,ivar.size), dtype=float)
    # NOTE: scipy issues a SparseEfficiencyWarning suggesting the next
    # operation is more efficient using an lil_matrix instead of a csr_matrix;
    # that's the reason for the declaration above and the conversion in the
    # return type...
    _cov[np.ix_(gpm.ravel(),gpm.ravel())] = cov
    return gpm, _cov.tocsr()


class MaNGAKinematics(Kinematics):
    """
    Base class for MaNGA derived classes the provides common functionality.

    This class *should not* be instantiated by itself.
    """

    @classmethod
    def from_plateifu(cls, plate, ifu, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-10',
                      redux_path=None, cube_path=None, image_path=None, analysis_path=None,
                      maps_path=None, ignore_psf=False, **kwargs):
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
                Additional arguments passed directly to the nominal
                instantiation method.
        """
        maps_file, cube_file, image_file \
                = manga_files_from_plateifu(plate, ifu, daptype=daptype, dr=dr,
                                            redux_path=redux_path, cube_path=cube_path,
                                            image_path=image_path, analysis_path=analysis_path,
                                            maps_path=maps_path)
        if ignore_psf:
            cube_file = None
        elif not os.path.isfile(cube_file):
            warnings.warn(f'Datacube file {cube_file} does not exist!')
            cube_file = None

        if image_file is not None and not os.path.isfile(image_file):
            warnings.warn(f'Image file {image_file} does not exist!')
            image_file = None

        return cls(maps_file, cube_file=cube_file, image_file=image_file, **kwargs)


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
            :class:`~nirvana.data.kinematics.Kinematics` object.
        image_file (:obj:`str`, optional):
            Name of the PNG file containing the image of the galaxy.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
        line (:obj:`str`, optional):
            The name of the emission-line to use for the kinematics.
        mask_flags (:obj:`str`, :obj:`list`, optional):
            One or more named bits used to select the data that should be
            masked. If 'any', mask any bit value larger than 0. If None, no
            spaxels are masked (a *bad* idea for MaNGA data). Otherwise, the
            list of strings are used with the ``sdssMaskbits.par`` file to
            determine which spaxels should be masked. For MaNGA data, the
            primary distinction is whether or not you flag everything or if
            you only flag the spaxels marked as ``DONOTUSE``.
        covar (:obj:`bool`, optional):
            Construct the covariance matrices for the map data.
        positive_definite (:obj:`bool`, optional):
            Force the construction of the covariance matrix to be positive
            definite. Practically speaking, this should generally be False if
            the covariance matrix is going to be inverted and used in
            chi-square/Gaussian likelihood calculations.
        quiet (:obj:`bool`, optional):
            Suppress printed output.
    """
    def __init__(self, maps_file, cube_file=None, image_file=None, psf_ext='RPSF', line='Ha-6564',
                 mask_flags='any', covar=False, positive_definite=False, quiet=False):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError(f'File does not exist: {maps_file}')

        # Get the PSF, if possible
        psf, fwhm = (None, None) if cube_file is None \
                        else read_manga_psf(cube_file, psf_ext, fwhm=True)
        # Get the 3-color galaxy thumbnail image
        image = None if image_file is None else img.imread(image_file)

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
        if not quiet:
            print('Reading {0} ... '.format(maps_file))
        with fits.open(maps_file) as hdu:
            eml = channel_dictionary(hdu, 'EMLINE_GVEL')
            if line not in eml:
                raise KeyError('{0} does not contain channel {1}.'.format(maps_file, line))

            x = hdu[coo_ext].data[0]
            y = hdu[coo_ext].data[1]
            binid = hdu['BINID'].data[3]
            grid_x = hdu['SPX_SKYCOO'].data[0]
            grid_y = hdu['SPX_SKYCOO'].data[1]
            sb = hdu['EMLINE_GFLUX'].data[eml[line]]
            sb_ivar = hdu['EMLINE_GFLUX_IVAR'].data[eml[line]]
            sb_anr = hdu['EMLINE_GANR'].data[eml[line]]
            vel = hdu['EMLINE_GVEL'].data[eml[line]]
            vel_ivar = hdu['EMLINE_GVEL_IVAR'].data[eml[line]]
            sig = hdu['EMLINE_GSIGMA'].data[eml[line]]
            sig_ivar = hdu['EMLINE_GSIGMA_IVAR'].data[eml[line]]
            sig_corr = hdu['EMLINE_INSTSIGMA'].data[eml[line]]
            reff = hdu[0].header['REFF']
            phot_ell = hdu[0].header['ECOOELL']
            phot_inc = np.degrees(np.arccos(1 - phot_ell))
            pri, sec, anc, oth = parse_manga_targeting_bits(hdu[0].header['MNGTARG1'], hdu[0].header['MNGTARG3'])
            maxr = 2.5 if sec else 1.5

            # Get the masks
            if mask_flags is None:
                sb_mask = np.zeros(sb.shape, dtype=bool)
                vel_mask = np.zeros(vel.shape, dtype=bool)
                sig_mask = np.zeros(sig.shape, dtype=bool)
            elif mask_flags == 'any':
                sb_mask = hdu['EMLINE_GFLUX_MASK'].data[eml[line]] > 0
                vel_mask = hdu['EMLINE_GVEL_MASK'].data[eml[line]] > 0
                sig_mask = hdu['EMLINE_GSIGMA_MASK'].data[eml[line]] > 0
            else:
                bitmask = sdss_bitmask('MANGA_DAPPIXMASK')
                sb_mask = bitmask.flagged(hdu['EMLINE_GFLUX_MASK'].data[eml[line]],
                                          flag=mask_flags)
                vel_mask = bitmask.flagged(hdu['EMLINE_GVEL_MASK'].data[eml[line]],
                                           flag=mask_flags)
                sig_mask = bitmask.flagged(hdu['EMLINE_GSIGMA_MASK'].data[eml[line]],
                                           flag=mask_flags)

        if not quiet:
            print('Done')

        if covar:
            if not quiet:
                print('Building covariance matrices ... ')
            sb_gpm, sb_covar = manga_map_covar(np.ma.MaskedArray(sb_ivar, mask=sb_mask),
                                               binid=binid, positive_definite=False, fill=True)
            vel_gpm, vel_covar = manga_map_covar(np.ma.MaskedArray(vel_ivar, mask=vel_mask),
                                                 binid=binid, positive_definite=False, fill=True)
            sig_gpm, sig_covar = manga_map_covar(np.ma.MaskedArray(sig_ivar, mask=sig_mask),
                                                 binid=binid, positive_definite=False, fill=True)
            if not quiet:
                print('Done')
        else:
            sb_covar, vel_covar, sig_covar = None, None, None

        super().__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask, vel_covar=vel_covar, x=x, y=y, 
                         sb=sb, sb_ivar=sb_ivar, sb_mask=sb_mask, sb_covar=sb_covar, sb_anr=sb_anr,
                         sig=sig, sig_ivar=sig_ivar, sig_mask=sig_mask, sig_covar=sig_covar,
                         sig_corr=sig_corr, psf=psf, binid=binid, grid_x=grid_x, 
                         grid_y=grid_y, reff=reff , fwhm=fwhm, image=image, 
                         phot_inc=phot_inc, maxr=maxr, positive_definite=positive_definite)


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
            :class:`~nirvana.data.kinematics.Kinematics` object.
        image_file (:obj:`str`, optional):
            Name of the PNG file containing the image of the galaxy.
        psf_ext (:obj:`str`, optional):
            The name of the extension with the reconstructed PSF.
        mask_flags (:obj:`str`, :obj:`list`, optional):
            One or more named bits used to select the data that should be
            masked. If 'any', mask any bit value larger than 0. If None, no
            spaxels are masked (a *bad* idea for MaNGA data). Otherwise, the
            list of strings are used with the ``sdssMaskbits.par`` file to
            determine which spaxels should be masked. For MaNGA data, the
            primary distinction is whether or not you flag everything or if
            you only flag the spaxels marked as ``DONOTUSE``.
        quiet (:obj:`bool`, optional):
            Suppress printed output.
    """
    def __init__(self, maps_file, cube_file=None, image_file=None, psf_ext='GPSF',
                 mask_flags='any', covar=False, positive_definite=False, quiet=False):

        if not os.path.isfile(maps_file):
            raise FileNotFoundError(f'File does not exist: {maps_file}')

        # Get the PSF, if possible
        psf, fwhm = (None,None) if cube_file is None \
                        else read_manga_psf(cube_file, psf_ext, fwhm=True)
        image = img.imread(image_file) if image_file else None

        # Establish whether or not the stellar kinematics were
        # determined on a spaxel-by-spaxel basis, which determines
        # which extensions to use for the on-sky coordinates and mean
        # flux for each unique measurement.
        # TODO: Actually, the BIN_* extensions are always right for the
        # stellar kinematics. I.e., in the SPX case, the BIN_* and
        # SPX_* extensions are identical.  Leaving it for now...
        bintype = maps_file.split('.')[0].split('-')[-3]
        coo_ext = 'SPX_SKYCOO' if bintype == 'SPX' else 'BIN_LWSKYCOO'
        flux_ext = 'SPX_MFLUX' if bintype == 'SPX' else 'BIN_MFLUX'

        # Read the kinematic maps
        if not quiet:
            print('Reading {0} ... '.format(maps_file))
        with fits.open(maps_file) as hdu:
            x = hdu[coo_ext].data[0]
            y = hdu[coo_ext].data[1]
            binid = hdu['BINID'].data[1]
            grid_x = hdu['SPX_SKYCOO'].data[0]
            grid_y = hdu['SPX_SKYCOO'].data[1]
            sb = hdu[flux_ext].data
            sb_ivar = hdu['{0}_IVAR'.format(flux_ext)].data
            sb_mask = np.logical_not((sb > 0) & (sb_ivar > 0))
            vel = hdu['STELLAR_VEL'].data
            vel_ivar = hdu['STELLAR_VEL_IVAR'].data
            sig = hdu['STELLAR_SIGMA'].data
            sig_ivar = hdu['STELLAR_SIGMA_IVAR'].data
            sig_corr = hdu['STELLAR_SIGMACORR'].data[0]
            reff = hdu[0].header['REFF']
            phot_ell = hdu[0].header['ECOOELL']
            phot_inc = np.degrees(np.arccos(1 - phot_ell))
            pri, sec, anc, oth = parse_manga_targeting_bits(hdu[0].header['MNGTARG1'], hdu[0].header['MNGTARG3'])
            maxr = 2.5 if sec else 1.5

            # Get the masks
            if mask_flags is None:
                vel_mask = np.zeros(vel.shape, dtype=bool)
                sig_mask = np.zeros(sig.shape, dtype=bool)
            elif mask_flags == 'any':
                vel_mask = hdu['STELLAR_VEL_MASK'].data > 0
                sig_mask = hdu['STELLAR_SIGMA_MASK'].data > 0
            else:
                bitmask = sdss_bitmask('MANGA_DAPPIXMASK')
                vel_mask = bitmask.flagged(hdu['STELLAR_VEL_MASK'].data, flag=mask_flags)
                sig_mask = bitmask.flagged(hdu['STELLAR_SIGMA_MASK'].data, flag=mask_flags)

        if not quiet:
            print('Done')

        if covar:
            if not quiet:
                print('Building covariance matrices ... ')
            sb_gpm, sb_covar = manga_map_covar(np.ma.MaskedArray(sb_ivar, mask=sb_mask),
                                               binid=binid, positive_definite=False, fill=True)
            vel_gpm, vel_covar = manga_map_covar(np.ma.MaskedArray(vel_ivar, mask=vel_mask),
                                                 binid=binid, positive_definite=False, fill=True)
            sig_gpm, sig_covar = manga_map_covar(np.ma.MaskedArray(sig_ivar, mask=sig_mask),
                                                 binid=binid, positive_definite=False, fill=True)
            if not quiet:
                print('Done')
        else:
            sb_covar, vel_covar, sig_covar = None, None, None

        super().__init__(vel, vel_ivar=vel_ivar, vel_mask=vel_mask, vel_covar=vel_covar, x=x, y=y,
                         sb=sb, sb_ivar=sb_ivar, sb_mask=sb_mask, sb_covar=sb_covar, sig=sig, 
                         sig_ivar=sig_ivar, sig_mask=sig_mask, sig_covar=sig_covar,
                         sig_corr=sig_corr, psf=psf, binid=binid, grid_x=grid_x, 
                         grid_y=grid_y, reff=reff, fwhm=fwhm, image=image,
                         phot_inc=phot_inc, maxr=maxr, positive_definite=positive_definite)


class MaNGAGlobalPar(GlobalPar):
    """
    Provides MaNGA-specific implementation of global parameters.

    Args:
        plate (:obj:`int`):
            Plate identifier
        ifu (:obj:`int`):
            IFU identifier
        redux_path (:obj:`str`, optional):
            The top-level directory with all DRP output. If None,
            this will be set to the ``MANGA_SPECTRO_REDUX``
            environmental variable, if it is defined.
        dr (:obj:`str`, optional):
            Data release identifier that matches the directory with
            the data.
        drpall_file (:obj:`str`, optional):
            DRPall filename. If None, the filename is assumed to be
            ``drpall*.fits`` and the path is constructed from other
            parameters.
        drpall_path (:obj:`str`, optional):
            This provides the *direct* path to the drpall file,
            circumventing the use of ``dr`` and ``redux_path``.
    """
    def __init__(self, plate, ifu, redux_path=None, dr=None, drpall_file=None, drpall_path=None, 
                 **kwargs):

        if drpall_file is None:
            # Set the path to the DRPall file
            if drpall_path is None:
                _redux_path = os.getenv('MANGA_SPECTRO_REDUX') \
                                if redux_path is None else redux_path
                if _redux_path is None:
                    raise ValueError('Could not define top-level root for DRP output.')
                drpall_path = os.path.join(_redux_path, dr)
            drpall_file = glob.glob(os.path.join(drpall_path, 'drpall*.fits'))[0]

        # Read the table
        plateifu = f'{plate}-{ifu}'
        print('Reading DRPall file...')
        with fits.open(drpall_file) as hdu:
            drpall = hdu['MANGA'].data
        print('    DONE')

        # Find the relevant row
        indx = np.where(drpall['PLATEIFU'] == plateifu)[0]
        if len(indx) != 1:
            raise ValueError(f'Could not find {plateifu} in {drpall_file}.')

        # Instantiate the object
        indx = indx[0]
        super().__init__(ra=drpall['objra'][indx], dec=drpall['objdec'][indx],
                         mass=drpall['nsa_sersic_mass'][indx], z=drpall['z'][indx],
                         pa=drpall['nsa_elpetro_phi'][indx], ell=1.-drpall['nsa_elpetro_ba'][indx],
                         reff=drpall['nsa_elpetro_th50_r'][indx],
                         sersic_n=drpall['nsa_sersic_n'][indx], **kwargs)

        # Save MaNGA-specific attributes
        self.mangaid = drpall['mangaid'][indx]
        self.plate = plate
        self.ifu = ifu
        self.drpall_file = drpall_file
        self.primaryplus, self.secondary, self.ancillary, self.other \
                = parse_manga_targeting_bits(drpall['mngtarg1'][indx],
                                             mngtarg3=drpall['mngtarg3'][indx])
        self.psf_band = np.array(['g', 'r', 'i', 'z'])
        self.psf_fwhm = np.array([drpall['gfwhm'][indx], drpall['rfwhm'][indx],
                                  drpall['ifwhm'][indx], drpall['zfwhm'][indx]])

        self.mag_band = np.array(['NUV', 'r', 'i'])
        self.mag = np.array([drpall['nsa_elpetro_absmag'][indx][1],
                             drpall['nsa_elpetro_absmag'][indx][4],
                             drpall['nsa_elpetro_absmag'][indx][5]])


