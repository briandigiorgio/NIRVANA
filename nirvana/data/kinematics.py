"""
Implements base class to hold observational data fit by the kinematic
model.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np
from scipy import sparse
from scipy import linalg
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import warnings

try:
    import theano.tensor as tt
except:
    tt = None
#try:
#    import cupy as cp
#except:
#    cp = None
cp = None

from .util import get_map_bin_transformations, impose_positive_definite

from ..models.beam import construct_beam, ConvolveFFTW, smear
from ..models.geometry import projected_polar
from ..models import oned

class Kinematics():
    r"""
    Base class to hold data fit by the kinematic model.

    All data to be fit by this package must be contained in a class
    that inherits from this one.

    On the coordinate grids: If the data are binned, the provided
    ``x`` and ``y`` arrays are assumed to be the coordinates of the
    unique bin measurements. I.e., all the array elements in the same
    bin should have the same ``x`` and ``y`` values. However, when
    modeling the data we need the coordinates of each grid cell, not
    the (irregular) binned grid coordinate. These are provided by the
    ``grid_x`` and ``grid_y`` arguments; these two arrays are
    *required* if ``binid`` is provided.

    Args:
        vel (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
            The velocity measurements of the kinematic tracer to be
            modeled.  Must be a square 2D array.
        vel_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the velocity measurements. If None,
            all values are set to 1.
        vel_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the velocity measurements. If
            None, all pixels are considered valid. If ``vel`` is
            provided as a masked array, this mask is combined with
            ``vel.mask``.
        x (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The on-sky Cartesian :math:`x` coordinates of each
            velocity measurement. Units are irrelevant, but they
            should be consistent with any expectations of the fitted
            model. If None, ``x`` is just the array index, except
            that it is assumed to be sky-right (increasing from
            *large to small* array indices; aligned with right
            ascension coordinates). Also, the coordinates are offset
            such that ``x==0`` is at the center of the array and
            increase along the first axis of the velocity array.
        y (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The on-sky Cartesian :math:`y` coordinates of each
            velocity measurement. Units are irrelevant, but they
            should be consistent with any expectations of the fitted
            model. If None, ``y`` is just the array index, offset
            such that ``y==0`` is at the center of the array and
            increase along the second axis of the velocity array.
        sb (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The observed surface brightness of the kinematic tracer.
            Ignored if None.
        sb_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the surface-brightness measurements.
            If None and ``sb`` is provided, all values are set to 1.
        sb_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the surface-brightness
            measurements. If None, all pixels are considered valid.
        sig (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The velocity dispersion of the kinematic tracer. Ignored
            if None.
        sig_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the velocity dispersion measurements.
            If None and ``sig`` is provided, all values are set to 1.
        sig_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the velocity-dispersion
            measurements. If None, all measurements are considered
            valid.

        sig_corr (`numpy.ndarray`_, optional):
            A quadrature correction for the velocity dispersion
            measurements. If None, velocity dispersions are assumed
            to be the *astrophysical* Doppler broadening of the
            kinematic tracer. If provided, the corrected velocity
            dispersion is:

            .. math::
                \sigma^2 = \sigma_{\rm obs}^2 - \sigma_{\rm corr}^2

            where :math:`\sigma_{\rm obs}` is provided by ``sig``.

        psf_name (:obj:`str`, optional):
            Identifier for the psf used. For example, this can be the
            wavelength band where the PSF was measured. If provided, this
            identifier is only used for informational purposes in output
            files.
        psf (`numpy.ndarray`_, optional):
            An image of the point-spread function of the
            observations. If ``aperture`` is not provided, this
            should be the effective smoothing kernel for the
            kinematic fields. Otherwise, this is the on-sky seeing
            kernel and the effective smoothing kernel is constructed
            as the convolution of this image with ``aperture``. If
            None, any smearing of the kinematic data is ignored.
            Shape must match ``vel`` and the extent of the PSF map
            must identically match ``vel``.
        aperture (`numpy.ndarray`_, optional):
            Monochromatic image of the spectrograph aperture. See
            ``psf`` for how this is used.
        binid (`numpy.ndarray`_, optional):
            Integer array associating each measurement with a unique
            bin number. Measurements not associated with any bin
            should have a value of -1 in this array. If None, all
            (unmasked) measurements are considered unique.
        grid_x (`numpy.ndarray`_, optional):
            The on-sky Cartesian :math:`x` coordinates of *each*
            element in the data grid. If the data are unbinned, this
            array is identical to `x` (except that *every* value
            should be valid). This argument is *required* if
            ``binid`` is provided.
        grid_y (`numpy.ndarray`_, optional):
            The on-sky Cartesian :math:`y` coordinates of *each*
            element in the data grid. See the description of
            ``grid_x``.
        grid_sb (`numpy.ndarray`_, optional):
            The relative surface brightness of the kinematic tracer over the
            full coordinate grid.  If None, this is either assumed to be unity
            or set by the provided ``sb``.  When fitting the data with, e.g., 
            :class:`~nirvana.model.axisym.AxisymmetricDisk` via the ``sb_wgt``
            parameter in its fitting method, this will be the weighting used.
            The relevance of this array is to enable the weighting used in
            constructing the model velocity field to be *unbinned* for otherwise
            binned kinematic data.
        grid_wcs (`astropy.wcs.WCS`_, optional):
            World coordinate system for the on-sky grid. Currently, this is
            only used for output files.
        reff (:obj:`float`, optional):
            Effective radius in same units as :attr:`x` and :attr:`y`.
        fwhm (:obj:`float`, optional):
            The FWHM of the PSF of the galaxy in the same units as :attr:`x` and
            :attr:`y`.
        phot_inc (:obj:`float`, optional):
            Photometric inclination in degrees.
        maxr (:obj:`float`, optional):
            Maximum radius of useful data in effective radii.

    Raises:
        ValueError:
            Raised if the input arrays are not 2D or square, if any
            of the arrays do not match the shape of ``vel``, if
            either ``x`` or ``y`` is provided but not both or
            neither, or if ``binid`` is provided but ``grid_x`` or
            ``grid_y`` is None.
    """
    def __init__(self, vel, vel_ivar=None, vel_mask=None, vel_covar=None, x=None, y=None, sb=None,
                 sb_ivar=None, sb_mask=None, sb_covar=None, sb_anr=None, sig=None, sig_ivar=None,
                 sig_mask=None, sig_covar=None, sig_corr=None, psf_name=None, psf=None,
                 aperture=None, binid=None, grid_x=None, grid_y=None, grid_sb=None, grid_wcs=None,
                 reff=None, fwhm=None, image=None, phot_inc=None, phot_pa=None, maxr=None,
                 positive_definite=False, quiet=False):

        # Check shape of input arrays
        self.nimg = vel.shape[0]
        if len(vel.shape) != 2:
            raise ValueError('Input arrays to Kinematics must be 2D.')
        # TODO: I don't remember why we have this restriction (maybe it was
        # just because I didn't want to have to worry about having to
        # accommodate anything but MaNGA kinematic fields yet), but we should
        # look to get rid of this constraint of a square map.
        if vel.shape[1] != self.nimg:
            raise ValueError('Input arrays to Kinematics must be square.')
        for a in [vel_ivar, vel_mask, x, y, sb, sb_ivar, sb_mask, sig, sig_ivar, sig_mask,
                  sig_corr, psf, aperture, binid, grid_x, grid_y, grid_sb]:
            if a is not None and a.shape != vel.shape:
                raise ValueError('All arrays provided to Kinematics must have the same shape.')
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError('Must provide both x and y or neither.')
        if binid is not None and grid_x is None or grid_y is None:
            raise ValueError('If the data are binned, you must provide the pixel-by-pixel input '
                             'coordinate grids, grid_x and grid_y.')

        # Basic properties
        self.spatial_shape = vel.shape
        self.psf_name = 'unknown' if psf_name is None else psf_name
        self._set_beam(psf, aperture)
        self.reff = reff
        self.fwhm = fwhm
        self.image = image
        self.sb_anr = sb_anr
        self.phot_inc = phot_inc
        self.phot_pa = phot_pa
        self.maxr = maxr

        # Build coordinate arrays
        if x is None:
            # No coordinate arrays provided, so just assume a
            # coordinate system with 0 at the center. Ensure that
            # coordinates mimic being "sky-right" (i.e., x increases
            # toward lower pixel indices).
            self.x, self.y = map(lambda x : x - self.nimg//2,
                                 np.meshgrid(np.arange(self.nimg)[::-1], np.arange(self.nimg)))
        else:
            self.x, self.y = x, y

        # Build map data
        self.sb, self.sb_ivar, self.sb_mask = self._ingest(sb, sb_ivar, sb_mask)
        self.vel, self.vel_ivar, self.vel_mask = self._ingest(vel, vel_ivar, vel_mask)
        self.sig, self.sig_ivar, self.sig_mask = self._ingest(sig, sig_ivar, sig_mask)
        # Have to treat sig_corr separately
        if isinstance(sig_corr, np.ma.MaskedArray):
            self.sig_mask |= np.ma.getmaskarray(sig_corr)
            self.sig_corr = sig_corr.data
        else:
            self.sig_corr = sig_corr

        # The following are arrays used to convert between arrays
        # holding the data for the unique bins to arrays with the full
        # data map.
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_wcs = grid_wcs
        self.binid, self.nspax, self.bin_indx, self.grid_indx, self.bin_inverse, \
            self.bin_transform \
                = get_map_bin_transformations(spatial_shape=self.spatial_shape, binid=binid)

        # Unravel and select the valid values for all arrays
        for attr in ['x', 'y', 'sb', 'sb_ivar', 'sb_mask', 'vel', 'vel_ivar', 'vel_mask', 'sig', 
                     'sig_ivar', 'sig_mask', 'sig_corr', 'sb_anr']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).ravel()[self.bin_indx])

        # Set the surface-brightness grid.  This needs to be after the
        # unraveling of the attributes done in the lines above so that I can use
        # self.remap in the case that grid_sb is not provided directly.
        self.grid_sb = self.remap('sb').filled(0.0) if grid_sb is None else grid_sb

        # Calculate the square of the astrophysical velocity
        # dispersion. This is just the square of the velocity
        # dispersion if no correction is provided. The error
        # calculation assumes there is no error on the correction.
        # TODO: Change this to sig2 or sigsqr
        # TODO: Need to keep track of mask...
        self.sig_phys2 = self.sig**2 if self.sig_corr is None else self.sig**2 - self.sig_corr**2
        self.sig_phys2_ivar = None if self.sig_ivar is None \
                                    else self.sig_ivar/(2*self.sig + (self.sig == 0.0))**2

        # Ingest the covariance matrices, if they're provided
        self.vel_covar = self._ingest_covar(vel_covar, positive_definite=positive_definite)
        self.sb_covar = self._ingest_covar(sb_covar, positive_definite=False) #positive_definite)
        self.sig_covar = self._ingest_covar(sig_covar, positive_definite=positive_definite)

        # Construct the covariance in the square of the astrophysical velocity
        # dispersion.
        if self.sig_covar is None:
            self.sig_phys2_covar = None
        else:
            jac = sparse.diags(2*self.sig, format='csr')
            self.sig_phys2_covar = jac.dot(self.sig_covar.dot(jac.T))

        # TODO: Should issue a some warning/error if the user has provided both
        # ivar and covar and they are not consistent


    def _set_beam(self, psf, aperture):
        """
        Instantiate :attr:`beam` and :attr:`beam_fft`.

        If both ``psf`` and ``aperture`` are None, the convolution
        kernel for the data is assumed to be unknown.

        Args:
            psf (`numpy.ndarray`_):
                An image of the point-spread function of the
                observations. If ``aperture`` is None, this should be
                the effective smoothing kernel for the kinematic
                fields. Otherwise, this is the on-sky seeing kernel
                and the effective smoothing kernel is constructed as
                the convolution of this image with ``aperture``. If
                None, the kernel will be set to the ``aperture``
                value (if provided) or None.
            aperture (`numpy.ndarray`_):
                Monochromatic image of the spectrograph aperture. If
                ``psf`` is None, this should be the effective
                smoothing kernel for the kinematic fields. Otherwise,
                this is the on-sky representation of the spectrograph
                aperture and the effective smoothing kernel is
                constructed as the convolution of this image with
                ``psf``. If None, the kernel will be set to the
                ``psf`` value (if provided) or None.
        """
        if psf is None and aperture is None:
            self.beam = None
            self.beam_fft = None
            return
        if psf is None:
            self.beam = aperture/np.sum(aperture)
            self.beam_fft = np.fft.fftn(np.fft.ifftshift(aperture))
            return
        if aperture is None:
            self.beam = psf/np.sum(psf)
            self.beam_fft = np.fft.fftn(np.fft.ifftshift(psf))
            return
        self.beam_fft = construct_beam(psf/np.sum(psf), aperture/np.sum(aperture), return_fft=True)
        self.beam = np.fft.fftshift(np.fft.ifftn(self.beam_fft).real)

    def _ingest(self, data, ivar, mask):
        """
        Check the data for ingestion into the object.

        Args:
            data (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Kinematic measurements. Can be None.
            ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Inverse variance in the kinematic measurements.
                Regardless of the input, any pixel with an inverse
                variance that is not greater than 0 is automatically
                masked.
            mask (`numpy.ndarray`_):
                A boolean bad-pixel mask (i.e., values to ignore are
                set to True). This is the baseline mask that is
                combined with any masks provide by ``data`` and
                ``ivar`` if either are provided as
                `numpy.ma.MaskedArray`_ objects. The returned mask
                also automatically masks any bad inverse-variance
                values. If None, the baseline mask is set to be False
                for all pixels.

        Returns:
            :obj:`tuple`: Return three `numpy.ndarray`_ objects with the
            ingested data, inverse variance, and boolean mask. The first two
            arrays are forced to have type ``numpy.float64``.
        """
        if data is None:
            # No data, so do nothing
            return None, None, None

        # Initialize the mask
        _mask = np.zeros(self.spatial_shape, dtype=bool) if mask is None else mask.copy()

        # Set the data and incorporate the mask for a masked array
        if isinstance(data, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(data)
            _data = data.data.astype(np.float64)
        else:
            _data = data.astype(np.float64)

        # Set the error and incorporate the mask for a masked array
        if ivar is None:
            # Don't instantiate the array if we don't need to.
            _ivar = None
        elif isinstance(ivar, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(ivar)
            _ivar = ivar.data.astype(np.float64)
        else:
            _ivar = ivar.astype(np.float64)
        # Make sure to mask any measurement with ivar <= 0
        if _ivar is not None:
            _mask |= np.logical_not(_ivar > 0)

        return _data, _ivar, _mask

    def _ingest_covar(self, covar, positive_definite=True, quiet=False):
        """
        Ingest an input covariance matrix for use when fitting the data.

        The covariance matrix is forced to be positive everywhere (any negative
        values are set to 0) and to be identically symmetric.  

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix. It's shape must match the input map shape.
                If None, the returned value is also None.
            positive_definite (:obj:`bool`, optional):
                Use :func:`~nirvana.data.util.impose_positive_definite` to force
                the provided covariance matrix to be positive definite.
            quiet (:obj:`bool`, optional):
                Suppress output to stdout.

        Returns:
            `scipy.sparse.csr_matrix`_: The covariance matrix of the good bin
            values.
        """
        if covar is None:
            return None

        if not quiet:
            print('Ingesting covariance matrix ... ')

        nspax = np.prod(self.spatial_shape)
        if covar.shape != (nspax,nspax):
            raise ValueError('Input covariance matrix has incorrect shape: {0}'.format(covar.shape))

        _covar = covar.copy() if isinstance(covar, sparse.csr.csr_matrix) \
                    else sparse.csr_matrix(covar)

        # It should be the case that, on input, the covariance matrix should
        # demonstrate that the map values that are part of the same bin by being
        # perfectly correlated. The matrix operation below constructs the
        # covariance in the *binned* data, but you should also be able to
        # obtain this just by selecting the appropriate rows/columns of the
        # covariance matrix. You should be able to recover the input covariance
        # matrix (or at least the populated regions of it) like so:

        #   # Covariance matrix of the binned data
        #   vc = self.bin_transform.dot(vel_covar.dot(self.bin_transform.T))
        #   # Revert
        #   gpm = np.logical_not(vel_mask)
        #   _bt = self.bin_transform[:,gpm.ravel()].T.copy()
        #   _bt[_bt > 0] = 1.
        #   ivc = _bt.dot(vc.dot(_bt.T))
        #   assert np.allclose(ivc.toarray(),
        #                      vel_covar[np.ix_(gpm.ravel(), gpm.ravel())].toarray())

        _covar = self.bin_transform.dot(_covar.dot(self.bin_transform.T))

        # Deal with possible numerical error
        # - Force it to be positive
        _covar[_covar < 0] = 0.
        # - Force it to be identically symmetric
        _covar = (_covar + _covar.T)/2
        # - Force it to be positive definite if requested
        return impose_positive_definite(_covar) if positive_definite else _covar

    def remap(self, data, mask=None, masked=True, fill_value=0):
        """
        Remap the requested attribute to the full 2D array.

        Args:
            data (`numpy.ndarray`_, :obj:`str`):
                The data or attribute to remap. If the object is a
                string, the string must be a valid attribute.
            mask (`numpy.ndarray`_, optional):
                Boolean mask with the same shape as ``data`` or the selected
                ``data`` attribute. If ``data`` is provided as a
                `numpy.ndarray`_, this provides an associated mask. If
                ``data`` is provided as a string, this is a mask is used *in
                addition to* any mask associated with selected attribute.
                Ignored if set to None.
            masked (:obj:`bool`, optional):
                Return data as a masked array, where data that are not filled
                by the provided data. If ``data`` is a string selecting an
                attribute and an associated mask exists for that attribute
                (called "{data}_mask"), also include the mask in the output.
            fill_value (scalar-like, optional):
                Value used to fill the masked pixels, if a masked array is
                *not* requested. Warning: The value is automatically
                converted to be the same data type as the input array or
                attribute.

        Returns:
            `numpy.ndarray`_, `numpy.ma.MaskedArray`_: 2D array with
            the attribute remapped to the original on-sky locations.

        Raises:
            ValueError:
                Raised if ``data`` is a `numpy.ndarray`_ and the
                shape does not match the expected 1d shape.
            AttributeError:
                Raised if ``data`` is a string and the requested
                attribute is invalid.

        """
        if isinstance(data, str):
            # User attempting to select an attribute. First check it exists.
            if not hasattr(self, data):
                raise AttributeError('No attribute called {0}.'.format(data))
            # Get the data
            d = getattr(self, data)
            if d is None:
                # There is no data, so just return None
                return None
            # Try to find the mask
            m = '{0}_mask'.format(data)
            if not masked or not hasattr(self, m) or getattr(self, m) is None:
                # If there user doesn't want the mask, there is no mask, or the
                # mask is None, ignore it
                m = None if mask is None else mask
            else:
                # Otherwise, get it
                m = getattr(self, m)
                if mask is not None:
                    m |= mask
        else:
            # User provided arrays directly
            d = data
            m = mask

        # Check the shapes (overkill if the user selected an attribute...)    
        if d.shape != self.vel.shape and tt is not None and type(d) is not tt.TensorVariable:
            raise ValueError('To remap, data must have the same shape as the internal data '
                             'attributes: {0}'.format(self.vel.shape))
        if m is not None and m.shape != self.vel.shape:
            raise ValueError('To remap, mask must have the same shape as the internal data '
                             'attributes: {0}'.format(self.vel.shape))

        # Construct the output map
#        _data = np.ma.masked_all(self.spatial_shape, dtype=d.dtype)
        # NOTE: np.ma.masked_all sets the initial data array to
        # 2.17506892e-314, which just leads to trouble. I've replaced this with
        # the line below to make sure that the initial value is just 0.
        _data = np.ma.MaskedArray(np.zeros(self.spatial_shape, dtype=d.dtype), mask=True)
        _data[np.unravel_index(self.grid_indx, self.spatial_shape)] = d[self.bin_inverse]
        if m is not None:
            np.ma.getmaskarray(_data)[np.unravel_index(self.grid_indx, self.spatial_shape)] \
                    = m[self.bin_inverse]
        # Return a masked array if requested; otherwise, fill the masked values
        # with the equivalent of 0. WARNING: this will be False for a boolean
        # array...
        return _data if masked else _data.filled(d.dtype.type(fill_value))

    # TODO: Include an optional weight map.  E.g., to mimic the luminosity
    # weighting of the kinematics in data.
    def bin(self, data):
        """
        Provided a set of mapped data, rebin it to match the internal vectors.

        This method is most often used to bin maps of model data to match the
        binning of the kinematic data.  The operation takes the average of
        ``data`` within a bin defined by the kinematic data.  For unbinned data,
        this operation simply selects and reorders the data from the input map
        to match the internal vectors with the kinematic data.
        
        Args:
            data (`numpy.ndarray`_):
                Data to rebin. Shape must match :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: A vector with the data rebinned to match the
            number of unique measurements available.

        Raises:
            ValueError:
                Raised if the shape of the input array is incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        return self.bin_transform.dot(data.ravel())

    # TODO: Include an optional weight map.  E.g., to mimic the luminosity
    # weighting of the kinematics in data.
    def deriv_bin(self, data, deriv):
        """
        Provided a set of mapped data, rebin it to match the internal vectors.

        This method is most often used to bin maps of model data to match the
        binning of the kinematic data.  The operation takes the average of
        ``data`` within a bin defined by the kinematic data.  For unbinned data,
        this operation simply selects and reorders the data from the input map
        to match the internal vectors with the kinematic data.

        This method is identical to :func:`bin`, except that it allows for
        propagation of derivatives of the provided model with respect to its
        parameters.  The propagation of derivatives for any single parameter is
        identical to calling :func:`bin` on that derivative map.
        
        Args:
            data (`numpy.ndarray`_):
                Data to rebin. Shape must match :attr:`spatial_shape`.
            deriv (`numpy.ndarray`_):
                If the input data is a kinematic model, this provides the
                derivatives of model w.r.t. its parameters.  The first two axes
                of the array must have a shape that matches
                :attr:`spatial_shape`.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ arrays.  The first provides the
            vector with the data rebinned to match the number of unique
            measurements available, and the second is a 2D array with the binned
            derivatives for each model parameter.

        Raises:
            ValueError:
                Raised if the spatial shapes of the input arrays are incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        if deriv.shape[:2] != self.spatial_shape:
            raise ValueError('Derivative shape is incorrect; expected {0}, found {1}.'.format(
                              self.spatial_shape, deriv.shape[:2]))
        return self.bin_transform.dot(data.ravel()), \
                    np.stack(tuple([self.bin_transform.dot(deriv[...,i].ravel())
                                    for i in range(deriv.shape[-1])]), axis=-1)

    def unique(self, data):
        """
        Provided a set of binned and remapped data (i.e., each element in a
        bin has the same value in the map), select the unique values from the
        map.

        This is the same operation performed on the input 2D maps of data to
        extract the unique data vectors; e.g.,::

            assert np.array_equal(self.vel, self.unique(self.remap('vel', masked=False)))

        Args:
            data (`numpy.ndarray`_):
                The 2D data array from which to extract the unique data.
                Shape must be :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: The 1D vector with the unique data.

        Raises:
            ValueError:
                Raised if the spatial shape is wrong.
        """
        if data.shape != self.spatial_shape:
            raise ValueError(f'Input has incorrect shape; found {data.shape}, '
                             f'expected {self.spatial_shape}.')
        return data.flat[self.bin_indx]

    def max_radius(self):
        """
        Calculate and return the maximum *on-sky* radius of the valid data.
        Note this not the in-plane disk radius; however, the two are the same
        along the major axis.
        """
        minx = np.amin(self.x)
        maxx = np.amax(self.x)
        miny = np.amin(self.y)
        maxy = np.amax(self.y)
        return np.sqrt(max(abs(minx), maxx)**2 + max(abs(miny), maxy)**2)

    def reject(self, vel_rej=None, sig_rej=None):
        r"""
        Reject/Mask data.

        This is a simple wrapper that incorporates the provided vectors into
        the kinematic masks.

        Args:
            vel_rej (`numpy.ndarray`_, optional):
                Boolean vector selecting the velocity measurements to reject.
                Shape must be :math:`N_{\rm bin}`. If None, no additional
                data are rejected.
            sig_rej (`numpy.ndarray`_, optional):
                Boolean vector selecting the velocity dispersion measurements
                to reject. Shape must be :math:`N_{\rm bin}`. If None, no
                additional data are rejected. Ignored if :attr:`sig` is None.
        """
        if vel_rej is not None:
            self.vel_mask |= vel_rej
        if self.sig is not None and sig_rej is not None:
            self.sig_mask |= sig_rej

    def remask(self, mask):
        '''
        Apply a given mask to the masks that are already in the object.

        Args:
            mask (`numpy.ndarray`):
                Mask to apply to the data. Should be the same shape as the
                data (either 1D binned or 2D). Will be interpreted as boolean.

        Raises:
            ValueError:
                Thrown if input mask is not the same shape as the data.
        '''

        if mask.ndim > 1 and mask.shape != self.spatial_shape:
            raise ValueError('Mask is not the same shape as data.')
        if mask.ndim == 1 and len(mask) != len(self.vel):
            raise ValueError('Mask is not the same length as data')

        for m in ['sb_mask', 'vel_mask', 'sig_mask']:
            if m is None: continue
            if mask.ndim > 1: mask = self.bin(mask)
            setattr(self, m, np.array(getattr(self, m) + mask, dtype=bool))

    def clip_err(self, max_vel_err=None, max_sig_err=None):
        """
        Reject data with large errors.

        The rejection is directly incorporated into :attr:`vel_mask` and
        :attr:`sig_mask`.

        Args:
            max_vel_err (:obj:`float`, optional):
                Maximum allowed velocity error. If None, no additional
                masking is performed.
            max_sig_err (:obj:`float`, optional):
                Maximum allowed *observed* velocity dispersion error. I.e.,
                this is the measurement error before any velocity dispersion
                correction.  If None, no additional masking is performed.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data that
            were rejected. If :attr:`sig` is None, the returned object
            selecting the velocity dispersion data that was rejected is also
            None.
        """
        vel_rej = np.zeros(self.vel.size, dtype=bool) if max_vel_err is None else \
                    self.vel_ivar < 1/max_vel_err**2
        sig_rej = None if self.sig is None else \
                    (np.zeros(self.sig.size, dtype=bool) if max_sig_err is None else
                     self.sig_ivar < 1/max_sig_err**2)
        self.reject(vel_rej=vel_rej, sig_rej=sig_rej)
        return vel_rej, sig_rej

    # TODO: Include a separate S/N measurement, like as done with A/N for the
    # gas.
    def clip_snr(self, min_vel_snr=None, min_sig_snr=None):
        """
        Reject data with low S/N.

        The S/N of a given spaxel is given by the ratio of its surface
        brightness to the error in the surface brightness. An exception is
        raised if the surface-brightness or surface-brightness error are not
        defined.

        The rejection is directly incorporated into :attr:`vel_mask` and
        :attr:`sig_mask`.

        Args:
            min_vel_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for velocity measurements. If
                None, no additional masking is performed.
            min_sig_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for dispersion measurements.
                If None, no additional masking is performed.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data that
            were rejected. If :attr:`sig` is None, the returned object
            selecting the velocity dispersion data that was rejected is also
            None.
        """
        if self.sb is None or self.sb_ivar is None:
            raise ValueError('Cannot perform S/N rejection; no surface brightness and/or '
                             'surface brightness error data.')
        snr = self.sb * np.sqrt(self.sb_ivar)
        vel_rej = np.zeros(self.vel.size, dtype=bool) if min_vel_snr is None else snr < min_vel_snr
        sig_rej = None if self.sig is None else \
                    (np.zeros(self.sig.size, dtype=bool) if min_sig_snr is None else
                     snr < min_sig_snr)
        self.reject(vel_rej=vel_rej, sig_rej=sig_rej)
        return vel_rej, sig_rej

    def cupy_convert(self):
        '''
        Convert all attributes that are numpy arrays to cupy arrays.

        If CuPy is installed correctly and a GPU is available, converting
        things to CuPy arrays should allow certain tasks to be offloaded to
        the GPU, potentially speeding up computation considerably.
        '''
        if cp is None:
            raise ImportError('CuPy did not import correctly. Cannot convert arrays')

        attrs = [a for a in dir(self) if '__' not in a]
        for a in attrs:
            if isinstance(getattr(self, a), np.ndarray):
                try: setattr(self, cp.array(getattr(self, a)))
                except: pass
