"""
Implements base class to hold observational data fit by the kinematic
model.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

# TODO: 
#   - Do not *require* that Kinematics use (square) 2D maps
#   - Allow Kinematics to hold more than one component (i.e., gas and
#     stars or multiple gas lines)
#   - Allow Kinematics to include (inverse) covariance
#   - Allow for errors in the sigma correction?
#   - Create an automated way of constructing `grid_x` and `grid_y`. Will
#     be required when/if kinematics are not provided in a regular array
#     grid. This is needed so that we know where to calculate the model.

from IPython import embed

import numpy as np


class Kinematics:
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

    Raises:
        ValueError:
            Raised if the input arrays are not 2D or square, if any
            of the arrays do not match the shape of ``vel``, if
            either ``x`` or ``y`` is provided but not both or
            neither, or if ``binid`` is provided but ``grid_x`` or
            ``grid_y`` is None.
    """
    def __init__(self, vel, vel_ivar=None, vel_mask=None, x=None, y=None, sb=None, sb_ivar=None,
                 sb_mask=None, sig=None, sig_ivar=None, sig_mask=None, sig_corr=None, psf=None,
                 aperture=None, binid=None, grid_x=None, grid_y=None):

        # Check shape of input arrays
        self.nimg = vel.shape[0]
        if len(vel.shape) != 2:
            raise ValueError('Input arrays to Kinematics must be 2D.')
        if vel.shape[1] != self.nimg:
            raise ValueError('Input arrays to Kinematics must be square.')
        for a in [vel_ivar, vel_mask, x, y, sb, sb_ivar, sb_mask, sig, sig_ivar, sig_mask,
                  sig_corr, psf, aperture, binid, grid_x, grid_y]:
            if a is not None and a.shape != vel.shape:
                raise ValueError('All arrays provided to Kinematics must have the same shape.')
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError('Must provide both x and y or neither.')
        if binid is not None and grid_x is None or grid_y is None:
            raise ValueError('If the data are binned, you must provide the pixel-by-pixel input '
                             'coordinate grids, grid_x and grid_y.')

        # Basic properties
        self.spatial_shape = vel.shape

        self.psf = psf
        self.aperture = aperture

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
        #   - The input BIN IDs
        self.binid = binid
        self.grid_x = grid_x
        self.grid_y = grid_y
        #   - bin_transform is used to bin data in a map with
        #     per-spaxel data to match the binning applied to the
        #     measurements. E.g., this is used to match a per-spaxel model
        #     to the binned data. By default (binid is None), this is just
        #     an identity vector. See :func:`bin` for how this is used.
        self.bin_transform = np.ones(np.prod(self.spatial_shape), dtype=float)
        #   - grid_indx and bin_inverse serve similar purposes and are
        #   identical if the data are unbinned. grid_index gives the
        #   flattened index of each unique measurement in the input map.
        #   bin_inverse gives the indices that can be used to reconstruct
        #   the input maps based on the values extracted for the unique
        #   bin IDs. See :func:`remap` for how these arrays are used to
        #   reconstruct the input maps.
        self.grid_indx = np.arange(np.prod(self.spatial_shape))
        self.bin_inverse = self.grid_indx.copy()

        if self.binid is None:
            indx = self.grid_indx.copy()
        else:
            # Get the indices of measurements with unique bin IDs, ignoring any
            # IDs set to -1
            binid_map = self.binid.ravel()
            self.binid, indx, self.bin_inverse, nbin \
                = np.unique(binid_map, return_index=True, return_inverse=True, return_counts=True)
            if np.any(self.binid == -1):
                self.binid = self.binid[1:]
                indx = indx[1:]
                self.grid_indx = self.grid_indx[self.bin_inverse > 0]
                self.bin_inverse = self.bin_inverse[self.bin_inverse > 0] - 1
                nbin = nbin[1:]

            # NOTE: In most cases, self.binid[self.bin_inverse] is
            # identical to self.bin_inverse. The exception is if the
            # bin numbers are not sequential, i.e., the bin numbers are
            # not identical to np.arange(nbin).

            # Construct the bin transform
            # TODO: Might be a faster way of doing this...
            # TODO: Set this to be a scipy.sparse.csr_matrix?
            self.bin_transform = np.array([(binid_map == b).astype(float) 
                                            for b in self.binid])/nbin.astype(float)[:,None]

        # Unravel and select the valid values for all arrays
        for attr in ['x', 'y', 'sb', 'sb_ivar', 'sb_mask', 'vel', 'vel_ivar', 'vel_mask', 'sig', 
                     'sig_ivar', 'sig_mask']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).ravel()[indx])

    def _ingest(self, data, ivar, mask):
        """
        Ingest data.
        """
        if data is None:
            # No data, so do nothing
            return None, None, None

        # Initialize the mask
        _mask = np.zeros(self.spatial_shape, dtype=bool) if mask is None else mask.copy()

        # Set the data and incorporate the mask for a masked array
        if isinstance(data, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(data)
            _data = data.data
        else:
            _data = data

        # Set the error and incorporate the mask for a masked array
        if ivar is None:
            _ivar = np.ones(self.spatial_shape, dtype=float)
        elif isinstance(ivar, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(ivar)
            _ivar = ivar.data
        else:
            _ivar = ivar
        # Make sure to mask any measurement with ivar <= 0
        _mask |= np.logical_not(_ivar > 0)

        return _data, _ivar, _mask

    # TODO: include sigma correction when attr='sig'?
    def remap(self, attr, masked=True):
        """
        Remap the requested attribute to the full 2D array.

        Args:
            attr (:obj:`str`):
                The attribute to remap.  Must be a valid attribute.
            masked (:obj:`bool`, optional):
                If an associated mask exists for the selected
                attribute, return the map as a
                `numpy.ma.MaskedArray`_.

        Returns:
            `numpy.ndarray`_, `numpy.ma.MaskedArray`_: 2D array with
            the attribute remapped to the original on-sky locations.

        Raises:
            AttributeError:
                Raised if the requested attribute is invalid.
        """
        if not hasattr(self, attr):
            raise AttributeError('No attribute called {0}.'.format(attr))
        if getattr(self, attr) is None:
            return None

        data = np.zeros(self.spatial_shape, dtype=float)
        data[np.unravel_index(self.grid_indx, self.spatial_shape)] \
                = getattr(self, attr)[self.bin_inverse]
        mask_attr = '{0}_mask'.format(attr)
        if not masked or not hasattr(self, mask_attr) or getattr(self, mask_attr) is None:
            return data

        mask = np.ones(self.spatial_shape, dtype=bool)
        mask[np.unravel_index(self.grid_indx, self.spatial_shape)] \
                = getattr(self, mask_attr)[self.bin_inverse]
        return np.ma.MaskedArray(data, mask=mask)

    def bin(self, data):
        """
        Provided a set of mapped data, rebin it to match the internal
        vectors.

        Args:
            data (`numpy.ndarray`_):
                Data to rebin. Shape must match
                :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: A vector with the data rebinned to
            match the number of unique measurements available.

        Raises:
            ValueError:
                Raised if the shape of the input array is incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        return np.dot(self.bin_transform, data.ravel())

