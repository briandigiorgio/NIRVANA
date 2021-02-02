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

from .fitargs import FitArgs
from .util import get_map_bin_transformations, impose_positive_definite

from ..models.beam import construct_beam, ConvolveFFTW, smear
from ..models.geometry import projected_polar
from ..models import oned, axisym

# TODO: We should separate the needs of the model from the needs of the
# data. I.e., I don't think that Kinematics should inherit from
# FitArgs.
class Kinematics(FitArgs):
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
        reff (:obj:`float`, optional):
            Effective radius in same units as :attr:`x` and :attr:`y`.
        fwhm (:obj:`float`, optional):
            The FWHM of the PSF of the galaxy in the same units as :attr:`x` and
            :attr:`y`.
        bordermask (`numpy.ndarray`_):
            Boolean array containing the mask for a ring around the outside of
            the data. Meant to mask bad data from convolution errors.

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
                 sig_mask=None, sig_covar=None, sig_corr=None, psf=None, aperture=None, binid=None,
                 grid_x=None, grid_y=None, reff=None, fwhm=None, bordermask=None, image=None,
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
        self._set_beam(psf, aperture)
        self.reff = reff
        self.fwhm = fwhm
        self.image = image
        self.sb_anr = sb_anr

        # TODO: This has more to do with the model than the data, so we
        # should put in the relevant model class/method
        self.bordermask = bordermask.astype(bool) if bordermask is not None else None

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
        self.binid, self.bin_indx, self.grid_indx, self.bin_inverse, self.bin_transform \
                = get_map_bin_transformations(spatial_shape=self.spatial_shape, binid=binid)

        # Unravel and select the valid values for all arrays
        for attr in ['x', 'y', 'sb', 'sb_ivar', 'sb_mask', 'vel', 'vel_ivar', 'vel_mask', 'sig', 
                     'sig_ivar', 'sig_mask', 'sig_corr', 'bordermask','sb_anr']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).ravel()[self.bin_indx])

        # Calculate the square of the astrophysical velocity
        # dispersion. This is just the square of the velocity
        # dispersion if no correction is provided. The error
        # calculation assumes there is no error on the correction.
        self.sig_phys2 = self.sig**2 if self.sig_corr is None else self.sig**2 - self.sig_corr**2
        self.sig_phys2_ivar = None if self.sig_ivar is None \
                                    else self.sig_ivar/(2*self.sig + (self.sig == 0.0))**2

        #if self.bordermask is not None:
        #    self.sb_mask  |= self.bordermask
        #    self.vel_mask |= self.bordermask
        #    self.sig_mask |= self.bordermask
        self.vel_covar = self._ingest_covar(vel_covar, positive_definite=positive_definite)
        self.sb_covar = self._ingest_covar(sb_covar, positive_definite=positive_definite)
        self.sig_covar = self._ingest_covar(sig_covar, positive_definite=positive_definite)

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
            :obj:`tuple`: Return three `numpy.ndarray`_ objects with
            the ingested data, inverse variance, and boolean mask.
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
            # Don't instantiate the array if we don't need to.
            _ivar = None
        elif isinstance(ivar, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(ivar)
            _ivar = ivar.data
        else:
            _ivar = ivar
        # Make sure to mask any measurement with ivar <= 0
        if _ivar is not None:
            _mask |= np.logical_not(_ivar > 0)

        return _data, _ivar, _mask

    def _ingest_covar(self, covar, positive_definite=True, quiet=False):
        """
        Ingest an input covariance matrix for use when fitting the data.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix. It's shape must match the input map shape.
                If None, the returned value is also None.

        Returns:
            `scipy.sparse.csr_matrix`_: The covariance matrix of the good bin values.
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

    def remap(self, data, masked=True):
        """
        Remap the requested attribute to the full 2D array.

        Args:
            data (`numpy.ndarray`_, :obj:`str`):
                The data or attribute to remap. If the object is a
                string, the string must be a valid attribute.
            masked (:obj:`bool`, optional):
                Return data as a masked array, where data that are
                not filled by the provided data. If ``data`` is a
                string selecting an attribute and an associated mask
                exists for that attribute, also include the mask in
                the output.

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
        if isinstance(data, np.ndarray):
            if data.shape != self.vel.shape:
                raise ValueError('To remap, must have the same shape as the internal data '
                                 'attributes.')
            _data = np.ma.masked_all(self.spatial_shape, dtype=float) \
                        if masked else np.zeros(self.spatial_shape, dtype=float)
            _data[np.unravel_index(self.grid_indx, self.spatial_shape)] = data[self.bin_inverse]
            return _data

        if not hasattr(self, data):
            raise AttributeError('No attribute called {0}.'.format(data))
        if getattr(self, data) is None:
            return None

        _data = np.ma.masked_all(self.spatial_shape, dtype=float) \
                        if masked else np.zeros(self.spatial_shape, dtype=float)
#        _data = np.zeros(self.spatial_shape, dtype=float)
        _data[np.unravel_index(self.grid_indx, self.spatial_shape)] \
                = getattr(self, data)[self.bin_inverse]
        mask_data = '{0}_mask'.format(data)
        if not masked or not hasattr(self, mask_data) or getattr(self, mask_data) is None:
            return _data

        mask = np.ones(self.spatial_shape, dtype=bool)
        mask[np.unravel_index(self.grid_indx, self.spatial_shape)] \
                = getattr(self, mask_data)[self.bin_inverse]
        _data[mask] = np.ma.masked
#        return np.ma.MaskedArray(_data, mask=mask)
        return _data

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
        return self.bin_transform.dot(data.ravel())

    def max_radius(self):
        """
        Calculate and return the maximum on-sky radius of the valid data.
        """
        minx = np.amin(self.x)
        maxx = np.amax(self.x)
        miny = np.amin(self.y)
        maxy = np.amax(self.y)
        return np.sqrt(max(abs(minx), maxx)**2 + max(abs(miny), maxy)**2)

    # TODO: This should be in a different method/class
    @classmethod
    def mock(cls, size, inc, pa, pab, vsys, vt, v2t, v2r, sig, xc=0, yc=0, reff=10, maxr=15, psf=None, border=3, fwhm=2.44):
        """
        Makes a :class:`nirvana.data.kinematics.Kinematics` object with a
        mock velocity field with input parameters using similar code to
        :func:`nirvana.fiting.bisym_model`.

        Args:
            size (:obj:`int`):
                length of each side of the output arrays.
            inc (:obj:`float`):
                Inclination in degrees.
            pa (:obj:`float`):
                Position angle in degrees.
            pab (:obj:`float`):
                Relative position angle of bisymmetric features.
            vsys (:obj:`float`):
                Systemic velocity.
            vt (`numpy.ndarray`_):
                First order tangential velocities. Must have same length as
                :attr:`v2t` and :attr:`v2r`.
            v2t (`numpy.ndarray`_):
                Second order tangential velocities. Must have same length as
                :attr:`vt` and :attr:`v2r`.
            v2r (`numpy.ndarray`_):
                Second order radial velocities. Must have same length as
                :attr:`vt` and :attr:`v2t`.
            sig (`numpy.ndarray`_):
                Velocity dispersion values for each radial bin. Must have same
                length as other velocity arrays. 
            xc (:obj:`float`, optional):
                Offset of center on x axis. Optional, defaults to 0.
            yc (:obj:`float`, optional):
                Offset of center on y axis. Optional, defaults to 0.
            reff (:obj:`float`, optional):
                Effective radius of the mock galaxy. Units are arbitrary
                but must be the same as :attr:`r`. Defaults to 10.
            maxr (:obj:`float`, optional):
                Maximum absolute value for the x and y arrays. Defaults to 15.
            psf (`numpy.ndarray`_, optional):
                2D array of the point-spread function of the simulated galaxy.
                Must have dimensions of `size` by `size`. If not given, it will
                load a default PSF taken from a MaNGA observation that is 55 by
                55.
            border (:obj:`float`, optional):
                How many FWHM widths of a border to make around the central
                part of the galaxy you actually care about. This is to mitigate
                the edge effects of the PSF convolution that create erroneous
                values. Bigger borders will lead to smaller edge effects but
                will cost computational time in model fitting. Defaults to 3. 
            fwhm (:obj:`float`, optional):
                FWHM of PSF in same units as :attr:`size`. Defaults to 2.44 for
                example MaNGA PSF.

        Returns:
            :class:`nirvana.data.kinematics.Kinematics`: Object with the
            velocity field and x and y coordinates of the mock galaxy.

        Raises:
            ValueError:
                Raises if input velocity arrays are not the same length.
                
        """

        #check that velocities are compatible
        if len(vt) != len(v2t) or len(vt) != len(v2r) or len(vt) != len(sig):
            raise ValueError('Velocity arrays must be the same length.')

        #if the border needs to be masked, increase the size of the array to
        #make up for it so it ends up the right size in the end 
        if border: 
            _r = maxr + border * fwhm
            _size = int(_r/maxr * size)+1
            _bsize = (_size - size)//2
        else: _r,_size = (maxr,size)

        #make grid of x and y and define edges
        a = np.linspace(-_r,_r,_size)
        x,y = np.meshgrid(a,a)
        edges = np.linspace(0, maxr, len(vt)+1)

        #convert angles to polar
        _inc,_pa,_pab = np.radians([inc, pa, pab])
        r, th = projected_polar(x - xc, y - yc, _pa, _inc)

        #interpolate velocity values for all r 
        bincents = (edges[:-1] + edges[1:])/2
        vtvals  = np.interp(r, bincents, vt)
        v2tvals = np.interp(r, bincents, v2t)
        v2rvals = np.interp(r, bincents, v2r)
        sig = np.interp(r, bincents, sig)
        sb = oned.Sersic1D([1,10,1]).sample(r) #sersic profile for flux

        #spekkens and sellwood 2nd order vf model (from andrew's thesis)
        vel = vsys + np.sin(_inc) * (vtvals * np.cos(th) - 
              v2tvals * np.cos(2*(th - _pab)) * np.cos(th) -
              v2rvals * np.sin(2*(th - _pab)) * np.sin(th))

        #load example MaNGA PSF if none is provided
        #TODO: construct a general PSF instead
        if psf is None: psf = np.load('psfexample56.npy')

        #make border around PSF if necessary
        if border:
            #make the mask for the border
            bordermask = np.ones((_size, _size))
            bordermask[_bsize:-_bsize, _bsize:-_bsize] = 0

            #define masked versions of all the arrays
            _vel = np.ma.array(vel, mask=bordermask)
            _x   = np.ma.array(x,   mask=bordermask)
            _y   = np.ma.array(y,   mask=bordermask)
            _sig = np.ma.array(sig, mask=bordermask)
            _sb  = np.ma.array(sb,  mask=bordermask)

            #make bigger masked psf
            _psf = np.zeros((_size, _size))
            _psf[_bsize:-_bsize, _bsize:-_bsize] = psf
           
        else: _vel, _x, _y, _sig, _sb = [vel, x, y, sig, sb]

        binid = np.arange(np.product(_vel.shape)).reshape(_vel.shape)
        return cls(_vel, x=_x, y=_y, grid_x=_x, grid_y=_y, reff=reff, binid=binid, sig=_sig, psf=_psf, sb=_sb, bordermask=bordermask)

    def clip(self, sigma=10, sb=.03, anr=5, maxiter=10, smear_dv=50, smear_dsig=50, verbose=False):
        '''
        Filter out bad spaxels in kinematic data.
        
        Looks for features smaller than PSF by reconvolving PSF and looking for
        outlier points. Iteratively fits axisymmetric velocity field models and
        sigma clips residuals and chisq to get rid of outliers. Also clips
        based on surface brightness flux and ANR ratios. Applies new mask to
        galaxy.

        Args: 
            sigma (:obj:`float`, optional): 
                Significance threshold to be passed to
                `astropy.stats.sigma_clip` for sigma clipping the residuals
                and chi squared. Can't be too low or it will cut out
                nonaxisymmetric features. 
            sb (:obj:`float`, optional): 
                Flux threshold below which spaxels are masked.
            sb (:obj:`float`, optional): 
                Surface brightness amplitude/noise ratio threshold below which
                spaxels are masked.
            maxiter (:obj:`int`, optional):
                Maximum number of iterations to allow clipping process to go
                through.
            smear_dv (:obj:`float`, optional):
                Threshold for clipping residuals of resmeared velocity data
            smear_dsig (:obj:`float`, optional):
                Threshold for clipping residuals of resmeared velocity
                dispersion data.
            verbose (:obj:`bool`, optional):
                Flag for printing out information on iterations.
        '''

        #reconvolve psf on top of velocity and dispersion
        cnvfftw = ConvolveFFTW(self.spatial_shape)
        vel = self.remap('vel')
        smeared = smear(vel, self.beam_fft, beam_fft=True, 
                sig=self.remap('sig'), sb=self.remap('sb'), cnvfftw=cnvfftw)

        #cut out spaxels with too high residual because they're probably bad
        dvmask = self.bin(np.abs(vel - smeared[1]) > smear_dv)
        masks = [dvmask]
        labels = ['dv']
        if self.sig is not None: 
            dsigmask = self.bin(np.abs(self.remap('sig') - smeared[2]) > smear_dsig)
            masks += [dsigmask]
            labels += ['dsig']

        #clip on surface brightness and ANR
        if self.sb is not None: 
            sbmask = self.sb < sb
            masks += [sbmask]
            labels += ['sb']

        if self.sb_anr is not None:
            anrmask = self.sb_anr < anr
            masks += [anrmask]
            labels += ['anr']

        #combine all masks and apply to data
        mask = np.zeros(dvmask.shape)
        for m in masks: mask += m
        mask = mask.astype(bool)
        self.remask(mask)

        #iterate through rest of clips until mask converges
        nmaskedold = -1
        nmasked = np.sum(mask)
        niter = 0
        while nmaskedold != nmasked:
            #quick axisymmetric least squares fit
            fit = axisym.AxisymmetricDisk()
            fit.lsq_fit(self)

            #quick axisymmetric fit
            model = self.bin(fit.model())
            resid = self.vel - model

            #clean up the data by sigma clipping residuals and chisq
            chisq = resid**2 * self.vel_ivar if self.vel_ivar is not None else resid**2
            residmask = sigma_clip(resid, sigma=sigma, masked=True).mask
            chisqmask = sigma_clip(chisq, sigma=sigma, masked=True).mask
            clipmask = (mask + residmask + chisqmask).astype(bool)

            #iterate
            nmaskedold = nmasked
            nmasked = np.sum(clipmask)
            niter += 1
            if verbose: print(f'Performed {niter} clipping iterations...', end='\r')

            #apply mask to data
            self.remask(clipmask)

            if len(mask) == mask.sum(): verbose = True
            if niter > maxiter: 
                if verbose: print(f'Reached maximum clipping iterations: {niter}')
                break

        #make a plot of all of the masks if desired
        if verbose: 
            masks += [residmask, chisqmask]
            labels += ['resid', 'chisq']
            print(f'Clipping converged after {niter} iterations')
            plt.figure(figsize = (12,8))
            for i in range(len(masks)):
                plt.subplot(231+i)
                plt.axis('off')
                plt.imshow(self.remap(masks[i]), origin='lower')
                plt.title(labels[i])
            plt.tight_layout()
            plt.show()
            if len(mask) == mask.sum(): 
                raise ValueError(f'All data clipped after {niter} iterations. No good data')

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
