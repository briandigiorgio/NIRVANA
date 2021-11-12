"""

.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np

try:
    import pyfftw
except:
    pyfftw = None

def gauss2d_kernel(n, sigma):
    """
    Return a circular 2D Gaussian.

    The center of the Gaussian is centered at ``n//2``.

    Args:
        n (:obj:`int`):
            The length of the axis of the square (n x n) output
            array.
        sigma (:obj:`float`):
            The circular (symmetric) standard deviation of the 2D
            Gaussian

    Returns:
        `numpy.ndarray`_: Gridded representation of the 2D Gaussian.
    """
    x, y = np.meshgrid(*(np.arange(n, dtype=float) - n//2,)*2)
    d = 2*sigma**2
    g = np.exp(-(x**2 + y**2) / d) / d / np.pi
    return g / np.sum(g)


def convolve_fft(data, kernel, kernel_fft=False, return_fft=False):
    """
    Convolve data with a kernel.

    This is inspired by astropy.convolution.convolve_fft, but
    stripped down to what's needed for the expected application. That
    has the benefit of cutting down on the execution time, but limits
    its use.

    Beware:
        - ``data`` and ``kernel`` must have the same shape.
        - For the sum of all pixels in the convolved image to be the
          same as the input data, the kernel must sum to unity.
        - Padding is never added by default.

    Args:
        data (`numpy.ndarray`_):
            Data to convolve.
        kernel (`numpy.ndarray`_):
            The convolution kernel, which must have the same shape as
            ``data``. If ``kernel_fft`` is True, this is the FFT of
            the kernel image; otherwise, this is the direct kernel
            image with the center of the kernel at the center of the
            array.
        kernel_fft (:obj:`bool`, optional):
            Flag that the provided ``kernel`` array is actually the
            FFT of the kernel, not its direct image.
        return_fft (:obj:`bool`, optional):
            Flag to return the FFT of the convolved image, instead of
            the direct image.

    Returns:
        `numpy.ndarray`_: The convolved image, or its FFT, with the
        same shape as the provided ``data`` array.

    Raises:
        ValueError:
            Raised if ``data`` and ``kernel`` do not have the same
            shape or if any of their values are not finite.
    """
    if data.shape != kernel.shape:
        raise ValueError('Data and kernel must have the same shape.')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(kernel)):
        print('**********************************')
        print(f'nans in data: {(~np.isfinite(data)).sum()}, nans in kernel: {(~np.isfinite(kernel)).sum()}')
        raise ValueError('Data and kernel must both have valid values.')

    datafft = np.fft.fftn(data)
    kernfft = kernel if kernel_fft else np.fft.fftn(np.fft.ifftshift(kernel))
    fftmult = datafft * kernfft

    return fftmult if return_fft else np.fft.ifftn(fftmult).real


class ConvolveFFTW:
    """
    Class to perform convolutions using the FFTW library.

    Speed gains with FFTW depend on allocating a set block in memory that is
    used repeatedly for the FFT computations.  If you're planning on just
    convolving one image once, this method is likely slower than the standard
    approach using `numpy`_ routines (see :func:`convolve_fft`).
    
    
    Instantiation of this object requires the shape of the arrays that will be
    convolved.  When using this class to perform the convolution, both the
    convolved image and the convolution kernel must have this same shape.  Also,
    the instantiation of the memory workspace assumes that both the image to be
    convolved and the convolution kernel are made up of 64-bit floats.  The
    methods check the data type and raise an exception if this is not true.

    Args:
        shape (:obj:`tuple`):
            Shape of the arrays to be convolved. Any arrays passed to
            member functions of this instance must have this shape.
        flags (:obj:`tuple`):
            Flags passed to pyFFTW when setting up the FFTW
            instances, describing how the FFTW performance
            optimization is determined. The default is FFTW_MEASURE.
    """
    def __init__(self, shape, flags=None):
        if pyfftw is None:
            raise ImportError('pyfftw package must be available to use ConvolveFFTW.  Ensure '
                              'that both the FFTW library and the PyFFTW interface are installed.')
        # Dimensionality
        self.shape = shape
        self.ndim = len(self.shape)
        # Array workspace
        self.data = pyfftw.empty_aligned(shape, dtype='complex128')
        self.data.imag[...] = 0.
        self.kern = pyfftw.empty_aligned(shape, dtype='complex128')
        self.kern.imag[...] = 0.

        self.data_fft = pyfftw.empty_aligned(shape, dtype='complex128')
        self.kern_fft = pyfftw.empty_aligned(shape, dtype='complex128')

        self.dcnv = pyfftw.empty_aligned(shape, dtype='complex128')
        self.dcnv.imag[...] = 0.

        # FFTW algorithms
        self.flags = ('FFTW_MEASURE',) if flags is None else flags
        if not isinstance(self.flags, tuple):
            raise TypeError('Provided flags must be strings in a tuple instance.')
        self.dfft = pyfftw.FFTW(self.data, self.data_fft,
                                axes=tuple(np.arange(self.ndim).tolist()),
                                direction='FFTW_FORWARD', flags=self.flags)
        self.kfft = pyfftw.FFTW(self.kern, self.kern_fft,
                                axes=tuple(np.arange(self.ndim).tolist()),
                                direction='FFTW_FORWARD', flags=self.flags)
        self.ifft = pyfftw.FFTW(self.data_fft, self.dcnv,
                                axes=tuple(np.arange(self.ndim).tolist()),
                                direction='FFTW_BACKWARD', flags=self.flags)

    def __call__(self, data, kernel, kernel_fft=False, return_fft=False):
        """
        Convolve data with a kernel using FFTW.

        This method is identical to :func:`convolve_fft`, but uses
        the pre-established memory working space setup during the
        instantiation of the object.

        Beware:
            - ``data`` and ``kernel`` must have the same shape.
            - For the sum of all pixels in the convolved image to be the
              same as the input data, the kernel must sum to unity.
            - Padding is never added by default.

        Args:
            data (`numpy.ndarray`_):
                Data to convolve.  Data type must be `numpy.float64` and shape
                must match :attr:`shape`.
            kernel (`numpy.ndarray`_):
                The convolution kernel, which must have the same shape as
                ``data``. If ``kernel_fft`` is True, this is the FFT of the
                kernel image and must have type `numpy.complex128`_; otherwise,
                this is the direct kernel image with the center of the kernel at
                the center of the array and must have type `numpy.float64`_.
            kernel_fft (:obj:`bool`, optional):
                Flag that the provided ``kernel`` array is actually the
                FFT of the kernel, not its direct image.
            return_fft (:obj:`bool`, optional):
                Flag to return the FFT of the convolved image, instead of
                the direct image.

        Returns:
            `numpy.ndarray`_: The convolved image, or its FFT, with the
            same shape as the provided ``data`` array.

        Raises:
            ValueError:
                Raised if ``data`` and ``kernel`` do not have the expected shape
                or if any of their values are not finite.
            TypeError:
                Raised if the data types of either ``data`` or ``kernel`` do not
                match the expected values (numpy.float64 for direct data,
                numpy.complex128 for Fourier Transform data).
        """
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(kernel)):
            raise ValueError('Data and kernel must both have valid values.')
        self.fft(data)

        if kernel.shape != self.shape:
            raise ValueError('Kernel has incorrect shape for this instance of ConvolveFFTW.')
        if kernel_fft:
            if kernel.dtype.type is not np.complex128:
                raise TypeError('Kernel FFT must be of type numpy.complex128.')
            self.kern_fft[...] = kernel
        else:
            if kernel.dtype.type is not np.float64:
                raise TypeError('Kernel must be of type numpy.float64.')
            self.kern.real[...] = np.fft.ifftshift(kernel)
            self.kfft()
        if return_fft:
            return self.data_fft * self.kern_fft
        self.data_fft *= self.kern_fft
        self.ifft()
        return self.dcnv.real.copy()

    def __reduce__(self):
        '''
        Internal method for pickling.

        Returns:
            :obj:`tuple`: Tuple of the class type and the arguments needed for
            instantiating the class.
        '''
            
        return (self.__class__, (self.shape, ))

    def fft(self, data, copy=True, shift=False):
        """
        Calculate the FFT of the provided data array.

        Args:
            data (`numpy.ndarray`_):
                Data for FFT computation.  Data type must be `numpy.float64` and
                shape must match :attr:`shape`.
            copy (:obj:`bool`, optional):
                The result of the FFT is computed using the
                :attr:`data_fft` workspace. If False, the
                :attr:`data_fft` *is* the returned array; if True,
                returned array is a copy.
            shift (:obj:`bool`, optional):
                Before computing, use ``numpy.fft.iffshift`` to shift
                the spatial coordinates of the image such that the 0
                frequency component of the FFT is shifted to the
                center of the image.

        Returns:
            `numpy.ndarray`_: The FFT of the provided data.

        Raises:
            ValueError:
                Raised if the shape of the data array does not match
                :attr:`shape`.
            TypeError:
                Raised if the type of the array is not np.float64.
        """
        if data.shape != self.shape:
            raise ValueError('Data has incorrect shape for this instance of ConvolveFFTW.')
        if data.dtype.type is not np.float64:
            raise TypeError('Data must be of type numpy.float64.')

        self.data.real[...] = np.fft.ifftshift(data) if shift else data
        self.dfft()
        return self.data_fft.copy() if copy else self.data_fft
        

def construct_beam(psf, aperture, return_fft=False):
    """
    Construct the beam profile.

    This is a simple wrapper for :func:`convolve_fft`. Nominally,
    both arrays should sum to unity.

    Args:
        psf (`numpy.ndarray`_):
            An image of the point-spread function of the
            observations. Must have the same shape as ``aperture``.
        aperture (`numpy.ndarray`_):
            Monochromatic image of the spectrograph aperture. Must
            have the same shape as ``psf``.
        return_fft (:obj:`bool`, optional):
            Flag to return the FFT of the beam profile, instead of
            its the direct image.

    Returns:
        `numpy.ndarray`_: The 2D image of the beam profile, or its
        FFT, with the same shape as the provided ``psf`` and
        ``aperture`` arrays.
    """
    return convolve_fft(psf, aperture, return_fft=return_fft)


# TODO: Include higher moments?
def smear(v, beam, beam_fft=False, sb=None, sig=None, cnvfftw=None, verbose=False):
    """
    Get the beam-smeared surface brightness, velocity, and velocity
    dispersion fields.
    
    Args:
        v (`numpy.ndarray`_):
            2D array with the discretely sampled velocity field. Must
            be square.
        beam (`numpy.ndarray`_):
            An image of the beam profile or its precomputed FFT. Must
            be the same shape as ``v``. If the beam profile is
            provided, it is expected to be normalized to unity.
        beam_fft (:obj:`bool`, optional):
            Flag that the provided data for ``beam`` is actually the
            precomputed FFT of the beam profile.
        sb (`numpy.ndarray`_, optional):
            2D array with the surface brightness of the object. This
            is used to weight the convolution of the kinematic fields
            according to the luminosity distribution of the object.
            Must have the same shape as ``v``. If None, the
            convolution is unweighted.
        sig (`numpy.ndarray`_, optional):
            2D array with the velocity dispersion measurements. Must
            have the same shape as ``v``.
        cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
            An object that expedites the convolutions using
            FFTW/pyFFTW. If None, the convolution is done using numpy
            FFT routines.

    Returns:
        :obj:`tuple`: Tuple of three objects, which are nominally the
        beam-smeared surface brightness, velocity, and velocity
        dispersion fields. The first and last objects in the tuple
        can be None, if ``sb`` or ``sig`` are not provided,
        respectively. The 2nd returned object is always the
        beam-smeared velocity field.

    Raises:
        ValueError:
            Raised if the provided arrays are not 2D or if the shapes
            of the arrays are not all the same.
    """
    if v.ndim != 2:
        raise ValueError('Can only accept 2D images.')
    if beam.shape != v.shape:
        raise ValueError('Input beam and velocity field array sizes must match.')
    if sb is not None and sb.shape != v.shape:
        raise ValueError('Input surface-brightness and velocity field array sizes must match.')
    if sig is not None and sig.shape != v.shape:
        raise ValueError('Input velocity dispersion and velocity field array sizes must match.')

    _cnv = convolve_fft if cnvfftw is None else cnvfftw

    # Pre-compute the beam FFT
    bfft = beam if beam_fft else (np.fft.fftn(np.fft.ifftshift(beam))
                                    if cnvfftw is None else cnvfftw.fft(beam, shift=True))

    # Get the first moment of the beam-smeared intensity distribution
    if verbose: print('Convolving surface brightness...')
    mom0 = _cnv(np.ones(v.shape, dtype=float) if sb is None else sb, bfft, kernel_fft=True)
#    mom0 = None if sb is None else _cnv(sb, bfft, kernel_fft=True)

    # First moment
    if verbose: print('Convolving velocity field...',sb,v)
    mom1 = _cnv(v if sb is None else sb*v, bfft, kernel_fft=True)
    if mom0 is not None:
        mom1 /= (mom0 + (mom0 == 0.0))

    if sig is None:
        # Sigma not provided so we're done
        return mom0, mom1, None

    # Second moment
    _sig = np.square(v) + np.square(sig)
    if verbose: print('Convolving velocity dispersion...')
    mom2 = _cnv(_sig if sb is None else sb*_sig, bfft, kernel_fft=True)
    if mom0 is not None:
        mom2 /= (mom0 + (mom0 == 0.0))
    mom2 -= mom1**2
    mom2[mom2 < 0] = 0.0
    return mom0, mom1, np.sqrt(mom2)


def deriv_smear(v, dv, beam, beam_fft=False, sb=None, dsb=None, sig=None, dsig=None, cnvfftw=None):
    """
    Get the beam-smeared surface brightness, velocity, and velocity
    dispersion fields and their derivatives.

    Args:
        v (`numpy.ndarray`_):
            2D array with the discretely sampled velocity field. Must be square.
        dv (`numpy.ndarray`_):
            2D arrays with velocity field derivatives with respect to a set of
            model parameters.  The shape of the first two axes must match ``v``;
            the third axis is the number of parameters.  The `numpy.ndarray`_
            *must* have three dimensions, even if the derivative is w.r.t. a
            single parameter.
        beam (`numpy.ndarray`_):
            An image of the beam profile or its precomputed FFT. Must be the
            same shape as ``v``. If the beam profile is provided, it is expected
            to be normalized to unity.
        beam_fft (:obj:`bool`, optional):
            Flag that the provided data for ``beam`` is actually the precomputed
            FFT of the beam profile.
        sb (`numpy.ndarray`_, optional):
            2D array with the surface brightness of the object. This is used to
            weight the convolution of the kinematic fields according to the
            luminosity distribution of the object.  Must have the same shape as
            ``v``. If None, the convolution is unweighted.
        dsb (`numpy.ndarray`_, optional):
            2D arrays with the derivative of the surface brightness of the
            object with respect to a set of parameters.  Must have the same
            shape as ``dv``. If None, the surface brightness derivatives are
            assumed to be 0.
        sig (`numpy.ndarray`_, optional):
            2D array with the velocity dispersion measurements. Must have the
            same shape as ``v``.
        dsig (`numpy.ndarray`_, optional):
            2D arrays with the derivative of the velocity dispersion
            measurements with respect to a set of model parameters. Must have
            the same shape as ``dv``.
        cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
            An object that expedites the convolutions using FFTW/pyFFTW. If
            None, the convolution is done using numpy FFT routines.

    Returns:
        :obj:`tuple`: Tuple of six `numpy.ndarray`_ objects, which are nominally
        the beam-smeared surface brightness, velocity, and velocity dispersion
        fields, and their derivatives, respectively.

    Raises:
        ValueError:
            Raised if the provided arrays are not 2D or if the shapes
            of the arrays are not all the same.
    """
    if v.ndim != 2:
        raise ValueError('Can only accept 2D images.')
    if dv.ndim != 3:
        raise ValueError('Velocity-field derivative array must be 3D.')
    if v.shape != dv.shape[:2]:
        raise ValueError('Shape of first two axes of dv must match shape of v.')
    if beam.shape != v.shape:
        raise ValueError('Input beam and velocity field array sizes must match.')
    if sb is not None and sb.shape != v.shape:
        raise ValueError('Input surface-brightness and velocity field array sizes must match.')
    if sb is None and dsb is not None:
        raise ValueError('Must provide surface-brightness if providing its derivative.')
    if dsb is not None and dsb.shape != dv.shape:
        raise ValueError('Surface-brightness derivative array shape must match dv.')
    if sig is not None and sig.shape != v.shape:
        raise ValueError('Input velocity dispersion and velocity field array sizes must match.')
    if sig is None and dsig is not None:
        raise ValueError('Must provide velocity dispersion if providing its derivative.')
    if dsig is not None and dsig.shape != dv.shape:
        raise ValueError('Velocity dispersion derivative array shape must match dv.')

    _cnv = convolve_fft if cnvfftw is None else cnvfftw

    # Pre-compute the beam FFT
    bfft = beam if beam_fft else (np.fft.fftn(np.fft.ifftshift(beam))
                                    if cnvfftw is None else cnvfftw.fft(beam, shift=True))

    # Number of parameters is the length of the last axis of 'dv'
    npar = dv.shape[-1]

    # Get the zeroth moment of the beam-smeared intensity distribution
    _sb = np.ones(v.shape, dtype=float) if sb is None else sb
    mom0 = _cnv(_sb, bfft, kernel_fft=True)

    # Get the zeroth moment derivatives, if possible
    dmom0 = None 
    if dsb is not None:
        dmom0 = dsb.copy()
        for i in range(npar):
            dmom0[...,i] = _cnv(dsb[...,i], bfft, kernel_fft=True)

    inv_mom0 = 1./(mom0 + (mom0 == 0.0))

    # First moment
    mom1 = _cnv(_sb*v, bfft, kernel_fft=True) * inv_mom0
    dmom1 = dv.copy()
    for i in range(npar):
        dmom1[...,i] = _cnv(_sb*dv[...,i], bfft, kernel_fft=True) * inv_mom0
        if dsb is not None:
            dmom1[...,i] += _cnv(v*dsb[...,i], bfft, kernel_fft=True) * inv_mom0
            dmom1[...,i] -= mom1 * inv_mom0 * dmom0[...,i]

    if sig is None:
        # Sigma not provided so we're done
        return mom0, mom1, None, dmom0, dmom1, None

    # Second moment
    _sig = np.square(v) + np.square(sig)
    mom2 = _cnv(_sb*_sig, bfft, kernel_fft=True) * inv_mom0 - mom1**2
    mom2[mom2 < 0] = 0.0
    _mom2 = np.sqrt(mom2)
    _inv_mom2 = 1./(_mom2 + (_mom2 == 0.0))
    dmom2 = dv.copy()
    for i in range(npar):
        # dv terms
        dmom2[...,i] = _cnv(2*_sb*v*dv[...,i], bfft, kernel_fft=True) * inv_mom0 \
                            - 2 * mom1 * dmom1[...,i]
        # dsb terms
        if dsb is not None:
            dmom2[...,i] += _cnv(_sig*dsb[...,i], bfft, kernel_fft=True) * inv_mom0
            dmom2[...,i] -= mom2 * inv_mom0 * dmom0[...,i]
        # dsig terms
        if dsig is not None:
            dmom2[...,i] += _cnv(2*_sb*sig*dsig[...,i], bfft, kernel_fft=True) * inv_mom0
        # sqrt operation
        dmom2[...,i] *= _inv_mom2 / 2

    return mom0, mom1, _mom2, dmom0, dmom1, dmom2


