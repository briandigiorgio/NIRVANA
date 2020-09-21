"""

.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np


def gauss2d_kernel(n, sigma):
    """
    Return a circular 2D Gaussian.

    Shape of the returned map is nxn, and the sigma is provided in pixels.
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
        - Kernel is assumed to sum to unity.
        - Padding is never added by default.

    """
    if data.shape != kernel.shape:
        raise ValueError('Data and kernel must have the same shape.')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(kernel)):
        raise ValueError('Data and kernel must both have valid values.')

    datafft = np.fft.fftn(data)
    kernfft = kernel if kernel_fft else np.fft.fftn(np.fft.ifftshift(kernel))
    fftmult = datafft * kernfft

    return fftmult if return_fft else np.fft.ifftn(fftmult).real


def construct_beam(psf, aperture, return_fft=False):
    """
    Construct the beam profile.

    psf and aperture must have the same shape.
    psf and aperture are both expected to sum to 1.
    """
    return convolve_fft(psf, aperture, return_fft=return_fft)


def smear(v, beam, beam_fft=False, sb=None, sig=None):
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

    # Pre-compute the beam FFT
    bfft = beam if beam_fft else np.fft.fftn(np.fft.ifftshift(beam))

    # Get the first moment of the beam-smeared intensity distribution
    mom0 = None if sb is None else convolve_fft(sb, bfft, kernel_fft=True)

    # First moment
    mom1 = convolve_fft(v if sb is None else sb*v, bfft, kernel_fft=True)
    if mom0 is not None:
        mom1 /= (mom0 + (mom0 == 0.0))

    if sig is None:
        # Sigma not provided so we're done
        return mom0, mom1, None

    # Second moment
    _sig = np.square(v) + np.square(sig)
    mom2 = convolve_fft(_sig if sb is None else sb*_sig, bfft, kernel_fft=True)
    mom2 = mom2 / (mom0 + (mom0 == 0.0)) - mom1**2
    mom2[mom2 < 0] = 0.0
    return mom0, mom1, np.sqrt(mom2)


def apply_beam_smearing_old(v, psf, sb=None, sig=None, aperture=None, mask=None, fftw=False):
    """
    Get the beam-smeared surface brightness, velocity, and velocity
    dispersion fields.
    
    Args:
        v (np.ndarray):
            2D array with the discretely sampled velocity field. Must
            be square.
        psf (np.ndarray):
            An image of the point-spread function. If ``aperture`` is
            not provided, this should be the effective smoothing
            kernel for the kinematic fields. Otherwise, this is the
            on-sky seeing kernel and the effective smoothing kernel
            is constructed as the convolution of this image with
            ``aperture``.  Must be the same shape as ``v``.
        sb (np.ndarray, optional):
            2D array with the surface brightness of the object. This
            is used to weight the convolution of the kinematic fields
            according to the luminosity distribution of the object.
            Must have the same shape as ``v``. If None, the
            convolution is unweighted.
        sig (np.ndarray, optional):
            2D array with the velocity dispersion measurements. Must
            have the same shape as ``v``.
        aperture (np.ndarray, optional):
            Monochromatic image of the spectrograph aperture. See
            ``psf`` for how this is used.
        mask (np.ndarray, optional):
            2D Boolean array to mask the results with.
        fftw (bool, optional):
            Will use ``pyfftw`` FFT library if set to true which
            should save some time in the convolution but doesn't in this
            implementation and causes the fit to hang. Will use
            ``scipy.signal.fftconvolve`` if set to false (default).

    Returns:
        :obj:`tuple`: Tuple of three objects, which are nominally the
        beam-smeared surface brightness, velocity, and velocity
        dispersion fields. The first and last objects in the tuple
        can be None, if ``sb`` or ``sig`` are not provided,
        respectively. The 2nd returned object is always the
        beam-smeared velocity field.

    Raises:
        ValueError:
            Raised if the provided arrays are not 2D, if they are not
            square, or if the shapes of the arrays are not all the
            same.
    """
    nimg = v.shape[0]
    if len(v.shape) != 2:
        raise ValueError('Can only accept 2D images.')
    if v.shape[1] != nimg:
        raise ValueError('Input array must be square.')
    if psf.shape != v.shape:
        raise ValueError('Input point-spread function and velocity field array sizes must match.')
    if sb is not None and sb.shape != v.shape:
        raise ValueError('Input surface-brightness and velocity field array sizes must match.')
    if sig is not None and sig.shape != v.shape:
        raise ValueError('Input velocity dispersion and velocity field array sizes must match.')
    if aperture is not None and aperture.shape != v.shape:
        raise ValueError('Input spectrograph aperture and velocity field array sizes must match.')

    # Get the image of the beam and ensure it's normalized to unity
    beam = psf if aperture is None else scipy.signal.fftconvolve(psf, aperture, mode='same')
    beam /= np.sum(beam)

    if fftw:
        import pyfftw
        scipy.fftpack = pyfftw.interfaces.scipy_fftpack
        v = pyfftw.byte_align(v)
        beam = pyfftw.byte_align(beam)
        if sb is not None: sb = pyfftw.byte_align(sb)

    # Get the first moment of the beam-smeared intensity distribution
    mom0 = None if sb is None else scipy.signal.fftconvolve(sb, beam, mode='same')
    if mask is not None and mom0 is not None: mom0 = np.ma.array(mom0, mask=mask)

    # First moment
    if fftw: mom1 = scipy.signal.fftconvolve(v if sb is None else pyfftw.byte_align(sb*v), beam, mode='same')
    else: mom1 = scipy.signal.fftconvolve(v if sb is None else sb*v, beam, mode='same')
    if mom0 is not None:
        mom1 = np.ma.divide(mom1, mom0).filled(0.0)
        if mask is not None: mom1 = np.ma.array(mom1, mask=mask)

    if sig is None:
        # Sigma not provided so we're done
        return mom0, mom1, None

    # Second moment
    if fftw: 
        _sig = pyfftw.byte_align(np.square(v) + np.square(sig))
        mom2 = scipy.signal.fftconvolve(_sig if sb is None else pyfftw.byte_align(sb*_sig), beam, mode='same')
    else:
        _sig = np.square(v) + np.square(sig)
        mom2 = scipy.signal.fftconvolve(_sig if sb is None else sb*_sig, beam, mode='same')
    if mom2 is not None and mom0 is not None:
        mom2 = np.ma.sqrt(np.ma.divide(mom2,mom0) - np.square(mom1)).filled(0.0)
        if mask is not None: mom2 = np.ma.array(mom2, mask=mask)

    # Finish
    return mom0, mom1, mom2


