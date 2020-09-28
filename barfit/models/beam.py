"""

.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np


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
        raise ValueError('Data and kernel must both have valid values.')

    datafft = np.fft.fftn(data)
    kernfft = kernel if kernel_fft else np.fft.fftn(np.fft.ifftshift(kernel))
    fftmult = datafft * kernfft

    return fftmult if return_fft else np.fft.ifftn(fftmult).real


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

