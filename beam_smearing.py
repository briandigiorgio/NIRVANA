
import numpy
from scipy import signal

def apply_beam_smearing(v, psf, sb=None, sig=None, aperture=None):
    """
    Get the beam-smeared surface brightness, velocity, and velocity
    dispersion fields.
    
    Args:
        v (numpy.ndarray):
            2D array with the discretely sampled velocity field. Must
            be square.
        psf (numpy.ndarray):
            An image of the point-spread function. If ``aperture`` is
            not provided, this should be the effective smoothing
            kernel for the kinematic fields. Otherwise, this is the
            on-sky seeing kernel and the effective smoothing kernel
            is constructed as the convolution of this image with
            ``aperture``.  Must be the same shape as ``v``.
        sb (numpy.ndarray, optional):
            2D array with the surface brightness of the object. This
            is used to weight the convolution of the kinematic fields
            according to the luminosity distribution of the object.
            Must have the same shape as ``v``. If None, the
            convolution is unweighted.
        sig (numpy.ndarray, optional):
            2D array with the velocity dispersion measurements. Must
            have the same shape as ``v``.
        aperture (numpy.ndarray, optional):
            Monochromatic image of the spectrograph aperture. See
            ``psf`` for how this is used.

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
    beam = psf if aperture is None else signal.fftconvolve(psf, aperture, mode='same')
    beam /= numpy.sum(beam)

    # Get the first moment of the beam-smeared intensity distribution
    mom0 = None if sb is None else signal.fftconvolve(sb, beam, mode='same')

    # First moment
    mom1 = signal.fftconvolve(v if sb is None else sb*v, beam, mode='same')
    if mom0 is not None:
        mom1 = numpy.ma.divide(mom1, mom0).filled(0.0)

    if sig is None:
        # Sigma not provided so we're done
        return mom0, mom1, None

    # Second moment
    _sig = numpy.square(v) + numpy.square(sig)
    mom2 = signal.fftconvolve(_sig if sb is None else sb*_sig, beam, mode='same')

    # Finish
    return mom0, mom1, numpy.ma.sqrt(numpy.ma.divide(mom2,mom0) - numpy.square(mom1)).filled(0.0)


