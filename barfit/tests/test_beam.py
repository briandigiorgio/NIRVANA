
from IPython import embed

import numpy

from astropy import convolution

from barfit.models.beam import convolve_fft, gauss2d_kernel

def test_convolve():
    """
    Test that the results of the convolution match astropy.
    """
    synth = gauss2d_kernel(73, 3.)
    astsynth = convolution.convolve_fft(synth, synth, fft_pad=False, psf_pad=False,
                                        boundary='wrap')
    intsynth = convolve_fft(synth, synth)
    assert numpy.all(numpy.isclose(astsynth, intsynth)), 'Difference wrt astropy convolution'


def test_beam():
    """
    Test that the convolution doesn't shift the center (at least when
    the kernel is constructed with gauss2d_kernel).

    Note this test fails if you use scipy.fftconvolve because the
    kernels are treated differently.
    """
    n = 50
    beam = gauss2d_kernel(n, 3.)
    _beam = convolve_fft(beam, beam)
    assert numpy.argmax(beam) == numpy.argmax(_beam), \
            'Beam kernel shifted the center for an even image size.'

    n = 51
    beam = gauss2d_kernel(n, 3.)
    _beam = convolve_fft(beam, beam)
    assert numpy.argmax(beam) == numpy.argmax(_beam), \
            'Beam kernel shifted the center for an odd image size.'



