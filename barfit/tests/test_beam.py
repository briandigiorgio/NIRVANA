
from IPython import embed

import numpy

from astropy import convolution

from barfit.models.beam import convolve_fft, gauss2d_kernel, ConvolveFFTW

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


def test_fft():
    synth = gauss2d_kernel(73, 3.)
    synth_fft = numpy.fft.fftn(numpy.fft.ifftshift(synth))
    _convolve_fft = ConvolveFFTW(synth.shape)

    # Compare numpy with direct vs. FFT kernel input
    np_synth2 = convolve_fft(synth, synth)
    _np_synth2 = convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(np_synth2, _np_synth2), 'Difference if FFT is passed'

    # Compare numpy and FFTW with direct input
    _np_synth2 = _convolve_fft(synth, synth)
    assert numpy.allclose(np_synth2, _np_synth2), 'Difference between numpy and FFTW'

    # Compare FFTW with direct vs. FFT kernel input
    np_synth2 = _convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(np_synth2, _np_synth2), 'Difference between numpy and FFTW'

    # Compare numpy and FFTW with direct input and FFT output
    np_synth2 = convolve_fft(synth, synth, return_fft=True)
    _np_synth2 = _convolve_fft(synth, synth, return_fft=True)
    assert numpy.allclose(np_synth2, _np_synth2), 'Difference between numpy and FFTW'


