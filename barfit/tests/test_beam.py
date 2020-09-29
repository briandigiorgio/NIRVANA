
from IPython import embed

import numpy

from astropy import convolution

from barfit.models import beam
from barfit.models import oned
from barfit.models.geometry import projected_polar

def test_convolve():
    """
    Test that the results of the convolution match astropy.
    """
    synth = beam.gauss2d_kernel(73, 3.)
    astsynth = convolution.convolve_fft(synth, synth, fft_pad=False, psf_pad=False,
                                        boundary='wrap')
    intsynth = beam.convolve_fft(synth, synth)
    assert numpy.all(numpy.isclose(astsynth, intsynth)), 'Difference wrt astropy convolution'


def test_beam():
    """
    Test that the convolution doesn't shift the center (at least when
    the kernel is constructed with gauss2d_kernel).

    Note this test fails if you use scipy.fftconvolve because the
    kernels are treated differently.
    """
    n = 50
    synth = beam.gauss2d_kernel(n, 3.)
    _synth = beam.convolve_fft(synth, synth)
    assert numpy.argmax(synth) == numpy.argmax(_synth), \
            'Beam kernel shifted the center for an even image size.'

    n = 51
    synth = beam.gauss2d_kernel(n, 3.)
    _synth = beam.convolve_fft(synth, synth)
    assert numpy.argmax(synth) == numpy.argmax(_synth), \
            'Beam kernel shifted the center for an odd image size.'


def test_fft():
    synth = beam.gauss2d_kernel(73, 3.)
    synth_fft = numpy.fft.fftn(numpy.fft.ifftshift(synth))
    _convolve_fft = beam.ConvolveFFTW(synth.shape)

    # Compare numpy with direct vs. FFT kernel input
    synth2 = beam.convolve_fft(synth, synth)
    _synth2 = beam.convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference if FFT is passed for numpy'

    # Compare numpy and FFTW with direct input
    _synth2 = _convolve_fft(synth, synth)
    assert numpy.allclose(synth2, _synth2), 'Difference between numpy and FFTW'

    # Compare FFTW with direct vs. FFT kernel input
    synth2 = _convolve_fft(synth, synth_fft, kernel_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference if FFT is passed for FFTW'

    # Compare numpy and FFTW with direct input and FFT output
    synth2 = beam.convolve_fft(synth, synth, return_fft=True)
    _synth2 = _convolve_fft(synth, synth, return_fft=True)
    assert numpy.allclose(synth2, _synth2), 'Difference between numpy and FFTW'


def test_smear():

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    r, theta = projected_polar(x, y, *numpy.radians([45., 30.]))

    rc = oned.HyperbolicTangent(par=numpy.array([100., 1.]))
    sig = oned.Exponential(par=numpy.array([100., 20.]))
    sb = oned.Sersic1D(par=numpy.array([1., 10., 1.]))

    sb_field = sb.sample(r)
    vel_field = rc.sample(r)*numpy.cos(theta)
    sig_field = sig.sample(r)

    cnvlv = beam.ConvolveFFTW(x.shape)
    synth = beam.gauss2d_kernel(n, 3.)
    synth_fft = cnvlv.fft(synth, shift=True)

    vel_smear = beam.smear(vel_field, synth)[1]
    _vel_smear = beam.smear(vel_field, synth, cnvfftw=cnvlv)[1]
    assert numpy.allclose(vel_smear, _vel_smear), 'Velocity-field-only convolution difference.'

    vel_smear = beam.smear(vel_field, synth)[1]
    _vel_smear = beam.smear(vel_field, synth_fft, beam_fft=True)[1]
    assert numpy.allclose(vel_smear, _vel_smear), \
            'Velocity-field difference w/ vs. w/o precomputing the beam FFT for numpy.'

    vel_smear = beam.smear(vel_field, synth, cnvfftw=cnvlv)[1]
    _vel_smear = beam.smear(vel_field, synth_fft, beam_fft=True, cnvfftw=cnvlv)[1]
    assert numpy.allclose(vel_smear, _vel_smear), \
            'Velocity-field difference w/ vs. w/o precomputing the beam FFT for ConvolveFFTW.'

    sb_smear, vel_smear, _ = beam.smear(vel_field, synth, sb=sb_field)
    _sb_smear, _vel_smear, _ = beam.smear(vel_field, synth, sb=sb_field, cnvfftw=cnvlv)
    assert numpy.allclose(sb_smear, _sb_smear), 'SB+Vel convolution difference in SB.'
    assert numpy.allclose(vel_smear, _vel_smear), 'SB+Vel convolution difference in Vel.'

    sb_smear, vel_smear, sig_smear = beam.smear(vel_field, synth, sb=sb_field, sig=sig_field)
    _sb_smear, _vel_smear, _sig_smear = beam.smear(vel_field, synth, sb=sb_field, sig=sig_field,
                                                   cnvfftw=cnvlv)
    assert numpy.allclose(sb_smear, _sb_smear), 'SB+Vel+Sig convolution difference in SB.'
    assert numpy.allclose(vel_smear, _vel_smear), 'SB+Vel+Sig convolution difference in vel.'
    assert numpy.allclose(sig_smear, _sig_smear), 'SB+Vel+Sig convolution difference in sig.'

