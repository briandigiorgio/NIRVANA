"""
Module for testing the geometry module.
"""

from IPython import embed

import numpy

from nirvana.models import geometry

def test_polar():

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    # Parameters are xc, yc, rotation, pa, inclination
    par = [10., 10., 10., 45., 30.]
    if par[2] > 0:
        xf, yf = map(lambda x,y : x - y,
                     geometry.rotate(x, y, numpy.radians(par[2]), clockwise=True), par[:2])
    else:
        xf, yf = x - par[0], y - par[1]

    r, th = geometry.projected_polar(xf, yf, *numpy.radians(par[3:]))

    indx = numpy.unravel_index(numpy.argmin(r), r.shape)
    assert indx[0] > n//2 and indx[1] < n//2, 'Offset in wrong direction.'

    assert numpy.amax(r) > n/2, 'Inclination should result in larger radius'
    assert not numpy.any(th < 0) and not numpy.any(th > 2*numpy.pi), 'Theta has wrong range'


def test_deriv_rotate():

    x = numpy.array([1., 0., -1., 0.])
    y = numpy.array([0., 1., 0., -1.])

    rot = numpy.pi/3.
    drot = 1e-3
    xrp, yrp = geometry.rotate(x, y, rot+drot)
    xrn, yrn = geometry.rotate(x, y, rot)
    xr, yr, dxr, dyr = geometry.deriv_rotate(x, y, rot)
    assert numpy.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'
    assert numpy.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'

    # Check clockwise rotation
    rot = numpy.pi/3.
    drot = 1e-3
    xrp, yrp = geometry.rotate(x, y, rot+drot, clockwise=True)
    xrn, yrn = geometry.rotate(x, y, rot, clockwise=True)
    xr, yr, dxr, dyr = geometry.deriv_rotate(x, y, rot, clockwise=True)
    assert numpy.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'
    assert numpy.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'

    # Check defining rotation in degrees
    rot = 30.
    drot = 0.1
    dxdp = numpy.zeros(x.shape+(1,), dtype=float)
    dydp = numpy.zeros(y.shape+(1,), dtype=float)
    drotdp = numpy.atleast_1d(numpy.radians(1))
    xr, yr, dxr, dyr \
            = geometry.deriv_rotate(x, y, numpy.radians(rot), dxdp=dxdp, dydp=dydp, drotdp=drotdp)
    xrp, yrp = geometry.rotate(x, y, numpy.radians(rot+drot))
    xrn, yrn = geometry.rotate(x, y, numpy.radians(rot))
    assert numpy.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-2, atol=0.), \
            'X derivative does not match finite difference'
    assert numpy.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-2, atol=0.), \
            'Y derivative does not match finite difference'


def test_deriv_projected_polar():

    # NOTE: A point at x = 1. and y = 0. will cause this to fail because the
    # brute force calculation of the derivative of theta results in a wrap
    # degeneracy.
#    x = numpy.array([2., 0., -2., 0., 1., 0., -1., 0.])
#    y = numpy.array([0., 2., 0., -2., 0., 1., 0., -1.])

    x = numpy.array([2., 0., -2., 0., 0., -1., 0.])
    y = numpy.array([0., 2., 0., -2., 1., 0., -1.])

    # Parameter vector is x0, y0, position angle, inclination.  x0 and y0 have
    # the same units as x and y; pa and inclination are in degrees.
    p = numpy.array([0.5, -0.5, 45., 30.])
    dp = numpy.array([0.0001, 0.0001, 0.001, 0.001])

    _x = x - p[0]
    _y = y - p[1]
    _pa = numpy.radians(p[2])
    _inc = numpy.radians(p[3])

    _dx = numpy.tile(numpy.array([-1., 0., 0., 0.]), (x.size, 1))
    _dy = numpy.tile(numpy.array([0., -1., 0., 0.]), (x.size, 1))
    _dpa = numpy.array([0., 0., numpy.radians(1.), 0.])
    _dinc = numpy.array([0., 0., 0., numpy.radians(1.)])

    r, t, dr, dt = geometry.deriv_projected_polar(_x, _y, _pa, _inc, dxdp=_dx, dydp=_dy,
                                                  dpadp=_dpa, dincdp=_dinc)

    # Finite difference approach
    rp = numpy.zeros((x.size, p.size), dtype=float)
    tp = numpy.zeros((x.size, p.size), dtype=float)
    for i in range(p.size):
        _p = p.copy()
        _p[i] += dp[i]
        rp[:,i], tp[:,i] = geometry.projected_polar(x - _p[0], y - _p[1], numpy.radians(_p[2]),
                                                    numpy.radians(_p[3]))

    assert numpy.allclose(dr, (rp-r[:,None])/dp[None,:], rtol=0., atol=1e-4), \
        'Radius derivative does not match finite difference'
    assert numpy.allclose(dt, (tp-t[:,None])/dp[None,:], rtol=0., atol=1e-3), \
        'Azimuth derivative does not match finite difference'


