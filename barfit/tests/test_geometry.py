"""
Module for testing the geometry module.
"""

from IPython import embed

import numpy

from barfit.models.geometry import rotate, projected_polar

def test_polar():

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    # Parameters are xc, yc, rotation, pa, inclination
    par = [10., 10., 10., 45., 30.]
    if par[2] > 0:
        xf, yf = map(lambda x,y : x - y, rotate(x, y, numpy.radians(par[2]), clockwise=True), par[:2])
    else:
        xf, yf = x - par[0], y - par[1]

    r, th = projected_polar(xf, yf, *numpy.radians(par[3:]))

    indx = numpy.unravel_index(numpy.argmin(r), r.shape)
    assert indx[0] > n//2 and indx[1] < n//2, 'Offset in wrong direction.'

    assert numpy.amax(r) > n/2, 'Inclination should result in larger radius'
    assert not numpy.any(th < 0) and not numpy.any(th > 2*numpy.pi), 'Theta has wrong range'

