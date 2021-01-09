"""
Methods for geometric projection.

.. include:: ../include/links.rst
"""

import numpy as np

def rotate(x, y, rot, clockwise=False):
    r"""
    Rotate a set of coordinates about :math:`(x,y) = (0,0)`.

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.
        rot (:obj:`float`):
            Rotation angle in radians.
        clockwise (:obj:`bool`, optional):
            Perform a clockwise rotation. Rotation is
            counter-clockwise by default.

    Returns:
        :obj:`tuple`: Two `numpy.ndarray`_ objects with the rotated x
        and y coordinates.
    """
    cosr = np.cos(rot)
    sinr = np.sin(rot)
    _x = np.asarray(x)
    _y = np.asarray(y)
    if clockwise:
        return _x*cosr + _y*sinr, _y*cosr - _x*sinr
    return _x*cosr - _y*sinr, _y*cosr + _x*sinr


def projected_polar(x, y, pa, inc):
    r"""
    Calculate the in-plane polar coordinates of an inclined plane.

    The position angle, :math:`\phi_0`, is the rotation from the
    :math:`y=0` axis through the :math:`x=0` axis. I.e.,
    :math:`\phi_0 = \pi/2` is along the :math:`+x` axis and
    :math:`\phi_0 = \pi` is along the :math:`-y` axis.

    The inclination, :math:`i`, is the angle of the plane normal with
    respect to the line-of-sight. I.e., :math:`i=0` is a face-on
    (top-down) view of the plane and :math:`i=\pi/2` is an edge-on
    view.

    The returned coordinates are the projected distance from the
    :math:`(x,y) = (0,0)` and the project azimuth. The projected
    azimuth, :math:`\theta`, is defined to increase in the same
    direction as :math:`\phi_0`, with :math:`\theta = 0` at
    :math:`\phi_0`.

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.
        pa (:obj:`float`)
            Position angle, as defined above, in radians.
        inc (:obj:`float`)
            Inclination, as defined above, in radians.

    Returns:
        :obj:`tuple`: Returns two arrays with the projected radius
        and in-plane azimuth. The radius units are identical to the
        provided cartesian coordinates. The azimuth is in radians
        over the range :math:`[0,2\pi]`.
    """
    xd, yd = rotate(x, y, np.pi/2-pa, clockwise=True)
    yd /= np.cos(inc)
    return np.sqrt(xd**2 + yd**2), np.arctan2(-yd,xd) % (2*np.pi)


