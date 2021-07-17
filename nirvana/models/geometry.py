"""
Methods for geometric projection.

.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np
from scipy.spatial import KDTree

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

def rotate(x, y, rot, clockwise=False):
    r"""
    Rotate a set of coordinates about :math:`(x,y) = (0,0)`.

    .. warning::

        The ``rot`` argument should be a float.  If it is an array, the code
        will either fault if ``rot`` cannot be broadcast to match ``x`` and
        ``y`` or the rotation will be different for each ``x`` and ``y``
        element.

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.  Shape must match ``x``, but this is not
            checked.
        rot (:obj:`float`):
            Rotation angle in radians.
        clockwise (:obj:`bool`, optional):
            Perform a clockwise rotation.  Rotation is counter-clockwise by
            default.  By definition and implementation, setting this to True is
            identical to calling the function with a negative counter-clockwise
            rotation.  I.e.::

                xr, yr = rotate(x, y, rot, clockwise=True)
                _xr, _yr = rotate(x, y, -rot)
                assert numpy.array_equal(xr, _xr) and numpy.array_equal(yr, _yr)

    Returns:
        :obj:`tuple`: Two `numpy.ndarray`_ objects with the rotated x
        and y coordinates.
    """
    if clockwise:
        return rotate(x, y, -rot)
    cosr = np.cos(rot)
    sinr = np.sin(rot)
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)
    return _x*cosr - _y*sinr, _y*cosr + _x*sinr


def deriv_rotate(x, y, rot, dxdp=None, dydp=None, drotdp=None, clockwise=False):
    r"""
    Rotate the provided coordinates about :math:`(x,y) = (0,0)`, and calculate
    the derivatives of the returned coordinates with respect to a set of input
    parameters.
    
    The set of input parameters can be unknown, i.e., defined in the calling
    method, or the single parameter of this function (``rot``).  That is, if the
    arguments of this function depend on upstream set of parameters, their
    derivatives with respect to these parameters should be passed to the
    relevant keyword arguments.  Importantly, if you supply one of either
    ``dxdp``, ``dydp``, or ``drotdp``, you must supply them all.

    See additional documentation of the :func:`rotate` method.

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.  Shape must match ``x``, but this is not
            checked.
        rot (:obj:`float`):
            Rotation angle in radians.
        dxdp (array-like, optional):
            Derivative of the Cartesian x coordinates w.r.t. a set of unknown
            parameters.  Shape has one more dimension than ``x``, where the size
            of that dimension, :math:`m`, is is the number of parameters.  If
            None, the provided x coordinates are assumed to be independent of
            any model parameters.
        dydp (array-like, optional):
            Derivative of the Cartesian y coordinates w.r.t. a set of unknown
            parameters.  Shape has one more dimension than ``x``, where the size
            of that dimension, :math:`m`, is is the number of parameters.  If
            None, the provided y coordinates are assumed to be independent of
            any model parameters.
        drotdp (array-like, optional):
            Derivative of the rotation angle w.r.t. a set of unknown parameters.
            Shape is :math:`(m,)`, where :math:`m` is the number of parameters.
            If None, the rotation is considered to be the only model parameter.
        clockwise (:obj:`bool`, optional):
            Perform a clockwise rotation. Rotation is counter-clockwise by
            default.  By definition and implementation, setting this to True is
            identical to calling the function with a negative counter-clockwise
            rotation.  

    Returns:
        :obj:`tuple`: Four `numpy.ndarray`_ objects: the rotated x coordinates,
        the rotated y coordinates, the derivate of the rotated x coordinates
        w.r.t. a set of parameters, and the derivative of the rotated y
        coordinates w.r.t. a set of parameters.
    """
    # Check derivative input
    isNone = [i is None for i in [dxdp, dydp, drotdp]]
    if np.any(isNone) and not np.all(isNone):
        raise ValueError('Must provide all of dxdp, dydp, and drotdp, or none of them.')

    # Convert to numpy arrays
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)

    # If dxdp is None, they all should be.
    if dxdp is None:
        # Set the input derivatives to be with respect to rot
        dxdp = np.zeros(_x.shape+(1,), dtype=float)
        dydp = np.zeros(_y.shape+(1,), dtype=float)
        drotdp = np.ones((1,), dtype=float)

    if clockwise:
        return deriv_rotate(x, y, -rot, dxdp=dxdp, dydp=dydp, drotdp=-drotdp)

    _dxdp = np.atleast_1d(dxdp)
    _dydp = np.atleast_1d(dydp)
    _drotdp = np.atleast_1d(drotdp)

    cosr = np.cos(rot)
    sinr = np.sin(rot)
    xr = _x*cosr - _y*sinr
    yr = _y*cosr + _x*sinr
    dxrdp = _dxdp*cosr - _x[...,None]*sinr*_drotdp[None,:] - _dydp*sinr \
                - _y[...,None]*cosr*_drotdp[None,:]
    dyrdp = _dydp*cosr - _y[...,None]*sinr*_drotdp[None,:] + _dxdp*sinr \
                + _x[...,None]*cosr*_drotdp[None,:]
    return xr, yr, dxrdp, dyrdp


def projected_polar(x, y, pa, inc):
    r"""
    Calculate the in-plane polar coordinates of an inclined plane.

    The position angle, :math:`\phi_0`, is the rotation from the :math:`y=0`
    axis through the :math:`x=0` axis. I.e., :math:`\phi_0 = \pi/2` is along the
    :math:`+x` axis and :math:`\phi_0 = \pi` is along the :math:`-y` axis.

    The inclination, :math:`i`, is the angle of the plane normal with respect to
    the line-of-sight. I.e., :math:`i=0` is a face-on (top-down) view of the
    plane and :math:`i=\pi/2` is an edge-on view.

    The returned coordinates are the projected distance from the :math:`(x,y) =
    (0,0)` and the project azimuth. The projected azimuth, :math:`\theta`, is
    defined to increase in the same direction as :math:`\phi_0`, with
    :math:`\theta = 0` at :math:`\phi_0`.

    .. warning::

        Calculation of the disk-plane y coordinate is undefined at :math:`i =
        \pi/2`.  Only use this function with :math:`i < \pi/2`!

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.  Shape must match ``x``, but this is not
            checked.
        pa (:obj:`float`)
            Position angle, as defined above, in radians.
        inc (:obj:`float`)
            Inclination, as defined above, in radians.

    Returns:
        :obj:`tuple`: Returns two arrays with the projected radius
        and in-plane azimuth. The radius units are identical to the
        provided cartesian coordinates. The azimuth is in radians
        over the range :math:`[0,2\pi)`.
    """
    xd, yd = rotate(x, y, np.pi/2-pa, clockwise=True)
    yd /= np.cos(inc)
    return np.sqrt(xd**2 + yd**2), np.arctan2(-yd,xd) % (2*np.pi)


def deriv_projected_polar(x, y, pa, inc, dxdp=None, dydp=None, dpadp=None, dincdp=None):
    r"""
    Calculate the in-plane polar coordinates of an inclined plane and their
    derivatives with respect to a set of input parameters.

    The set of input parameters can be unknown, i.e. defined in the calling
    method, or the two parameters of this function (``pa``, ``inc``).  That is,
    if the arguments of this function depend on upstream set of parameters,
    their derivatives with respect to these parameters should be passed to the
    relevant keyword arguments.  Importantly, if you supply one of either
    ``dxdp``, ``dydp``, ``dpadp``, ``dincdp``, you must supply them all.

    See additional documentation of the :func:`projected_polar` method.  The
    same warning there about the calculation when :math:`i = \pi/2` holds for
    the derivatives, as well.

    Args:
        x (array-like):
            Cartesian x coordinates.
        y (array-like):
            Cartesian y coordinates.  Shape must match ``x``, but this is not
            checked.
        pa (:obj:`float`)
            Position angle in radians; see :func:`projected_polar`.
        inc (:obj:`float`)
            Inclination in radians; see :func:`projected_polar`.
        dxdp (array-like, optional):
            Derivative of the Cartesian x coordinates w.r.t. a set of unknown
            parameters.  Shape has one more dimension than ``x``, where the size
            of that dimension, :math:`m`, is is the number of parameters.  If
            None, the provided x coordinates are assumed to be independent of
            any model parameters.
        dydp (array-like, optional):
            Derivative of the Cartesian y coordinates w.r.t. a set of unknown
            parameters.  Shape has one more dimension than ``x``, where the size
            of that dimension, :math:`m`, is is the number of parameters.  If
            None, the provided y coordinates are assumed to be independent of
            any model parameters.
        dpadp (array-like, optional):
            Derivative of the position angle w.r.t. a set of unknown parameters.
            Shape is :math:`(m,)`, where :math:`m` is the number of parameters.
            If None, the position angle is considered to be the one of the model
            parameters.
        dincdp (array-like, optional):
            Derivative of the inclination w.r.t. a set of unknown parameters.
            Shape is :math:`(m,)`, where :math:`m` is the number of parameters.
            If None, the inclination is considered to be the one of the model
            parameters.

    Returns:
        :obj:`tuple`: Returns four arrays with the projected radius and in-plane
        azimuth and their derivatives (order is radius, aziumth, radius
        derivative, azimuth derivative). The radius units are identical to the
        provided cartesian coordinates. The azimuth is in radians over the range
        :math:`[0,2\pi)`.
    """
    # Check derivative input
    isNone = [i is None for i in [dxdp, dydp, dpadp, dincdp]]
    if np.any(isNone) and not np.all(isNone):
        raise ValueError('Must provide all of dxdp, dydp, dpadp, and dincdp, or none of them.')    

    # Convert to numpy arrays
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)

    # If dxdp is None, they all should be.
    if dxdp is None:
        # Set the input derivatives to be with respect to pa and inc
        dxdp = np.zeros(_x.shape+(2,), dtype=float)
        dydp = np.zeros(_y.shape+(2,), dtype=float)
        dpadp = np.array([1., 0.], dtype=float)
        dincdp = np.array([0., 1.], dtype=float)

    _dxdp = np.atleast_1d(dxdp)
    _dydp = np.atleast_1d(dydp)
    _dpadp = np.atleast_1d(dpadp)
    _dincdp = np.atleast_1d(dincdp)

    # Calculate the rotated coordinates (note the propagation of the derivative
    # given the calculation of the applied rotation based on the position angle)
    xd, yr, dxd, dyr = deriv_rotate(_x, _y, np.pi/2-pa, dxdp=_dxdp, dydp=_dydp, drotdp=-_dpadp,
                                    clockwise=True)

    # Project the y axis
    cosi = np.cos(inc)
    yd = yr / cosi
    dyd = dyr / cosi + yr[...,None] * _dincdp[None,:] * np.sin(inc) / cosi**2

    # Calculate the polar coordinates
    r = np.sqrt(xd**2 + yd**2)
    dr = drdx(xd, yd, r=r)[...,None]*dxd + drdy(xd, yd, r=r)[...,None]*dyd

    t = np.arctan2(-yd,xd) % (2*np.pi)
    dt = dthetadx(xd, -yd, r=r)[...,None]*dxd - dthetady(xd, -yd, r=r)[...,None]*dyd

    return r, t, dr, dt


def drdx(x, y, r=None):
    r"""
    Compute the derivative of :math:`r=\sqrt{x^2+y^2}` w.r.t. :math:`x`.
    
    When both :math:`x` and :math:`y` are non-zero, the result is:

    .. math::

        \frac{\partial r}{\partial x} = \frac{x}{r}

    In the limit where both :math:`x` and :math:`y` approach 0, this reduces to
    :math:`\pm 1`, where the sign degeneracy is due to different left/right
    limits.  This function choose the one-sided limit approaching from the right
    (:math:`+1`).

    Args:
        x (`numpy.ndarray`_):
            Cartesian x coordinates.
        y (`numpy.ndarray`_):
            Cartesian y coordinates.
        r (`numpy.ndarray`_, optional):
            Pre-calculated radius.  If None, the radius is calculated using
            ``x`` and ``y``.

    Returns:
        `numpy.ndarray`_: The derivative of :math:`r` w.r.t. :math:`x`.
    """
    _r = np.sqrt(x**2 + y**2) if r is None else r
    indx = _r == 0.
    if not np.any(indx):
        return x / _r

    # Need special treatment for r==0 elements 
    dr = np.zeros(x.shape, dtype=float)

    # When both x and r are 0, y is 0 by definition.  The limit of x/r as x,y->0
    # is +/-1.  For a finite-difference approach you would measure
    # (sqrt(dx**2)-r)/dx, where r=0.  However, the sign discontinuity is because
    # different limits are reached if you approach from -x or +x.  This chooses
    # the one-sided limit approaching from +x.
    indx &= (x == 0.)
    dr[indx] = 1.
    # The rest are computed normally
    indx = np.logical_not(indx)
    dr[indx] = x[indx] / _r[indx]
    return dr


def drdy(x, y, r=None):
    r"""
    Compute the derivative of :math:`r=\sqrt{x^2+y^2}` w.r.t. :math:`y`.
    
    When both :math:`x` and :math:`y` are non-zero, the result is:

    .. math::

        \frac{\partial r}{\partial y} = \frac{y}{r}

    In the limit where both :math:`x` and :math:`y` approach 0, this reduces to
    :math:`\pm 1`, where the sign degeneracy is due to different left/right
    limits.  This function choose the one-sided limit approaching from the right
    (:math:`+1`).

    Args:
        x (`numpy.ndarray`_):
            Cartesian x coordinates.
        y (`numpy.ndarray`_):
            Cartesian y coordinates.
        r (`numpy.ndarray`_, optional):
            Pre-calculated radius.  If None, the radius is calculated using
            ``x`` and ``y``.

    Returns:
        `numpy.ndarray`_: The derivative of :math:`r` w.r.t. :math:`y`.
    """
    return drdx(y, x, r=r)


def dthetady(x, y, r=None):
    r"""
    Compute the derivative of :math:`\theta=\tan^{-1}(y/x)` w.r.t. :math:`y`.
    
    When both :math:`x` and :math:`y` are non-zero, the result is:

    .. math::

        \frac{\partial \theta}{\partial y} = \frac{x}{r^2}

    In the limit where both :math:`x` and :math:`y` approach 0, this reduces to
    :math:`\pm \infinity`, where the sign degeneracy is due to different
    left/right limits.  This function choose the one-sided limit approaching
    from the right (:math:`+\infinity`) but replaces the value with a large
    float (``1e20``).

    Args:
        x (`numpy.ndarray`_):
            Cartesian x coordinates.
        y (`numpy.ndarray`_):
            Cartesian y coordinates.
        r (`numpy.ndarray`_, optional):
            Pre-calculated radius.  If None, the radius is calculated using
            ``x`` and ``y``.

    Returns:
        `numpy.ndarray`_: The derivative of :math:`\theta` w.r.t. :math:`y`.
    """
    r2 = x**2 + y**2 if r is None else r**2
    indx = r2 == 0.
    if not np.any(indx):
        return x / r2

    # Need special treatment for r==0 elements 
    dtheta = np.zeros(x.shape, dtype=float)

    # The limit of x/r^2 as r->0 is +/- infinity.  The sign discontinuity is
    # because different limits are reached if you approach from -x or +x.  If x
    # == 0., this chooses the one-sided limit approaching from +x; otherwise, it
    # uses the value of x to set the sign.
    _indx = indx & (x == 0.)
    dtheta[_indx] = 1e20
    _indx = indx & (x != 0.)
    dtheta[_indx] = np.sign(x[_indx]) * 1e20
    # The rest are computed normally
    _indx = np.logical_not(indx)
    dtheta[_indx] = x[_indx] / r2[_indx]
    return dtheta


def dthetadx(x, y, r=None):
    r"""
    Compute the derivative of :math:`\theta=\tan^{-1}(y/x)` w.r.t. :math:`x`.
    
    When both :math:`x` and :math:`y` are non-zero, the result is:

    .. math::

        \frac{\partial \theta}{\partial y} = \frac{-y}{r^2}

    In the limit where both :math:`x` and :math:`y` approach 0, this reduces to
    :math:`\pm \infinity`, where the sign degeneracy is due to different
    left/right limits.  This function choose the one-sided limit approaching
    from the right (:math:`+\infinity`) but replaces the value with a large
    float (``1e20``).

    Args:
        x (`numpy.ndarray`_):
            Cartesian x coordinates.
        y (`numpy.ndarray`_):
            Cartesian y coordinates.
        r (`numpy.ndarray`_, optional):
            Pre-calculated radius.  If None, the radius is calculated using
            ``x`` and ``y``.

    Returns:
        `numpy.ndarray`_: The derivative of :math:`\theta` w.r.t. :math:`x`.
    """
    return dthetady(-y, x, r=r)


def polygon_winding_number(polygon, point):
    """
    Determine the winding number of a 2D polygon about a point.
    
    The code does **not** check if the polygon is simple (no interesecting line
    segments).  Algorithm taken from Numerical Recipes Section 21.4.

    Args:
        polygon (`numpy.ndarray`_):
            An Nx2 array containing the x,y coordinates of a polygon.
            The points should be ordered either counter-clockwise or
            clockwise.
        point (`numpy.ndarray`_):
            One or more points for the winding number calculation.
            Must be either a 2-element array for a single (x,y) pair,
            or an Nx2 array with N (x,y) points.

    Returns:
        :obj:`int`, `numpy.ndarray`: The winding number of each point with
        respect to the provided polygon. Points inside the polygon have winding
        numbers of 1 or -1; see :func:`point_inside_polygon`.

    Raises:
        ValueError:
            Raised if ``polygon`` is not 2D, if ``polygon`` does not have two
            columns, or if the last axis of ``point`` does not have 2 and only 2
            elements.
    """
    # Check input shape is for 2D only
    if len(polygon.shape) != 2:
        raise ValueError('Polygon must be an Nx2 array.')
    if polygon.shape[1] != 2:
        raise ValueError('Polygon must be in two dimensions.')
    _point = np.atleast_2d(point)
    if _point.shape[1] != 2:
        raise ValueError('Point must contain two elements.')

    # Get the winding number
    nvert = polygon.shape[0]
    npnt = _point.shape[0]

    dl = np.roll(polygon, 1, axis=0)[None,:,:] - _point[:,None,:]
    dr = polygon[None,:,:] - point[:,None,:]
    dx = dl[...,0]*dr[...,1] - dl[...,1]*dr[...,0]

    indx_l = dl[...,1] > 0
    indx_r = dr[...,1] > 0

    wind = np.zeros((npnt, nvert), dtype=int)
    wind[indx_l & np.logical_not(indx_r) & (dx < 0)] = -1
    wind[np.logical_not(indx_l) & indx_r & (dx > 0)] = 1

    return np.sum(wind, axis=1)[0] if point.ndim == 1 else np.sum(wind, axis=1)


def point_inside_polygon(polygon, point):
    """
    Determine if one or more points is inside the provided polygon.

    Primarily a wrapper for :func:`polygon_winding_number`, that
    returns True for each point that is inside the polygon.

    Args:
        polygon (`numpy.ndarray`_):
            An Nx2 array containing the x,y coordinates of a polygon.
            The points should be ordered either counter-clockwise or
            clockwise.
        point (`numpy.ndarray`_):
            One or more points for the winding number calculation.
            Must be either a 2-element array for a single (x,y) pair,
            or an Nx2 array with N (x,y) points.

    Returns:
        :obj:`bool`, `numpy.ndarray`: Boolean indicating whether or not each
        point is within the polygon.
    """
    return np.absolute(polygon_winding_number(polygon, point)) == 1

def proj_angle(angle, inc, pa):
    '''
    Take an angle in the plane of a galaxy and project it on sky.

    Args:
        angle (:obj:`float`):
            In-plane position angle to be projected (in degrees)
        inc (:obj:`float`):
            Inclination angle relative to observer (in degrees)
        pa (:obj:`float`):
            Position angle of major axis (in degrees)

        Returns:
            :obj:`float`: the position angle supplied but projected on sky (in
            degrees)
    '''

    _angle, _inc, _pa = np.radians([angle, inc, pa])
    return np.degrees(_pa + np.arctan2(np.tan(_angle) * np.cos(_inc)))
