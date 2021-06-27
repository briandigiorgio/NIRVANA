"""
Methods for geometric projection.

.. include:: ../include/links.rst
"""

import numpy as np
from scipy.spatial import KDTree

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
        over the range :math:`[0,2\pi)`.
    """
    xd, yd = rotate(x, y, np.pi/2-pa, clockwise=True)
    yd /= np.cos(inc)
    return np.sqrt(xd**2 + yd**2), np.arctan2(-yd,xd) % (2*np.pi)


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


# TODO: The rest of these functions should probably be in their own module

def asymmetry(args, pa, vsys, xc=0, yc=0):
    '''
    Calculate asymmetry parameter and maps for major/minor axis reflection.
    '''

    #construct KDTree of spaxels for matching
    x = args.x - xc
    y = args.y - yc
    tree = KDTree(list(zip(x,y)))
    
    #compute major and minor axis asymmetry 
    arc2d = []
    for axis in [0,90]:
        #match spaxels to their reflections, mask out ones without matches
        d,i = tree.query(reflect(pa - axis, x, y).T)
        mask = np.ma.array(np.ones(len(args.vel)), mask = (d>.5) | args.vel_mask)

        #compute Andersen & Bershady (2013) A_RC parameter 2D maps
        vel = args.remap(args.vel * mask) - vsys
        ivar = args.remap(args.vel_ivar * mask)
        velr = args.remap(args.vel[i] * mask - vsys)
        ivarr = args.remap(args.vel_ivar[i] * mask)
        arc2d += [A_RC(vel, velr, ivar, ivarr)]
    
    #mean of maps to get global asym
    arc = np.mean([np.sum(a) for a in arc2d])
    asymmap = np.ma.array(arc2d).mean(axis=0)
    return arc, asymmap

def A_RC(vel, velr, ivar, ivarr):
    '''
    Compute velocity field asymmetry for a velocity field and its reflection.

    From Andersen & Bershady (2013) equation 7 but doesn't sum over whole galaxy so asymmmetry is spatially resolved. 
    '''
    return (np.abs(np.abs(vel) - np.abs(velr))/np.sqrt(1/ivar + 1/ivarr) 
         / (.5*np.sum(np.abs(vel) + np.abs(velr))/np.sqrt(1/ivar + 1/ivarr)))

def reflect(pa, x, y):
    '''
    Reflect arrays of x and y coordinates across a line at angle position angle pa.
    '''

    th = np.radians(90 - pa) #turn position angle into a regular angle

    #reflection matrix across arbitrary angle
    ux = np.cos(th) 
    uy = np.sin(th)
    return np.dot([[ux**2 - uy**2, 2*ux*uy], [2*ux*uy, uy**2 - ux**2]], [x, y])

def fig2data(fig):
    # draw the renderer
    fig.canvas.draw( )
 
    # Get the RGBA buffer from the figure
    h,w = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


