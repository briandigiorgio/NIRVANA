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

def asymmetry(args, pa, vsys, xc=0, yc=0, maxd=.5):
    '''
    Calculate global asymmetry parameter and map of asymmetry.

    Using Equation 7 from Andersen & Bershady (2007), the symmetry of a galaxy
    is calculated. This is done by reflecting it across the supplied position
    angle (assumed to be the major axis) and the angle perpendicular to that
    (assumed to be the minor axis). The value calculated is essentially a
    normalized error weighted difference between the velocity measurements on
    either side of a reflecting line. 

    The function tries to find the nearest neighbor to a spaxel's reflected
    coordinates. If the distance is too far, the spaxel is masked.

    The returned value is the mean of the sums of all of the normalized
    asymmetries over the whole velocity field for the major and minor axes and
    should scale between 0 and 1, wih 0 being totally symmetrical and 1 being
    totally asymmetrical. The map is the average of the asymmetry maps for the
    major and minor axes.

    Args:
        args (:class:`_nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and fit parameters for the galaxy
        pa (:obj:`float`):
            Position angle of galaxy (in degrees)
        vsys (:obj:`float`):
            Systemic velocity of galaxy in the same units as the velocity
        xc (:obj:`float`, optional):
            x position of the center of the velocity field. Must be in the same
            units as `args.x`.
        yc (:obj:`float`, optional):
            y position of the center of the velocity field. Must be in the same
            units as `args.y`.
        maxd (:obj:`float`, optional):
            Maximum distance to allow the nearest neighbor finding to look for
            a reflexted spaxel. Any neighbor pairs at a radius larger than this
            will be masked. Must be in the same units as the spatial
            coordinates `args.x` and `args.y`.

    Returns:
        :obj:`float`: Global asymmetry value for the entire galaxy. Scales
        between 0 and 1. 
        `numpy.ndarray`_: Array of spatially resolved
        asymmetry values for every possible point in the velocity field. Should
        be the same shape as the input velocity fields.

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
        mask = np.ma.array(np.ones(len(args.vel)), mask = (d>maxd) | args.vel_mask)

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

    From Andersen & Bershady (2013) equation 7 but doesn't sum over whole
    galaxy so asymmmetry is spatially resolved. 

    Using Equation 7 from Andersen & Bershady (2007), the symmetry of a galaxy
    is calculated. This is done by reflecting it across the supplied position
    angle (assumed to be the major axis) and the angle perpendicular to that
    (assumed to be the minor axis). The value calculated is essentially a
    normalized error weighted difference between the velocity measurements on
    either side of a reflecting line. 

    The returned array is  of all of the normalized asymmetries over the whole
    velocity field for the major and minor axes and should scale between 0 and
    1, wih 0 being totally symmetrical and 1 being totally asymmetrical. The
    map is the average of the asymmetry maps for the major and minor axes.

    Args:
        vel (`numpy.ndarray`_):
            Array of velocity measurements for a galaxy.
        velr (`numpy.ndarray`_):
            Array of velocity measurements for a galaxy reflected across the desired axis.
        ivar (`numpy.ndarray`_):
            Array of velocity inverse variances for a galaxy.
        ivarr (`numpy.ndarray`_):
            Array of velocity inverse variances for a galaxy reflected across
            the desired axis.

    Returns:
        `numpy.ndarray`_: Array of rotational asymmetries for all of the
        velocity measurements supplied. Should be the same shape as the input
        arrays.
    '''
    return (np.abs(np.abs(vel) - np.abs(velr))/np.sqrt(1/ivar + 1/ivarr) 
         / (.5*np.sum(np.abs(vel) + np.abs(velr))/np.sqrt(1/ivar + 1/ivarr)))

def reflect(pa, x, y):
    '''
    Reflect arrays coordinates across a given position angle.

    Args:
        pa (:obj:`float`):
            Position angle to reflect across (in degrees)
        x (`numpy.ndarray`_):
            Array of x coordinate positions. Must be the same shape as `y`.
        y (`numpy.ndarray`_):
            Array of y coordinate positions. Must be the same shape as `x`.

    Returns:
        :obj:`tuple`: Two `numpy.ndarray`_s of the new reflected x and y
        coordinates

    '''

    th = np.radians(90 - pa) #turn position angle into a regular angle

    #reflection matrix across arbitrary angle
    ux = np.cos(th) 
    uy = np.sin(th)
    return np.dot([[ux**2 - uy**2, 2*ux*uy], [2*ux*uy, uy**2 - ux**2]], [x, y])

def fig2data(fig):
    '''
    Take a `matplolib` figure and return it as an array of RGBA values.

    Stolen from somewhere on Stack Overflow.

    Args:
        fig (`matplotlib.figure.Figure`_):
            Figure to be turned into an array.

    Returns:
        `numpy.ndarray`_: RGBA array representation of the figure.
    '''

    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    h,w = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

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
