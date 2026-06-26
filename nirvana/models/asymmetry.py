"""
Functions for parameterizing asymmetry.

..inclde:: ../include/links.rst
"""

import numpy as np
from scipy.spatial import KDTree

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
        :obj:`tuple`: Two `numpy.ndarray`_ objects of the new reflected x and y
        coordinates
    '''

    th = np.radians(90 - pa) #turn position angle into a regular angle

    #reflection matrix across arbitrary angle
    ux = np.cos(th) 
    uy = np.sin(th)
    return np.dot([[ux**2 - uy**2, 2*ux*uy], [2*ux*uy, uy**2 - ux**2]], [x, y])
