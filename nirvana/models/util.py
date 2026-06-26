"""
Utility function for modeling.

.. include:: ../include/links.rst
"""

import numpy as np
from scipy import linalg, stats

def cov_err(jac):
    """
    Provided the Jacobian matrix from a least-squares minimization
    routine, construct the parameter covariance matrix. See e.g.
    Press et al. 2007, Numerical Recipes, 3rd ed., Section 15.4.2

    This is directly pulled from ppxf.capfit.cov_err, but only
    returns the covariance matrix:

    https://pypi.org/project/ppxf/

    Args:
        jac (`numpy.ndarray`_):
            Jacobian matrix

    Returns:
        `numpy.ndarray`_: Parameter covariance matrix.
    """
    U, s, Vh = linalg.svd(jac, full_matrices=False)
    w = s > np.spacing(s[0])*max(jac.shape)
    return (Vh[w].T/s[w]**2) @ Vh[w]


def lin_interp(x, x1, y1, x2, y2):
    """
    Linearly interpolate a new value at position x given two points
    that define the line.

    Nominally, the abscissa values for the two reference points
    should be to either side of the new points.

    .. warning::

        Will raise runtime warnings if ``np.any(x1 == x2)`` due to a
        division by 0, but this not checked.

    Args:
        x (:obj:`float`, `numpy.ndarray`_):
            Coordinate(s) at which to sample the new value.
        x1 (:obj:`float`, `numpy.ndarray`_):
            Abscissa value of the first reference point.
        y1 (:obj:`float`, `numpy.ndarray`_):
            Ordinate value of the first reference point.
        x2 (:obj:`float`, `numpy.ndarray`_):
            Abscissa value of the second reference point.
        y2 (:obj:`float`, `numpy.ndarray`_):
            Ordinate value of the second reference point.

    Returns:
        :obj:`float`, `numpy.ndarray`_: Interpolated y value(s) at
        the provided x value(s).
    """
    return y1 + (y2-y1) * (x-x1) / (x2-x1)


def deriv_lin_interp(x, x1, y1, x2, y2):
    """
    Linearly interpolate a new value at position x given two points that define
    the line.

    This function also calculates the derivatives of the result with respect to
    ``y1`` and ``y2``.  I.e., assuming ``y1`` and ``y2`` are parameters of a
    model (and ``x1`` and ``x2`` are *not*), this returns the derivative of the
    computation w.r.t. the model parameters.

    Nominally, the abscissa values for the two reference points should be to
    either side of the new points.

    .. warning::

        Will raise runtime warnings if ``np.any(x1 == x2)`` due to a division by
        0, but this not checked.

    Args:
        x (:obj:`float`, `numpy.ndarray`_):
            Coordinate(s) at which to sample the new value.
        x1 (:obj:`float`, `numpy.ndarray`_):
            Abscissa value of the first reference point.
        y1 (:obj:`float`, `numpy.ndarray`_):
            Ordinate value of the first reference point.
        x2 (:obj:`float`, `numpy.ndarray`_):
            Abscissa value of the second reference point.
        y2 (:obj:`float`, `numpy.ndarray`_):
            Ordinate value of the second reference point.

    Returns:
        :obj:`float`, `numpy.ndarray`_: Interpolated y value(s) at
        the provided x value(s).
    """
    dx = (x-x1) / (x2-x1)
    return y1 + (y2-y1) * dx, np.stack((1 - dx, dx), axis=-1)


def sech2(x):
    r"""
    Calculate the squared hyperbolic secant function using `numpy.cosh`_, while
    controlling for overflow errors.

    Overflow is assumed to occur whenever :math:`|x| \geq 100`.

    Args:
        x (array-like):
            Values at which to calculate :math:`\sech(x)`.

    Returns:
        `numpy.ndarray`_: Result of :math:`\sech(x)` where any values of
        :math:`x` that would control overflow errors are set to 0.
    """
    _x = np.atleast_1d(x)
    indx = np.absolute(_x) < 100
    if np.all(indx):
        return 1/np.cosh(_x)**2
    s = np.zeros_like(_x)
    s[indx] = 1/np.cosh(_x[indx])**2
    return s


def trunc(q, mean, std, left, right):
    """
    Wrapper function for the ``ppf`` method of the `scipy.stats.truncnorm`_
    function. This makes defining edges easier.
    
    Args:
        q (:obj:`float`):
            Desired quantile.
        mean (:obj:`float`):
            Mean of distribution
        std (:obj:`float`):
            Standard deviation of distribution.
        left (:obj:`float`):
            Left bound of truncation.
        right (:obj:`float`):
            Right bound of truncation.

    Returns:
        :obj:`float`: Value of the distribution at the desired quantile
    """
    a,b = (left-mean)/std, (right-mean)/std #transform to z values
    return stats.truncnorm.ppf(q,a,b,mean,std)


