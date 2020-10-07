
"""
Utility function for modeling.
"""

import numpy as np
from scipy import linalg

def cov_err(jac):
    """
    Provided the Jacobian matrix from a least-squares minimization
    routine, construct the parameter covariance matrix. See e.g.
    Press et al. 2007, Numerical Recipes, 3rd ed., Section 15.4.2

    This is directly pulled from ppxf.capfit.cov_err, but only
    returns the covariance matrix.

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


