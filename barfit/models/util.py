
"""
Utility function for modeling.
"""

import numpy as np
from scipy import linalg

def cov_err(jac):
    """
    Pulled from ppxf.capfit.cov_err, but only returns the covariance matrix.

    Covariance and 1sigma formal errors calculation (i.e. ignoring covariance).
    See e.g. Press et al. 2007, Numerical Recipes, 3rd ed., Section 15.4.2
    """
    U, s, Vh = linalg.svd(jac, full_matrices=False)
    w = s > np.spacing(s[0])*max(jac.shape)
    return (Vh[w].T/s[w]**2) @ Vh[w]


def lin_interp(x, x1, y1, x2, y2):
    """
    Linearly interpolate a new value at position x given two points
    that define the line.

    .. warning::

        Will raise runtime warnings if ``np.any(x1 == x2)`` due to a
        division by 0, but this not checked.

    """
    return y1 + (y2-y1) * (x-x1) / (x2-x1)
