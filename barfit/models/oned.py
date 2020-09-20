"""
Implements one-dimensional functions for modeling.
"""
import warnings

from IPython import embed

import numpy as np
from .util import lin_interp


class StepFunction:
    def __init__(self, edges, par=None):
        self.edges = np.sort(edges)
        if not np.array_equal(self.edges, edges):
            warnings.warn('As provided, edges for PiecewiseLinear were not sorted.')
        self.np = self.edges.size
        self.par = np.ones(self.np, dtype=float)
        self._set_par(par)

    def __call__(self, r, par=None):
        return self.sample(r, par=par)

    def _set_par(self, par):
        if par is not None:
            if len(par) != self.np:
                raise ValueError('Incorrect number of parameters.')
            self.par = np.atleast_1d(par)

    def _sort(self, r, check):
        i2 = np.searchsorted(self.edges, r, side='right')
        if not check:
            return i2
        if not np.all(np.isin(np.arange(self.np-1)+1, np.unique(i2))):
            raise ValueError('Not all segments of the piece-wise linear function are constrained.')

    def sample(self, r, par=None, check=False):
        if par is not None:
            self._set_par(par)
        f = np.full(r.size, self.edges[0], dtype=float)
        i2 = self._sort(r, check)
        indx = (i2 > 0)
        f[indx] = self.par[i2[indx]-1]
        return f

    def ddr(self, r, par=None):
        return np.zeros(r.size, dtype=float)

    def d2dr2(self, x, par=None):
        return np.zeros(r.size, dtype=float)


class PiecewiseLinear:
    def __init__(self, edges, par=None):
        self.edges = np.sort(edges)
        if not np.array_equal(self.edges, edges):
            warnings.warn('As provided, edges for PiecewiseLinear were not sorted.')
        self.np = self.edges.size
        self.par = np.ones(self.np, dtype=float)
        self._set_par(par)

    def __call__(self, r, par=None):
        return self.sample(r, par=par)

    def _set_par(self, par):
        if par is not None:
            if len(par) != self.np:
                raise ValueError('Incorrect number of parameters.')
            self.par = np.atleast_1d(par)

    def _sort(self, r, check):
        i2 = np.searchsorted(self.edges, r, side='right')
        if not check:
            return i2
        if not np.all(np.isin(np.arange(self.np-1)+1, np.unique(i2))):
            raise ValueError('Not all segments of the piece-wise linear function are constrained.')

    def sample(self, r, par=None, check=False):
        if par is not None:
            self._set_par(par)
        f = np.full(r.size, self.edges[0], dtype=float)
        i2 = self._sort(r, check)
        indx = (i2 > 0) & (i2 < self.np)
        f[indx] = lin_interp(r[indx], self.edges[i2[indx]-1], self.par[i2[indx]-1],
                             self.edges[i2[indx]], self.par[i2[indx]])
        f[i2 == self.np] = self.par[-1]
        return f

    def ddr(self, r, par=None):
        if par is not None:
            self._set_par(par)
        f = np.zeros(r.size, dtype=float)
        i2 = self._sort(r, check)
        indx = (i2 > 0) & (i2 < self.np)
        f[indx] = (self.par[i2[indx]] - self.par[i2[indx]-1]) \
                    / (self.edges[i2[indx]] - self.edges[i2[indx]-1])
        return f

    def d2dr2(self, x, par=None):
        return np.zeros(r.size, dtype=float)


class HyperbolicTangent:
    """
    Instantiates a hyperbolic tangent function.
    """
    def __init__(self, par=None):
        self.np = 2
        self.par = self.guess_par()
        self._set_par(par)

    def __call__(self, r, par=None):
        return self.sample(r, par=par)
        
    def _set_par(self, par):
        if par is not None:
            if len(par) != self.np:
                raise ValueError('Incorrect number of parameters.')
            self.par = np.atleast_1d(par)

    @staticmethod
    def guess_par():
        return np.array([100., 10.])

    @staticmethod
    def par_bounds(maxr):
        return np.array([0., 1e-3]), np.array([500., maxr])

    def sample(self, r, par=None):
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.tanh(r/self.par[1])

    def ddr(self, r, par=None):
        if par is not None:
            self._set_par(par)
        sech2 = (1./numpy.cosh(r/self.par[1]))**2
        return self.par[0] * sech2 / self.par[1]

    def d2dr2(self, r, par=None):
        if par is not None:
            self._set_par(par)
        rh = r/self.hrot
        sech2 = (1./numpy.cosh(rh))**2
        tanh = np.tanh(rh)
        return -2. * self.par[0] * sech2 * tanhr / self.par[1]**2 


class Exponential:
    """
    Exponential profile.
    """
    def __init__(self, par=None):
        self.np = 2
        self.par = np.ones(self.np, dtype=float)
        self._set_par(par)

    def __call__(self, r, par=None):
        return self.sample(r, par=par)
        
    def _set_par(self, par):
        if par is not None:
            if len(par) != self.np:
                raise ValueError('Incorrect number of parameters.')
            self.par = np.atleast_1d(par)

    def sample(self, r, par=None):
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.exp(-r/self.par[1])

    def ddr(self, r, par=None):
        if par is not None:
            self._set_par(par)
        return -self.sample(r)/self.par[1]



