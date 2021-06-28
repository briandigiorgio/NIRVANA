"""
Implements one-dimensional functions for modeling.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import special

from .util import lin_interp


class Func1D:
    """
    Base class that *should not* be instantiated by itself.
    """
    def __init__(self, par):
        self.np = len(par)
        self._set_par(par)

    def __call__(self, x, par=None, check=False):
        return self.sample(x, par=par, check=check)

    def _set_par(self, par):
        if par is not None:
            if len(par) != self.np:
                raise ValueError('Incorrect number of parameters.')
            self.par = np.atleast_1d(par)

    def sample(self, x, par=None, check=False):
        raise NotImplementedError('Sample function not defined for {0}!'.format(
                                    self.__class__.__name__))

    def ddx(self, x, par=None, check=False):
        raise NotImplementedError('Function derivative not defined for {0}!'.format(
                                    self.__class__.__name__))

    def d2dx2(self, x, par=None, check=False):
        raise NotImplementedError('Function second derivative not defined for {0}!'.format(
                                    self.__class__.__name__))


class StepFunction(Func1D):
    """
    Defines a step function.

    Args:
        edges (array-like):
            The *left* edges of each step. Samples to the left and right,
            respectively, of the first and last edge are given the same value
            as those edges.
        par (array-like, optional):
            The values of the step function. Shape must be the same
            as ``edges``. If None, step levels set by
            :func:`guess_par`.
        minv (:obj:`float`, optional):
            Uniform lower bound for all steps. If None, set by
            :func:`par_bounds`.
        maxv (:obj:`float`, optional):
            Uniform upper bound for all steps. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, edges, par=None, minv=None, maxv=None):
        self.edges = np.sort(edges)
        if not np.array_equal(self.edges, edges):
            warnings.warn('As provided, edges for StepFunction were not sorted.')
        if par is not None and len(par) != self.edges.size:
            raise ValueError('Provided number of parameters does not match the number of edges.')
        super().__init__(self.guess_par(self.edge.size) if par is None else par)
        self.lb, self.ub = self.par_bounds(minv=minv, maxv=maxv)

    @staticmethod
    def guess_par(npar):
        """
        Guess parameters for the step function.

        Args:
            npar (:obj:`int`):
                Number of step function edges.

        Returns:
            `numpy.ndarray`_: Guess parameters.
        """
        return np.ones(npar, dtype=float)

    def par_names(self, short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return [f'v{i+1}' for i in range(self.np)]
        return [f'Value at left edge of step {i+1}' for i in range(self.np)]

    def par_bounds(self, minv=None, maxv=None):
        """
        Function parameter boundaries.

        Args:
            minv (:obj:`float`, optional):
                Uniform lower boundary. If None, set to ``-np.inf``.
            maxv (:obj:`float`, optional):
                Uniform upper boundary. If None, set to ``np.inf``.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper boundary for the
            function.
        """
        return np.full(self.np, -np.inf if minv is None else minv, dtype=float), \
                np.full(self.np, np.inf if maxv is None else maxv, dtype=float)

    def _sort(self, x, check):
        """
        Sort the provided coordinates into each edge.

        Args:
            x (array-like):
                Locations at which to sample the function.
            check (:obj:`bool`):
                Check that at least one ``x`` value falls in each
                step.

        Returns:
            `numpy.ndarray`_: Indices of the *right* edge associated
            with each ``x`` value.
        """
        i2 = np.searchsorted(self.edges, x, side='right')
        if not check:
            return i2
        if not np.all(np.isin(np.arange(self.np-1)+1, np.unique(i2))):
            raise ValueError('Not all segments of the piece-wise linear function are constrained.')

    def sample(self, x, par=None, check=False):
        """
        Sample the step function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The step function parameters. If None, the current
                values of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Check that at least one ``x`` value falls in each
                step.

        Returns:
            `numpy.ndarray`_: Values of the step function at each
            ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        f = np.full(len(x), self.par[0], dtype=float)
        i2 = self._sort(x, check)
        indx = (i2 > 0)
        f[indx] = self.par[i2[indx]-1]
        return f

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the step function. See
        :func:`sample` for the argument descriptions.
        """
        return np.zeros(len(x), dtype=float)

    def d2dx2(self, x, par=None, check=False):
        """
        Sample the second derivative of the step function. See
        :func:`sample` for the argument descriptions.
        """
        return np.zeros(len(x), dtype=float)


class PiecewiseLinear(Func1D):
    """
    Defines a piece-wise linear function.

    Args:
        edges (array-like):
            The vertices of linearly connected function. Samples to
            the left and right, respectively, first and last edge are
            given the same value as those edges.
        par (array-like, optional):
            The values of the linear function at each vertex. Shape
            must be the same as ``edges``. If None, step levels set
            by :func:`guess_par`.
        minv (:obj:`float`, optional):
            Uniform lower bound for all steps. If None, set by
            :func:`par_bounds`.
        maxv (:obj:`float`, optional):
            Uniform upper bound for all steps. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, edges, par=None, minv=None, maxv=None):
        self.edges = np.sort(edges)
        if not np.array_equal(self.edges, edges):
            warnings.warn('As provided, edges for PiecewiseLinear were not sorted.')
        if par is not None and len(par) != self.edges.size:
            raise ValueError('Provided number of parameters does not match the number of edges.')
        super().__init__(self.guess_par(self.edge.size) if par is None else par)
        self.lb, self.ub = self.par_bounds(minv=minv, maxv=maxv)

    @staticmethod
    def guess_par(npar):
        """
        Guess parameters for the function.

        Args:
            npar (:obj:`int`):
                Number of linear function vertices.

        Returns:
            `numpy.ndarray`_: Guess parameters.
        """
        return np.ones(npar, dtype=float)

    def par_names(self, short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return [f'v{i+1}' for i in range(self.np)]
        return [f'Value at vertex {i+1}' for i in range(self.np)]

    def par_bounds(self, minv=None, maxv=None):
        """
        Function parameter boundaries.

        Args:
            minv (:obj:`float`, optional):
                Uniform lower boundary. If None, set to ``-np.inf``.
            maxv (:obj:`float`, optional):
                Uniform upper boundary. If None, set to ``np.inf``.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper boundary for the
            function.
        """
        return np.full(self.np, -np.inf if minv is None else minv, dtype=float), \
                np.full(self.np, np.inf if maxv is None else maxv, dtype=float)

    def _sort(self, x, check):
        """
        Sort the provided coordinates into each edge.

        Args:
            x (array-like):
                Locations at which to sample the function.
            check (:obj:`bool`):
                Check that at least one ``x`` value falls in each
                step.

        Returns:
            `numpy.ndarray`_: Indices of the *right* edge associated
            with each ``x`` value.
        """
        i2 = np.searchsorted(self.edges, x, side='right')
        if not check:
            return i2
        if not np.all(np.isin(np.arange(self.np-1)+1, np.unique(i2))):
            raise ValueError('Not all segments of the piece-wise linear function are constrained.')

    def sample(self, x, par=None, check=False):
        """
        Sample the piece-wise linear function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Check that at least one ``x`` value falls in each
                step.

        Returns:
            `numpy.ndarray`_: Values of the step function at each
            ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        f = np.full(len(x), self.par[0], dtype=float)
        i2 = self._sort(x, check)
        indx = (i2 > 0) & (i2 < self.np)
        f[indx] = lin_interp(x[indx], self.edges[i2[indx]-1], self.par[i2[indx]-1],
                             self.edges[i2[indx]], self.par[i2[indx]])
        f[i2 == self.np] = self.par[-1]
        return f

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the step function. See
        :func:`sample` for the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        f = np.zeros(len(x), dtype=float)
        i2 = self._sort(x, check)
        indx = (i2 > 0) & (i2 < self.np)
        f[indx] = (self.par[i2[indx]] - self.par[i2[indx]-1]) \
                    / (self.edges[i2[indx]] - self.edges[i2[indx]-1])
        return f

    def d2dx2(self, x, par=None, check=False):
        """
        Sample the second derivative of the step function. See
        :func:`sample` for the argument descriptions.
        """
        return np.zeros(len(x), dtype=float)


class HyperbolicTangent(Func1D):
    """
    Instantiates a hyperbolic tangent function.

    Args:
        par (array-like, optional):
            The two model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if ub is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([100., 10.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['asymp', 'scl']
        return ['Asymptotic value', 'Scale']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3]), np.array([500., 100.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.tanh(np.asarray(x)/self.par[1])

    def ddx(self, x, par=None):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        sech2 = (1./numpy.cosh(np.asarray(x)/self.par[1]))**2
        return self.par[0] * sech2 / self.par[1]

    def d2dx2(self, x, par=None):
        """
        Sample the second derivative of the function. See
        :func:`sample` for the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        xh = np.asarray(x)/self.hrot
        sech2 = (1./numpy.cosh(xh))**2
        return -2. * self.par[0] * sech2 * np.tanh(xh) / self.par[1]**2 


class PolyEx(Func1D):
    r"""
    Instantiates a "PolyEx" rotation curve function.

    The three-parameter functional form is:

    .. math::

        V(r) = V_0 (1 - e^{-r/h}) (1 + \alpha \frac{r}{h})

    The parameter vector is ordered: :math:`V_0, h, \alpha`.

    Args:
        par (array-like, optional):
            The three model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if ub is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([100., 10., 0.1])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['asymp', 'scl', 'slp']
        return ['Characteristic value', 'Inner scale', 'Outer slope']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3, -1.]), np.array([500., 100., 1.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        s = np.asarray(x)/self.par[1]
        return self.par[0] * (1 - np.exp(-s)) * (1 + self.par[2] * s)

    def ddx(self, x, par=None):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        s = np.asarray(x)/self.par[1]
        return self.par[0] * (np.exp(-s) * (1 + self.par[2]*(s-1)) + self.par[2]) / self.par[1]

    def d2dx2(self, x, par=None):
        """
        Sample the second derivative of the function. See
        :func:`sample` for the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        s = np.asarray(x)/self.par[1]
        return -self.par[0] * np.exp(-s) * (1+ self.par[2]*(s-2)) / self.par[1]**2 


class Exponential(Func1D):
    """
    Instantiates an exponential function.

    Args:
        par (array-like, optional):
            The two model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if lb is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([100., 10.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['cen', 'h']
        return ['Center value', 'e-folding length']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3]), np.array([500., 100.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.exp(-np.asarray(x)/self.par[1])

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        return -self.sample(x)/self.par[1]


class ExpBase(Func1D):
    r"""
    Instantiates an exponential function with an additional parameter for a
    constant baseline.

    Functional form is

    .. math::

        F(x) = a*e^{-x/b} + c,

    where :math:`a,b,c` are the three parameters (in that order).

    Args:
        par (array-like, optional):
            The three model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if lb is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([100., 10., 0.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['cen', 'h', 'base']
        return ['Center value', 'e-folding length', 'Baseline']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3, -250.]), np.array([500., 100., 250.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.exp(-np.asarray(x)/self.par[1]) + self.par[2]

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        return -self.sample(x)/self.par[1]


class PowerExp(Func1D):
    r"""
    Instantiates a function combining a power-law and an exponential.

    Functional form is:

    .. math::

        f(x) = f_0 \frac{e}{h \gamma}^{\gamma} x^{\gamma} e^{-x/h}

    where :math:`\gamma \geq 0` and :math:`h > 0`. Parameter vectors are
    ordered: :math:`f_0, h, \gamma`. The maximum value of the function,
    :math:`f_0`, occurs at :math:`x = h \gamma`.

    Args:
        par (array-like, optional):
            The two model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if lb is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([100., 10., 1.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['a', 'h', 'g']
        return ['Amplitude', 'e-folding length', 'Power index']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3, 0.]), np.array([500., 100., 5.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        _x = np.asarray(x)
        c = (np.e / self.par[1] / self.par[2])**self.par[2]
        return c * self.par[0] * np.exp(-_x/self.par[1]) * _x**self.par[2]

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        c = (np.e / self.par[1] / self.par[2])**self.par[2]
        indx = np.absolute(x) > 1e-10
        deriv = np.zeros(x.size, dtype=float)
        deriv[indx] = self.sample(x[indx]) * (self.par[2]/x[indx] - 1/self.par[1]) 
        return deriv


class Const(Func1D):
    r"""
    A function that always returns the same constant.

    Args:
        par (array-like, optional):
            Constant value. If None, set by :func:`guess_par`.
        lb (array-like, optional):
            Lower bound. If None, set by :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bound. If None, set by :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if lb is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([1.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['c']
        return ['Constant']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([-250.]), np.array([250.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        return np.full(x.shape, self.par[0], dtype=float)

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        return np.zeros(x.shape, dtype=float)


class Sersic1D(Func1D):
    """
    Instantiates a 1D Sersic profile.

    Parameters are (1) the surface brightness at 1 half-light radius,
    (2) the half-light radius, and (3) the Sersic index.

    Args:
        par (array-like, optional):
            The three model parameters. If None, set by
            :func:`guess_par`.
        lb (array-like, optional):
            Lower bounds for the model parameters. If None, set by
            :func:`par_bounds`.
        ub (:obj:`float`, optional):
            Upper bounds for the model parameters. If None, set by
            :func:`par_bounds`.
    """
    def __init__(self, par=None, lb=None, ub=None):
        super().__init__(self.guess_par() if par is None else par)
        if lb is not None and len(lb) != self.np:
            raise ValueError('Number of lower bounds does not match the number of parameters.')
        if lb is not None and len(ub) != self.np:
            raise ValueError('Number of upper bounds does not match the number of parameters.')
        _lb, _ub = self.par_bounds()
        self.lb = _lb if lb is None else np.atleast_1d(lb)
        self.ub = _ub if ub is None else np.atleast_1d(ub)
        self._set_par(self.par)

    def _set_par(self, par):
        super()._set_par(par)
        self.bn = special.gammaincinv(2. * self.par[2], 0.5) 

    @staticmethod
    def guess_par():
        """Return default guess parameters."""
        return np.array([1., 10., 1.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['norm', 'reff', 'n']
        return ['Profile value at effective radius', 'Effective radius', 'Sersic Index']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3, 1e-2]), np.array([500., 100., 10.])

    def sample(self, x, par=None, check=False):
        """
        Sample the function.

        Args:
            x (array-like):
                Locations at which to sample the function.
            par (array-like, optional):
                The function parameters. If None, the current values
                of :attr:`par` are used. Must have a length of
                :attr:`np`.
            check (:obj:`bool`, optional):
                Ignored. Only included for a uniform interface with
                other subclasses of :class:`Func1D`.

        Returns:
            `numpy.ndarray`_: Function evaluated at each ``x`` value.
        """
        if par is not None:
            self._set_par(par)
        return self.par[0]*np.exp(-self.bn * ((x/self.par[1])**(1/self.par[2]) - 1))



