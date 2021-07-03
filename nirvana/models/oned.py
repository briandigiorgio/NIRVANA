"""
Implements one-dimensional functions for modeling.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import special

from .util import lin_interp, deriv_lin_interp, sech2


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

    def deriv_sample(self, x, par=None, check=False):
        raise NotImplementedError('Sample function and parameter derivatives not defined for '
                                  '{0}!'.format(self.__class__.__name__))

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
            as ``edges``. If None, step levels are set by
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
            raise ValueError('Not all segments of the step function are constrained.')

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
        i2 = self._sort(x, check)   # 1-indexed step associated with each x
        indx = (i2 > 0)
        f[indx] = self.par[i2[indx]-1]
        return f

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

        The derivative of the step function w.r.t. each parameter is 1 in the
        radial range of the relevant step and 0 elsewhere.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        f = np.full(len(x), self.par[0], dtype=float)
        df = np.zeros((len(x), self.np), dtype=float)
        i2 = self._sort(x, check)   # 1-indexed step associated with each x
        indx = (i2 > 0)
        f[indx] = self.par[i2[indx]-1]
        df[np.where(indx)[0],i2[indx]-1] = 1.
        indx = i2 == 0
        df[np.where(indx)[0],np.array([0]*np.sum(indx))] = 1.
        return f, df


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
        Sample the piecewise linear function.

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
            `numpy.ndarray`_: Values of the piecewise function at each
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

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

        The derivative of the step function w.r.t. each parameter is 1 in the
        radial range of the relevant step and 0 elsewhere.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        f = np.full(len(x), self.par[0], dtype=float)
        df = np.zeros((len(x), self.np), dtype=float)
        i2 = self._sort(x, check)
        indx = (i2 > 0) & (i2 < self.np)
        f[indx], _df = deriv_lin_interp(x[indx], self.edges[i2[indx]-1], self.par[i2[indx]-1],
                                        self.edges[i2[indx]], self.par[i2[indx]])
        df[indx,i2[indx]-1] = _df[:,0]
        df[indx,i2[indx]] = _df[:,1]
        indx = i2 == self.np
        df[np.where(indx)[0],np.array([self.np-1]*np.sum(indx))] = 1.
        f[indx] = self.par[-1]
        indx = i2 == 0
        df[np.where(indx)[0],np.array([0]*np.sum(indx))] = 1.
        return f, df

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
        m = np.diff(self.par)/np.diff(self.edges)
        f[indx] = m[i2[indx]-1]
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
        return self.par[0]*np.tanh(np.atleast_1d(x)/self.par[1])

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        _x = np.atleast_1d(x)/self.par[1]
        f = self.par[0]*np.tanh(_x)
        _sech2 = sech2(_x)
        return f, np.stack((f/self.par[0], -self.par[0]*_x*_sech2/self.par[1]), axis=-1)

    def ddx(self, x, par=None):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        _sech2 = sech2(np.atleast_1d(x)/self.par[1])
        return self.par[0] * _sech2 / self.par[1]

    def d2dx2(self, x, par=None):
        """
        Sample the second derivative of the function. See
        :func:`sample` for the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        xh = np.atleast_1d(x)/self.par[1]
        _sech2 = sech2(xh)
        return -2. * self.par[0] * _sech2 * np.tanh(xh) / self.par[1]**2 


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
        s = np.atleast_1d(x)/self.par[1]
        return self.par[0] * (1 - np.exp(-s)) * (1 + self.par[2] * s)

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        s = np.atleast_1d(x)/self.par[1]
        e = np.exp(-s)
        t = 1 - e
        a = 1 + self.par[2] * s
        f = self.par[0] * t * a
        return f, np.stack((f/self.par[0], -f*s*(e/t + self.par[2]/a)/self.par[1], f*s/a), axis=-1)

    def ddx(self, x, par=None):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        s = np.atleast_1d(x)/self.par[1]
        return self.par[0] * (np.exp(-s) * (1 + self.par[2]*(s-1)) + self.par[2]) / self.par[1]

    def d2dx2(self, x, par=None):
        """
        Sample the second derivative of the function. See
        :func:`sample` for the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        s = np.atleast_1d(x)/self.par[1]
        return -self.par[0] * np.exp(-s) * (1+ self.par[2]*(s-2)) / self.par[1]**2 


class ConcentratedRotationCurve(Func1D):
    r"""
    Instantiates a rotation curve that enables a sharp rise that declines to a
    flat outer rotations speed.

    The four-parameter functional form is:

    .. math::

        V(x) = V_0 \frac{(1+x)^\beta}{(1+x^{-\gamma})^{1/\gamma}}

    where :math:`x = r/h` for radius :math:`r`.  This equation is provided in
    Eqn 8 of Rix et al. (1997, MNRAS, 285, 779) and is very close to Eqn 2 from
    Courteau (1997, AJ, 114, 2402).  The order of the parameter vector
    is:math:`V_0`, :math:`h`, :math:`\gamma`, and :math:`\beta`.

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
        return np.array([100., 10., 1., 0.1])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['norm', 'scl', 'peak', 'slp']
        return ['Normalization', 'Inner scale', 'Peakedness', 'Outer slope']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 1e-3, 1e-3, -1.]), np.array([500., 100., 100., 1.])

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
        xh = np.atleast_1d(x)/self.par[1]
        return self.par[0] * (1+xh)**self.par[3] * (1+xh**-self.par[2])**(-1/self.par[2])

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        xh = np.atleast_1d(x)/self.par[1]
        u = 1 + xh**-self.par[2]
        w = u**(-1/self.par[2])
        f = self.par[0] * (1+xh)**self.par[3] * w

        dxhd1 = -xh/self.par[1]
        dud1 = -self.par[2] * dxhd1 * xh**(-self.par[2]-1)
        dwd1 = -dud1 * u**(-1-1/self.par[2]) / self.par[2]

        dud2 = -np.log(x) * x**-self.par[2]
        dwd2 = np.log(u) * u**(-1/self.par[2]) / self.par[2]**2 \
                    - dud2 * u**(-1-1/self.par[2]) / self.par[2]

        return f, np.stack((f/self.par[0], f*self.par[3]*dxhd1/(1+xh) + f*dwd1/w, f*dwd2/w,
                           f*np.log(1+xh)), axis=-1)

    def ddx(self, x, par=None):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        xh = np.atleast_1d(x)/self.par[1]
        u = 1 + xh**-self.par[2]
        w = u**(-1/self.par[2])
        f = self.par[0] * (1+xh)**self.par[3] * w

        dxhdr = 1/self.par[1]
        dudr = -self.par[2] * dxhdr * xh**(-self.par[2]-1)
        dwdr = -dudr * u**(-1-1/self.par[2]) / self.par[2]

        return f*self.par[3]*dxhdr/(1+xh) + f*dwdr/w


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
        return self.par[0]*np.exp(-np.atleast_1d(x)/self.par[1])

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        _x = np.atleast_1d(x)/self.par[1]
        f = self.par[0]*np.exp(-_x)
        return f, np.stack((f/self.par[0], _x*f/self.par[1]), axis=-1)

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
        return self.par[0]*np.exp(-np.atleast_1d(x)/self.par[1]) + self.par[2]

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        _x = np.atleast_1d(x)/self.par[1]
        f = self.par[0]*np.exp(-_x) + self.par[2]
        return f, np.stack((np.exp(-_x), _x*(f-self.par[2])/self.par[1], np.ones(_x.shape)),
                           axis=-1)

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        return -np.exp(-np.atleast_1d(x)/self.par[1])/self.par[1]


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
        _x = np.atleast_1d(x)
        c = (np.e / self.par[1] / self.par[2])**self.par[2]
        return c * self.par[0] * np.exp(-_x/self.par[1]) * _x**self.par[2]

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)

        _x = np.atleast_1d(x)
        xh = _x/self.par[1]
        c0 = np.e / self.par[1] / self.par[2]
        c = c0**self.par[2]
        f = self.par[0] * c * np.exp(-xh) * _x**self.par[2]
        return f, np.stack((f/self.par[0], f*(xh/self.par[1] - self.par[2]/self.par[1]),
                            f*(np.log(c0) + np.log(_x) - 1)), axis=-1)

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


class PowerLaw(Func1D):
    r"""
    Instantiates a power-law function

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
        return np.array([1., 1.])

    @staticmethod
    def par_names(short=False):
        """
        Return a list of strings with the parameter names.
        """
        if short:
            return ['norm', 'gamma']
        return ['Normalization', 'Power-law index']

    @staticmethod
    def par_bounds():
        """
        Return default parameter boundaries.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ objects with,
            respectively, the lower and upper bounds for the
            parameters.
        """
        return np.array([0., 0.]), np.array([500., 5.])

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
        return self.par[0] * np.atleast_1d(x)**self.par[1]

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        _x = np.atleast_1d(x)
        f = self.par[0] * _x**self.par[1]
        return f, np.stack((f/self.par[0], f*np.log(_x)), axis=-1)

    def ddx(self, x, par=None, check=False):
        """
        Sample the derivative of the function. See :func:`sample` for
        the argument descriptions.
        """
        if par is not None:
            self._set_par(par)
        # TODO: Deal with the case when self.par[1] == 0 and x == 0.
        return self.par[0] * self.par[1] * np.atleast_1d(x)**(self.par[1]-1.)


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

    def deriv_sample(self, x, par=None, check=False):
        """
        Calculate the function and its derivative w.r.t. the parameters.

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
            :obj:`tuple`: Two `numpy.ndarray`_ objects: (1) the function
            evaluated at each ``x`` value and (2) the derivative of the function
            with respect to each parameter.  The object with the derivatives has
            one more dimension than the function data, with a length that is the
            number of functional parameters.
        """
        if par is not None:
            self._set_par(par)
        return np.ones((x.size,1), dtype=float)

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



