from IPython import embed

import numpy

from nirvana.models import oned

def test_step():
    n = 10
    edges = numpy.arange(n, dtype=float)+1
    rng = numpy.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    # Sample
    x = numpy.arange(n+2, dtype=float)+0.5
    y = f.sample(x)

    assert numpy.allclose(numpy.concatenate(([steps[0]], steps, [steps[-1]])), y), \
            'Evaluation is bad'

    # Check that the sampling coordinate doesn't need to be sorted on input
    _x = x.copy()
    rng.shuffle(_x)
    _y = f.sample(_x)
    srt = numpy.argsort(_x)

    assert numpy.array_equal(_y[srt], y), 'sorting of input coordinates should not matter'


def test_step_deriv():
    n = 10
    edges = numpy.arange(n, dtype=float)+1
    rng = numpy.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    x = numpy.arange(n+2, dtype=float)+0.5
    y = f.sample(x)
    _y, dy = f.deriv_sample(x)

    assert numpy.array_equal(y, _y), 'sample and deriv_sample give different function values'
    _dy = numpy.zeros_like(dy)
    _dy[0,0] = 1.
    _dy[-1,-1] = 1.
    _dy[1:-1] = numpy.identity(n)
    assert numpy.array_equal(dy, _dy), 'bad derivative'

    _x = x.copy()
    rng.shuffle(_x)
    _y, _dy = f.deriv_sample(_x)
    srt = numpy.argsort(_x)

    assert numpy.array_equal(y, _y[srt]), 'sorting of the input coordinates should not matter'
    assert numpy.array_equal(dy, _dy[srt]), 'sorting of the input coordinates should not matter'


def test_lin():
    n = 10
    edges = numpy.arange(n, dtype=float)+1
    rng = numpy.random.default_rng()
    anchors = rng.uniform(low=0., high=n+2., size=n)

    f = oned.PiecewiseLinear(edges, par=anchors)

    x = numpy.arange(n+1, dtype=float)+0.5
    y = f.sample(x)

    assert numpy.allclose(numpy.concatenate(([anchors[0]], (anchors[1:] + anchors[:-1])/2,
                                             [anchors[-1]])), y), 'Evaluation is bad'

    # Check that the sampling coordinate doesn't need to be sorted on input
    _x = x.copy()
    rng.shuffle(_x)
    _y = f.sample(_x)
    srt = numpy.argsort(_x)

    assert numpy.array_equal(_y[srt], y), 'sorting of input coordinates should not matter'


def test_lin_deriv():
    n = 10
    edges = numpy.arange(n, dtype=float)+1
    rng = numpy.random.default_rng()
    anchors = rng.uniform(low=0., high=n+2., size=n)

    f = oned.PiecewiseLinear(edges, par=anchors)

    x = numpy.arange(n+1, dtype=float)+0.5
    y = f.sample(x)
    _y, dy = f.deriv_sample(x)

    assert numpy.array_equal(y, _y), 'sample and deriv_sample give different function values'
    _dy = numpy.zeros_like(dy)
    _dy[0,0] = 1.
    _dy[1:] = numpy.diag(numpy.full(n-1, 0.5), 1) + numpy.diag(numpy.full(n, 0.5))
    _dy[-1,-1] = 1.
    assert numpy.array_equal(dy, _dy), 'bad derivative'
    assert numpy.array_equal(y, _y), 'sample and deriv_sample give different function values'

    _x = x.copy()
    rng.shuffle(_x)
    _y, _dy = f.deriv_sample(_x)
    srt = numpy.argsort(_x)

    assert numpy.array_equal(y, _y[srt]), 'sorting of the input coordinates should not matter'
    assert numpy.array_equal(dy, _dy[srt]), 'sorting of the input coordinates should not matter'


def test_tanh():
    f = oned.HyperbolicTangent(par=[1.,1.])
    y = f.sample([1.])

    assert numpy.isclose(y[0], numpy.tanh(1.)), 'Function changed.'


def test_tanh_deriv():
    par = numpy.array([1., 1.])
    dp = numpy.array([0.0001, 0.0001])
    f = oned.HyperbolicTangent(par=par)
    rng = numpy.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = numpy.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert numpy.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_plex():
    f = oned.PolyEx(par=[1.,1.,0.1])
    y = f.sample([1.])

    assert numpy.isclose(y[0], 1.1*(1-numpy.exp(-1.))), 'Function changed.'


def test_plex_deriv():
    par = numpy.array([1., 1., 0.1])
    dp = numpy.array([0.0001, 0.0001, 0.0001])
    f = oned.PolyEx(par=par)
    rng = numpy.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = numpy.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert numpy.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_exp():
    f = oned.Exponential(par=[1.,1.])
    y = f.sample([1.])

    assert numpy.isclose(y[0], numpy.exp(-1.)), 'Function changed.'


def test_exp_deriv():
    par = numpy.array([1., 1.])
    dp = numpy.array([0.0001, 0.0001])
    f = oned.Exponential(par=par)
    rng = numpy.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = numpy.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert numpy.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_expbase():
    f = oned.ExpBase(par=[1.,1.,1.])
    y = f.sample([1.])

    assert numpy.isclose(y[0], numpy.exp(-1.)+1), 'Function changed.'


def test_expbase_deriv():
    par = numpy.array([1., 1., 1.])
    dp = numpy.array([0.0001, 0.0001, 0.001])
    f = oned.ExpBase(par=par)
    rng = numpy.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = numpy.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert numpy.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_powexp():
    f = oned.PowerExp(par=[1.,1.,1.])
    y = f.sample([1.])
    assert numpy.isclose(y[0], 1.), 'Function changed.'


def test_powexp_deriv():
    par = numpy.array([1., 1., 1.])
    dp = numpy.array([0.0001, 0.0001, 0.0001])
    f = oned.PowerExp(par=par)
    rng = numpy.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = numpy.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert numpy.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


