from IPython import embed

import numpy

from barfit.models import oned

def test_step():
    n = 10
    edges = numpy.arange(n, dtype=float)+1
    rng = numpy.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    x = numpy.arange(n+2, dtype=float)+0.5
    y = f.sample(x)

    assert numpy.allclose(numpy.concatenate(([steps[0]], steps, [steps[-1]])), y), \
            'Evaluation is bad'


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


def test_tanh():
    f = oned.HyperbolicTangent(par=[1.,1.])
    y = f.sample([1.])

    assert numpy.isclose(y[0], numpy.tanh(1.)), 'Function changed.'


def test_exp():
    f = oned.Exponential(par=[1.,1.])
    y = f.sample([1.])

    assert numpy.isclose(y[0], numpy.exp(-1.)), 'Function changed.'

