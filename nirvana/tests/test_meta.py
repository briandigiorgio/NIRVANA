
from IPython import embed

import numpy

from nirvana.data.meta import GlobalPar
from nirvana.models.axisym import AxisymmetricDisk

def test_inc():
    gp = GlobalPar(pa=45, ell=0.5, q0=0.)
    assert numpy.isclose(gp.guess_inclination(), 60.), 'Wrong guess inclination'

    gp = GlobalPar(pa=45, ell=0.5, q0=0.2)
    assert gp.guess_inclination() > 60., 'Wrong guess inclination'

def test_kinpa():
    disk = AxisymmetricDisk()
    disk.par[:2] = 0.       # Ensure that the center is at 0,0
    disk.par[3] = 45.       # Set the kinematic position angle

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    vel = disk.model(disk.par, x=x, y=y)

    # Set the global parameters with a flipped position angle
    gp = GlobalPar(pa=-135., ell=0.5, q0=0.)
    pa = gp.guess_kinematic_pa(x, y, vel)

    # Make sure the correct position angle was recovered
    assert numpy.isclose(pa, 45.), 'Position angle should have been flipped'


#if __name__ == '__main__':
    #test_kinpa()

