
from IPython import embed

import numpy

from barfit.data import manga
from barfit.tests.util import remote_data_file, requires_remote, dap_test_daptype
from barfit.models.oned import HyperbolicTangent
from barfit.models.axisym import AxisymmetricDisk
from barfit.models.beam import convolve_fft, smear, gauss2d_kernel

def test_disk():
    disk = AxisymmetricDisk()
    disk.par[:2] = 0.       # Ensure that the center is at 0,0
    disk.par[-1] = 1.       # Put in a quickly rising RC

    n = 51
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(x, y)

    vel = disk.model(disk.par, x=x, y=y)
    beam = gauss2d_kernel(n, 3.)
    _vel = disk.model(disk.par, x=x, y=y, beam=beam)

    assert numpy.isclose(vel[n//2,n//2], _vel[n//2,n//2]), 'Smearing moved the center.'

@requires_remote
def test_lsq_nopsf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root, ignore_psf=True)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 53. < disk.par[3] < 54., 'Inclination changed'
    assert 242. < disk.par[5] < 244., 'Projected rotation changed'


@requires_remote
def test_lsq_psf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 56. < disk.par[3] < 57., 'Inclination changed'
    assert 250. < disk.par[5] < 252., 'Projected rotation changed'

