
from IPython import embed

import numpy

from scipy import stats, special
from nirvana import data
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.models.oned import HyperbolicTangent, Exponential
from nirvana.models.axisym import AxisymmetricDisk
from nirvana.models.beam import gauss2d_kernel


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
    kin = data.manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                      maps_path=data_root, ignore_psf=True)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 55. < disk.par[3] < 57., 'Inclination changed'
    assert 245. < disk.par[5] < 247., 'Projected rotation changed'


@requires_remote
def test_lsq_psf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = data.manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                      maps_path=data_root)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 57. < disk.par[3] < 59., 'Inclination changed'
    assert 252. < disk.par[5] < 255., 'Projected rotation changed'


@requires_remote
def test_lsq_with_sig():

    # Read the data to fit
    data_root = remote_data_file()
    kin = data.manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                      maps_path=data_root)
    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the dispersion profile
    dc = Exponential(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc, dc=dc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin, sb_wgt=True)

    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 58. < disk.par[3] < 60., 'Inclination changed'
    assert 250. < disk.par[5] < 253., 'Projected rotation changed'
    assert 35. < disk.par[7] < 37., 'Central velocity dispersion changed'


@requires_remote
def test_lsq_with_covar():
    # NOTE: This only fits the velocity field....

    # Read the data to fit
    data_root = remote_data_file()
    kin = data.manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                      maps_path=data_root, covar=True)

    kin.vel_covar = data.util.impose_positive_definite(kin.vel_covar)

    # Set the rotation curve
    rc = HyperbolicTangent(lb=numpy.array([0., 1e-3]), ub=numpy.array([500., kin.max_radius()]))
    # Set the disk velocity field
    disk = AxisymmetricDisk(rc=rc) #, dc=dc)
    # Fit it with a non-linear least-squares optimizer
    disk.lsq_fit(kin, sb_wgt=True) #, verbose=2)
    # Rejected based on error-weighted residuals, accounting for intrinsic scatter
    resid = kin.vel - kin.bin(disk.model())
    err = 1/numpy.sqrt(kin.vel_ivar)
    scat = data.scatter.IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm)
    sig, rej, gpm = scat.iter_fit(fititer=5) #, verbose=2)
    # Check
    assert sig > 6., 'Different intrinsic scatter'
    assert numpy.sum(rej) == 19, 'Different number of pixels were rejected'

    # Refit with new mask, include scatter and covariance
    kin.vel_mask = numpy.logical_not(gpm)
    p0 = disk.par
    disk.lsq_fit(kin, scatter=sig, sb_wgt=True, p0=p0, ignore_covar=False,
                 assume_posdef_covar=True) #, verbose=2)
    # Reject
    resid = kin.vel - kin.bin(disk.model())
    scat = data.scatter.IntrinsicScatter(resid, covar=kin.vel_covar, gpm=disk.vel_gpm,
                                         assume_posdef_covar=True)
    sig, rej, gpm = scat.iter_fit(fititer=5) #, verbose=2)
    # Check
    assert sig > 1., 'Different intrinsic scatter'
    assert numpy.sum(rej) == 5, 'Different number of pixels were rejected'
    # Model parameters
    assert numpy.all(numpy.absolute(disk.par[:2]) < 0.1), 'Center changed'
    assert 165. < disk.par[2] < 167., 'PA changed'
    assert 56. < disk.par[3] < 58., 'Inclination changed'
    assert 249. < disk.par[5] < 252., 'Projected rotation changed'


