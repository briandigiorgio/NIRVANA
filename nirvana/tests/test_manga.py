
from IPython import embed

import numpy

from astropy.io import fits

from scipy import signal
from astropy import convolution

from nirvana import data
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.tests.util import drp_test_version, dap_test_daptype

from nirvana.models.beam import convolve_fft, gauss2d_kernel

# TODO: Test remapping

@requires_remote
def test_manga_gas_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    kin = data.manga.MaNGAGasKinematics(maps_file, cube_file=cube_file)
    _vel = kin.remap('vel', masked=False)

    with fits.open(maps_file) as hdu:
        eml = data.manga.channel_dictionary(hdu, 'EMLINE_GVEL')
        assert numpy.array_equal(_vel, hdu['EMLINE_GVEL'].data[eml['Ha-6564']]), 'Bad read'

    # Check that the binning works. NOTE: This doesn't use
    # numpy.array_equal because the matrix multiplication used by bin()
    # leads to differences that are of order the numerical precision.
    assert numpy.allclose(kin.bin(_vel), kin.vel), 'Rebinning is bad'


@requires_remote
def test_manga_stellar_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    kin = data.manga.MaNGAStellarKinematics(maps_file, cube_file=cube_file, covar=True)
    _vel = kin.remap('vel', masked=False)

    # Check that the input map is correctly reproduced
    with fits.open(maps_file) as hdu:
        assert numpy.array_equal(_vel, hdu['STELLAR_VEL'].data), 'Bad read'

    # Check that the binning works. NOTE: This doesn't use
    # numpy.array_equal because the matrix multiplication used by bin()
    # leads to differences that are of order the numerical precision.
    assert numpy.allclose(kin.bin(_vel), kin.vel), 'Rebinning is bad'


@requires_remote
def test_from_plateifu():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    data_root = remote_data_file()

    _maps_file, _cube_file, _image_file \
            = data.manga.manga_files_from_plateifu(8138, 12704, cube_path=data_root,
                                                   maps_path=data_root, check=False)

    assert maps_file == _maps_file, 'MAPS file name incorrect'
    assert cube_file == _cube_file, 'CUBE file name incorrect'


@requires_remote
def test_targeting_bits():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    hdu = fits.open(maps_file)

    assert data.manga.parse_manga_targeting_bits(hdu[0].header['MNGTARG1']) \
                == (True, False, False, False), 'Incorrect targeting parsing'
    assert data.manga.parse_manga_targeting_bits(hdu[0].header['MNGTARG1'],
                                            mngtarg3=hdu[0].header['MNGTARG1']) \
                == (True, False, True, False), 'Incorrect targeting parsing'

    mngtarg1 = numpy.array([hdu[0].header['MNGTARG1'], hdu[0].header['MNGTARG1']])
    mngtarg3 = numpy.array([hdu[0].header['MNGTARG3'], hdu[0].header['MNGTARG3']])

    pp, s, a, o = data.manga.parse_manga_targeting_bits(mngtarg1, mngtarg3=mngtarg3)
    assert numpy.array_equal(pp, [True, True]), 'Bad array parsing'


def test_covar_fake():

    # Make a fake inverse variance map
    ivar = numpy.zeros((30,30), dtype=float)
    ivar[10:20,10:20] = data.util.boxcar_replicate(numpy.sqrt(numpy.arange(25)+1).reshape(5,5), 2)

    # And an associated fake binning map
    binid = numpy.full(ivar.shape, -1, dtype=int)
    binid[10:20,10:20] = data.util.boxcar_replicate(numpy.arange(25).reshape(5,5), 2)

    _, covar = data.manga.manga_map_covar(ivar, binid=binid, fill=True)
    gpm, subcovar = data.manga.manga_map_covar(ivar, binid=binid, fill=False)

    # NOTE: this assert is already done by the manga_map_covar method at the moment.
    assert numpy.allclose(subcovar.diagonal(), 1./ivar[gpm]), \
            'Incorrect variances along the diagonal of the covariance matrix'

    assert numpy.array_equal(covar[numpy.ix_(gpm.ravel(),gpm.ravel())].toarray(),
                             subcovar.toarray()), 'Failed in filling the covariance array'


@requires_remote
def test_inv_covar():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    kin = data.manga.MaNGAGasKinematics(maps_file, covar=True)

    # Force the covariance matrix to be positive definite
    cov = data.util.impose_positive_definite(kin.vel_covar)
    # Invert it
    icov = data.util.cinv(cov.toarray())
    # Check that you get back the identity matrix
    assert numpy.std(numpy.diag(numpy.dot(icov, cov.toarray())) - 1.) < 1e-4, \
            'Multiplication by inverse matrix does not accurately produce identity matrix'


@requires_remote
def test_meta():
    drpall_file = remote_data_file(f'drpall-{drp_test_version}.fits')
    meta = data.manga.MaNGAGlobalPar(8138, 12704, drpall_file=drpall_file)

    embed()
    exit()

if __name__ == '__main__':
    test_meta()


