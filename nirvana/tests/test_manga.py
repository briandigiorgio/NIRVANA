
from IPython import embed

import numpy

from astropy.io import fits

from scipy import signal
from astropy import convolution

from nirvana.data import manga
from nirvana.tests.util import remote_data_file, requires_remote, dap_test_daptype

from nirvana.models.beam import convolve_fft, gauss2d_kernel

# TODO: Test remapping

@requires_remote
def test_manga_gas_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    kin = manga.MaNGAGasKinematics(maps_file, cube_file)
    _vel = kin.remap('vel', masked=False)

    with fits.open(maps_file) as hdu:
        eml = manga.channel_dictionary(hdu, 'EMLINE_GVEL')
        assert numpy.array_equal(_vel, hdu['EMLINE_GVEL'].data[eml['Ha-6564']]), 'Bad read'

    # Check that the binning works. NOTE: This doesn't use
    # numpy.array_equal because the matrix multiplication used by bin()
    # leads to differences that are of order the numerical precision.
    assert numpy.allclose(kin.bin(_vel), kin.vel), 'Rebinning is bad'

@requires_remote
def test_manga_stellar_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    kin = manga.MaNGAStellarKinematics(maps_file, cube_file)
    _vel = kin.remap('vel', masked=False)

    # Check that the input map is correctly reproduced
    with fits.open(maps_file) as hdu:
        assert numpy.array_equal(_vel, hdu['STELLAR_VEL'].data), 'Bad read'

    # Check that the binning works. NOTE: This doesn't use
    # numpy.array_equal because the matrix multiplication used by bin()
    # leads to differences that are of order the numerical precision.
    assert numpy.allclose(kin.bin(_vel), kin.vel), 'Rebinning is bad'

def test_from_plateifu():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    data_root = remote_data_file()

    _maps_file, _cube_file = manga.manga_files_from_plateifu(8138, 12704, cube_path=data_root,
                                                             maps_path=data_root, check=False)

    assert maps_file == _maps_file, 'MAPS file name incorrect'
    assert cube_file == _cube_file, 'CUBE file name incorrect'


