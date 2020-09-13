
from IPython import embed

from barfit.data import manga
from barfit.tests.util import remote_data_file, requires_remote, dap_test_daptype

# TODO: Test remapping

@requires_remote
def test_manga_gas_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    gaskin = manga.MaNGAGasKinematics(maps_file, cube_file)


@requires_remote
def test_manga_stellar_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    strkin = manga.MaNGAStellarKinematics(maps_file, cube_file)


def test_from_plateifu():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    data_root = remote_data_file()

    _maps_file, _cube_file = manga.manga_files_from_plateifu(8138, 12704, cube_path=data_root,
                                                             maps_path=data_root)

    assert maps_file == _maps_file, 'MAPS file name incorrect'
    assert cube_file == _cube_file, 'CUBE file name incorrect'

