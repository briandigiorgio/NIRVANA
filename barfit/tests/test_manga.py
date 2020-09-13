
from IPython import embed

from barfit.data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from barfit.tests.util import remote_data_file, requires_remote, dap_test_daptype

@requires_remote
def test_manga_gas_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    gaskin = MaNGAGasKinematics(maps_file, cube_file)


@requires_remote
def test_manga_stellar_kinematics():
    maps_file = remote_data_file('manga-8138-12704-MAPS-{0}.fits.gz'.format(dap_test_daptype))
    cube_file = remote_data_file('manga-8138-12704-LOGCUBE.fits.gz')

    strkin = MaNGAStellarKinematics(maps_file, cube_file)

