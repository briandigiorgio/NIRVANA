
from IPython import embed

import numpy

from barfit.data import manga
from barfit.tests.util import remote_data_file
from barfit.models.oned import HyperbolicTangent
from barfit.models.axisym import AxisymmetricDisk

# Benchmarking test for the least-squares fit
def test_lsq_nopsf():

    # Read the data to fit
    data_root = remote_data_file()
    kin = manga.MaNGAGasKinematics.from_plateifu(8078, 12703, cube_path=data_root,
                                                 maps_path=data_root)

    nrun = 100
    for i in range(nrun):
        print('{0}/{1}'.format(i+1,nrun), end='\r')
        # Set the rotation curve
        rc = HyperbolicTangent()
        # Set the disk velocity field
        disk = AxisymmetricDisk(rc)
        # Fit it with a non-linear least-squares optimizer
        disk.lsq_fit(kin)
    print('{0}/{0}'.format(nrun))

if __name__ == '__main__':
    test_lsq_nopsf()



