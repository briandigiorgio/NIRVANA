
0.0.1dev
--------

 - Initial version
 - Add beam-smearing
 - Restructure into python package
 - Added `Kinematics` class for presenting the data to be modeled
 - Added the MaNGA-specific I/O objects that subclass from `Kinematics`
 - Changes the geometry computation; see `barfit.models.geometry`
 - Speeds up the beam-smearing convolution, and checks that the
   convolution with the MaNGA PSF does not shift the model
 - Dramatically speeds up the `Kinematics.bin` function using
   `scipy.sparse` matrices.
 - Adds a bunch of test code.
 - Begins the implementation of a general `AxisymmetricDisk` class.
 - Adds fixing of center
 - Adds mock data set
