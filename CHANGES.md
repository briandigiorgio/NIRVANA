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
 - Added in fitting of dispersion in radial bins
 - Changed `barfit.barfit.unpack` to use a dictionary
 - Lots of changes to `barfit.barfit` code to utilize dictionary
 - Various changes to `FitArgs` and script to accommodate dictionary
 - Huge changes to `barfit.plotting` for dispersion and dictionary
 - Switched to using `MPL-10` by default
 - Tests and docs for `barfit.models.oned`.
 - Added `barfit.models.beam.ConvolveFFTW` class for doing
   convolutions with the FFTW library.
 - Added masking of edges of mocks to mitigate convolution edge effects
 - Renamed to NIRVANA
 - Removed MCMC and associated arguments
 - Enable fit to velocity dispersion using `AxisymmetricDisk`.
 - Force calculation of `mom0` in `nirvana.model.beam.smear` even if the
   surface brightness is not provided, using a unity array.
 - Add continuous integration tests using GitHub Workflow

