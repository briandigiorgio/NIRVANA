
0.1.2dev
--------

 - Include `PowerExp` 1D model
 - Allow the surface-brightness used during the model construction to be
   defined separately from the binned surface-brightness data.  This is
   useful for modeling the stellar kinematics.
 - Allow the surface-brightness masks to be patched by a Gaussian
   smoothing algorithm to fill surface-brightness holes.
 - `AxisymmetriDisk` now uses `ConvolveFFTW` by default.
 - Included analytic calculations of derivatives for use with the
   `AxisymmetricDisk` fits.  Analytic calculations are used by default,
   but the `lsq_fit` function allows for a fall back to the
   finite-difference methods provided by `scipy.optimize.least_squares`.
 - Included additional 1D functions for, e.g., rotation curves.
 - Bug fixes when using covariance and fitting without dispersion in
   `nirvana_manga_axisym` script

0.1.0
-----

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
 - Added `barfit.models.beam.ConvolveFFTW` class for doing convolutions
   with the FFTW library.
 - Added masking of edges of mocks to mitigate convolution edge effects
 - Renamed to NIRVANA
 - Removed MCMC and associated arguments
 - Enable fit to velocity dispersion using `AxisymmetricDisk`.
 - Force calculation of `mom0` in `nirvana.model.beam.smear` even if the
   surface brightness is not provided, using a unity array.
 - Add continuous integration tests using GitHub Workflow
 - Add initial docs
 - Copy over the BitMask class from the sdss-mangadap
 - Update masking and add method that determines the target sample given
   the targeting bits
 - Added construction of MaNGA covariance matrices and incorporated them
   into `Kinematics`.
 - Added handling of covariance matrices in `AxisymmetricDisk`.
 - Added `nirvana.data.scatter.IntrinsicScatter` used to both reject
   outlying residuals and determine the intrinsic scatter in the data.
 - Added galaxy pngs, drpall, and dapall to test data
 - Added `nirvana.data.meta.GlobalPar` and
   `nirvana.data.manga.MaNGAGlobalPar` classes to hold metadata
 - Added iterative axisymmetric disk fitting, assessment plots, and
   output file.
 - Fix `nirvana.data.manga.MaNGAGlobalPar` to handle when photometry is
   unavailable.  New header keyword `PHOT_KEY` add to binary table
   output for axisymmetric fit giving which photometry was used.
 - Axisymmetric fit changes:
    - Include flag to set minimum number of valid spaxels for
      axisymmetric fit to proceed.
    - set lower bound on inclination to 1 degree to avoid div-by-zero
      errors.
    - Added option to find the largest coherent region of valid spaxels
      for the fit.
 - Added manga download scripts


