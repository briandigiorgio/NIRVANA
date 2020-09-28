# TODO

## Main tasks

- ~~recheck and merge Kyle's code~~
- ~~test on mock data (no bad points?)~~
- ~~fix edges of model data (interpolation errors)~~
- ~~error bars on plots~~
- multiple mocks with different characteristics
- pick out bad points
- binning schemes
- add error term
- stellar fits
- ~~adding in dispersion~~
- ~~add in surface brightness~~
- allow Kinematics to include (inverse) covariance
    - Include nominal MaNGA covariance for MaNGAKinematics
- Treatment of the velocity dispersion and correction

## Side tasks

- optimize runtime by switching on beam smearing running?
- ~~fix multiprocessing~~ (don't know why it works now)
- Renaming
    - NASKAR: NonAxiSymmetric Kinematic Analysis Routine
    - NIRVANA: Nonaxisymmetric Irregular Rotational Velocity ANAlysis
- Jacobians?
- Fisher matrix?

## Back-burner development

- Think about how to efficiently keep fully mapped data in Kinematics.
- Do not *require* that Kinematics objects use (square) 2D maps
- Construct a Kinematics subclass that can hold more than one kinematic
  component (i.e., gas and stars or multiple gas lines).
- Errors in the sigma correction?
- Allow Kinematics to be provided with line-of-sight kinematics that
  have not be regridded to a square image.
    - Create an automated way of constructing `grid_x` and `grid_y`
      based on the `x` and `y` locations of the irregularly sampled
      data.

