# TODO

## Main tasks

- ~~multiple mocks with different characteristics~~
- ~~binning schemes~~
- projected velocities
- pick out bad points
- add error term
- stellar fits
- ~~adding in dispersion~~
- ~~add in surface brightness~~
- allow Kinematics to include (inverse) covariance
    - Include nominal MaNGA covariance for MaNGAKinematics
- Treatment of the velocity dispersion and correction

## Side tasks

- optimize runtime by switching on beam smearing running?
- Renaming
    - NASKAR: NonAxiSymmetric Kinematic Analysis Routine
    - *NIRVANA: Nonaxisymmetric Irregular Rotational Velocity ANAlysis*
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

