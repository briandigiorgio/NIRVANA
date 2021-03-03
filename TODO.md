# TODO

## Main tasks

- output files
- computation time
- compare fits with axisym
- projected velocities
    - just take out sini
- allow Kinematics to include (inverse) covariance
    - Include nominal MaNGA covariance for MaNGAKinematics
- Treatment of the velocity dispersion and correction
- smarter surface brightness with all gas channels
- put prior, clipping, etc parameters into one config file

## Side tasks

- optimize runtime by switching on beam smearing running?
- Jacobians?
- Fisher matrix?
- Only one mask?

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

