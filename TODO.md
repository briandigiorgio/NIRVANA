# TODO

## Main tasks

- ~~squaring sigma:~~ caused weird behavior
    - new variable sigma2 = sigma^2 - sigmacorr^2
    - new error by propagating errors through
    - in likelihood, compute chisq of square of sigma instead
- ~~maximum radius to avoid edge effects~~ still needs refining
- ~~pick out bad points~~ not good enough yet
    - sigma clip outliers
- projected velocities
    - just take out sini
- ~~add error term~~ not working yet
    -hogg 2010
- stellar fits
- allow Kinematics to include (inverse) covariance
    - Include nominal MaNGA covariance for MaNGAKinematics
- Treatment of the velocity dispersion and correction
- smarter surface brightness with all gas channels

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

