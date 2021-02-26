
import warnings

from IPython import embed

import numpy as np

from scipy import optimize

from .oned import HyperbolicTangent
from .geometry import projected_polar
from .beam import smear
from .util import cov_err
from ..data.util import impose_positive_definite, cinv, inverse


class AxisymmetricDisk:
    r"""
    Simple model for an axisymmetric disk.

    The model assumes the disk is infinitely thin and has a single set of
    geometric parameters:

        - :math:`x_c, y_c`: The coordinates of the galaxy dynamical center.
        - :math:`\phi`: The position angle of the galaxy (the angle from N
          through E)
        - :math:`i`: The inclination of the disk; the angle of the disk
          normal relative to the line-of-sight such that :math:`i=0` is a
          face-on disk.
        - :math:`V_{\rm sys}`: The systemic (bulk) velocity of the galaxy
          taken as the line-of-sight velocity at the dynamical center.

    In addition to these parameters, the model instantiation requires class
    instances that define the rotation curve and velocity dispersion profile.
    These classes must have:

        - an ``np`` attribute that provides the number of parameters in the
          model
        - a ``guess_par`` method that provide initial guess parameters for
          the model, and
        - ``lb`` and ``ub`` attributes that provide the lower and upper
          bounds for the model parameters.

    Importantly, note that the model fits the parameters for the *projected*
    rotation curve. I.e., that amplitude of the fitted function is actually
    :math:`V_{\rm rot} \sin i`.

    .. todo::
        Describe the attributes

    """
    def __init__(self, rc=None, dc=None):
        # Rotation curve
        self.rc = HyperbolicTangent() if rc is None else rc
        # Velocity dispersion curve (can be None)
        self.dc = dc

        # Number of "base" parameters
        self.nbp = 5
        # Total number parameters
        self.np = self.nbp + self.rc.np
        if self.dc is not None:
            self.np += self.dc.np
        # Initialize the parameters
        self.par = self.guess_par()
        self.par_err = None
        # Flag which parameters are freely fit
        self.free = np.ones(self.np, dtype=bool)
        self.nfree = np.sum(self.free)

        # Workspace
        self.x = None
        self.y = None
        self.beam_fft = None
        self.kin = None
        self.sb = None
        self.vel_gpm = None
        self.sig_gpm = None

    def guess_par(self):
        """
        Return a list of generic guess parameters.

        .. todo::
            Could enable this to base the guess on the data to be fit, but at
            the moment these are hard-coded numbers.
        """
        # Return the 
        gp = np.concatenate(([0., 0., 45., 30., 0.], self.rc.guess_par()))
        return gp if self.dc is None else np.append(gp, self.dc.guess_par())

    def base_par(self):
        """
        Return the base (largely geometric) parameters. Returns None if
        parameters are not defined yet.
        """
        return None if self.par is None else self.par[:self.nbp]

    def par_bounds(self):
        """
        Return the lower and upper boundaries on the model parameters.

        .. todo::
            Could enable this to base the bounds on the data to be fit, but
            at the moment these are hard-coded numbers.
        """
        minx = np.amin(self.x)
        maxx = np.amax(self.x)
        miny = np.amin(self.y)
        maxy = np.amax(self.y)
        maxr = np.sqrt(max(abs(minx), maxx)**2 + max(abs(miny), maxy)**2)
        # Minimum and maximum allowed values for xc, yc, pa, inc, vsys, vrot, hrot
        lb = np.concatenate(([minx, miny, -350., 0., -300.], self.rc.lb))
        ub = np.concatenate(([maxx, maxy, 350., 89., 300.], self.rc.ub))
        return (lb, ub) if self.dc is None \
                    else (np.append(lb, self.dc.lb), np.append(ub, self.dc.ub))

    def _set_par(self, par):
        """
        Set the full parameter vector, accounting for any fixed parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
        """
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.np:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError('Must provide {0} or {1} parameters.'.format(self.np, self.nfree))
        self.par[self.free] = par.copy()

    def _init_coo(self, x, y, beam, is_fft):
        """
        Initialize the coordinate arrays and beam-smearing kernel.

        Args:
            x (`numpy.ndarray`_):
                The 2D x-coordinates at which to evaluate the model.
            y (`numpy.ndarray`_):
                The 2D y-coordinates at which to evaluate the model.
            beam (`numpy.ndarray`_):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT).
            is_fft (:obj:`bool`):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if beam is not None:
            self.beam_fft = beam if is_fft else np.fft.fftn(np.fft.ifftshift(beam))

        if self.x.shape != self.y.shape:
            raise ValueError('Input coordinates must have the same shape.')
        if self.beam_fft is not None:
            if self.x.ndim != 2:
                raise ValueError('To perform convolution, must provide 2d coordinate arrays.')
            if self.beam_fft.shape != self.x.shape:
                raise ValueError('Currently, convolution requires the beam map to have the same '
                                 'shape as the coordinate maps.')

    def _init_par(self, p0, fix):
        """
        Initialize the relevant parameter vectors that track the full set of
        model parameters and which of those are freely fit by the model.

        Args:
            p0 (`numpy.ndarray`_):
                The initial parameters for the model. Length must be
                :attr:`np`.
            fix (`numpy.ndarray`_):
                A boolean array selecting the parameters that should be fixed
                during the model fit.
        """
        if p0 is None:
            p0 = self.guess_par()
        _p0 = np.atleast_1d(p0)
        if _p0.size != self.np:
            raise ValueError('Incorrect number of model parameters.')
        self.par = _p0
        self.par_err = None
        _free = np.ones(self.np, dtype=bool) if fix is None else np.logical_not(fix)
        if _free.size != self.np:
            raise ValueError('Incorrect number of model parameter fitting flags.')
        self.free = _free
        self.nfree = np.sum(self.free)

    def model(self, par=None, x=None, y=None, beam=None, is_fft=False, cnvfftw=None,
              ignore_beam=False):
        """
        Evaluate the model.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`x` is used.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`y` is used.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT).
                  If not provided, the internal :attr:`beam_fft` is used.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            `numpy.ndarray`_, :obj:`tuple`: The velocity field model, and the
            velocity dispersion field model, if the latter is included
        """
        if x is not None or y is not None or beam is not None:
            self._init_coo(x, y, beam, is_fft)
        if self.x is None or self.y is None:
            raise ValueError('No coordinate grid defined.')
        if par is not None:
            self._set_par(par)

        r, theta = projected_polar(self.x - self.par[0], self.y - self.par[1],
                                   *np.radians(self.par[2:4]))

        # NOTE: The velocity-field construction does not include the
        # sin(inclination) term because this is absorbed into the
        # rotation curve amplitude.
        ps = self.nbp
        pe = ps + self.rc.np
        vel = self.rc.sample(r, par=self.par[ps:pe])*np.cos(theta) + self.par[4]
        if self.dc is None:
            # Only fitting the velocity field
            return vel if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb,
                                   cnvfftw=cnvfftw)[1]

        # Fitting both the velocity and velocity-dispersion field
        ps = pe
        pe = ps + self.dc.np
        sig = self.dc.sample(r, par=self.par[ps:pe])
        return (vel, sig) if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb, sig=sig,
                                   cnvfftw=cnvfftw)[1:]

    def _v_resid(self, model_vel):
        return self.kin.vel[self.vel_gpm] - model_vel[self.vel_gpm]

    def _v_chisqr(self, model_vel):
        return self._v_resid(model_vel) / self._v_err[self.vel_gpm]

    def _v_chisqr_covar(self, model_vel):
        return np.dot(self._v_resid(model_vel), self._v_ucov)

#    def _v_chisqr_covar(self, model_vel):
#        dv = self._v_resid(model_vel)
#        return np.sqrt(np.dot(dv, np.dot(self._v_icov, dv)))

    def _s_resid(self, model_sig):
        return self.kin.sig_phys2[self.sig_gpm] - model_sig[self.sig_gpm]**2

    def _s_chisqr(self, model_sig):
        return self._s_resid(model_sig) / self._s_err[self.sig_gpm]

    def _s_chisqr_covar(self, model_sig):
        return np.dot(self._s_resid(model_sig), self._s_ucov)

    def _resid(self, par):
        """
        Calculate the residuals between the data and the current model.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.

        Returns:
            `numpy.ndarray`_: Difference between the data and the model for
            all measurements.
        """
        self._set_par(par)
        vel, sig = (self.kin.bin(self.model()), None) if self.dc is None \
                        else map(lambda x : self.kin.bin(x), self.model())
        return self._v_resid(vel) if self.dc is None \
                    else np.append(self._v_resid(vel), self._s_resid(sig))

    def _chisqr(self, par):
        """
        Calculate the error-normalized residual (close to the signed
        chi-square metric) between the data and the current model.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.

        Returns:
            `numpy.ndarray`_: Difference between the data and the model for
            all measurements, normalized by their errors.
        """
        self._set_par(par)
        vel, sig = (self.kin.bin(self.model()), None) if self.dc is None \
                        else map(lambda x : self.kin.bin(x), self.model())
        if self.has_covar:
            return self._v_chisqr_covar(vel) if self.dc is None \
                    else np.append(self._v_chisqr_covar(vel), self._s_chisqr_covar(sig))
        else:
            return self._v_chisqr(vel) if self.dc is None \
                    else np.append(self._v_chisqr(vel), self._s_chisqr(sig))

    def _fit_prep(self, kin, p0, fix, scatter, sb_wgt, assume_posdef_covar, ignore_covar):
        """
        Prepare the object for fitting the provided kinematic data.

        Args:
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                The object providing the kinematic data to be fit.
            p0 (`numpy.ndarray`_):
                The initial parameters for the model. Length must be
                :attr:`np`.
            fix (`numpy.ndarray`_):
                A boolean array selecting the parameters that should be fixed
                during the model fit.
            scatter (:obj:`float`, optional):
                Introduce a fixed intrinsic-scatter term into the model. This
                single value is added in quadrature to all measurement errors
                in the calculation of the merit function. If no errors are
                available, this has the effect of renormalizing the
                unweighted merit function by 1/scatter.
            sb_wgt (:obj:`bool`):
                Flag to use the surface-brightness data provided by ``kin``
                to weight the model when applying the beam-smearing.
            assume_posdef_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, this forces the code to proceed assuming
                the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, ignore them and just use the inverse
                variance.
        """
        self._init_par(p0, fix)
        self.kin = kin
        self.x = self.kin.grid_x
        self.y = self.kin.grid_y
        # TODO: This should be changed for binned data. I.e., we should be
        # weighting by the *unbinned* surface-brightness map.
        self.sb = self.kin.remap('sb').filled(0.0) if sb_wgt else None
        self.beam_fft = self.kin.beam_fft
        self.vel_gpm = np.logical_not(self.kin.vel_mask)
        self.sig_gpm = None if self.dc is None else np.logical_not(self.kin.sig_mask)

        # Determine which errors were provided
        self.has_err = self.kin.vel_ivar is not None if self.dc is None \
                        else self.kin.vel_ivar is not None and self.kin.sig_ivar is not None
        if not self.has_err and (self.kin.vel_err is not None or self.kin.sig_err is not None):
            warnings.warn('Some errors being ignored if both velocity and velocity dispersion '
                          'errors are not provided.')
        self.has_covar = self.kin.vel_covar is not None if self.dc is None \
                            else self.kin.vel_covar is not None and self.kin.sig_covar is not None
        if not self.has_covar \
                and (self.kin.vel_covar is not None or self.kin.sig_covar is not None):
            warnings.warn('Some covariance matrices being ignored if both velocity and velocity '
                          'dispersion covariances are not provided.')
        if ignore_covar:
            # Force ignoring the covariance
            # TODO: This requires that, e.g., kin.vel_ivar also be defined...
            self.has_covar = False

        # Check the intrinsic scatter input
        self.scatter = None
        if scatter is not None:
            self.scatter = np.atleast_1d(scatter)
            if self.scatter.size > 2:
                raise ValueError('Should provide, at most, one scatter term for each kinematic '
                                 'moment being fit.')
            if self.dc is not None and self.scatter.size == 1:
                warnings.warn('Using single scatter term for both velocity and velocity '
                              'dispersion.')
                self.scatter = np.array([scatter, scatter])

        # Set the internal error attributes
        if self.has_err:
            self._v_err = np.sqrt(inverse(self.kin.vel_ivar))
            self._s_err = None if self.dc is None \
                                else np.sqrt(inverse(self.kin.sig_phys2_ivar))
            if self.scatter is not None:
                self._v_err = np.sqrt(self._v_err**2 + self.scatter[0]**2)
                if self.dc is not None:
                    self._s_err = np.sqrt(self._s_err**2 + self.scatter[1]**2)
        elif not self.has_err and not self.has_covar and self.scatter is not None:
            self.has_err = True
            self._v_err = np.full(self.kin.vel.shape, self.scatter[0], dtype=float)
            self._s_err = None if self.dc is None \
                                else np.full(self.kin.sig.shape, self.scatter[1], dtype=float)
        else:
            self._v_err = None
            self._s_err = None

        # Set the internal covariance attributes
        if self.has_covar:
            # Construct the matrices used to calculate the merit function in
            # the presence of covariance.
            if not assume_posdef_covar:
                # Force the matrices to be positive definite
                print('Forcing vel covar to be pos-def')
                vel_pd_covar = impose_positive_definite(self.kin.vel_covar[
                                                                np.ix_(self.vel_gpm,self.vel_gpm)])
                # TODO: This needs to be fixed to be for sigma**2, not sigma
                print('Forcing sig covar to be pos-def')
                sig_pd_covar = None if self.dc is None \
                                    else impose_positive_definite(self.kin.sig_covar[
                                                                np.ix_(self.sig_gpm,self.sig_gpm)])
                
            else:
                vel_pd_covar = self.kin.vel_covar[np.ix_(self.vel_gpm,self.vel_gpm)]
                sig_pd_covar = None if self.dc is None \
                                else self.kin.sig_covar[np.ix_(self.vel_gpm,self.vel_gpm)]

            if self.scatter is not None:
                # A diagonal matrix with only positive values is, by
                # definition, positive difinite; and the sum of two positive
                # definite matrices is also positive definite.
                vel_pd_covar += np.diag(np.full(vel_pd_covar.shape[0], self.scatter[0]**2,
                                                dtype=float))
                if self.dc is not None:
                    sig_pd_covar += np.diag(np.full(sig_pd_covar.shape[0], self.scatter[1]**2,
                                                    dtype=float))

            self._v_ucov = cinv(vel_pd_covar, upper=True)
            self._s_ucov = None if sig_pd_covar is None else cinv(sig_pd_covar, upper=True)
        else:
            self._v_ucov = None
            self._s_ucov = None

        return self._chisqr if self.has_err or self.has_covar else self._resid

    def lsq_fit(self, kin, sb_wgt=False, p0=None, fix=None, scatter=None, verbose=0,
                assume_posdef_covar=False, ignore_covar=True):
        """
        Use `scipy.optimize.least_squares`_ to fit the model to the provided
        kinematics.

        Once complete, the best-fitting parameters are saved to :attr:`par`
        and the parameter errors (estimated by the parameter covariance
        matrix constructed as a by-product of the least-squares fit) are
        saved to :attr:`par_err`.

        Args:
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                Object with the kinematic data to fit.
            sb_wgt (:obj:`bool`, optional):
                Flag to use the surface-brightness data provided by ``kin``
                to weight the model when applying the beam-smearing.
            p0 (`numpy.ndarray`_, optional):
                The initial parameters for the model. Length must be
                :attr:`np`.
            fix (`numpy.ndarray`_, optional):
                A boolean array selecting the parameters that should be fixed
                during the model fit.
            scatter (:obj:`float`, `numpy.ndarray`_, optional):
                Introduce a fixed intrinsic-scatter term into the model. This
                single value per kinematic moment (v, sigma) is added in
                quadrature to all measurement errors in the calculation of
                the merit function. If no errors are available, this has the
                effect of renormalizing the unweighted merit function by
                1/scatter.
            verbose (:obj:`int`, optional):
                Verbosity level to pass to `scipy.optimize.least_squares`_.
            assume_posdef_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, this forces the code to proceed assuming
                the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, ignore them and just use the inverse
                variance.
        """
        # Prepare to fit the data; the returned object is the callable function
        # used to generate the figure-of-merit.
        fom = self._fit_prep(kin, p0, fix, scatter, sb_wgt, assume_posdef_covar, ignore_covar)
        # Parameter boundaries
        lb, ub = self.par_bounds()
        # This means the derivative of the merit function wrt each parameter is
        # determined by a 1% change in each parameter.
        diff_step = np.full(self.np, 0.01, dtype=float)
        # Run the optimization
        result = optimize.least_squares(fom, self.par[self.free], method='trf',
                                        bounds=(lb[self.free], ub[self.free]), 
                                        diff_step=diff_step[self.free], verbose=verbose)
        # Save the best-fitting parameters
        self._set_par(result.x)
        try:
            # Calculate the nominal parameter errors using the precision matrix
            cov = cov_err(result.jac)
            self.par_err = np.zeros(self.np, dtype=float)
            self.par_err[self.free] = np.sqrt(np.diag(cov))
        except:
            warnings.warn('Unable to compute parameter errors from precision matrix.')
            self.par_err = None


