
from IPython import embed

import numpy as np

from scipy import optimize

from .oned import HyperbolicTangent
from .geometry import projected_polar
from .beam import smear
from .util import cov_err


def rotcurveeval(x,y,vmax,inc,pa,h,vsys=0,xc=0,yc=0,reff=1):
    '''
    Evaluate a simple tanh rotation curve with asymtote vmax, inclination inc
    in degrees, position angle pa in degrees, rotation scale h, systematic
    velocity vsys, and x and y offsets xc and yc. Returns array in same shape
    as input x andy.
    '''

    inc, pa = np.radians([inc,pa])
    r,th = projected_polar(x-xc,y-yc, pa, inc)
    r /= reff
    # TODO: Why was there a negative here? (it used to be `-vmax`)
    model = vmax * np.tanh(r/h) * np.cos(th) * np.sin(inc) + vsys
    return model


class AxisymmetricDisk:
    """
    Simple model for an axisymmetric disk.

    Base parameters are xc, yc, pa, inc, vsys.

    Full parameters include number of *projected* rotation curve
    parameters.
    """
    def __init__(self, rc=None):
        # Rotation curve
        self.rc = HyperbolicTangent() if rc is None else rc

        # Number of "base" parameters
        self.nbp = 5
        # Total number parameters
        self.np = self.nbp + self.rc.np
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

    def guess_par(self):
        return np.concatenate(([0., 0., 45., 30., 0.], self.rc.guess_par()))

    def base_par(self):
        """
        Return the base parameters.  Returns None if parameters are not defined yet.
        """
        return None if self.par is None else self.par[:self.nbp]

    def par_bounds(self):
        minx = np.amin(self.x)
        maxx = np.amax(self.x)
        miny = np.amin(self.y)
        maxy = np.amax(self.y)
        maxr = np.sqrt(max(abs(minx), maxx)**2 + max(abs(miny), maxy)**2)
        rclb, rcub = self.rc.par_bounds(maxr)
        # Minimum and maximum allowed values for xc, yc, pa, inc, vsys, vrot, hrot
        return np.concatenate(([minx, miny, -350., 0., -300.], rclb)), \
               np.concatenate(([maxx, maxy, 350., 89., 300.], rcub))

    def _set_par(self, par):
        """
        Set the parameters by accounting for any fixed parameters.
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

    def model(self, par=None, x=None, y=None, beam=None, is_fft=False):
        """
        Evaluate the model.
        """
        if x is not None or y is not None or beam is not None:
            self._init_coo(x, y, beam, is_fft)
        if self.x is None or self.y is None:
            raise ValueError('No coordinate grid defined.')
        if par is not None:
            self._set_par(par)

        r, theta = projected_polar(self.x - self.par[0], self.y - self.par[1],
                                   *np.radians(self.par[2:4]))

        # NOTE: This doesn't include the sin(inclination) term because
        # this is absorbed into the rotation curve amplitude.
        vel = self.rc.sample(r, par=self.par[self.nbp:])*np.cos(theta) + self.par[4]
        return vel if self.beam_fft is None else smear(vel, self.beam_fft, beam_fft=True)[1]

    def _resid(self, par):
        self._set_par(par)
        return self.kin.vel[self.vel_gpm] - self.kin.bin(self.model())[self.vel_gpm]

    def _chisqr(self, par): 
        return self._resid(par) * np.sqrt(self.kin.vel_ivar[self.vel_gpm])

    def _fit_prep(self, kin, p0, fix):
        self._init_par(p0, fix)
        self.kin = kin
        self.x = self.kin.grid_x
        self.y = self.kin.grid_y
        self.beam_fft = self.kin.beam_fft
        self.vel_gpm = np.logical_not(self.kin.vel_mask)
        return self._resid if self.kin.vel_ivar is None else self._chisqr

    def lsq_fit(self, kin, p0=None, fix=None, verbose=0):
        """
        Use least_squares to fit kinematics.
        """
        fom = self._fit_prep(kin, p0, fix)
        lb, ub = self.par_bounds()
        diff_step = np.full(self.np, 0.1, dtype=float)
        result = optimize.least_squares(fom, self.par[self.free], method='trf',
                                        bounds=(lb[self.free], ub[self.free]), 
                                        diff_step=diff_step[self.free], verbose=verbose)
        self._set_par(result.x)

        cov = cov_err(result.jac)
        self.par_err = np.zeros(self.np, dtype=float)
        self.par_err[self.free] = np.sqrt(np.diag(cov))


        