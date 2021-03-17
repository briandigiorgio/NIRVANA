"""
Provides a random set of utility methods.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import sparse, stats, optimize
from matplotlib import pyplot, patches

from astropy.stats import sigma_clip

from . import util
from ..util import plot


class IntrinsicScatter:
    """
    Fit/Measure the intrinsic scatter in fit residuals.

    Args:
        resid (`numpy.ndarray`_):
            Fit residuals.
        err (`numpy.ndarray`_, optional):
            1-sigma errors in the measurements. Must have the same shape as
            ``resid``. If not provided, scatter is determined based on the
            un-normalized residuals.
        covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_, optional):
            Data covariance matrix. If provided, ``err`` is ignored and must
            be a square matrix with each axis having the same length as the
            flattened ``resid`` array.
        gpm (`numpy.ndarray`_, optional):
            Pre-existing "good-pixel" mask for the provided residuals
            selecting the values to use in the intrinsic scatter
            measurements. Must have the same shape as ``resid``.
        npar (:obj:`int`, optional):
            Number of model parameters used to fit the data. Used to
            determine the degrees of freedom.
        assume_posdef_covar (:obj:`bool`, optional):
            If ``covar`` is provided, assume it is already positive-definite.
            If not, the instantiation of the object passes the covariance
            matrix to :func:`~nirvana.data.util.impose_positive_definite` to
            ensure that the covariance matrix is positive-definite. Ignored
            if ``covar`` is not provided.
    """
    def __init__(self, resid, err=None, covar=None, gpm=None, npar=0, assume_posdef_covar=False):

        self.resid = resid
        self.size = self.resid.size
        self.err = err
        if self.err is not None and self.err.size != self.size:
            raise ValueError('Size of the error array must match the residual array.')
        self.covar = covar if covar is None or assume_posdef_covar \
                        else util.impose_positive_definite(sparse.csr_matrix(covar))
        if self.covar is not None and self.covar.shape != (self.size,self.size):
            raise ValueError('Covariance array must have shape ({0},{0}).'.format(self.size))
        if isinstance(self.covar, sparse.csr_matrix):
            self.covar = self.covar.toarray()
        self.gpm = np.ones(self.size, dtype=bool) if gpm is None else gpm.copy()
        if self.gpm.size != self.size:
            raise ValueError('Size of the good-pixel mask must match the residual array.')
        self.inp_gpm = self.gpm.copy()
        self.npar = npar

        # Work-space arrays
        self._x = None
        self._res = None
        self._rsq = None
        self._cov = None
        self._var = None
        self._rho = None
        self.fixed_rho = False
        self.debug = False

    def _merit_vec_err(self, x):
        return self._res / np.sqrt(self._var + x[0]**2)

    def _merit_err(self, x):
        """
        Calculate the merit function without covariance.
        """
        merit = abs(np.sum(self._rsq / (self._var + x[0]**2)) - self._dof)
        if self.debug:
            print('Par={0:7.3f}, Merit={1:9.3e}'.format(x[0], merit))
        return merit
        
    def _adj_covar(self, x):
        if self.fixed_rho:
            _var = self._var + x[0]**2
            return self._rho*np.sqrt(_var[:,None]*_var[None,:])
        return self._cov + self._rho * x[0]**2

    def _merit_vec_covar(self, x):
        return np.dot(self._res, util.cinv(self._adj_covar(x), upper=True))

    def _merit_covar(self, x):
        """
        Calculate the merit function with covariance.
        """
        merit = abs(np.dot(self._res, np.dot(util.cinv(self._adj_covar(x)), self._res)) - self._dof)
        if self.debug:
            print('Par={0:7.3f}, Merit={1:9.3e}'.format(x[0], merit))
        return merit
        
    def _fit_init(self, sig0=None, fixed_rho=False):
        """
        Initialize the fitting workspace objects.
        """
        self._dof = np.sum(self.gpm) - self.npar
        self._res = self.resid[self.gpm]
        self._rsq = self._res**2
        self.fixed_rho = fixed_rho
        if self.covar is None:
            # NOTE: Shouldn't get here if self.err is None; see fit()
            self._var = self.err[self.gpm]**2
            self._cov = None
            self._rho = None
        else:
            self._cov = self.covar[np.ix_(self.gpm,self.gpm)]
            self._var = np.diag(self._cov)
            self._rho = self._cov / np.sqrt(self._var[:,None]*self._var[None,:]) \
                            if self.fixed_rho else np.identity(np.sum(self.gpm), dtype=float)
        self._x = np.array([util.sigma_clip_stdfunc_mad(self._res/np.sqrt(self._var))
                                if sig0 is None else sig0])

    def fit(self, sig0=None, sigma_rej=5, rejiter=None, fixed_rho=False, verbose=0):
        """
        Find the intrinsic scatter in the residuals.

        If no errors are provided (either independent or in a covariance
        matrix; see the instantiation of the object), the result is simply a
        clipped standard deviation.

        Rejections use `astropy.stats.sigma_clip` with
        :func:`~nirvana.data.util.sigma_clip_stdfunc_mad` as the method used
        to compute the standard deviation.

        Algorithm is:
            - Perform an iterative sigma-clipping of the error-weighted
              residuals using `astropy.stats.sigma_clip`; uses ``sigma_rej``
              and ``rejiter``.
            - Perform a least-squares fit of the unclipped data to find the
              best-fitting intrinsic scatter.

        To iteratively perform the fit by repeating the above algorithm
        multiple times, use :func:`iter_fit`.

        Args:
            sig0 (scalar-like, optional):
                Starting estimate of the 1-sigma intrinsic scatter.
            sigma_rej (scalar-like, optional):
                The standard deviation rejection threshold.
            rejiter (:obj:`int`, optional):
                For a fixed set of data and intrinsic scatter value, this is
                the maximum number of rejection iterations performed. If
                None, iterations are performed until no more rejections occur
                (see `astropy.stats.sigma_clip`_).
            fixed_rho (:obj:`bool`, optional):
                Force the correlation matrix of the *intrinsic scatter* term
                to be identical to the correlation matrix of the data.
                Ignored if no covariance matrix is available. If False, the
                correlation matrix of the intrinsic scatter term is the
                identity matrix. *Barring further testing, we recommend that
                this always be False.*
            verbose (:obj:`int`, optional):
                Verbosity level.  0 mean no output.

        Returns:
            :obj:`tuple`: Returns the value of the intrinsic scatter, a
            `numpy.ndarray`_ selecting the rejected data points, and a
            `numpy.ndarray`_ selecting all good data points; the latter is
            the intersection of the input good-pixel mask and those data
            *not* rejected by the function.
        """
        if sig0 is not None and not sig0 > 0:
            warnings.warn('Initial guess for sigma must be greater than 0.  Ignoring input.')
            _sig0 = None
        else:
            _sig0 = sig0

        self.debug = verbose > 0
        _sigma_rej = float(sigma_rej)

        # Do a first pass at clipping the data based on the standard deviation
        # in the (error-normalized) residuals
        _chi = self.resid.copy()
        if _sig0 is not None and _sig0 > 0:
            _chi[self.gpm] /= _sig0 if self.err is None \
                                else np.sqrt(_sig0**2 + self.err[self.gpm]**2)
        elif self.err is not None:
            _chi[self.gpm] /= self.err[self.gpm]
        clip = sigma_clip(np.ma.MaskedArray(_chi, mask=np.logical_not(self.gpm)),
                          sigma=_sigma_rej, stdfunc=util.sigma_clip_stdfunc_mad, maxiters=rejiter)
        clipped = np.ma.getmaskarray(clip)
        self.rej = self.inp_gpm & clipped
        self.gpm = np.logical_not(clipped)

        if self.err is None and self.covar is None:
            # No need to fit; just return the clipped standard deviation.
            self.sig = np.std(self.resid[self.gpm])
            return self.sig, self.rej, self.gpm

        # Initialize the fit workspace objects
        self._fit_init(sig0=_sig0, fixed_rho=fixed_rho)

        # Assign the merit function to use based on the availability of the
        # covariance
        fom = self._merit_err if self.covar is None else self._merit_covar

        # Run the fit
        result = optimize.least_squares(fom, self._x, method='lm', diff_step=np.array([1e-5]),
                                        verbose=verbose)
        # Save the result
        self.sig = abs(result.x[0])
        # TODO: Save the success somehow?

        return self.sig, self.rej, self.gpm

    def iter_fit(self, sig0=None, sigma_rej=5, rejiter=None, fixed_rho=False, verbose=0,
                 fititer=None, sticky=False):
        """
        Iteratively fit the scatter by clipping data and refitting the
        intrinsic scatter.

        This method just runs multiple iterations of :func:`fit`.

        Args:
            sig0 (scalar-like, optional):
                Starting estimate of the 1-sigma intrinsic scatter.
            sigma_rej (scalar-like, optional):
                The standard deviation rejection threshold.
            rejiter (:obj:`int`, optional):
                For a fixed set of data and intrinsic scatter value, this is
                the maximum number of rejection iterations performed. If
                None, iterations are performed until no more rejections occur
                (see `astropy.stats.sigma_clip`_).
            fixed_rho (:obj:`bool`, optional):
                Force the correlation matrix of the *intrinsic scatter* term
                to be identical to the correlation matrix of the data.
                Ignored if no covariance matrix is available. If False, the
                correlation matrix of the intrinsic scatter term is the
                identity matrix. *Barring further testing, we recommend that
                this always be False.*
            verbose (:obj:`int`, optional):
                Verbosity level.  0 mean no output.
            fititer (:obj:`int`, optional):
                Number of iterations of :func:`fit` to iteration. If None,
                iterate until no data are rejected. Setting ``fititer`` to 1,
                is identical to running :func:`fit` once.
            sticky (:obj:`bool`, optional):
                Cumulatively build up the set of rejected data points through
                each iteration. If False, any data rejected in a give
                iteration is still included in the analysis of subsequent
                iterations.

        Returns:

            :obj:`tuple`: Returns the value of the intrinsic scatter, a
            `numpy.ndarray`_ selecting the rejected data points, and a
            `numpy.ndarray`_ selecting all good data points; the latter is
            the intersection of the input good-pixel mask and those data
            *not* rejected by the function.
        
        """
        # In debug mode?
        self.debug = verbose > 0
        # Perform the first fit
        sig, rej, gpm = self.fit(sig0=sig0, sigma_rej=sigma_rej, rejiter=rejiter,
                                 fixed_rho=fixed_rho, verbose=verbose)
        if self.debug:
            print('sig={0:5.2f}, Nrej={1:>3}, Nrej_diff=   0'.format(sig, np.sum(rej)))
        i = 1
        nrej = np.sum(rej)
        _rej = rej.copy()
        while np.sum(_rej) != 0 and (fititer is None or i < fititer):
            if not sticky:
                self.gpm = self.inp_gpm.copy()
            _rej = rej.copy()
            sig, rej, gpm = self.fit(sig0=sig, sigma_rej=sigma_rej, rejiter=rejiter,
                                     fixed_rho=fixed_rho, verbose=verbose)
            _rej = rej != _rej
            if self.debug:
                print('sig={0:5.2f}, Nrej={1:>3}, Nrej_diff={2:>3}'.format(
                            sig, np.sum(rej), np.sum(_rej)))
            i += 1
        return sig, rej, gpm

#    def sample_merit(self):
#        x = np.logspace(-1, 2, 50)
#        chisqr_ncv = np.zeros(x.size, dtype=float)
#        chisqr_icv = np.zeros(x.size, dtype=float)
#        chisqr_fcv = np.zeros(x.size, dtype=float)
#        for i in range(x.size):
#            chisqr_ncv[i] = self._merit_err([x[i]])
#            chisqr_fcv[i] = self._merit_covar([x[i]])
#        self._rho = np.identity(np.sum(self.gpm), dtype=float)
#        for i in range(x.size):
#            chisqr_icv[i] = self._merit_covar([x[i]])
#
#        from matplotlib import pyplot
#        pyplot.plot(x, chisqr_ncv)
#        pyplot.plot(x, chisqr_icv)
#        pyplot.plot(x, chisqr_fcv)
#        pyplot.xscale('log')
#        pyplot.show()

    def show(self, sig=None, rej=None, gpm=None, ofile=None, title=None):
        """
        """
        # Save the current gpm so that the method doesn't change it.
        sv_gpm = self.gpm.copy()
        if gpm is not None:
            self.gpm = gpm

        # Initialize the fit workspace objects
        _rej = self.rej if rej is None else rej
        self.gpm &= np.logical_not(_rej)
        self._fit_init(sig0=self.sig if sig is None else sig, fixed_rho=self.fixed_rho)

        # Assign the merit function to use based on the availability of the
        # covariance
        fom_vec = self._merit_vec_err if self.covar is None else self._merit_vec_covar
        fom = self._merit_err if self.covar is None else self._merit_covar

        mean_eps = np.mean(np.sqrt(self._var))

        enres_def = fom_vec(np.array([1.]))
        rng_def = util.growth_lim(enres_def, 0.95, 1.1, midpoint=0.0)
        mean_enres_def = np.mean(enres_def)
        sigma_enres_def = np.std(enres_def)

        enres = fom_vec(self._x)
        rng = util.growth_lim(enres, 0.99, 1.3, midpoint=0.0)
        mean_enres = np.mean(enres)
        sigma_enres = np.std(enres)
        nrej = np.sum(_rej)
        ntot = np.sum(self.inp_gpm)

        logformatter = plot.get_logformatter()

        w,h = pyplot.figaspect(1)
        fig = pyplot.figure(figsize=(2*w,h))

        ax = plot.init_ax(fig, [0.03, 0.1, 0.45, 0.87])
        by, bx, _ = ax.hist(enres_def, bins=100, range=rng_def, density=True, color='k', lw=0,
                            alpha=0.3, zorder=4, histtype='stepfilled')
        maxy = np.amax(by)
        by, bx, _ = ax.hist(enres, bins=100, range=rng_def, density=True, color='k', lw=0,
                            alpha=0.6, zorder=5, histtype='stepfilled')
        maxy = max(maxy, np.amax(by))
        bc = bx[:-1]+np.diff(bx)/2
        ax.step(bc, util.pixelated_gaussian(bc, density=True), where='mid', color='C3',
                zorder=6)
        ax.set_xlim(rng_def)
        ax.set_ylim([0, 1.05*maxy])
        plot.rotate_y_ticks(ax, 90., 'center')
        ax.text(0.5, -0.07, r'$\Delta/\epsilon$', ha='center', va='center',
                transform=ax.transAxes)

        if title is not None:
            ax.text(0.02, 0.96, title, ha='left', va='center', transform=ax.transAxes, fontsize=12)

        ax = plot.init_ax(fig, [0.54, 0.1, 0.45, 0.87])
        ax.set_xlim([0., rng[1]])
        ax.set_ylim([0.9*2*(1 - stats.norm.cdf(rng[1])), 1.05])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(logformatter)
        plot.rotate_y_ticks(ax, 90., 'center')
        abs_enres_def = np.absolute(enres_def)
        srt = np.argsort(abs_enres_def)
        ax.step(abs_enres_def[srt], 1-np.arange(srt.size)/srt.size, where='post', color='0.6',
                zorder=5)
        abs_enres = np.absolute(enres)
        srt = np.argsort(abs_enres)
        ax.step(abs_enres[srt], 1-np.arange(srt.size)/srt.size, where='post', color='0.3',
                zorder=6)

        ax.plot(abs_enres[srt], 2 - stats.norm.cdf(abs_enres[srt])*2, color='C3',
                zorder=4)

        ax.text(-0.07, 0.5, '1-Growth', ha='center', va='center', rotation='vertical',
                transform=ax.transAxes)
        ax.text(0.5, -0.07, r'$|\Delta|/\epsilon$', ha='center', va='center',
                transform=ax.transAxes)
        ax.axhline(y=1-0.6826, color='0.5', linestyle='--', zorder=3)
        ax.axhline(y=1-0.9544, color='0.5', linestyle='--', zorder=3)
        ax.axhline(y=1-0.9973, color='0.5', linestyle='--', zorder=3)

        ax.add_patch(patches.Rectangle((0.03,0.02), 0.29, 0.50, facecolor='w', lw=0,
                                       edgecolor='none', zorder=7, alpha=0.7,
                                       transform=ax.transAxes))

        ax.text(0.04, 0.49, r'$N_{\rm tot}$:', ha='left', va='center',
                transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.49, f'{ntot}', ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.45, r'$N_{\rm rej}$:', ha='left', va='center',
                transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.45, f'{nrej}', ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.41, r'$\nu$:', ha='left', va='center',
                transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.41, f'{self._dof}', ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.37, r'$\langle\epsilon\rangle$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.37, '{0:.2f}'.format(mean_eps), ha='right',
                va='center', transform=ax.transAxes, zorder=8)

        ax.text(0.04, 0.31, r'$\langle\Delta/\epsilon\rangle_0$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.31, '{0:.2f}'.format(mean_enres_def), ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.27, r'$\sigma_0$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.27, '{0:.2f}'.format(sigma_enres_def), ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.23, r'$|\chi_0^2-\nu|$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.23, '{0:.0f}'.format(fom(np.array([0]))), ha='right',
                va='center', transform=ax.transAxes, zorder=8)

        ax.text(0.04, 0.17, r'$\epsilon_i$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.17, '{0:.2f}'.format(self._x[0]), ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.13, r'$\langle\Delta/\epsilon\rangle_i$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.13, '{0:.2f}'.format(mean_enres), ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.09, r'$\sigma_i$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.09, '{0:.2f}'.format(sigma_enres), ha='right',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.04, 0.05, r'$|\chi_i^2-\nu|$:', ha='left',
                va='center', transform=ax.transAxes, zorder=8)
        ax.text(0.31, 0.05, '{0:.0f}'.format(fom(self._x)), ha='right',
                va='center', transform=ax.transAxes, zorder=8)

        if ofile is not None:
            fig.canvas.print_figure(ofile, bbox_inches='tight')
        else:
            pyplot.show()
        fig.clear()
        pyplot.close(fig)

        # Revert to original gpm
        self.gpm = sv_gpm

