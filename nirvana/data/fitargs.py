#!/usr/bin/env python

from IPython import embed

import numpy as np
from scipy.optimize import leastsq, least_squares
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from ..models.geometry import projected_polar
from ..models.axisym import rotcurveeval, AxisymmetricDisk
from ..models.beam import ConvolveFFTW

class FitArgs:
    '''
    Parent class for :class:`nirvana.data.kinematics.Kinematics` class to hold
    information that is necessary for fitting but is not related to the actual
    data.
    '''

    def setnglobs(self, nglobs):
        '''
        Set number of global variables in fit. Global variables are assumed to
        be inclination, first order position angle, second order position angle,
        systemic velocity, and (optionally) x and y coordinates of the center.
        
        Should be 4 when not fitting the position of the center and 6 when the
        center position is being fit.

        This should probably be turned into a flag.

        Args:
            nglobs (:obj:`int`):
                Number of global parameters. Should be 4 if center position is
                fixed and 6 if center position is not fit.
        '''

        self.nglobs = nglobs

    def setweight(self, weight):
        '''
        Set weight to assign to smoothness of rotation curves in fit.

        Args:
            weight (:obj:`float`):
                Normalization factor to multiply smoothing penalty by.
        '''

        self.weight = weight

    def setedges(self, inc, maxr=None, nbin=False):
        '''
        Construct array of bin edges for the galaxy.

        Defaults to making bins that Nyquist sample the FWHM of the galaxy at
        the minor axis where the elliptical bins are the narrowest, but will not
        let the deprojected bin size get larger than the FWHM itself. This
        transition should kick in at at an inclination of 60 degrees. Can also
        be manually set to a specific number of bins.

        Args:
            inc (:obj:`float`):
                Inclination to construct the elliptical bins at. If `nbin ==
                True`, this is instead the number of bins to make (must be an
                :obj:`int`)
            maxr (:obj:`float`, optional):
                Maximum radius for the bins in effective radii. If not
                specified, it will default to the maximum unmasked radius of
                the galaxy.
            nbin (:obj:`bool`):
                Flag for whether or not to set the number of bins manually.
        '''

        if maxr is None: 
            #mask outside pixels if necessary
            if self.bordermask is not None:
                x = self.x * (1-self.bordermask)
                y = self.y * (1-self.bordermask)
            else: x,y = (self.x, self.y)

            #calculate maximum radius of image if none is given
            maxr = np.max(np.sqrt(x**2 + y**2))/self.reff
        self.maxr = maxr
        maxr *= self.reff #change to arcsec

        #specify number of bins manually if desired
        if nbin: self.edges = np.linspace(0, maxr, inc+1)/self.reff

        #calculate nyquist bin width based off fwhm and inc
        else:
            binwidth = min(self.fwhm/2/np.cos(np.radians(inc)), self.fwhm)
            self.edges = np.arange(0, maxr, binwidth)/self.reff

    def setdisp(self, disp):
        '''
        Whether or not to fit dispersion.

        Args:
            disp (:obj:`bool`):
                Flag for whether to fit dispersion.
        '''

        self.disp = disp

    def setmix(self, mix):
        self.mix = mix

    def getguess(self, fill=10, clip=True):
        '''
        Generate a set of guess parameters for the galaxy using a simple least
        squares fit.

        Takes the velocity data associated with the object and fits a hyperbolic
        tangent rotation curve to the data to get global parameters then uses
        predefined edges to get guesses of the first order tangential velocity
        at each of the bin centers (if :attr:`edges` has been defined). Just
        uses a fill value for second order velocities since they can't be fit
        simply.
        
        Isn't meant to be totally accurate (especially for galaxies with large
        kinematic irregularities), just meant to give a good starting point for
        fitting and provide an inclination for defining bins.

        Args:
            fill (:obj:`float`, optional):
                Fill value for second order velocities.
            clip (:obj:`bool`, optional):
                Whether to clean up the data. if `True`, it will do a 7 sigma
                clip on the residuals from the initial round of the fit and
                the chi squared of that fit. This is arbitrary but seems to
                remove regions of bad data without removing regions of
                legitimate but weird data.

        Returns:
            :obj:`tuple`: Tuple of guesses for the parameters. Will be in the
            format laid out in :func:`nirvana.fitting.fit` and expected by
            :func:`nirvana.fitting.unpack` if :attr:`edges` are given. Otherwise
            it will just be [asymptotic velocity, inclination, position angle,
            rotation scale, systemic velocity]. All angles are in degrees and
            all velocities are in consistent units.
        '''

        if self.vel_ivar is None: ivar = np.ones_like(self.vel)
        else: ivar = self.vel_ivar

        #quick fit of data
        if clip: self.clip()
        fit = AxisymmetricDisk()
        fit.lsq_fit(self)

        #get fit params
        xc, yc, pa, inc, vsys, vsini, h = fit.par
        vmax = vsini/np.sin(np.radians(inc))

        #generate model velocity field, start assembling array of guess values
        #model = rotcurveeval(self.grid_x,self.grid_y,vmax,inc,pa,h,vsys,reff=self.reff)
        model = fit.model()
        guess = [inc,pa,pa,vsys,0,0,0]
        if self.nglobs == 6: guess += [0,0]

        #if edges have not been defined, just return global parameters
        if not hasattr(self, 'edges'): return [vmax,inc,pa,h,vsys]

        #define polar coordinates and normalize to effective radius
        r,th = projected_polar(self.grid_x, self.grid_y, *np.radians([pa,inc]))
        r /= self.reff

        #iterate through bins and get vt value for each bin, 
        #dummy value for v2t and v2r since there isn't a good guess
        nbin = len(self.edges)
        vt = np.zeros(nbin)
        v2t = np.array([fill] * nbin)
        v2r = np.array([fill] * nbin)
        for i in range(1,nbin):
            cut = (r > self.edges[i-1]) * (r < self.edges[i])
            vt[i] = np.max(model[cut])
            guess += [vt[i], v2t[i], v2r[i]]
        
        #clean and return
        guess = np.array(guess)
        guess[np.isnan(guess)] = 100
        self.guess = guess
        return self.guess
