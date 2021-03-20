#!/usr/bin/env python

from IPython import embed

import numpy as np
from scipy.optimize import leastsq, least_squares
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from ..models.geometry import projected_polar, asymmetry
from ..models.axisym import AxisymmetricDisk
from ..models.beam import ConvolveFFTW

class FitArgs:
    '''
    Parent class for :class:`nirvana.data.kinematics.Kinematics` class to hold
    information that is necessary for fitting but is not related to the actual
    data.
    '''

    def __init__(self, nglobs=6, weight=10, edges=None, disp=True,
            fixcent=True, mix=False, guess=None, nbins=None, bounds=None,
            arc=None, asymmap=None, maxr=None, noisefloor=5, penalty=100,
            npoints=500, smearing=True):

        self.nglobs = nglobs
        self.weight = weight
        self.edges = edges
        self.disp = disp
        self.fixcent = fixcent
        self.mix = mix
        self.guess = guess
        self.nbins = nbins
        self.bounds = bounds
        self.maxr = maxr
        self.noise_floor = noisefloor
        self.penalty = penalty
        self.npoints = npoints
        self.smearing = smearing


    def setedges(self, inc, maxr=0, nbin=False, clipmasked=True):
        '''
        Construct array of bin edges for the galaxy.

        Defaults to making bins that Nyquist sample the FWHM of the galaxy at
        the minor axis where the elliptical bins are the narrowest, but will not
        let the deprojected bin size get larger than the FWHM itself. This
        transition should kick in at at an inclination of 60 degrees. Can also
        be manually set to a specific number of bins.

        Args:
            inc (:obj:`float`):
                Inclination in degrees to construct the elliptical bins at. If
                `nbin == True`, this is instead the number of bins to make
                (must be an :obj:`int`)
            maxr (:obj:`float`, optional):
                Maximum radius for the bins in effective radii. If not
                specified, it will default to the maximum unmasked radius of
                the galaxy.
            nbin (:obj:`bool`, optional):
                Whether or not to set the number of bins manually.
            clipmasked (:obj:`bool`, optional):
                Whether to clip off data outside of the last bin and in bins
                where too many spaxels are masked 
        '''

        #figure out max radius if not set
        if maxr == 0: 
            if self.maxr is not None: maxr = self.maxr #check if it is set
            else:
                #mask outside pixels if necessary
                if self.bordermask is not None:
                    x = self.x * (1-self.bordermask)
                    y = self.y * (1-self.bordermask)
                else: x,y = (self.x, self.y)

                #calculate maximum radius of image if none is given
                maxr = np.max(np.sqrt(x**2 + y**2))/self.reff
        maxr *= self.reff #change to arcsec

        #specify number of bins manually if desired
        if nbin: self.edges = np.linspace(0, maxr, inc+1)/self.reff

        #calculate nyquist bin width based off fwhm and inc
        else:
            binwidth = min(self.fwhm/2/np.cos(np.radians(inc)), self.fwhm)
            self.edges = np.arange(0, maxr, binwidth)/self.reff

        #clip outer bins that have too many masked spaxels
        if clipmasked:
            #find radial coordinates of each spaxel
            guess = self.getguess(simple=True)
            r,th = projected_polar(self.x, self.y, *np.radians((guess[2], inc)))
            r /= self.reff
            mr = np.ma.array(r, mask=self.vel_mask)

            #calculate the number of spaxels in each bin 
            #and what fraction of them are masked
            nspax = np.zeros(len(self.edges)-1)
            maskfrac = np.zeros_like(nspax)
            for i in range(len(self.edges)-1):
                mcut = (mr > self.edges[i]) * (mr < self.edges[i+1])
                cut = (r > self.edges[i]) * (r < self.edges[i+1])
                nspax[i] = np.sum(mcut)
                maskfrac[i] = np.sum(self.vel_mask[cut])/cut.sum()
            
            #cut bins where too many spaxels are masked
            bad = (maskfrac > .75) #| (nspax < 10)
            self.edges = [self.edges[0], *self.edges[1:][~bad]]

            #mask spaxels outside last bin edge
            self.vel_mask[r > self.edges[-1]] = True

    def getguess(self, fill=10, clip=False, simple=False):
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
            simple (:obj:`bool`, optional):
                Whether to produce the simple 6 parameter fit regardless of
                whether the edges have been edfined 

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
        model = fit.model()
        guess = [inc,pa,pa,vsys]
        if hasattr(self, 'nglobs') and self.nglobs == 6: guess += [xc, yc]

        #if edges have not been defined, just return global parameters
        if not hasattr(self, 'edges') or simple: return [vmax,inc,pa,h,vsys]

        #define polar coordinates and normalize to effective radius
        r,th = projected_polar(self.grid_x, self.grid_y, *np.radians([pa,inc]))
        r /= self.reff

        #iterate through bins and get vt value for each bin, 
        #dummy value for v2t and v2r since there isn't a good guess
        nbin = len(self.edges)
        vt = [0] if not self.fixcent else []
        for i in range(1,nbin):
            cut = (r > self.edges[i-1]) * (r < self.edges[i])
            vt += [np.max(model[cut])]
        v2t = [0] + [fill] * (nbin - 1) if not self.fixcent else [fill] * (nbin - 1)
        v2r = [0] + [fill] * (nbin - 1) if not self.fixcent else [fill] * (nbin - 1)

        guess += vt
        guess += v2t
        guess += v2r
        
        #clean and return
        guess = np.array(guess)
        guess[np.isnan(guess)] = 100
        self.guess = guess
        return self.guess

    def setbounds(self, incpad=20, papad=30, vsyspad=30, cenpad=2, velpad = 1.5,
            velmax=400, sigmax=300):
        '''
        Set the bounds for the prior of the fit.

        Takes in guess values from `self.guess` and sets bounds on either side
        of the guess value, with the padding size set by the input. Also caps
        velocity values based on the maximum values seen in the data.

        Args:
            incpad (:obj:`float`, optional):
                Padding on either side of guess inclination in degrees.
            papad (:obj:`float`, optional):
                Padding on either side of guess position angle in degrees.
            vsyspad (:obj:`float`, optional):
                Padding on either side of guess systemic velocity in km/s.
            cenpad (:obj:`float`, optional):
                Padding on either side of 0 for the center positions in arcsec.
            vsyspad (:obj:`float`, optional):
                Multiplicative factor to apply to max value in velocity data to
                determine upper bound for velocity prior.
            velmax (:obj:`float`, optional):
                Maximum allowed value for the velocity values to take
                regardless of what the data says.
            sigmax (:obj:`float`, optional):
                Maximum allowed value for the velocity dispersion values
                regardless of what the data says.
        '''

        try: theta0 = self.guess
        except: raise AttributeError('Must define guess first')
        if not hasattr(self, 'nbins'): 
            raise AttributeError('Must define nbins first')

        inc = self.guess[1] if self.phot_inc is None else self.phot_inc
        ndim = len(self.guess) + (self.nbins + self.fixcent) * self.disp

        #prior bounds defined based off of guess
        bounds = np.zeros((ndim, 2))
        bounds[0] = (max(inc - incpad, 5), min(inc + incpad, 85))
        bounds[1] = (theta0[1] - papad, theta0[1] + papad)
        bounds[2] = (0, 180) #uninformed
        bounds[3] = (theta0[3] - vsyspad, theta0[3] + vsyspad)
        if self.nglobs == 6: #assumes (0,0) is the best guess for center
            bounds[4] = (-cenpad, cenpad)
            bounds[5] = (-cenpad, cenpad)

        #if pa is near the wraparound, just unbound the prior
        if bounds[1][0] < 0 or bounds[1][1] > 360: bounds[1] = (0,360)

        #cap velocities at maximum in vf plus a padding factor
        vmax = min(np.max(self.vel)/np.cos(np.radians(inc)) * velpad, velmax)
        bounds[self.nglobs:self.nglobs + self.nbins] = (0, vmax)
        bounds[self.nglobs + self.nbins:self.nglobs + 3*self.nbins] = (0, vmax)
        if self.disp: bounds[self.nglobs + 3*self.nbins:] = (0, min(np.max(self.sig), sigmax))
        self.bounds = bounds

    def getasym(self):
        '''
        Calculate the asymmetry parameter and asymmetry map for the galaxy
        based on the Andersen & Bershady (2011) A_RC parameter.
        '''

        if not hasattr(self, 'guess'):
            raise AttributeError('Must define guess first')

        #get relevant galaxy parameters
        if self.nglobs == 6: 
            inc, pa, pab, vsys, xc, yc = self.guess[:6]
        elif self.nglobs == 4: 
            inc, pa, pab, vsys = self.guess[:4]
            xc, yc = [0, 0]

        #calculate asymmetry
        self.arc, self.asymmap = asymmetry(self, pa, vsys, xc, yc)

    def setnglobs(self, nglobs):
        '''
        Set number of global variables in fit. Global variables are assumed to
        be inclination, first order position angle, second order position angle,
        systemic velocity, and (optionally) x and y coordinates of the center.
        
        Should be 4 when not fitting the position of the center and 6 when the
        center position is being fit.

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

    def setdisp(self, disp):
        '''
        Whether or not to fit dispersion.

        Args:
            disp (:obj:`bool`):
                Whether to fit dispersion.
        '''

        self.disp = disp

    def setfixcent(self, fixcent):
        '''
        Whether or not to fix the center velocity bin at 0.

        Args:
            fixcent (:obj:`bool`):
                Whether or not to fix the center velocity bin at 0.
        '''

        self.fixcent = fixcent

    def setnbins(self, nbins):
        '''
        Set the number of radial bins the galaxy has

        Args:
            nbins (:obj:`bool`):
                Number of radial bins the galaxy has
        '''
        self.nbins = nbins

    def setnoisefloor(self, floor):
        '''
        Set intrinsic error to add to `vel_ivar` in quadrature.

        Args:
            floor (:obj:`float`):
                Intrinsic error to add to vel ivar in quadrature.   
        '''

        self.noise_floor = floor

    def setpenalty(self, penalty):
        '''
        Set penalty to use in :func:`~nirvana.fitting.loglike` if 2nd order
        velocity terms get too large

        Args:
            penalty (:obj:`float`):
                penalty if 2nd order velocity terms get too large    
        '''
        self.penalty = penalty

