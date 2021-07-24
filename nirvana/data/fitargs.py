#!/usr/bin/env python

from IPython import embed

import numpy as np
import matplotlib.pyplot as plt
import warnings

from astropy.stats import sigma_clip

from ..models.geometry import projected_polar
from ..models.asymmetry import asymmetry
from ..models.axisym import AxisymmetricDisk, axisym_iter_fit
from ..models.beam import ConvolveFFTW, smear

class FitArgs:
    '''
    Parent class for :class:`nirvana.data.kinematics.Kinematics` class to hold
    information that is necessary for fitting but is not related to the actual
    data.
    '''

    def __init__(self, kinematics, nglobs=6, weight=10, disp=True,
                fixcent=True, noisefloor=5, penalty=100, npoints=500,
                smearing=True, maxr=None, edges=None, guess=None, nbins=None, bounds=None,
                arc=None, asymmap=None):

        self.nglobs = nglobs
        self.weight = weight
        self.edges = edges
        self.disp = disp
        self.fixcent = fixcent
        self.guess = guess
        self.nbins = nbins
        self.bounds = bounds
        self.arc = arc
        self.asymmap = asymmap
        self.maxr = maxr
        self.noise_floor = noisefloor
        self.penalty = penalty
        self.npoints = npoints
        self.smearing = smearing
        self.kin = kinematics

    def setedges(self, inc, maxr=None, nbin=False, clipmasked=True):
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
                Maximum radius for the bins in arcsec. If not
                specified, it will default to the maximum unmasked radius of
                the galaxy.
            nbin (:obj:`bool`, optional):
                Whether or not to set the number of bins manually.
            clipmasked (:obj:`bool`, optional):
                Whether to clip off data outside of the last bin and in bins
                where too many spaxels are masked 
        '''

        #figure out max radius if not set
        if maxr == None: 
            x,y = (self.kin.x, self.kin.y)

            #calculate maximum radius of image if none is given
            maxr = np.max(np.sqrt(x**2 + y**2))

        #specify number of bins manually if desired
        if nbin: self.edges = np.linspace(0, maxr, inc+1)

        #calculate nyquist bin width based off fwhm and inc
        else:
            if self.kin.fwhm is None: 
                raise ValueError('Must either supply bin number or use PSF')
            binwidth = min(self.kin.fwhm/2/np.cos(np.radians(inc)), self.kin.fwhm)
            self.edges = np.arange(0, maxr, binwidth)

        #clip outer bins that have too many masked spaxels
        if clipmasked:
            #find radial coordinates of each spaxel
            guess = self.getguess(simple=True)
            r,th = projected_polar(self.kin.x, self.kin.y, *np.radians((guess[2], inc)))
            mr = np.ma.array(r, mask=self.kin.vel_mask)

            #calculate the number of spaxels in each bin 
            #and what fraction of them are masked
            nspax = np.zeros(len(self.edges)-1)
            maskfrac = np.zeros_like(nspax)
            for i in range(len(self.edges)-1):
                mcut = (mr > self.edges[i]) * (mr < self.edges[i+1])
                cut = (r > self.edges[i]) * (r < self.edges[i+1])
                nspax[i] = np.sum(mcut)
                maskfrac[i] = np.sum(self.kin.vel_mask[cut])/cut.sum()
            
            #cut bins where too many spaxels are masked
            bad = (maskfrac > .75)
            print(self.edges)
            self.edges = [self.edges[0], *self.edges[1:][~bad]]
            print(self.edges)

            #mask spaxels outside last bin edge
            self.kin.vel_mask[r > self.edges[-1]] = True

        self.maxr = max(self.edges)

    def getguess(self, fill=10, clip=False, simple=False, galmeta=None):
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

        if self.kin.vel_ivar is None: ivar = np.ones_like(self.kin.vel)
        else: ivar = self.kin.vel_ivar

        if clip: self.clip()

        #axisymmetric fit of data
        fit = None
        if galmeta is not None:
            try:
                fit = axisym_iter_fit(galmeta, self)[0]
                model = fit.model()[0]
            except Exception as e: 
                print(e)
                warnings.warn('Iterative fit failed, using noniterative fit instead')

        if fit is None:
            fit = AxisymmetricDisk()
            fit.lsq_fit(self.kin)
            model = fit.model()

        #get fit params
        xc, yc, pa, inc, vsys, vsini, h = fit.par[:7]
        vmax = vsini/np.sin(np.radians(inc))

        #generate model velocity field, start assembling array of guess values
        guess = [inc,pa,pa,vsys]
        if hasattr(self, 'nglobs') and self.nglobs == 6: guess += [xc, yc]

        #if edges have not been defined, just return global parameters
        if not hasattr(self, 'edges') or simple: return [vmax,inc,pa,h,vsys]

        #define polar coordinates and normalize to effective radius
        r,th = projected_polar(self.kin.grid_x, self.kin.grid_y, *np.radians([pa,inc]))

        #iterate through bins and get vt value for each bin, 
        #dummy value for v2t and v2r since there isn't a good guess
        nbin = len(self.edges)
        vt = [0] if not self.fixcent else []
        for i in range(1, nbin):
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

    def clip(self, galmeta=None, sigma=10, sbf=.03, anr=5, maxiter=10, smear_dv=50, smear_dsig=50, clip_thresh=.80, verbose=False):
        '''
        Filter out bad spaxels in kinematic data.
        
        Looks for features smaller than PSF by reconvolving PSF and looking for
        outlier points. Iteratively fits axisymmetric velocity field models and
        sigma clips residuals and chisq to get rid of outliers. Also clips
        based on surface brightness flux and ANR ratios. Applies new mask to
        galaxy.

        Args: 
            sigma (:obj:`float`, optional): 
                Significance threshold to be passed to
                `astropy.stats.sigma_clip` for sigma clipping the residuals
                and chi squared. Can't be too low or it will cut out
                nonaxisymmetric features. 
            sbf (:obj:`float`, optional): 
                Flux threshold below which spaxels are masked.
            anr (:obj:`float`, optional): 
                Surface brightness amplitude/noise ratio threshold below which
                spaxels are masked.
            maxiter (:obj:`int`, optional):
                Maximum number of iterations to allow clipping process to go
                through.
            smear_dv (:obj:`float`, optional):
                Threshold for clipping residuals of resmeared velocity data
            smear_dsig (:obj:`float`, optional):
                Threshold for clipping residuals of resmeared velocity
                dispersion data.
            clip_thresh (:obj:`float`, optional):
                Maximum fraction of the bins that can be clipped in order for
                the data to still be considered good. Will throw an error if
                it exceeds this level.
            verbose (:obj:`bool`, optional):
                Flag for printing out information on iterations.
        '''

        origvel = self.kin.remap('vel')
        origsig = self.kin.remap('sig')
        nmasked0 = self.kin.vel_mask.sum()
        ngood = (~self.kin.vel_mask).sum()

        #count spaxels in each bin and make 2d maps excluding large bins
        nspax = np.array([(self.kin.remap('binid') == self.kin.binid[i]).sum() for i in range(len(self.kin.binid))])
        binmask = self.kin.remap(nspax) > 10

        #axisymmetric fit of data
        fit = None
        if galmeta is not None:
            try:
                fit = axisym_iter_fit(galmeta, self)[0]
                avel, asig = fit.model()
            except Exception as e: 
                print(e)
                warnings.warn('Iterative fit failed, using noniterative fit instead')

        #failsafe simpler fit
        if fit is None:
            fit = AxisymmetricDisk()
            fit.lsq_fit(self.kin)
            avel = fit.model()
            asig = None

        #surface brightness
        sb  = np.ma.array(self.kin.remap('sb'), mask=binmask) if self.kin.sb is not None else None

        #get the vel field, fill masked areas with axisym model
        #have to do this so the convolution doesn't barf
        filledvel = np.ma.array(self.kin.remap('vel'), mask=binmask)
        mask = filledvel.mask | binmask.data | (filledvel == 0).data
        filledvel = filledvel.data
        filledvel[mask] = avel[mask]

        #same for sig
        filledsig = np.ma.array(np.ma.sqrt(self.kin.remap('sig_phys2')).filled(0.), mask=binmask) if self.kin.sig is not None else None
        if filledsig is not None and asig is not None:
            mask |= filledsig.mask | (filledsig == 0).data
            filledsig = filledsig.data
            filledsig[mask] = asig[mask]

        #reconvolve psf on top of velocity and dispersion
        cnvfftw = ConvolveFFTW(self.kin.spatial_shape)
        smeared = smear(filledvel, self.kin.beam_fft, beam_fft=True, sig=filledsig, sb=None, cnvfftw=cnvfftw)

        #cut out spaxels with too high residual because they're probably bad
        dvmask = self.kin.bin(np.abs(filledvel - smeared[1]) > smear_dv) 
        masks = [dvmask]
        labels = ['dv']
        if self.kin.sig is not None: 
            dsigmask = self.kin.bin(np.abs(filledsig - smeared[2]) > smear_dsig)
            masks += [dsigmask]
            labels += ['dsig']

        #clip on surface brightness and ANR
        if self.kin.sb is not None: 
            sbmask = self.kin.sb < sbf
            masks += [sbmask]
            labels += ['sb']

        if self.kin.sb_anr is not None:
            anrmask = self.kin.sb_anr < anr
            masks += [anrmask]
            labels += ['anr']

        #combine all masks and apply to data
        mask = np.zeros(dvmask.shape)
        for m in masks: mask += m
        mask = mask.astype(bool)
        self.kin.remask(mask)

        #iterate through rest of clips until mask converges
        nmaskedold = -1
        nmasked = np.sum(mask)
        niter = 0
        err = False
        while nmaskedold != nmasked and sigma:
            #quick axisymmetric least squares fit
            fit = AxisymmetricDisk()
            fit.lsq_fit(self.kin)

            #quick axisymmetric fit
            model = self.kin.bin(fit.model())
            resid = self.kin.vel - model

            #clean up the data by sigma clipping residuals and chisq
            chisq = resid**2 * self.kin.vel_ivar if self.kin.vel_ivar is not None else resid**2
            residmask = sigma_clip(resid, sigma=sigma, masked=True).mask
            chisqmask = sigma_clip(chisq, sigma=sigma, masked=True).mask
            clipmask = (mask + residmask + chisqmask).astype(bool)

            #iterate
            nmaskedold = nmasked
            nmasked = np.sum(clipmask)
            niter += 1
            if verbose: print(f'Performed {niter} clipping iterations...')

            #break if too many iterations
            if niter > maxiter: 
                if verbose: print(f'Reached maximum clipping iterations: {niter}')
                break

            #break if too much data has been clipped
            maskfrac = (nmasked - nmasked0)/ngood
            if maskfrac > clip_thresh:
                err = True
                break

            #apply mask to data
            self.kin.remask(clipmask)

        #make a plot of all of the masks if desired
        if verbose: 
            print(f'{round(maskfrac * 100, 1)}% of data clipped')
            if sigma:
                masks += [residmask, chisqmask]
                labels += ['resid', 'chisq']
                print(f'Clipping converged after {niter} iterations')

            plt.figure(figsize = (16,8))
            plt.subplot(241)
            plt.axis('off')
            plt.imshow(origvel, cmap='jet', origin='lower')
            plt.title('Original vel')
            plt.subplot(242)
            plt.axis('off')
            plt.imshow(origsig, cmap='jet', origin='lower')
            plt.title('Original sig')
            for i in range(len(masks)):
                plt.subplot(243+i)
                plt.axis('off')
                plt.imshow(self.kin.remap(masks[i]), origin='lower')
                plt.title(labels[i])
            plt.tight_layout()
            plt.show()

        if err:
            raise ValueError(f'Bad velocity field: {round(maskfrac * 100, 1)}% of data clipped after {niter} iterations')

    def setbounds(self, incpad=20, papad=30, vsyspad=30, cenpad=2, velpad = 1.5,
            velmax=400, sigmax=300, incgauss=False, pagauss=False):
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
            incgauss (:obj:`bool`, optional):
                If `True`, will treat `incpad` as the standard deviation of a
                Gaussian prior rather than bounds of a uniform prior.
            pagauss (:obj:`bool`, optional):
                If `True`, will treat `papad` as the standard deviation of a
                Gaussian prior rather than bounds of a uniform prior.
        '''

        try: theta0 = self.guess
        except: raise AttributeError('Must define guess first')
        if not hasattr(self, 'nbins'): 
            raise AttributeError('Must define nbins first')

        inc = self.guess[1] if self.kin.phot_inc is None else self.kin.phot_inc
        pa = self.guess[2] if self.kin.phot_pa is None else self.kin.phot_pa
        ndim = len(self.guess) + (self.nbins + self.fixcent) * self.disp

        #prior bounds defined based off of guess
        bounds = np.zeros((ndim, 2))
        if incgauss: bounds[0] = (inc, incpad)
        else: bounds[0] = (max(inc - incpad, 5), min(inc + incpad, 85))
        if pagauss: bounds[1] = (pa, papad)
        else: bounds[1] = (theta0[1] - papad, theta0[1] + papad)
        bounds[2] = (0, 180) #uninformed
        bounds[3] = (theta0[3] - vsyspad, theta0[3] + vsyspad)
        if self.nglobs == 6: #assumes (0,0) is the best guess for center
            bounds[4] = (-cenpad, cenpad)
            bounds[5] = (-cenpad, cenpad)

        #if pa is near the wraparound, just unbound the prior
        if bounds[1][0] < 0 or bounds[1][1] > 360: bounds[1] = (0,360)

        #cap velocities at maximum in vf plus a padding factor
        vmax = min(np.max(np.abs(self.kin.vel))/np.cos(np.radians(inc)) * velpad, velmax)
        bounds[self.nglobs:self.nglobs + self.nbins] = (0, vmax)
        bounds[self.nglobs + self.nbins:self.nglobs + 3*self.nbins] = (0, vmax)
        if self.disp: bounds[self.nglobs + 3*self.nbins:] = (0, min(np.max(self.kin.sig), sigmax))
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
        self.arc, self.asymmap = asymmetry(self.kin, pa, vsys, xc, yc)

    def setphotpa(self, galmeta):
        '''
        Correct the photometric PA by 180 degrees to align with the kinematics if necessary.
        '''

        self.kin.phot_pa = galmeta.guess_kinematic_pa(self.kin.x, self.kin.y, self.kin.vel) % 360

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

