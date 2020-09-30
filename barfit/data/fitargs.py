#!/usr/bin/env python

from IPython import embed

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from ..models.geometry import projected_polar
from ..models.axisym import rotcurveeval
from ..models.beam import ConvolveFFTW

class FitArgs:
    '''
    Extension of Galaxy class to carry around a few more useful variables when
    performing dynesty fit. 
    '''

    def setnglobs(self, nglobs):
        '''
        Set number of global variables in fit.
        '''

        self.nglobs = nglobs

    def setweight(self, weight):
        '''
        Set weight to assign to smoothness of rotation curves in fit.
        '''

        self.weight = weight

    def setedges(self, inc, maxr=None):
        '''
        Construct array of bin edges for the galaxy.


        '''

        if maxr is None: maxr = np.max(np.sqrt(self.x**2 + self.y**2))/self.reff
        maxr *= self.reff
        binwidth = min(self.fwhm/2/np.cos(np.radians(inc)), self.fwhm)
        self.edges = np.arange(0,maxr,binwidth)/self.reff

    def setfixcent(self, fixcent):
        '''
        Whether or not to fix the central velocity bin.
        '''

        self.fixcent = fixcent

    def setdisp(self, disp):
        '''
        Whether or not to fit dispersion.
        '''

        self.disp = disp

    def setconv(self):
        '''
        Define the ConvolveFFTW class for the galaxy.
        '''

        self.conv = ConvolveFFTW(self.spatial_shape)

    def getguess(self):
        '''
        Generate a set of guess parameters for a given velocity field vf with
        ivar by fitting it with a simple rotation curve using least squares
        and sampling the resulting cuve to generate values in bins specified
        by edges. If edges aren't defined, just return rotation curve fit. Sets
        guess parameter in format [inc,pa,pab,vsys] + [vt,v2t,v2r]*(number of
        bins). Inc and pa in degrees. Assumes pab = pa.
        '''

        #define a minimization function and feed it to simple leastsquares
        if self.vel_ivar is None: ivar = np.ones_like(self.vel)
        else: ivar = self.vel_ivar

        minfunc = lambda params,vf,x,y,e,reff: np.array((vf - \
                rotcurveeval(x,y,*params,reff=reff))/e).flatten()
        vmax,inc,pa,h,vsys = leastsq(minfunc, (200,45,180,3,0), 
                args = (self.vel,self.x,self.y,ivar**-.5,self.reff))[0]

        #check and fix signs if galaxy was fit upside down
        if np.product(np.sign([vmax,h])) < 0: pa += 180
        pa %= 360
        if inc%180 != inc%90: pa += 180
        inc %= 90
        vmax,inc,h = np.abs([vmax,inc,h])

        #generate model of vf and start assembling array of guess values
        model = rotcurveeval(self.grid_x,self.grid_y,vmax,inc,pa,h,vsys,reff=self.reff)
        guess = [inc,pa,pa,vsys,0,0,0]
        if not hasattr(self, 'edges'): return [vmax,inc,pa,h,vsys]

        r,th = projected_polar(self.grid_x, self.grid_y, *np.radians([pa,inc]))
        r /= self.reff

        #iterate through bins and get vt value for each bin, 
        #dummy value for v2t and v2r since there isn't a good guess
        vts = np.zeros(len(self.edges)-1)
        v2ts = np.array([10] * len(self.edges-1))
        v2rs = np.array([10] * len(self.edges-1))
        for i in range(1,len(self.edges)-1):
            cut = (r > self.edges[i]) * (r < self.edges[i+1])
            vts[i] = np.max(model[cut])
            guess += [vts[i], v2ts[i], v2rs[i]]
        
        #clean and return
        guess = np.array(guess)
        guess[np.isnan(guess)] = 100
        self.guess = guess
        return self.guess
