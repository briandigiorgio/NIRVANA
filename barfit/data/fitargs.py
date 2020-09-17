#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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

    def setedges(self, nbins, maxr):
        '''
        Construct array of nbin number of bin edges up to maximum radius maxr
        in Re.  
        '''

        self.edges = np.linspace(0,maxr,nbins+1)

    def getguess(self):
        '''
        Generate a set of guess parameters for a given velocity field vf with
        ivar by fitting it with a simple rotation curve using least squares
        and sampling the resulting cuve to generate values in bins specified
        by edges. Must have edges set before calling this. Sets guess
        parameter in format [inc,pa,pab,vsys] + [vt,v2t,v2r]*(number of bins).
        Inc and pa in degrees. Assumes pab = pa.
        '''
        from barfit.barfit import polar,rotcurveeval
        from scipy.optimize import leastsq

        if self.edges is None: raise ValueError('Must define edges first')

        #define a minimization function and feed it to simple leastsquares
        minfunc = lambda params,vf,x,y,e,reff: np.array((vf - \
                rotcurveeval(x,y,*params,reff=reff))/e).flatten()
        vmax,inc,pa,h,vsys = leastsq(minfunc, (200,45,180,3,0), 
                args = (self.vel,self.x,self.y,self.vel_ivar**-.5,self.reff))[0]

        #check and fix signs if galaxy was fit upside down
        if np.product(np.sign([vmax,inc,h])) < 0: pa += 180
        pa %= 360
        vmax,inc,h = np.abs([vmax,inc,h])

        #generate model of vf and start assembling array of guess values
        model = rotcurveeval(self.grid_x,self.grid_y,vmax,inc,pa,h,vsys,reff=self.reff)
        plt.imshow(model)
        guess = [inc,pa,pa,vsys,0,0,0]
        r,th = polar(self.grid_y, self.grid_x, inc, pa, reff=self.reff)

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
