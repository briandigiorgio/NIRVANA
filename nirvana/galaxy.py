#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from scipy.optimize import leastsq

class Galaxy:
    '''
    Class for fetching, calculating, and containing all of a galaxy's
    information that is relevant to the fit. Relies on standard MaNGA file
    structure to find MAPS files 
    '''

    def __init__(self, plate, ifu, stellar=False, spx=True, path=None):
        '''
        Get all of the information out of the MAPS file for a given plate ifu.
        Does Ha by default but can be set to do stellar. Does SPX by default
        but can be set to do HYB10. Assumes graymalkin file structure by
        default but can take in alternate path to a directory of MAPS files by
        plate ifu.
        '''

        self.plate = plate
        self.ifu = ifu

        #set desired extension
        if spx:
            self.bintype = 'SPX'
        else:
            self.bintype = 'HYB10'

        #set correct path to files and load, defaults to graymalkin
        if path is None:
            path = f'/data/manga/spectro/analysis/MPL-9/{self.bintype}-MILESHC-MASTARHC'
        f = fits.open(f'{path}/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-{self.bintype}-MILESHC-MASTARHC.fits.gz')

        #get relevant data from file
        if stellar:
            vf = f['STELLAR_VEL'].data #velocity field
            flux = f['SPX_MFLUX'].data #surface brightness
            ivar = f['STELLAR_VEL_IVAR'].data #vf inverse variance
            m = f['STELLAR_VEL_MASK'].data != 0 #masked spaxels
            sigma = f['STELLAR_SIGMA'].data #velocity dispersion
            sigmaivar = f['STELLAR_SIGMA_IVAR'].data #vel disp inverse variance
            inst = f['STELLAR_SIGMACORR'].data[0] #instrumental dispersion

        #same but for Ha
        else:
            vf = f['EMLINE_GVEL'].data[23]
            flux = f['EMLINE_GFLUX'].data[23]
            ivar = f['EMLINE_GVEL_IVAR'].data[23]
            m = f['EMLINE_GVEL_MASK'].data[23] != 0
            sigma = f['EMLINE_GSIGMA'].data[23]
            sigmaivar = f['EMLINE_GSIGMA_IVAR'].data[23]
            inst = f['EMLINE_INSTSIGMA'].data[23]

        #mask arrays
        m += (sigmaivar > inst) #mask pixels with unreasonable sigmaivar
        self.mask = m
        self.vf = np.ma.array(vf, mask = m)
        self.flux = np.ma.array(flux, mask = m)
        self.ivar = np.ma.array(ivar, mask = m)
        self.sigma = np.ma.array(sigma, mask = m)
        self.sigmaivar = np.ma.array(sigmaivar, mask = m)

        #on sky xy coordinates for spaxels
        x,y = [f['SPX_SKYCOO'].data[1], f['SPX_SKYCOO'].data[0]]
        self.x = np.ma.array(x, mask = m)
        self.y = np.ma.array(y, mask = m)
        
        #elliptical polar coordinates
        er,eth = [f['SPX_ELLCOO'].data[1], f['SPX_ELLCOO'].data[3]]
        self.er = np.ma.array(er, mask = m)
        self.eth = np.ma.array(eth, mask = m)

        self.psf = None

        f.close()

    def getpsf(self, path=None):
        '''
        Get the g band PSF calculated by the DRP. Assumes graymalkin file
        structure but can be given path to DRP products organized in standard
        way. 
        '''

        if path is None: path = '/data/manga/spectro/redux/MPL-9'
        psff = fits.open(f'{path}/{self.plate}/stack/manga-{self.plate}-{self.ifu}-LOGCUBE.fits.gz')
        self.psf = np.ma.array(psff['GPSF'].data,mask = self.mask) #g band psf
        psff.close()

    def flatten(self):
        '''
        Flatten all of the arrays stored as attributes.
        '''

        keys = list(vars(self).keys())
        vals = list(vars(self).values())
        for i in range(len(keys)):
            try: vars(self)[keys[i]] = vals[i].flatten()
            except: pass

    def makeedges(self, nbins, maxr):
        '''
        Construct array of nbin number of bin edges up to maximum radius maxr
        in Re.  
        '''

        self.edges = np.linspace(0,maxr,nbins+1)
