import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table,Column

from nirvana.plotting import fileprep
from nirvana.fitting import bisym_model
from nirvana.models.axisym import AxisymmetricDisk
from nirvana.models.geometry import projected_polar

def extractdir(directory = '/data/manga/digiorgio/nirvana/')
    '''
    Scan an entire directory for nirvana output files and extract useful data from them.
    '''

    #load up files
    fs = glob(directory + '*.nirv')
    medians = np.zeros(len(fs))
    galaxies = []
    asyms = []
    dicts = []
    for i in tqdm(range(len(fs))):
        try: 
            #get info out of each file and make bisym model
            args, resdict, chains, meds = fileprep(fs[i])
            velmodel,sigmodel = bisym_model(args, resdict, plot=True)

            #axisym fit holding xc, yc, pa, and inc constant
            fit = AxisymmetricDisk()
            fit.lsq_fit(args,p0=[resdict['xc'], resdict['yc'], resdict['pa'], resdict['inc'], 0, 100, 10], fix = [1, 1, 1, 1, 0, 0, 0])
            symmodel = fit.model()

            #fractional difference between bisym and axisym
            asym = np.abs((velmodel-symmodel)/velmodel)

            #store data
            medians[i] = np.ma.median(asym)
            galaxies += [args]
            asyms += [asym]
            dicts += [resdict]

        #failure if bad file
        except:
            galaxies += [None]
            asyms += [None]
            dicts += [None]

    return medians, galaxies, asyms, dicts

def makealltable(dicts, outfile=None):
    '''
    Take a list of dictionaries from extractdir and turn them into an astropy table (and a fits file if a filename is given).
    '''

    #load dapall and drpall
    drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data

    #make names and dtypes for columns
    names = list(dicts[0].keys()) + ['velmask','sigmask','drpindex','dapindex']
    dtypes = ['D','D','D','D','D','D','20D','20D','20D','20D',
              'D','D','D','D','D','D','D','D','D','D','D','D',
              '20D','20D','20D','20D','20D','20D','20D','20D',
              'I','I','S','20L','20L','I','I']

    t = Table(names=names,dtype=dtypes)
    for i in range(len(dicts)):
        try:
            data = list(dicts[i].values())
            for d in range(len(data)):
                #put arrays into longer array to make them the same length
                if type(data[d]) is np.ndarray:
                    dnew = np.zeros(20)
                    dnew[:len(data[d])] = data[d]
                    data[d] = dnew

            #make mask to get rid of extra padding in arrays
            velmask = np.ones(20,dtype=bool)
            velmask[:len(dicts[i]['vt'])] = False
            sigmask = np.ones(20,dtype=bool)
            sigmask[:len(dicts[i]['sig'])] = False

            #corresponding indicies in dapall and drpall
            drpindex = np.where(drp['plateifu'] == f"{dicts[i]['plate']}-{dicts[i]['ifu']}")[0][0]
            dapindex = np.where(dap['plateifu'] == f"{dicts[i]['plate']}-{dicts[i]['ifu']}")[0][0]
            data += [velmask,sigmask,drpindex,dapindex]

        #failure for empty dict
        except:
            data = None

        t.add_row(data)

    #correct bad data types
    for n in names:
        if t[n].dtype in [np.complex64, np.complex128]: t[n].dtype = np.float64

    #rearrange columns
    t = t['plate','ifu','type','drpindex','dapindex',
          'inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask',
          'incl','pal','pabl','vsysl','vtl','v2tl','v2rl','sigl',
          'incu','pau','pabu','vsysu','vtu','v2tu','v2ru','sigu']
    
    #write if desired
    if outfile is not None: t.write(outfile, format='fits', overwrite=True)
    return t

def imagefits(f, outfile=None):
    '''
    Make a fits file for an individual galaxy with its fit parameters and relevant data.
    '''

    #get relevant data
    args, resdict, chains, meds = fileprep(f)

    #dapall and drpall indices
    drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data
    drpindex = np.where(drp['plateifu'] == f"{resdict['plate']}-{resdict['ifu']}")[0][0]
    dapindex = np.where(dap['plateifu'] == f"{resdict['plate']}-{resdict['ifu']}")[0][0]

    velmask = np.ones(20,dtype=bool)
    velmask[:len(resdict['vt'])] = False
    sigmask = np.ones(20,dtype=bool)
    sigmask[:len(resdict['sig'])] = False

    names = list(dicts[0].keys()) + ['velmask','sigmask','drpindex','dapindex']
    dtypes = ['D','D','D','D','D','D','20D','20D','20D','20D',
              'D','D','D','D','D','D','D','D','D','D','D','D',
              '20D','20D','20D','20D','20D','20D','20D','20D',
              'I','I','S','20L','20L','I','I']

    t = Table(names=names,dtype=dtypes)
    hdus = [fits.PrimaryHDU(),fits.BinTableHDU(t)]
    for n in ['vel','sig_phys2','sb','vel_ivar','sig_ivar','sb_ivar','vel_mask']:
        hdus += [fits.ImageHDU(args.remap(n))]
    hdul = fits.HDUList(hdus)
    if outfile is None: 
        hdul.writeto(f"nirvana_{resdict['plate']}-{resdict['ifu']}_{resdict['type']}.fits")
    else: hdul.writeto(outfile)
