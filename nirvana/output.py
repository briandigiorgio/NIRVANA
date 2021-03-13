import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import multiprocessing as mp
import os

from astropy.io import fits
from astropy.table import Table,Column
from scipy.spatial import KDTree

from .plotting import fileprep, summaryplot
from .fitting import bisym_model
from .models.axisym import AxisymmetricDisk
from .models.geometry import projected_polar

def extractfile(f, remotedir=None):
    try: 
        #get info out of each file and make bisym model
        args, resdict, chains, meds = fileprep(f, remotedir=remotedir)

        #fractional difference between bisym and axisym
        arc, asymmap = asymmetry(args)
        resdict['a_rc'] = arc

    #failure if bad file
    except Exception as e:
        print(f'Extraction of {f} failed:', e)
        args, arc, asymmap, resdict = (None, None, None, None)

    return args, arc, asymmap, resdict

def extractdir(cores=10, directory='/data/manga/digiorgio/nirvana/'):
    '''
    Scan an entire directory for nirvana output files and extract useful data from them.
    '''

    #find nirvana files
    fs = glob(directory + '*.nirv')
    with mp.Pool(cores) as p:
        out = p.map(extractfile, fs)

    galaxies = np.zeros(len(fs), dtype=object)
    arcs = np.zeros(len(fs))
    asyms = np.zeros(len(fs), dtype=object)
    dicts = np.zeros(len(fs), dtype=object)
    for i in range(len(out)):
        galaxies[i], arcs[i], asyms[i], dicts[i] = out[i]

    return galaxies, arcs, asyms, dicts

def dictformatting(d, drp=None, dap=None, padding=20, fill=-9999):
    #load dapall and drpall
    if drp is None:
        drp = fits.open('/home/bdigiorg/dapall-v3_0_1-3.0.1.fits')[1].data
    if dap is None:
        dap = fits.open('/home/bdigiorg/drpall-v3_1_1.fits')[1].data
    try:
        data = list(d.values())
        for i in range(len(data)):
            #put arrays into longer array to make them the same length
            if padding and type(data[i]) is np.ndarray:
                dnew = np.ones(padding) * fill
                dnew[:len(data[i])] = data[i]
                data[i] = dnew

        #make mask to get rid of extra padding in arrays
        velmask = np.ones(padding,dtype=bool)
        velmask[:len(d['vt'])] = False
        sigmask = np.ones(padding,dtype=bool)
        sigmask[:len(d['sig'])] = False

        #corresponding indicies in dapall and drpall
        drpindex = np.where(drp['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        dapindex = np.where(dap['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        data += [velmask, sigmask, drpindex, dapindex]

    #failure for empty dict
    except:
        data = None

    return data

def makealltable(dicts, outfile=None, padding=20):
    '''
    Take a list of dictionaries from extractdir and turn them into an astropy table (and a fits file if a filename is given).
    '''

    #load dapall and drpall
    drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data

    #make names and dtypes for columns
    names = None
    i = 0
    while names is None:
        try: names = list(dicts[i].keys()) + ['velmask','sigmask','drpindex','dapindex']
        except: i += 1

    dtypes = ['f4','f4','f4','f4','f4','f4','20f4','20f4','20f4','20f4',
              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
              '20f4','20f4','20f4','20f4','20f4','20f4','20f4','20f4',
              'I','I','S','f4','20?','20?','I','I']

    data = []
    for d in dicts: data += [dictformatting(d, drp, dap, padding=padding)]

    t = Table(names=names, dtype=dtypes)
    for d in data: t.add_row(d)

    #rearrange columns
    t = t['plate','ifu','type','drpindex','dapindex',
          'xc','yc','inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask',
          'xcl','ycl','incl','pal','pabl','vsysl','vtl','v2tl','v2rl','sigl',
          'xcu','ycu','incu','pau','pabu','vsysu','vtu','v2tu','v2ru','sigu','a_rc']
    
    #write if desired
    if outfile is not None: t.write(outfile, format='fits', overwrite=True)
    return t

def maskedarraytofile(array, name=None, fill=0):
    '''
    Write a masked array to an HDU. 
    
    Numpy says it's not implemented yet so I'm implementing it.
    '''
    array[array.mask] = fill
    array = array.data
    arrayhdu = fits.ImageHDU(array)
    if name is not None: arrayhdu.name = name
    return arrayhdu

def imagefits(f, gal=None, outfile=None, padding=20, remotedir=None, outdir=''):
    '''
    Make a fits file for an individual galaxy with its fit parameters and relevant data.
    '''

    #get relevant data
    args, arc, asymmap, resdict = extractfile(f, remotedir=remotedir)
    if gal is not None: args = gal
    resdict['bin_edges'] = np.array(args.edges)
    data = dictformatting(resdict, padding=padding)

    data += [*np.delete(args.bounds.T, slice(7,-1), axis=1)]

    names = list(resdict.keys()) + ['velmask','sigmask','drpindex','dapindex','prior_lbound','prior_ubound']
    dtypes = ['f4','f4','f4','f4','f4','f4','20f4','20f4','20f4','20f4',
              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
              '20f4','20f4','20f4','20f4','20f4','20f4','20f4','20f4',
              'I','I','S','f4','20f4','20?','20?','I','I','8f4','8f4']

    #make table of fit data
    t = Table(names=names, dtype=dtypes)
    t.add_row(data)
    reordered = ['plate','ifu','type','drpindex','dapindex','bin_edges','prior_lbound','prior_ubound',
          'xc','yc','inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask',
          'xcl','ycl','incl','pal','pabl','vsysl','vtl','v2tl','v2rl','sigl',
          'xcu','ycu','incu','pau','pabu','vsysu','vtu','v2tu','v2ru','sigu','a_rc']
    t = t[reordered]
    bintable = fits.BinTableHDU(t)
    bintable.name = 'fit_params'
    hdus = [fits.PrimaryHDU(), bintable]

    #image = fits.ImageHDU(args.image)
    #image.name = 'image'
    #summplot = fits.ImageHDU(fig2data(summaryplot(f)))
    #summplot.name = 'summary'
    #hdus += [summplot, image]

    #add all data extensions from original data
    maps = ['vel', 'sig', 'sb', 'vel_ivar', 'sig_ivar', 'sb_ivar', 'vel_mask']
    for m in maps:
        if m == 'sig': 
            data = np.sqrt(args.remap('sig_phys2').data)
            mask = args.remap('sig_phys2').mask
        else:
            data = args.remap(m).data
            mask = args.remap(m).mask

        if data.dtype == bool: data = data.astype(int) #catch for bools
        data[mask] = 0 if 'mask' not in m else data[mask]
        hdu = fits.ImageHDU(data)
        hdu.name = m
        hdus += [hdu]
    hdus[-1].name = 'MaNGA_mask'

    #add mask from clipping
    args.clip()
    clipmask = fits.ImageHDU(np.array(args.remap('vel').mask, dtype=int))
    clipmask.name = 'clip_mask'
    hdus += [clipmask]

    #smeared and intrinsic velocity/dispersion models
    velmodel, sigmodel = bisym_model(args, resdict, plot=True)
    args.beam_fft = None
    intvelmodel, intsigmodel = bisym_model(args, resdict, plot=True)

    #unmask them, name them all, and add them to the list
    newnames = ['vel_model','sig_model','vel_int_model','sig_int_model','asymmetry']
    for i,a in enumerate([velmodel, sigmodel, intvelmodel, intsigmodel, asymmap]):
        hdus += [maskedarraytofile(a, name=newnames[i])]

    #add parameters to the header
    hdr = hdus[0].header
    hdr['maxr'] = args.maxr
    hdr['weight'] = args.weight
    hdr['fixcent'] = args.fixcent
    hdr['nbin'] = args.nbin
    hdr['npoints'] = args.npoints
    hdr['smearing'] = args.smearing
    hdr['avmax'], hdr['ainc'], hdr['apa'], hdr['ahrot'], hdr['avsys'] = args.getguess(simple=True)

    #write out
    hdul = fits.HDUList(hdus)
    if outfile is None: 
        hdul.writeto(f"{outdir}/nirvana_{resdict['plate']}-{resdict['ifu']}_{resdict['type']}.fits", overwrite=True)
    else: hdul.writeto(outdir + outfile)

def reflect(pa, x, y):
    '''
    Reflect arrays of x and y coordinates across a line at angle position angle pa.
    '''

    th = np.radians(90 - pa) #turn position angle into a regular angle

    #reflection matrix across arbitrary angle
    ux = np.cos(th) 
    uy = np.sin(th)
    return np.dot([[ux**2 - uy**2, 2*ux*uy], [2*ux*uy, uy**2 - ux**2]], [x, y])

def asymmetry(args, resdict=None):
    '''
    Calculate asymmetry parameter and maps for major/minor axis reflection.
    '''

    #use axisym fit if no nirvana fit is given
    if resdict is None:
        fit = AxisymmetricDisk()
        fit.lsq_fit(args)
        xc,yc,pa,inc,vsys,vmax,h = fit.par

    #get nirvana fit params
    else:
        xc,yc,pa,vsys = resdict['xc'], resdict['yc'], resdict['pa'], resdict['vsys']
        
    #construct KDTree of spaxels for matching
    x = args.x - xc
    y = args.y - xc
    tree = KDTree(list(zip(x,y)))
    
    #compute major and minor axis asymmetry 
    arc2d = []
    for axis in [0,90]:
        #match spaxels to their reflections, mask out ones without matches
        d,i = tree.query(reflect(pa - axis, x, y).T)
        mask = np.ma.array(np.ones(len(args.vel)), mask = (d>.5) | args.vel_mask)

        #compute Andersen & Bershady (2013) A_RC parameter 2D maps
        vel = args.remap(args.vel * mask) - vsys
        ivar = args.remap(args.vel_ivar * mask)
        velr = args.remap(args.vel[i] * mask - vsys)
        ivarr = args.remap(args.vel_ivar[i] * mask)
        arc2d += [A_RC(vel, velr, ivar, ivarr)]
    
    #sum over maps to get global params
    arc = np.sum([np.sum(a) for a in arc2d])
    asymmap = np.ma.array(arc2d).mean(axis=0)
    return arc, asymmap

def A_RC(vel, velr, ivar, ivarr):
    '''
    Compute velocity field asymmetry for a velocity field and its reflection.

    From Andersen & Bershady (2013) equation 7 but doesn't sum over whole galaxy so asymmmetry is spatially resolved. 
    '''
    return (np.abs(np.abs(vel) - np.abs(velr))/np.sqrt(1/ivar + 1/ivarr) 
         / (.5*np.sum(np.abs(vel) + np.abs(velr))/np.sqrt(1/ivar + 1/ivarr)))

def fig2data(fig):
    # draw the renderer
    fig.canvas.draw( )
 
    # Get the RGBA buffer from the figure
    h,w = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
