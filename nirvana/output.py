import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import multiprocessing as mp
import os
import traceback
import pickle

from astropy.io import fits
from astropy.table import Table,Column
from tqdm import tqdm

from .plotting import fileprep, summaryplot
from .fitting import bisym_model
from .models.axisym import AxisymmetricDisk
from .models.geometry import projected_polar, asymmetry
from .util import fileio
from .data.manga import MaNGAGlobalPar

def extractfile(f, remotedir=None, gal=None):
    try: 
        #get info out of each file and make bisym model
        args, resdict = fileprep(f, remotedir=remotedir, gal=gal)

        inc, pa, pab, vsys, xc, yc = args.guess[:6]
        arc, asymmap = asymmetry(args, pa, vsys, xc, yc)
        resdict['a_rc'] = arc

    #failure if bad file
    except Exception:
        print(f'Extraction of {f} failed:')
        print(traceback.format_exc())
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

def dictformatting(d, drp=None, dap=None, padding=20, fill=-9999, drpalldir='.', dapalldir='.'):
    #load dapall and drpall
    if drp is None:
        drpfile = glob(drpalldir + '/drpall*')[0]
        drp = fits.open(drpfile)[1].data
    if dap is None:
        dapfile = glob(dapalldir + '/dapall*')[0]
        dap = fits.open(dapfile)[1].data
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

def makealltable(fname='', dir='.', vftype='', outfile=None, padding=20):
    '''
    Take a list of dictionaries from extractdir and turn them into an astropy table (and a fits file if a filename is given).
    '''

    #load dapall and drpall
    drp = fits.open('/data/manga/spectro/redux/MPL-11/drpall-v3_1_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-11/dapall-v3_1_1-3.1.0.fits')[1].data

    fs = glob(f'{dir}/{fname}*{vftype}.fits')
    if len(fs) == 0:
        raise FileNotFoundError(f'No matching FITS files found in directory "{dir}"')
    else:
        print(len(fs), 'files found...')

    tables = []
    for f in tqdm(fs):
        try: 
            fi = fits.open(f)
            tables += [fi[1].data]
            fi.close()
        except Exception as e: print(f, 'failed:', e)

    #make names and dtypes for columns
    names = None
    i = 0
    while names is None:
        try: 
            names = list(tables[i].names)
            dtype = tables[i].dtype
        except: i += 1

    data = np.zeros(len(tables), dtype=dtype)
    for i in range(len(tables)):
        data[i] = tables[i]
    t = Table(data)

    #apparently numpy doesn't handle its own uint and bool dtypes correctly
    #so this is to fix them
    for k in ['plate', 'ifu', 'drpindex', 'dapindex']:
        t[k] += 2**31
    for k in ['velmask', 'sigmask']:
        t[k] //= 71

    #write if desired
    if outfile is not None: t.write(outfile, format='fits', overwrite=True)
    return t

def maskedarraytofile(array, name=None, fill=0, hdr=None):
    '''
    Write a masked array to an HDU. 
    
    Numpy says it's not implemented yet so I'm implementing it.
    '''
    array[array.mask] = fill
    array = array.data
    arrayhdu = fits.ImageHDU(array, name=name, header=hdr)
    return arrayhdu

def imagefits(f, galmeta=True, gal=None, outfile=None, padding=20, remotedir=None, outdir='', drpalldir='.', dapalldir='.'):
    '''
    Make a fits file for an individual galaxy with its fit parameters and relevant data.
    '''

    if gal==True: 
        try: gal = pickle.load(open(f[:-4] + 'gal', 'rb'))
        except: raise FileNotFoundError('Could not load .gal file')

    #get relevant data
    args, arc, asymmap, resdict = extractfile(f, remotedir=remotedir, gal=gal)
    if gal is not None: args = gal
    resdict['bin_edges'] = np.array(args.edges)
    r, th = projected_polar(args.x - resdict['xc'], args.y - resdict['yc'], *np.radians((resdict['pa'], resdict['inc'])))
    r = args.remap(r)
    th = args.remap(th)

    data = dictformatting(resdict, padding=padding, drpalldir=drpalldir, dapalldir=dapalldir)
    data += [*np.delete(args.bounds.T, slice(7,-1), axis=1)]

    names = list(resdict.keys()) + ['velmask','sigmask','drpindex','dapindex','prior_lbound','prior_ubound']
    dtypes = ['f4','f4','f4','f4','f4','f4','20f4','20f4','20f4','20f4',
              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
              '20f4','20f4','20f4','20f4','20f4','20f4','20f4','20f4',
              'I','I','S','f4','20f4','20?','20?','I','I','8f4','8f4']

    #add parameters to the header
    if galmeta==True:
        drpallfile = glob(drpalldir + '/drpall*')[0]
        galmeta = MaNGAGlobalPar(resdict['plate'], resdict['ifu'], drpall_file=drpallfile)
    hdr = fileio.initialize_primary_header(galmeta)
    maphdr = fileio.add_wcs(hdr, args)
    psfhdr = hdr.copy()
    psfhdr['PSFNAME'] = (args.psf_name, 'Original PSF name')

    hdr['MANGAID'] = (galmeta.mangaid, 'MaNGA ID')
    hdr['PLATE'] = (galmeta.plate, 'MaNGA plate')
    hdr['IFU'] = (galmeta.ifu, 'MaNGA IFU')
    hdr['OBJRA'] = (galmeta.ra, 'Galaxy center RA in deg')
    hdr['OBJDEC'] = (galmeta.dec, 'Galaxy center Dec in deg')
    hdr['Z'] = (galmeta.z, 'Galaxy redshift')
    hdr['ASEC2KPC'] = (galmeta.kpc_per_arcsec(), 'Kiloparsec to arcsec conversion factor')
    hdr['REFF'] = (galmeta.reff, 'Effective radius in arcsec')
    hdr['SERSICN'] = (galmeta.sersic_n, 'Sersic index')
    hdr['PHOT_PA'] = (galmeta.pa, 'Position angle derived from photometry in deg')
    hdr['PHOT_INC'] = (args.phot_inc, 'Photomentric inclination angle in deg')
    hdr['ELL'] = (galmeta.ell, 'Photometric ellipticity')
    hdr['guess_Q0'] = (galmeta.q0, 'Intrinsic oblateness (from population stats)')

    hdr['maxr'] = (args.maxr, 'Maximum observation radius in REFF')
    hdr['weight'] = (args.weight, 'Weight of profile smoothness')
    hdr['fixcent'] = (args.fixcent, 'Whether first velocity bin is fixed at 0')
    hdr['nbin'] = (args.nbins, 'Number of radial bins')
    hdr['npoints'] = (args.npoints, 'Number of dynesty live points')
    hdr['smearing'] = (args.smearing, 'Whether PSF smearing was used')
    hdr['ivar_flr'] = (args.noise_floor, 'Noise added to ivar arrays in quadrature')
    hdr['penalty'] = (args.penalty, 'Penalty for large 2nd order terms')

    avmax, ainc, apa, ahrot, avsys = args.getguess(simple=True, galmeta=galmeta)
    hdr['a_vmax'] = (avmax, 'Axisymmetric asymptotic velocity in km/s')
    hdr['a_pa'] = (apa, 'Axisymmetric position angle in deg')
    hdr['a_inc'] = (ainc, 'Axisymmetric inclination angle in deg')
    hdr['a_vsys'] = (avsys, 'Axisymmetric systemic velocity in km/s')

    #make table of fit data
    t = Table(names=names, dtype=dtypes)
    t.add_row(data)
    reordered = ['plate','ifu','type','drpindex','dapindex','bin_edges','prior_lbound','prior_ubound',
          'xc','yc','inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask',
          'xcl','ycl','incl','pal','pabl','vsysl','vtl','v2tl','v2rl','sigl',
          'xcu','ycu','incu','pau','pabu','vsysu','vtu','v2tu','v2ru','sigu','a_rc']
    t = t[reordered]
    bintable = fits.BinTableHDU(t, name='fit_params', header=hdr)
    hdus = [fits.PrimaryHDU(header=hdr), bintable]

    hdus += [maskedarraytofile(r, name='ell_r', hdr=fileio.finalize_header(maphdr, 'ell_r'))]
    hdus += [maskedarraytofile(th, name='ell_theta', hdr=fileio.finalize_header(maphdr, 'ell_th'))]

    #add all data extensions from original data
    mapnames = ['vel', 'sigsqr', 'sb', 'vel_ivar', 'sig_ivar', 'sb_ivar', 'vel_mask', 'sig_mask']
    units = ['km/s', '(km/2)^2', '1E-17 erg/s/cm^2/ang/spaxel', '(km/s)^{-2}', '(km/s)^{-4}', '(1E-17 erg/s/cm^2/ang/spaxel)^{-2}', None, None]
    errs = [True, True, True, False, False, False, True, True]
    quals = [True, True, False, True, True, False, False, False]
    hduclas2s = ['DATA', 'DATA', 'DATA', 'ERROR', 'ERROR', 'ERROR', 'QUALITY', 'QUALITY', 'QUALITY']
    bittypes = [None, None, None, None, None, None, np.bool, np.bool]

    for m, u, e, q, h, b in zip(mapnames, units, errs, quals, hduclas2s, bittypes):
        if m == 'sigsqr': 
            data = np.sqrt(args.remap('sig_phys2').data)
            mask = args.remap('sig_phys2').mask
        else:
            data = args.remap(m).data
            mask = args.remap(m).mask

        if data.dtype == bool: data = data.astype(int) #catch for bools
        data[mask] = 0 if 'mask' not in m else data[mask]
        hdus += [fits.ImageHDU(data, name=m, header=fileio.finalize_header(maphdr, m, u, h, e, q, None, b))]

    hdus += [fits.ImageHDU(args.beam, name='PSF', header=fileio.finalize_header(psfhdr, 'PSF'))]

    #smeared and intrinsic velocity/dispersion models
    velmodel, sigmodel = bisym_model(args, resdict, plot=True, relative_pab=False)
    args.beam_fft = None
    intvelmodel, intsigmodel = bisym_model(args, resdict, plot=True, relative_pab=False)

    #unmask them, name them all, and add them to the list
    models = [velmodel, sigmodel, intvelmodel, intsigmodel, asymmap]
    modelnames = ['vel_model','sig_model','vel_int_model','sig_int_model','asymmetry']
    units = ['km/s', '(km/s)^2', 'km/s', '(km/s)^2', None]
    for a, n, u in zip(models, modelnames, units):
        hdri = fileio.finalize_header(maphdr, n, u)
        hdus += [maskedarraytofile(a, name=n, hdr=hdri)]

    #write out
    hdul = fits.HDUList(hdus)
    if outfile is None: 
        outfile = f"nirvana_{resdict['plate']}-{resdict['ifu']}_{resdict['type']}.fits"
    hdul.writeto(outdir + outfile, overwrite=True, output_verify='fix', checksum=True)

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
