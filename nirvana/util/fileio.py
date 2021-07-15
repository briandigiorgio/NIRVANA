r"""
Various utilities for use with fits files.

----

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
import sys
import os
import gzip
import shutil
from glob import glob
import traceback
import pickle

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table,Column
from tqdm import tqdm

# For versioning
import scipy
import astropy
from .. import __version__

from ..models.higher_order import bisym_model
from ..models.geometry import projected_polar, asymmetry

def init_record_array(shape, dtype):
    r"""
    Utility function that initializes a record array using a provided
    input data type.  For example::

        dtype = [ ('INDX', np.int, (2,) ),
                  ('VALUE', np.float) ]

    Defines two columns, one named `INDEX` with two integers per row and
    the one named `VALUE` with a single float element per row.  See
    `np.recarray`_.
    
    Args:
        shape (:obj:`int`, :obj:`tuple`):
            Shape of the output array.
        dtype (:obj:`list`):
            List of the tuples that define each element in the record
            array.

    Returns:
        `np.recarray`_: Zeroed record array
    """
    data = np.zeros(shape, dtype=dtype)
    return data.view(np.recarray)


def rec_to_fits_type(rec_element):
    """
    Return the string representation of a fits binary table data type
    based on the provided record array element.
    """
    n = 1 if len(rec_element[0].shape) == 0 else rec_element[0].size
    if rec_element.dtype == np.bool:
        return '{0}L'.format(n)
    if rec_element.dtype == np.uint8:
        return '{0}B'.format(n)
    if rec_element.dtype == np.int16 or rec_element.dtype == np.uint16:
        return '{0}I'.format(n)
    if rec_element.dtype == np.int32 or rec_element.dtype == np.uint32:
        return '{0}J'.format(n)
    if rec_element.dtype == np.int64 or rec_element.dtype == np.uint64:
        return '{0}K'.format(n)
    if rec_element.dtype == np.float32:
        return '{0}E'.format(n)
    if rec_element.dtype == np.float64:
        return '{0}D'.format(n)
    
    # If it makes it here, assume its a string
    l = int(rec_element.dtype.str[rec_element.dtype.str.find('U')+1:])
#    return '{0}A'.format(l) if n==1 else '{0}A{1}'.format(l*n,l)
    return '{0}A'.format(l*n)


def rec_to_fits_col_dim(rec_element):
    """
    Return the string representation of the dimensions for the fits
    table column based on the provided record array element.

    The shape is inverted because the first element is supposed to be
    the most rapidly varying; i.e. the shape is supposed to be written
    as row-major, as opposed to the native column-major order in python.
    """
    return None if len(rec_element[0].shape) <= 1 else str(rec_element[0].shape[::-1])


def compress_file(ifile, overwrite=False):
    """
    Compress a file using gzip.  The output file has the same name as
    the input file with '.gz' appended.

    Any existing file will be overwritten if overwrite is true.

    An error is raised if the input file name already has '.gz' appended
    to the end.
    """
    if ifile.split('.')[-1] == 'gz':
        raise ValueError(f'{ifile} appears to already have been compressed!')

    ofile = f'{ifile}.gz'
    if os.path.isfile(ofile) and not overwrite:
        raise FileExistsError(f'File already exists: {ofile}.\nTo overwrite, set overwrite=True.')

    with open(ifile, 'rb') as f_in:
        with gzip.open(ofile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def create_symlink(ofile, symlink_dir, relative_symlink=True, overwrite=False, quiet=False):
    """
    Create a symlink to the input file in the provided directory.  If
    relative_symlink is True (default), the path to the file is relative
    to the directory with the symlink.
    """
    # Check if the file already exists
    olink_dest = os.path.join(symlink_dir, ofile.split('/')[-1])
    if os.path.isfile(olink_dest) or os.path.islink(olink_dest):
        if overwrite:
            os.remove(olink_dest)
        else:
            return

    # Make sure the symlink directory exists
    if not os.path.isdir(symlink_dir):
        os.makedirs(symlink_dir)

    # Set the relative path for the symlink, if requested
    olink_src = os.path.relpath(ofile, start=os.path.dirname(olink_dest)) \
                    if relative_symlink else ofile

    # Create the symlink
    os.symlink(olink_src, olink_dest)


# TODO: This is MaNGA specific and needs to be abstracted.
def initialize_primary_header(galmeta):
    hdr = fits.Header()

    hdr['MANGADR'] = (galmeta.dr, 'MaNGA Data Release')
    hdr['MANGAID'] = (galmeta.mangaid, 'MaNGA ID number')
    hdr['PLATEIFU'] = (f'{galmeta.plate}-{galmeta.ifu}', 'MaNGA observation plate and IFU')

    # Add versioning
    hdr['VERSPY'] = ('.'.join([ str(v) for v in sys.version_info[:3]]), 'Python version')
    hdr['VERSNP'] = (np.__version__, 'Numpy version')
    hdr['VERSSCI'] = (scipy.__version__, 'Scipy version')
    hdr['VERSAST'] = (astropy.__version__, 'Astropy version')
    hdr['VERSNIRV'] = (__version__, 'NIRVANA version')

    return hdr


def add_wcs(hdr, kin):
    if kin.grid_wcs is None:
        return hdr
    return hdr + kin.grid_wcs.to_header()


# TODO: Assumes uncertainties are provided as inverse variances...
def finalize_header(hdr, ext, bunit=None, hduclas2='DATA', err=False, qual=False, bm=None,
                    bit_type=None, prepend=True):

    # Don't change the input header
    _hdr = hdr.copy()

    # Add the units
    if bunit is not None:
        _hdr['BUNIT'] = (bunit, 'Unit of pixel value')

    # Add the common HDUCLASS keys
    _hdr['HDUCLASS'] = ('SDSS', 'SDSS format class')
    _hdr['HDUCLAS1'] = ('IMAGE', 'Data format')
    if hduclas2 == 'DATA':
        _hdr['HDUCLAS2'] = 'DATA'
        if err:
            _hdr['ERRDATA'] = (ext+'_IVAR' if prepend else 'IVAR',
                                'Associated inv. variance extension')
        if qual:
            _hdr['QUALDATA'] = (ext+'_MASK' if prepend else 'MASK',
                                'Associated quality extension')
        return _hdr

    if hduclas2 == 'ERROR':
        _hdr['HDUCLAS2'] = 'ERROR'
        _hdr['HDUCLAS3'] = ('INVMSE', 'Value is inverse mean-square error')
        _hdr['SCIDATA'] = (ext, 'Associated data extension')
        if qual:
            _hdr['QUALDATA'] = (ext+'_MASK' if prepend else 'MASK',
                                'Associated quality extension')
        return _hdr

    if hduclas2 == 'QUALITY':
        _hdr['HDUCLAS2'] = 'QUALITY'
        if bit_type is None:
            if bm is None:
                raise ValueError('Must provide the bit type or the bitmask object.')
            else:
                bit_type = bm.minimum_dtype()
        _hdr['HDUCLAS3'] = mask_data_type(bit_type)
        _hdr['SCIDATA'] = (ext, 'Associated data extension')
        if err:
            _hdr['ERRDATA'] = (ext+'_IVAR' if prepend else 'IVAR',
                                'Associated inv. variance extension')
        if bm is not None:
            # Add the bit values
            bm.to_header(_hdr)
        return _hdr
            
    raise ValueError('HDUCLAS2 must be DATA, ERROR, or QUALITY.')


def mask_data_type(bit_type):
    if bit_type in [np.uint64, np.int64]:
        return ('FLAG64BIT', '64-bit mask')
    if bit_type in [np.uint32, np.int32]:
        return ('FLAG32BIT', '32-bit mask')
    if bit_type in [np.uint16, np.int16]:
        return ('FLAG16BIT', '16-bit mask')
    if bit_type in [np.uint8, np.int8]:
        return ('FLAG8BIT', '8-bit mask')
    if bit_type == np.bool:
        return ('MASKZERO', 'Binary mask; zero values are good/unmasked')
    raise ValueError('Invalid bit_type: {0}!'.format(str(bit_type)))


def fileprep(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None,
        cen=True, fixcent=True, clip=True, remotedir=None,
        gal=None, galmeta=None):
    """
    Function to turn any nirvana output file into useful objects.

    Can take in `.fits`, `.nirv`, `dynesty.NestedSampler`_, or
    `dynesty.results.Results`_ along with any relevant parameters and spit
    out galaxy, result dictionary, all livepoint positions, and median values
    for each of the parameters.

    Args:
        f (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            `.fits` file, sampler, results, `.nirv` file of dumped results
            from :func:`~nirvana.fitting.fit`. If this is in the regular
            format from the automatic outfile generator in
            :func:`~nirvana.scripts.nirvana.main` then it will fill in most
            of the rest of the parameters by itself.
        plate (:obj:`int`, optional):
            MaNGA plate number for desired galaxy. Can be auto filled by `f`.
        ifu (:obj:`int`, optional):
            MaNGA IFU number for desired galaxy. Can be auto filled by `f`.
        smearing (:obj:`bool`, optional):
            Whether or not to apply beam smearing to models. Can be auto
            filled by `f`.
        stellar (:obj:`bool`, optional):
            Whether or not to use stellar velocity data instead of gas. Can
            be auto filled by `f`.
        maxr (:obj:`float`, optional):
            Maximum radius to make edges go out to in units of effective
            radii. Can be auto filled by `f`.
        cen (:obj:`bool`, optional):
            Whether the position of the center was fit. Can be auto filled by
            `f`.
        fixcent (:obj:`bool`, optional):
            Whether the center velocity bin was held at 0 in the fit. Can be
            auto filled by `f`.
        clip (:obj:`bool`, optional):
            Whether to apply clipping to the galaxy with
            :func:`~nirvana.data.kinematics.clip` as it is handling it.
        remotedir (:obj:`str`, optional):
            Directory to load MaNGA data files from, or save them if they are
            not found and are remotely downloaded.
        gal (:class:`~nirvana.data.fitargs.FitArgs`, optional):
            Galaxy object to use instead of loading the galaxy from scratch.
        
        Returns:
            :class:`~nirvana.data.fitargs.FitArgs`: Galaxy object containing
            relevant data and parameters. :obj:`dict`: Dictionary of results
            of the fit.
    """
    #unpack fits file
    if type(f) == str and '.fits' in f:
        isfits = True #tracker variable

        #open file and get relevant stuff from header
        with fits.open(f) as fitsfile:
            table = fitsfile[1].data
            maxr = fitsfile[0].header['maxr']
            smearing = fitsfile[0].header['smearing']

        #unpack bintable into dict
        keys = table.columns.names
        vals = [table[k][0] for k in keys]
        resdict = dict(zip(keys, vals))
        for v in ['vt','v2t','v2r','vtl','vtu','v2tl','v2tu','v2rl','v2ru']:
            resdict[v] = resdict[v][resdict['velmask'] == 0]
        for s in ['sig','sigl','sigu']:
            resdict[s] = resdict[s][resdict['sigmask'] == 0]

        #get galaxy object
        if gal is None:
            if resdict['type'] == 'Stars':
                args = MaNGAStellarKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
            else:
                args = MaNGAGasKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
        else:
            args = gal

        fill = len(resdict['velmask'])
        fixcent = resdict['vt'][0] == 0
        lenmeds = 6 + 3*(fill - resdict['velmask'].sum() - fixcent) + (fill - resdict['sigmask'].sum())
        meds = np.zeros(lenmeds)

    else:
        isfits = False

        #get sampler in right format
        if type(f) == str: chains = pickle.load(open(f,'rb'))
        elif type(f) == np.ndarray: chains = f
        elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results

        if gal is None and '.nirv' in f and os.path.isfile(f[:-5] + '.gal'):
            gal = f[:-5] + '.gal'
        if type(gal) == str: gal = np.load(gal, allow_pickle=True)

        #parse the automatically generated filename
        if plate is None or ifu is None:
            fname = re.split('/', f[:-5])[-1]
            info = re.split('/|-|_', fname)
            plate = int(info[0]) if plate is None else plate
            ifu = int(info[1]) if ifu is None else ifu
            stellar = True if 'stel' in info else False
            cen = True if 'nocen' not in info else False
            smearing = True if 'nosmear' not in info else False
            try: maxr = float([i for i in info if 'r' in i][0][:-1])
            except: maxr = None

            if 'fixcent' in info: fixcent = True
            elif 'freecent' in info: fixcent = False

        #mock galaxy using stored values
        if plate == 0:
            mock = np.load('mockparams.npy', allow_pickle=True)[ifu]
            print('Using mock:', mock['name'])
            params = [mock['inc'], mock['pa'], mock['pab'], mock['vsys'], mock['vts'], mock['v2ts'], mock['v2rs'], mock['sig']]
            args = Kinematics.mock(56,*params)
            cnvfftw = ConvolveFFTW(args.kin.spatial_shape)
            smeared = smear(args.kin.remap('vel'), args.kin.beam_fft, beam_fft=True, sig=args.kin.remap('sig'), sb=args.kin.remap('sb'), cnvfftw=cnvfftw)
            args.kin.sb  = args.kin.bin(smeared[0])
            args.kin.vel = args.kin.bin(smeared[1])
            args.kin.sig = args.kin.bin(smeared[2])
            args.fwhm  = 2.44

        #load input galaxy object
        elif gal is not None:
            args = gal

        #load in MaNGA data
        else:
            if stellar:
                args = MaNGAStellarKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)
            else:
                args = MaNGAGasKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)

    #set relevant parameters for galaxy
    args.setdisp(True)
    args.setnglobs(4) if not cen else args.setnglobs(6)
    args.setfixcent(fixcent)

    #clip data if desired
    if gal is not None: clip = False
    if clip: args.clip()

    vel_r = args.kin.remap('vel')
    sig_r = args.kin.remap('sig') if args.kin.sig_phys2 is None else np.sqrt(np.abs(args.kin.remap('sig_phys2')))

    if not isfits: meds = dynmeds(chains)

    #get appropriate number of edges  by looking at length of meds
    nbins = (len(meds) - args.nglobs - fixcent)/4
    if not nbins.is_integer(): 
        raise ValueError('Dynesty output array has a bad shape.')
    else: nbins = int(nbins)

    #calculate edges and velocity profiles, get basic data
    if not isfits:
        if gal is None: args.setedges(nbins - 1 + args.fixcent, nbin=True, maxr=maxr)
        resdict = profs(chains, args, stds=True)
        resdict['plate'] = plate
        resdict['ifu'] = ifu
        resdict['type'] = 'Stars' if stellar else 'Gas'
    else:
        args.edges = resdict['bin_edges'][~resdict['velmask']]

    args.getguess(galmeta=galmeta)
    args.getasym()

    return args, resdict

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

def imagefits(f, galmeta, gal=None, outfile=None, padding=20, remotedir=None, outdir='', drpalldir='.', dapalldir='.'):
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
    r, th = projected_polar(args.kin.x - resdict['xc'], args.kin.y - resdict['yc'], *np.radians((resdict['pa'], resdict['inc'])))
    r = args.kin.remap(r)
    th = args.kin.remap(th)

    data = dictformatting(resdict, padding=padding, drpalldir=drpalldir, dapalldir=dapalldir)
    data += [*np.delete(args.bounds.T, slice(7,-1), axis=1)]

    names = list(resdict.keys()) + ['velmask','sigmask','drpindex','dapindex','prior_lbound','prior_ubound']
    dtypes = ['f4','f4','f4','f4','f4','f4','20f4','20f4','20f4','20f4',
              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
              '20f4','20f4','20f4','20f4','20f4','20f4','20f4','20f4',
              'I','I','S','f4','20f4','20?','20?','I','I','8f4','8f4']

    #add parameters to the header
    #if galmeta==None:
        #drpallfile = glob(drpalldir + '/drpall*')[0]
        #galmeta = MaNGAGlobalPar(resdict['plate'], resdict['ifu'], drpall_file=drpallfile)
    hdr = initialize_primary_header(galmeta)
    maphdr = add_wcs(hdr, args)
    psfhdr = hdr.copy()
    psfhdr['PSFNAME'] = (args.kin.psf_name, 'Original PSF name')

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
    hdr['PHOT_INC'] = (args.kin.phot_inc, 'Photomentric inclination angle in deg')
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

    hdus += [maskedarraytofile(r, name='ell_r', hdr=finalize_header(maphdr, 'ell_r'))]
    hdus += [maskedarraytofile(th, name='ell_theta', hdr=finalize_header(maphdr, 'ell_th'))]

    #add all data extensions from original data
    mapnames = ['vel', 'sigsqr', 'sb', 'vel_ivar', 'sig_ivar', 'sb_ivar', 'vel_mask', 'sig_mask']
    units = ['km/s', '(km/2)^2', '1E-17 erg/s/cm^2/ang/spaxel', '(km/s)^{-2}', '(km/s)^{-4}', '(1E-17 erg/s/cm^2/ang/spaxel)^{-2}', None, None]
    errs = [True, True, True, False, False, False, True, True]
    quals = [True, True, False, True, True, False, False, False]
    hduclas2s = ['DATA', 'DATA', 'DATA', 'ERROR', 'ERROR', 'ERROR', 'QUALITY', 'QUALITY', 'QUALITY']
    bittypes = [None, None, None, None, None, None, np.bool, np.bool]

    for m, u, e, q, h, b in zip(mapnames, units, errs, quals, hduclas2s, bittypes):
        if m == 'sigsqr': 
            data = np.sqrt(args.kin.remap('sig_phys2').data)
            mask = args.kin.remap('sig_phys2').mask
        else:
            data = args.kin.remap(m).data
            mask = args.kin.remap(m).mask

        if data.dtype == bool: data = data.astype(int) #catch for bools
        data[mask] = 0 if 'mask' not in m else data[mask]
        hdus += [fits.ImageHDU(data, name=m, header=finalize_header(maphdr, m, u, h, e, q, None, b))]

    hdus += [fits.ImageHDU(args.kin.beam, name='PSF', header=finalize_header(psfhdr, 'PSF'))]

    #smeared and intrinsic velocity/dispersion models
    velmodel, sigmodel = bisym_model(args, resdict, plot=True, relative_pab=False)
    args.kin.beam_fft = None
    intvelmodel, intsigmodel = bisym_model(args, resdict, plot=True, relative_pab=False)

    #unmask them, name them all, and add them to the list
    models = [velmodel, sigmodel, intvelmodel, intsigmodel, asymmap]
    modelnames = ['vel_model','sig_model','vel_int_model','sig_int_model','asymmetry']
    units = ['km/s', '(km/s)^2', 'km/s', '(km/s)^2', None]
    for a, n, u in zip(models, modelnames, units):
        hdri = finalize_header(maphdr, n, u)
        hdus += [maskedarraytofile(a, name=n, hdr=hdri)]

    #write out
    hdul = fits.HDUList(hdus)
    if outfile is None: 
        outfile = f"nirvana_{resdict['plate']}-{resdict['ifu']}_{resdict['type']}.fits"
    hdul.writeto(outdir + outfile, overwrite=True, output_verify='fix', checksum=True)
