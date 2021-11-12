r"""
Various utilities for use with fits files.

----

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
import sys
import os
from glob import glob
import traceback
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import dynesty
from astropy.io import fits
from astropy.table import Table,Column
from tqdm import tqdm

from ..models.higher_order import bisym_model
from ..models.geometry import projected_polar
from ..models.asymmetry import asymmetry
from ..data.manga import MaNGAStellarKinematics, MaNGAGasKinematics
from ..data.fitargs import FitArgs
from ..data.util import unpack
from .fileio import initialize_primary_header, add_wcs, finalize_header

def dynmeds(samp, stds=False, fixcent=True):
    """
    Get median values for each variable's posterior in a
    `dynesty.NestedSampler`_ sampler.

    Args:
        samp (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            Sampler, results, or file of dumped results from `dynesty`_ fit.
        stds (:obj:`bool`, optional):
            Flag for whether or not to return standard deviations of the
            posteriors as well.

    Returns:
        `numpy.ndarray`_: Median values of all of the parameters in the
        `dynesty`_ sampler. If ``stds == True``, it will instead return a
        :obj:`tuple` of three `numpy.ndarray`_ objects. The first is the
        median values, the second is the lower 1 sigma bound for all of the
        posteriors, and the third is the upper 1 sigma bound.
    """

    #get samples and weights
    if type(samp) == str: res = pickle.load(open(samp,'rb'))
    elif type(samp)==dynesty.results.Results: res = samp
    else: res = samp.results
    samps = res.samples
    weights = np.exp(res.logwt - res.logz[-1])

    #iterate through and get 50th percentile of values
    meds = np.zeros(samps.shape[1])
    for i in range(samps.shape[1]):
        meds[i] = dynesty.utils.quantile(samps[:,i],[.5],weights)[0]

    #pull out 1 sigma values on either side of the mean as well if desired
    if stds:
        lstd = np.zeros(samps.shape[1])
        ustd = np.zeros(samps.shape[1])
        for i in range(samps.shape[1]):
            lstd[i] = dynesty.utils.quantile(samps[:,i], [.5-.6826/2], weights)[0]
            ustd[i] = dynesty.utils.quantile(samps[:,i], [.5+.6826/2], weights)[0]
        return meds, lstd, ustd

    return meds

def profs(samp, args, plot=None, stds=False, jump=None, **kwargs):
    '''
    Turn a sampler output by `nirvana` into a set of rotation curves.
    
    Args:
        samp (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            Sampler, results, or file of dumped results from
            :func:`~nirvana.fitting.fit`
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        plot (:class:`matplotlib.axes._subplots.Axes`, optional):
            Axis to plot the rotation curves on. If not specified, it will not
            try to plot anything.
        stds (:obj:`bool`, optional):
            Flag for whether to fetch the standard deviations as well.
        jump (:obj:`int`, optional):
            Number of radial bins in the sampler. Will be calculated
            automatically if not specified.
        **kwargs:
            args for :func:`plt.plot`.

    Returns:
        :obj:`dict`: Dictionary with all of the median values of the
        posteriors in the sampler. Has keys for inclination `inc`, first
        order position angle `pa`, second order position angle `pab`,
        systemic velocity `vsys`, x and y center coordinates `xc` and `yc`,
        `numpy.ndarray`_ of first order tangential velocities `vt`,
        `numpy.ndarray`_ objects of second order tangential and radial
        velocities `v2t` and `v2r`, and `numpy.ndarray`_ of velocity
        dispersions `sig`. If `stds == True` it will also contain keys for
        the 1 sigma lower bounds of the velocity parameters `vtl`, `v2tl`,
        `v2rl`, and `sigl` as well as their 1 sigma upper bounds `vtu`,
        `v2tu`, `v2ru`, and `sigu`. Arrays have lengths that are the same as
        the number of bins (determined automatically or from `jump`). All
        angles are in degrees and all velocities must be in consistent units.

        If `plot` is not `None`, it will also display a plot of the profiles.
    '''

    #get and unpack median values for params
    meds = dynmeds(samp, stds=stds, fixcent=args.fixcent)

    #get standard deviations and put them into the dictionary
    if stds:
        meds, lstd, ustd = meds
        paramdict = unpack(meds, args, jump=jump, relative_pab=False)
        paramdict['incl'], paramdict['pal'], paramdict['pabl'], paramdict['vsysl'] = lstd[:4]
        paramdict['incu'], paramdict['pau'], paramdict['pabu'], paramdict['vsysu'] = ustd[:4]
        print('*'*20, paramdict['pab'], paramdict['pabl'], paramdict['pabu'])
        if args.nglobs == 6:
            paramdict['xcl'], paramdict['ycl'] = lstd[4:6]
            paramdict['xcu'], paramdict['ycu'] = ustd[4:6]

        start = args.nglobs
        jump = len(args.edges) - args.fixcent
        vs = ['vt', 'v2t', 'v2r']
        for i,v in enumerate(vs):
            for b in ['l','u']:
                exec(f'paramdict["{v}{b}"] = {b}std[start + {i}*jump : start + {i+1}*jump]')
                if args.fixcent:
                    exec(f'paramdict["{v}{b}"] = np.insert(paramdict["{v}{b}"], 0, 0)')

        #dispersion stds
        if args.disp: 
            sigjump = jump + 1
            paramdict['sigl'] = lstd[start + 3*jump:start + 3*jump + sigjump]
            paramdict['sigu'] = ustd[start + 3*jump:start + 3*jump + sigjump]

    else: paramdict = unpack(meds, args, jump=jump)

    #plot profiles if desired
    if plot is not None: 
        if not isinstance(plot, matplotlib.axes._subplots.Axes): f,plot = plt.subplots()
        ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
        [plot.plot(args.edges, p, label=ls[i], **kwargs) 
                for i,p in enumerate([paramdict['vt'], paramdict['v2t'], paramdict['v2r']])]

        #add in lower and upper bounds
        if stds: 
            errors = [[paramdict['vtl'], paramdict['vtu']], [paramdict['v2tl'], paramdict['v2tu']], [paramdict['v2rl'], paramdict['v2ru']]]
            for i,p in enumerate(errors):
                plot.fill_between(args.edges, p[0], p[1], alpha=.5) 

        plt.xlabel(r'$R_e$')
        plt.ylabel(r'$v$ (km/s)')
        plt.legend(loc=2)

    return paramdict

def fileprep(f, plate=None, ifu=None, smearing=None, stellar=False, maxr=None,
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
        galmeta (:class:`~nirvana.data.manga.MaNGAGlobalPar`, optional):
            Info on MaNGA galaxy used for plate and ifu

        
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
            smearing = fitsfile[0].header['smearing'] if smearing is None else smearing
            scatter = fitsfile[0].header['scatter']

        #unpack bintable into dict
        keys = table.columns.names
        vals = [table[k][0] for k in keys]
        resdict = dict(zip(keys, vals))
        for v in ['vt','v2t','v2r','vtl','vtu','v2tl','v2tu','v2rl','v2ru']:
            resdict[v] = resdict[v][resdict['velmask'] == 0]
        for s in ['sig','sigl','sigu']:
            resdict[s] = resdict[s][resdict['sigmask'] == 0]

        #failsafe
        if 'Stars' in f: resdict['type'] = 'Stars'

        #get galaxy object
        if gal is None:
            if resdict['type'] == 'Stars':
                kin = MaNGAStellarKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
            else:
                kin = MaNGAGasKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
            scatter = ('vel_scatter' in resdict.keys()) and (resdict['vel_scatter'] != 0)
        else:
            kin = gal
            scatter = gal.scatter

        fill = len(resdict['velmask'])
        fixcent = resdict['vt'][0] == 0
        lenmeds = 6 + 3*(fill - resdict['velmask'].sum() - fixcent) + (fill - resdict['sigmask'].sum()) + 2*scatter
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

        #load input galaxy object
        if gal is not None:
            kin = gal

        #load in MaNGA data
        else:
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

            if stellar:
                kin = MaNGAStellarKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)
            else:
                kin = MaNGAGasKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)

        print(stellar)
    #set relevant parameters for galaxy
    if isinstance(kin, FitArgs): args = kin
    else: args = FitArgs(kin, smearing=smearing, scatter=scatter)
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
    nbins = (len(meds) - args.nglobs - fixcent - 2*args.scatter)/4
    if not nbins.is_integer(): 
        print(len(meds), args.nglobs, fixcent, 2*args.scatter, nbins)
        raise ValueError('Dynesty output array has a bad shape.')
    else: nbins = int(nbins)

    #calculate edges and velocity profiles, get basic data
    if not isfits:
        if gal is None: args.setedges(nbins - 1 + args.fixcent, nbin=True, maxr=maxr)
        resdict = profs(chains, args, stds=True)
        resdict['plate'] = galmeta.plate if galmeta is not None else None
        resdict['ifu'] = galmeta.ifu if galmeta is not None else None
        resdict['type'] = 'Stars' if stellar else 'Gas'
    else:
        args.edges = resdict['bin_edges'][~resdict['velmask']]

    args.getguess(galmeta=galmeta)
    args.getasym()

    return args, resdict

def extractfile(f, remotedir=None, gal=None, galmeta=None):
    '''
    Take a filename, open it, and extract the info needed for output files.

    Tries to open a file using :func:`~nirvana.util.fits_prep.fileprep` to get
    the :class:`~nirvana.data.fitargs.FitArgs` object and the dictionary of its
    fit parameters. Then get its asymmetry parameter and map, returning all of
    that.

    Args:
        f (:obj:`str`):
            Complete filename to be extracted with
            :func:`~nirvana.util.fits_prep.fileprep`.
        remotedir (:obj:`str`, optional):
            Directory to look for downloaded data in.
        gal (:class:`~nirvana.data.fitargs.FitArgs`, optional):
            Pre loaded galaxy object to use instead of reloading galaxy
        galmeta (:class:`~nirvana.data.manga.MaNGAGlobalPar`, optional):
            Global data used for :func:`~nirvana.util.fits_prep.fileprep`.

    Returns:
        :class:`~nirvana.data.fitargs.FitArgs`: Galaxy object

        :obj:`float`: Asymmetry parameter

        `numpy.ndarray`_: 2D map of spatially resolved asymmetry

        :obj:`dict`: Dictionary of fit parameters

    Raises:
        Exception:
            Raised if any part of the file extraction process fails. Should
            provide a useful error message.
    '''
    try: 
        #get info out of each file and make bisym model
        args, resdict = fileprep(f, remotedir=remotedir, gal=gal, galmeta=galmeta)

        inc, pa, pab, vsys, xc, yc = args.guess[:6]
        arc, asymmap = asymmetry(args.kin, pa, vsys, xc, yc)
        resdict['a_rc'] = arc

    #failure if bad file
    except Exception:
        print(f'Extraction of {f} failed:')
        print(traceback.format_exc())
        args, arc, asymmap, resdict = (None, None, None, None)

    return args, arc, asymmap, resdict

def extractdir(cores=10, directory='/data/manga/digiorgio/nirvana/'):
    '''
    Scan an entire directory for nirvana output files and extract useful data
    from them.

    Looks for all available .nirv files in a given directory and runs
    :func:`~nirvana.util.fits_prep.extractfile` on all of them, producing
    arrays of the outputs. Uses multiprocessing for parallel loading of files.

    Args:
        cores (:obj:`int`, optional):
            Number of cores to use for multiprocessing.
        directory (:obj:`str`, optional):
            Directory to look for .nirv files in.

    Returns:
        `numpy.ndarray`_: array of :class:`~nirvana.data.fitargs.FitArgs`
        galaxy objects

        `numpy.ndarray`_: array of :obj:`float` asymmetry parameters

        `numpy.ndarray`_: array of 2D maps of spatially resolved asymmetry

        `numpy.ndarray`_: array of :obj:`dict` dictionaries of fit parameters
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
    '''
    Reformat results dictionaries so they can be put into FITS tables.

    Take the dictionaries that come out of
    :func:`~nirvana.util.fits_prep.fileprep` and make their velocity profiles a
    standard length to accommodate limitations of the FITS format. Also apply a
    mask to the profiles with filler values.

    Args:
        d (:obj:`dict`):
            Dictionary of fit results.
        drp (`numpy.ndarray`_, optional):
            Data array from DRPAll file
        dap (`numpy.ndarray`_, optional):
            Data array from DAPAll file
        padding (:obj:`int`, optional):
            Maximum length to pad velocity profiles out to
        fill (:obj:`float`, optional):
            Value to fill in while padding velocity profiles
        drpalldir (:obj:`str`, optional):
            Path to look for DRPAll files for. If the DRPAll array isn't
            already supplied, it will look in this directory for files that
            match 'drpall*' and load the first one.
        dapalldir (:obj:`str`, optional):
            Path to look for DAPAll files for. If the DAPAll array isn't
            already supplied, it will look in this directory for files that
            match 'dapall*' and load the first one.

    Returns:
        :obj:`dict`: Reformatted dictionary.

    Raises:
        Exception:
            Raised if dictionary is empty or isn't in the correct format
    '''

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
        print(d['plate'],d['ifu'])
        drpindex = np.where(drp['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        dapindex = np.where(dap['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        data += [velmask, sigmask, drpindex, dapindex]

    #failure for empty dict
    except Exception as e:
        print(e)
        data = None

    return data

def makealltable(fname='', dir='.', vftype='', outfile=None):
    '''
    Combine the fit parameter tables for all of the FITS files in a given
    directory of a given velocity field type into one table.

    Looks for all of the FITS files matching a certain filename and velocity
    field type (stars or gas) and combines their tables of fit parameters into
    one giant Astropy table. This can also be saved as another FITS file if a
    filename is given.

    Args:
        fname (:obj:`str`, optional):
            All FITS files containing this string will be loaded and combined
        dir (:obj:`str`, optional):
            Directory to look for FITS files in
        vftype (:obj:`str`, optional):
            Specific type of velocity field to get the FITS files for ('Stars' or 'Gas')
        outfile (:obj:`str`, optional):
            If given, the output table will be written to this filename

    Returns:
        :class:`~astropy.table.Table`: table of combined fit data

    Raises:
        FileNotFoundError:
            Raised if no FITS files matching the description are found
    '''

    #load dapall and drpall
    drp = fits.open('/data/manga/spectro/redux/MPL-11/drpall-v3_1_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-11/dapall-v3_1_1-3.1.0.fits')[1].data

    #look for fits files
    fs = glob(f'{dir}/{fname}*{vftype}.fits')
    if len(fs) == 0:
        raise FileNotFoundError(f'No matching FITS files found in directory "{dir}"')
    else:
        print(len(fs), 'files found...')

    #combine all fits tables into one list
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

    #put all of the data from the tables into one array
    data = np.zeros(len(tables), dtype=dtype)
    for i in range(len(tables)):
        data[i] = tables[i]

    #turn array into an astropy table
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
    
    Numpy says it's not implemented yet so I'm implementing it. It takes a masked array, replaces the masked areas with a fill value, then writes it to an HDU.

    Args:
        array (`numpy.ma.array`_):
            Masked array to be written to an HDU
        name (:obj:`str`, optional):
            Name for the HDU
        fill (:obj:`float`, optional):
            Fill value to put in masked areas
        hdr (:class:`~astropy.fits.Header`, optional):
            Header for resulting HDU

    Returns:
        :class:`~astropy.fits.ImageHDU`: HDU of the masked array
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
    args, arc, asymmap, resdict = extractfile(f, remotedir=remotedir, gal=gal, galmeta=galmeta)
    if gal is not None: args = gal
    resdict['bin_edges'] = np.array(args.edges)
    r, th = projected_polar(args.kin.x - resdict['xc'], args.kin.y - resdict['yc'], *np.radians((resdict['pa'], resdict['inc'])))
    r = args.kin.remap(r)
    th = args.kin.remap(th)

    if not args.scatter or 'vel_scatter' not in resdict.keys():
        resdict['vel_scatter'] = 0
        resdict['sig_scatter'] = 0

    data = dictformatting(resdict, padding=padding, drpalldir=drpalldir, dapalldir=dapalldir)
    data += [*np.delete(args.bounds.T, slice(7,-1), axis=1)]

    names = list(resdict.keys()) + ['velmask','sigmask','drpindex','dapindex','prior_lbound','prior_ubound']
    dtypes = ['f4','f4','f4','f4','f4','f4','20f4','20f4','20f4','20f4',
              'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',
              '20f4','20f4','20f4','20f4','20f4','20f4','20f4','20f4',
              'I','I','S','f4','20f4','f4','f4','20?','20?','I','I','8f4','8f4']

    #add parameters to the header
    #if galmeta==None:
        #drpallfile = glob(drpalldir + '/drpall*')[0]
        #galmeta = MaNGAGlobalPar(resdict['plate'], resdict['ifu'], drpall_file=drpallfile)
    hdr = initialize_primary_header(galmeta)
    maphdr = add_wcs(hdr, args.kin)
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
    hdr['scatter'] = (args.scatter, 'Whether intrinsic scatter was included')

    avmax, ainc, apa, ahrot, avsys = args.getguess(simple=True, galmeta=galmeta)
    hdr['a_vmax'] = (avmax, 'Axisymmetric asymptotic velocity in km/s')
    hdr['a_pa'] = (apa, 'Axisymmetric position angle in deg')
    hdr['a_inc'] = (ainc, 'Axisymmetric inclination angle in deg')
    hdr['a_vsys'] = (avsys, 'Axisymmetric systemic velocity in km/s')

    #make table of fit data
    t = Table(names=names, dtype=dtypes)
    t.add_row(data)
    reordered = ['plate','ifu','type','drpindex','dapindex','bin_edges','prior_lbound','prior_ubound',
          'xc','yc','inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask', 'vel_scatter', 'sig_scatter',
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

def fig2data(fig):
    '''
    Take a `matplolib` figure and return it as an array of RGBA values.

    Stolen from somewhere on Stack Overflow.

    Args:
        fig (`matplotlib.figure.Figure`_):
            Figure to be turned into an array.

    Returns:
        `numpy.ndarray`_: RGBA array representation of the figure.
    '''

    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    h,w = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
