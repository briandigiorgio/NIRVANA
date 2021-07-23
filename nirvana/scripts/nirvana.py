#!/usr/bin/env python

"""
Script that runs the bisymmetric fit.
"""

import argparse
import pickle
import os
from glob import glob
import numpy as np

from nirvana.models.bisym import fit
from nirvana.util.fits_prep import fileprep, imagefits
from nirvana.data.manga import MaNGAGlobalPar

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plateifu', nargs=2, type=int, 
                        help='MaNGA plate and ifu identifiers')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-11', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the directory to the MaNGA galaxy data, and it will be '
                             'ignored if the root directory is set directly (using --root).')
    parser.add_argument('-c', '--cores', default=10, type=int,
                        help='Number of threads to utilize.')
    parser.add_argument('-f', '--outfile', default=None, type=str,
                        help='Outfile to dump results in.')
    parser.add_argument('-n', '--nbins', default=0, type=int,
                        help='Number of radial bins in fit.')
    parser.add_argument('-w', '--weight', default=10, type=int,
                        help='How much to weight smoothness of rotation curves in fit')
    parser.add_argument('-r', '--maxr', default=0.0, type=float,
                        help='Maximum radius in arcsec for bins')
    parser.add_argument('-p', '--points', default = 500, type=int,
                        help='Number of dynesty live points')
    parser.add_argument('--root', default=None, type=str,
                        help='Path with the fits files required for the fit; the DAP LOGCUBE '
                             'file is only required if the beam-smearing is included in the fit.')
    parser.add_argument('--nosmear', dest='smearing', default=True, action='store_false',
                        help='Don\'t use beam smearing to speed up fit')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Run dynesty sampling with verbose output.')
    parser.add_argument('--dir', type=str, default = '',
                        help='Directory to save the outfile in')
    parser.add_argument('--nodisp', dest='disp', default=True, action='store_false',
                        help='Turn off dispersion fitting')
    parser.add_argument('-s', '--stellar', default=False, action='store_true',
                        help='Fit stellar velocity field rather than gas')
    parser.add_argument('--nocen', dest='cen', default=True, action='store_false',
                        help='Fit the position of the center')
    parser.add_argument('--free', dest='fixcent', default=True, action='store_false',
                        help='Allow the center vel bin to be free')
    parser.add_argument('--nofits', dest='fits', default=True, action='store_false',
                        help='Save results as a much larger sampler output instead of as a FITS file')
    parser.add_argument('--remote', default=None, 
                        help='Download sas data into this dir instead of local')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Overwrite preexisting outfiles')
    parser.add_argument('--drpall_dir', default='.',
                        help='Path to drpall file. Will use first file in dir')
    parser.add_argument('--penalty', type=float, default=500,
                        help='Relative size of penalty for big 2nd order terms')
    parser.add_argument('--floor', type=float, default=5,
                        help='Error floor to add onto ivars')
    parser.add_argument('-m', '--mock', type=str, default='',
                        help='filepath to .fits output to turn into a mock observation and fit')
    parser.add_argument('-i', '--mock-inc', type=float, default=0,
                        help='change the inclination of the mock galaxy')
    parser.add_argument('--resid', type=str, default='',
                        help='Resuidual from residlib to add on top of vel')

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):
    #set save directory
    if args.dir == '':
        args.dir = '/data/manga/digiorgio/nirvana/'
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f'Outfile directory does not exist: {args.dir}')

    if args.nbins == 0: args.nbins = None
    if args.maxr == 0: args.maxr = None
    plate, ifu = args.plateifu

    try:
        drpall_file = glob(args.drpall_dir + '/drpall*.fits')[0]
        galmeta = MaNGAGlobalPar(plate, ifu, drpall_file=drpall_file)
    except Exception as e: 
        print(e)
        galmeta = None


    #make descriptive outfile name
    if args.outfile is None:
        args.outfile = f'{plate}-{ifu}_{args.weight}w_{args.points}p'
        if args.nbins is not None: args.outfile += f'_{args.nbins}bin'
        if args.maxr  is not None: args.outfile += f'_{args.maxr}r'
        if not args.smearing: args.outfile += '_nosmear'
        if not args.disp: args.outfile += '_nodisp'
        if args.stellar: args.outfile += '_stel'
        else: args.outfile += '_gas'
        if not args.cen: args.outfile += '_nocen'
        if not args.fixcent: args.outfile += '_freecent'
        if args.mock: args.outfile += '_mock'
        if args.mock_inc: args.outfile += f'_i{int(args.mock_inc)}'
        if args.resid: args.outfile += f'_r{args.resid}'

        fitsname = f"{args.dir}nirvana_{plate}-{ifu}_{vftype}" 
        if args.mock:
            fitsname += '_mock'
            if args.mock_inc: fitsname += f'_i{int(args.mock_inc)}'
            if args.resid: fitsname += f'_r{args.resid}'
    else: fitsname = args.dir + args.outfile

    print('File name:', args.outfile)
    if args.stellar: vftype = 'Stars'
    else: vftype = 'Gas'

    fname = args.dir + args.outfile + '.nirv'
    galname = args.dir + args.outfile + '.gal'
    fitsname += '.fits'

    #check if outfile already exists
    if not args.clobber:
        if os.path.isfile(fname) and not args.fits:
            raise FileExistsError(f'Output .nirv file already exists. Use --clobber to overwrite it: {fname}')
        elif args.fits and os.path.isfile(fitsname):
            raise FileExistsError(f'Output FITS file already exists. Use --clobber to overwrite it: {fitsname}')
    if args.fits and os.path.isfile(fname):
        print(f'Output .nirv file found, converting to FITS: {fname} --> {fitsname}')
        gal = np.load(galname, allow_pickle=True)
        imagefits(fname, galmeta, gal, outfile=fitsname, remotedir=args.remote) 
        os.remove(fname)
        os.remove(galname)
        return

    if args.mock: 
        mockgal, params = fileprep(args.mock, remotedir=args.remote)
        if args.mock_inc: 
            params['inc'] = args.mock_inc
            mockgal.kin.phot_inc = args.mock_inc
        mock = (mockgal, params, args.resid)
    else: mock = None

    #run fit with supplied args
    samp, gal = fit(plate, ifu, galmeta=galmeta, daptype=args.daptype, dr=args.dr, cores=args.cores, nbins=args.nbins,
                  weight=args.weight, maxr=args.maxr, smearing=args.smearing, root=args.root,
                  verbose=args.verbose, disp=args.disp, points=args.points, 
                  stellar=args.stellar, cen=args.cen, fixcent=args.fixcent,
                  remotedir=args.remote, mock=mock, penalty=args.penalty)

    #write out with sampler results or just FITS table
    pickle.dump(samp.results, open(fname, 'wb'))
    pickle.dump(gal, open(galname, 'wb'))
    if args.fits: 
        try:
            imagefits(fname, galmeta, gal, outfile=fitsname, remotedir=args.remote) 
            os.remove(fname)
            os.remove(galname)
        except Exception:
            raise ValueError('Unable to save as FITS. Output still available as .nirv and .gal')
