#!/usr/bin/env python

"""
Script that runs the fit.
"""

import argparse
import pickle
import os

from nirvana.fitting import fit
from nirvana.output import imagefits

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plateifu', nargs=2, type=int, 
                        help='MaNGA plate and ifu identifiers')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-10', type=str,
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
                        help='Maximum radius in Re for bins')
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
    parser.add_argument('--fits', default=False, action='store_true',
                        help='Save results as a much smaller FITS file instead')
    parser.add_argument('--remote', default=None, 
                        help='Download sas data into this dir instead of local')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Overwrite preexisting outfiles')

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):
    #set save directory
    if args.dir == '':
        args.dir = '/data/manga/digiorgio/nirvana/'
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f'Outfile directory does not exist: {args.dir}')
    if args.nbins == 0: args.nbins = None

    #make descriptive outfile name
    plate, ifu = args.plateifu
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
    fname = args.dir + args.outfile + '.nirv'
    fitsname = args.dir + args.outfile + '.fits'
    galname = args.dir + args.outfile + '.gal'

    #check if outfile already exists
    if not args.clobber and os.path.isfile(fname):
        raise FileExistsError(f'Output .nirv file already exists. Use --clobber to overwrite it: {fname}')
    elif args.clobber and args.fits and os.path.isfile(fitsname):
        raise FileExistsError(f'Output FITS file already exists. Use --clobber to overwrite it: {fitsname}')

    #run fit with supplied args
    samp, gal = fit(plate, ifu, daptype=args.daptype, dr=args.dr, cores=args.cores, nbins=args.nbins,
                  weight=args.weight, maxr=args.maxr, smearing=args.smearing, root=args.root,
                  verbose=args.verbose, disp=args.disp, points=args.points, 
                  stellar=args.stellar, cen=args.cen, fixcent=args.fixcent,
                  remotedir=args.remote)

    #write out with sampler results or just FITS table
    pickle.dump(samp.results, open(fname, 'wb'))
    pickle.dump(gal, open(galname, 'wb'))
    if args.fits: 
        try:
            imagefits(fname, gal, outfile=fitsname, remotedir=remotedir) 
            os.remove(fname)
            os.remove(galname)
        except:
            raise ValueError('Unable to save as FITS. Output still available as .nirv')
