#!/usr/bin/env python

"""
Script that runs the fit.
"""

import argparse
import pickle
import os

from nirvana.fitting import fit

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
    parser.add_argument('-r', '--maxr', default=1.5, type=float,
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

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):
    #set save directory
    if args.dir == '':
        args.dir = '/data/manga/digiorgio/nirvana/'
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f'Outfile directory does not exist: {args.dir}')
    if args.nbins == 0: args.nbins = None

    #run fit with supplied args
    plate, ifu = args.plateifu
    samp = fit(plate, ifu, daptype=args.daptype, dr=args.dr, cores=args.cores, nbins=args.nbins,
                  weight=args.weight, maxr=args.maxr, smearing=args.smearing, root=args.root,
                  verbose=args.verbose, disp=args.disp, points=args.points)

    #make descriptive outfile name
    if args.outfile is None:
        args.outfile = f'{plate}-{ifu}_{args.weight}w_{args.points}p'
        if args.nbins is not None: args.outfile += f'_{args.nbins}bin'
        if args.maxr  is not None: args.outfile += f'_{args.maxr}r'
        if not args.smearing: args.outfile += '_nosmear'
        if not args.disp: args.outfile += '_nodisp'
    args.outfile += '.nirv'

    # TODO: Do we need to use pickle?
    pickle.dump(samp.results, open(args.dir+args.outfile, 'wb'))
