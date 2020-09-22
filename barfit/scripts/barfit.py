#!/usr/bin/env python

"""
Script that runs the fit.
"""

import argparse
import pickle

from barfit.barfit import barfit

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plateifu', nargs=2, type=int, help='MaNGA plate and ifu identifiers')
    parser.add_argument('-d', '--daptype', default='HYB10-MILESHC-MASTARHC', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-9', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the directory to the MaNGA galaxy data, and it will be '
                             'ignored if the root directory is set directly (using --root).')
    parser.add_argument('-c', '--cores', default=20, type=int,
                        help='Number of threads to utilize.')
    parser.add_argument('-f', '--outfile', default=None, type=str,
                        help='Outfile to dump results in.')
    parser.add_argument('-n', '--nbins', default=10, type=int,
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

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):

    plate, ifu = args.plateifu
    samp = barfit(plate, ifu, daptype=args.daptype, dr=args.dr, cores=args.cores, nbins=args.nbins,
                  weight=args.weight, maxr=args.maxr, smearing=args.smearing, root=args.root,
                  verbose=args.verbose)
    if args.outfile is None:
        args.outfile = f'{plate}-{ifu}_{args.nbins}bin_{args.weight}w_{args.points}p.out'
        if args.smearing: args.outfile += '_s'
        else: args.outfile += '_ns'

    # TODO: Do we need to use pickle?
    pickle.dump(samp.results, open(args.outfile, 'wb'))
