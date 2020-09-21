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
    parser.add_argument('--nosmear', action='store_true', default=False,
                        help='Don\'t use beam smearing to speed up fit')

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):

    plate, ifu = args.plateifu
    samp = barfit(plate, ifu, cores=args.cores, nbins=args.nbins, weight=args.weight,
                  maxr=args.maxr, smearing=~args.nosmear)

    if args.outfile is None:
        args.outfile = f'{args.plate}-{args.ifu}_{args.nbin}bin_{args.weight}w_{args.points}p'
        if args.nosmear: args.outfile += '_ns'
        else: args.outfile += '_s'

    # TODO: Do we need to use pickle?
    pickle.dump(samp.results, open(args.outfile, 'wb'))
