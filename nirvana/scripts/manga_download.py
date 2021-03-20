"""
Script to download MaNGA data needed to run nirvana.
"""
import os
import argparse

from IPython import embed

from matplotlib import pyplot

from ..data import manga

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plate', default=None, type=int, 
                        help='MaNGA plate identifier (e.g., 8138)')
    parser.add_argument('ifu', default=None, type=int, 
                        help='MaNGA ifu identifier (e.g., 12704)')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.')
    parser.add_argument('--dr', default='MPL-11', type=str,
                        help='The MaNGA data release.')
    parser.add_argument('--redux', default=None, type=str,
                        help='Top-level directory with the MaNGA DRP output. This is used to '
                             'construct the output directory structure to mimic the SAS. If not '
                             'defined and the direct root for the output is also not defined '
                             '(see --root), this is set by the environmental variable '
                             'MANGA_SPECTRO_REDUX.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output. This is used to '
                             'construct the output directory structure to mimic the SAS. If not '
                             'defined and the direct root for the output is also not defined '
                             '(see --root), this is set by the environmental variable '
                             'MANGA_SPECTRO_ANALYSIS.') 
    parser.add_argument('--root', default=None, type=str,
                        help='Output path for *all* files.  This includes the DRP LOGCUBE file, '
                             'the DAP MAPS file, and the SDSS multicolor finding chart image.  '
                             'For the latter, note that the name if the file is just the IFU '
                             'number, meaning that files can be overwritten if you download '
                             'data into the same directory from multiple plates.')
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite any existing files.')

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):
    files = download_plateifu(args.plate, args.ifu, daptype=args.daptype, dr=args.dr,
                              oroot=args.root, redux_path=args.redux_path,
                              analysis_path=args.analysis_path, overwrite=args.overwrite)

