"""
Script to collate results from AxisymmetricDisk fits to MaNGA data into a single
fits file.
"""
import os
import pathlib
import argparse

from IPython import embed

import numpy as np
from astropy.io import fits

import nirvana
from nirvana.data.manga import manga_paths, manga_file_names
from nirvana.models.axisym import AxisymmetricDisk, _fit_meta_dtype
from nirvana.models.oned import Func1D
from nirvana.util import fileio
from nirvana.util.inspect import all_subclasses

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str,
                        help='Top-level directory with the output from the nirvana_manga_axisym '
                             'script.  This script will recursively comb through the directory '
                             'structure, looking for files produced by the script.  The file '
                             'names are expected to be '
                             '"nirvana-manga-axisym-[plate]-[ifu]-[tracer].fits.gz".  Each '
                             'combination of [plate], [ifu], and [tracer] *must* be unique.  '
                             'All files must also be the result of fitting the same '
                             'parameterization of the axisymmetric model.')
    parser.add_argument('ofile', type=str, help='Name for the output file.')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is used '
                             'to select the relevant extension of the DAPall file for '
                             'crossmatching.')
    parser.add_argument('--dr', default='MPL-11', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the MaNGA DRPall and DAPall file names.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output.  If not defined, '
                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.  '
                             'This is only used to automatically construct the MaNGA DRPall and '
                             'DAPall file names.')
    parser.add_argument('--dapall', default=None, type=str,
                        help='The full path to the MaNGA DAPall file.  If None, the file name '
                             'is constructed assuming the default paths.')
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help='Instead of trolling the directory for output files, attempt to '
                             'find output for all plateifus in the DAPall file, for both Gas '
                             'and Stars.')
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite existing an existing output file.')

    # TODO: Get rid of these two options once the fits files have been updated
    # to include this info in the output file header.
    parser.add_argument('--rc', default='HyperbolicTangent', type=str,
                        help='Class used to parameterize the rotation curve.')
    parser.add_argument('--dc', default=None, type=str,
                        help='Class used to parameterize the dispersion profile, if it was fit.')
    
    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):

    nirvana_root = 'nirvana-manga-axisym'

    # Check input directory and attempt to find files to collate
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f'{args.dir} does not exist!')
    files = [str(p.resolve()) 
             for p in pathlib.Path(args.dir).rglob(f'{nirvana_root}*.fits.gz')]
    if len(files) == 0:
        raise ValueError(f'No files found with the expected naming convention in {args.dir}!')

    oroot = np.unique([os.path.dirname(os.path.dirname(f)) for f in files])
    if len(oroot) > 1:
        raise ValueError('Currently cannot handle more than one root directory.')
    oroot = oroot[0]

    # Get the list of plates, ifus, and tracers
    if not args.full:
        pltifutrc = np.array(['-'.join(os.path.basename(f).split('.')[0].split('-')[-3:])
                              for f in files])
        if pltifutrc.size != np.unique(pltifutrc).size:
            raise ValueError(f'All plate-ifu-tracer must be unique for files found in {args.dir}!')
        plateifu = np.unique(['-'.join(p.split('-')[:2]) for p in pltifutrc])
    else:
        plateifu = None

    # Check the output file and determine if it is expected to be compressed
    if os.path.isfile(args.ofile) and not args.overwrite:
        raise FileExistsError(f'{args.ofile} already exists!')
    if args.ofile.split('.')[-1] == 'gz':
        _ofile = args.ofile[:args.ofile.rfind('.')]
        compress = True
    else:
        _ofile = args.ofile

    # Attempt to find the DRPall and DAPall files
    if args.dapall is None:
        _dapall_path = manga_paths(0, 0, dr=args.dr, analysis_path=args.analysis)[3]
        _dapall_file = manga_file_names(0, 0, dr=args.dr)[3]
        _dapall_file = os.path.join(_dapall_path, _dapall_file) \
                        if args.dapall is None else args.dapall
    else:
        _dapall_file = args.dapall
    if not os.path.isfile(_dapall_file):
        raise FileNotFoundError(f'{_dapall_file} does not exist!')

    # Check the input rotation curve and dispersion profile parameterization
    # TODO: Update this once the fits files have been updated to include the
    # parameterization names.
    func1d_classes = all_subclasses(Func1D)
    func1d_class_names = [c.__name__ for c in func1d_classes]
    if args.rc not in func1d_class_names:
        raise ValueError(f'{args.rc} is not a known parameterization.')
    if args.dc is not None and args.dc not in func1d_class_names:
        raise ValueError(f'{args.rc} is not a known parameterization.')
    disk = AxisymmetricDisk(rc=func1d_classes[func1d_class_names.index(args.rc)](),
                            dc=None if args.dc is None 
                                else func1d_classes[func1d_class_names.index(args.dc)]())

    # Get the data type for the output table
    _dtype = _fit_meta_dtype(disk.par_names(short=True))
    meta_keys = [d[0] for d in _dtype]
    _dtype += [('DRPALLINDX', np.int), ('DAPALLINDX', np.int), ('SUCCESS', np.int)]

    # Read the DAPall file
    with fits.open(_dapall_file) as hdu:
        dapall = hdu[args.daptype].data

    # Generate the list of observations to fit
    indx = np.where(dapall['DAPDONE'])[0] if args.full \
                else np.array([np.where(dapall['PLATEIFU'] == p)[0][0] for p in plateifu])

    gas_metadata = fileio.init_record_array(indx.size, _dtype)
    str_metadata = fileio.init_record_array(indx.size, _dtype)

    for i, j in enumerate(indx):
        print(f'Collating {i+1}/{indx.size}', end='\r')
        plate = dapall['PLATE'][j]
        ifu = dapall['IFUDESIGN'][j]

        gas_metadata['DRPALLINDX'][i] = dapall['DRPALLINDX'][j]
        gas_metadata['DAPALLINDX'][i] = j
        gas_file = os.path.join(oroot, str(plate), f'{nirvana_root}-{plate}-{ifu}-Gas.fits.gz')
        if os.path.isfile(gas_file):
            gas_metadata['SUCCESS'][i] = 1
            with fits.open(gas_file) as hdu:
                for k in meta_keys:
                    gas_metadata[k][i] = hdu['FITMETA'].data[k][0]
        else:
            gas_metadata['MANGAID'][i] = dapall['MANGAID'][j]
            gas_metadata['PLATE'][i] = plate
            gas_metadata['IFU'][i] = ifu
            

        str_metadata['DRPALLINDX'][i] = dapall['DRPALLINDX'][j]
        str_metadata['DAPALLINDX'][i] = j
        str_file = os.path.join(oroot, str(plate), f'{nirvana_root}-{plate}-{ifu}-Stars.fits.gz')
        if os.path.isfile(str_file):
            str_metadata['SUCCESS'][i] = 1
            with fits.open(str_file) as hdu:
                for k in meta_keys:
                    str_metadata[k][i] = hdu['FITMETA'].data[k][0]
        else:
            str_metadata['MANGAID'][i] = dapall['MANGAID'][j]
            str_metadata['PLATE'][i] = plate
            str_metadata['IFU'][i] = ifu

    print(f'Collating {indx.size}/{indx.size}')

    # TODO: Add to the primary header?
    fits.HDUList([fits.PrimaryHDU(),
                  fits.BinTableHDU.from_columns(
                        [fits.Column(name=n, format=fileio.rec_to_fits_type(gas_metadata[n]),
                                     array=gas_metadata[n]) for n in gas_metadata.dtype.names],
                        name='GAS'),
                  fits.BinTableHDU.from_columns(
                        [fits.Column(name=n, format=fileio.rec_to_fits_type(str_metadata[n]),
                                     array=str_metadata[n]) for n in str_metadata.dtype.names],
                        name='STARS')]).writeto(_ofile, overwrite=True, checksum=True)
    if compress:
        fileio.compress_file(_ofile, overwrite=True)
        os.remove(_ofile)



