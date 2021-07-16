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
from ..data.manga import MaNGAStellarKinematics, MaNGAGasKinematics

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
