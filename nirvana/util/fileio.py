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
import logging
import warnings

import numpy

from scipy import sparse

from astropy.wcs import WCS
from astropy.io import fits
import astropy.constants

from .. import __version__

from .bitmask import BitMask


def init_record_array(shape, dtype):
    r"""
    Utility function that initializes a record array using a provided
    input data type.  For example::

        dtype = [ ('INDX', numpy.int, (2,) ),
                  ('VALUE', numpy.float) ]

    Defines two columns, one named `INDEX` with two integers per row and
    the one named `VALUE` with a single float element per row.  See
    `numpy.recarray`_.
    
    Args:
        shape (:obj:`int`, :obj:`tuple`):
            Shape of the output array.
        dtype (:obj:`list`):
            List of the tuples that define each element in the record
            array.

    Returns:
        `numpy.recarray`_: Zeroed record array
    """
    data = numpy.zeros(shape, dtype=dtype)
    return data.view(numpy.recarray)


def rec_to_fits_type(rec_element):
    """
    Return the string representation of a fits binary table data type
    based on the provided record array element.
    """
    n = 1 if len(rec_element[0].shape) == 0 else rec_element[0].size
    if rec_element.dtype == numpy.bool:
        return '{0}L'.format(n)
    if rec_element.dtype == numpy.uint8:
        return '{0}B'.format(n)
    if rec_element.dtype == numpy.int16 or rec_element.dtype == numpy.uint16:
        return '{0}I'.format(n)
    if rec_element.dtype == numpy.int32 or rec_element.dtype == numpy.uint32:
        return '{0}J'.format(n)
    if rec_element.dtype == numpy.int64 or rec_element.dtype == numpy.uint64:
        return '{0}K'.format(n)
    if rec_element.dtype == numpy.float32:
        return '{0}E'.format(n)
    if rec_element.dtype == numpy.float64:
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


def create_symlink(ofile, symlink_dir, relative_symlink=True, clobber=False, loggers=None,
                   quiet=False):
    """
    Create a symlink to the input file in the provided directory.  If
    relative_symlink is True (default), the path to the file is relative
    to the directory with the symlink.
    """
    # Check if the file already exists
    olink_dest = os.path.join(symlink_dir, ofile.split('/')[-1])
    if os.path.isfile(olink_dest) or os.path.islink(olink_dest):
        if clobber:
            os.remove(olink_dest)
        else:
            return

    # Make sure the symlink directory exists
    if not os.path.isdir(symlink_dir):
        os.makedirs(symlink_dir)

    # Set the relative path for the symlink, if requested
    olink_src = os.path.relpath(ofile, start=os.path.dirname(olink_dest)) \
                    if relative_symlink else ofile
    if not quiet:
        log_output(loggers, 1, logging.INFO, 'Creating symlink: {0}'.format(olink_dest))

    # Create the symlink
    os.symlink(olink_src, olink_dest)



