"""
Testing utilities
"""

# TODO: Put most/all of this in nirvana/tests/__init__.py ?

import os
import warnings

import pytest

from pkg_resources import resource_filename

def data_file(filename=None):
    root = resource_filename('nirvana', 'data')
    return root if filename is None else os.path.join(root, filename)

def test_data_file(filename=None):
    root = os.path.join(data_file(), 'tests')
    return root if filename is None else os.path.join(root, filename)

def remote_data_file(filename=None):
    root = os.path.join(data_file(), 'remote')
    return root if filename is None else os.path.join(root, filename)

# TODO: It's super annoying that these file names do not include the plate
# number...
def remote_drp_test_images():
    return ['12704.png', '12703.png']

def remote_drp_test_files():
    return ['manga-8138-12704-LOGCUBE.fits.gz', 'manga-8078-12703-LOGCUBE.fits.gz']

def remote_dap_test_files(daptype='HYB10-MILESHC-MASTARHC2'):
    return ['manga-8138-12704-MAPS-{0}.fits.gz'.format(daptype),
            'manga-8078-12703-MAPS-{0}.fits.gz'.format(daptype)]

drp_test_version = 'v3_0_1'
dap_test_version = '3.0.1'
dap_test_daptype = 'HYB10-MILESHC-MASTARHC2'

test_files = remote_drp_test_files() + remote_dap_test_files(daptype=dap_test_daptype)

remote_available = all([os.path.isfile(remote_data_file(f)) for f in test_files])

requires_remote = pytest.mark.skipif(not remote_available, reason='Remote data files are missing.')

if not remote_available:
    warnings.warn('Remote data not available.  Some tests will be skipped.')

try:
    import pyfftw
except:
    pyfftw = None

requires_pyfftw = pytest.mark.skipif(pyfftw is None, reason='pyfftw is not installed')

