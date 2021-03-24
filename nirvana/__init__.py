
__version__ = '0.1.1dev'
__license__ = 'BSD3'
__author__ = 'Brian DiGiorgio'
__maintainer__ = 'Brian DiGiorgio'
__email__ = 'bdigiorg@ucsc.edu'
__copyright__ = '(c) 2020, Brian DiGiorgio'
__credits__ = ['Kyle B. Westfall']

def short_warning(message, category, filename, lineno, file=None, line=None):
    """
    Return the format for a short warning message.
    """
    return ' %s: %s\n' % (category.__name__, message)

import warnings
warnings.formatwarning = short_warning

