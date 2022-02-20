import logging
from .detection import *
from .features import *
from .hypno import *
from .numba import *
from .others import *
from .plotting import *
from .sleepstats import *
from .spectral import *
from .staging import *
from outdated import warn_if_outdated

# Define YASA logger
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

__author__  = "Raphael Vallat <raphaelvallat9@gmail.com>"
__version__ = "0.6.0"

# Warn if a newer version of YASA is available
warn_if_outdated("yasa", __version__)
