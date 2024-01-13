import logging
from .detection import *
from .features import *
from .heart import *
from .hypno import *
from .numba import *
from .others import *
from .plotting import *
from .sleepstats import *
from .spectral import *
from .staging import *

# Define YASA logger
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%d-%b-%y %H:%M:%S")

__author__ = "Raphael Vallat <raphaelvallat9@gmail.com>"
__version__ = "0.6.4"
