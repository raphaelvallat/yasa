import logging as _logging

from .detection import *
from .evaluation import *
from .fetchers import *
from .heart import *
from .hypno import *
from .others import *
from .plotting import *
from .sleepstats import *
from .spectral import *
from .staging import *

# Configure the "yasa" logger only, never the root logger of the application importing YASA.
# Messages are printed by YASA's own handler and are not propagated to the root logger, to avoid
# duplicated messages when the application also configures logging. To route YASA messages to
# your own handlers instead, use: ``logging.getLogger("yasa").propagate = True``.
_logger = _logging.getLogger("yasa")
if not _logger.handlers:
    _handler = _logging.StreamHandler()
    _handler.setFormatter(
        _logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    )
    _logger.addHandler(_handler)
    _logger.propagate = False

__author__ = "Raphael Vallat <raphaelvallat9@gmail.com>"
__version__ = "0.7.0"
