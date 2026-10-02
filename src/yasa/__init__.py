import logging as _logging
import warnings as _warnings

from . import others as _others
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
# Messages propagate to the root logger, so that the handlers configured by the application (e.g.
# ``logging.basicConfig(filename=...)``) receive them. YASA's own handler only prints messages when
# the root logger has no handler, which avoids duplicated messages once logging is configured.
_logger = _logging.getLogger("yasa")
if not _logger.handlers:  # pragma: no branch (False only if the module is reloaded)
    _handler = _logging.StreamHandler()
    _handler.setFormatter(
        _logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    )
    _handler.addFilter(lambda record: not _logging.getLogger().handlers)
    _logger.addHandler(_handler)

# Functions removed from the top-level namespace in v0.8, still available in yasa.others
_MOVED_TO_OTHERS = ("get_centered_indices", "trimbothstd")


def __getattr__(name):
    if name in _MOVED_TO_OTHERS:
        _warnings.warn(
            f"`yasa.{name}` is deprecated and will be removed in v0.9. "
            f"Please use `yasa.others.{name}` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(_others, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__author__ = "Raphael Vallat <raphaelvallat9@gmail.com>"
__version__ = "0.8.0"
