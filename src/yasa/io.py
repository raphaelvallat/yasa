"""Helper functions for YASA (e.g. logger)"""

import functools
import inspect
import logging
from contextlib import contextmanager

LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def _parse_log_level(verbose):
    """Convert ``verbose`` to a logging level (int), or None to leave the level unchanged."""
    if verbose is None:
        return None
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    if isinstance(verbose, int):
        return verbose
    if isinstance(verbose, str) and verbose.upper() in LOGGING_TYPES:
        return LOGGING_TYPES[verbose.upper()]
    raise ValueError("verbose must be a bool, an int, None or in %s" % ", ".join(LOGGING_TYPES))


def set_log_level(verbose=None):
    """Set the level of the YASA logger.

    The level persists until it is changed again. YASA functions with a ``verbose`` argument only
    change the level for the duration of the call.

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG, INFO, WARNING, ERROR,
        or CRITICAL (case-insensitive). ``True`` is the same as INFO and ``False`` the same as
        WARNING. An int is used as a :py:mod:`logging` level. If None, the level is unchanged.
    """
    level = _parse_log_level(verbose)
    if level is not None:
        logging.getLogger("yasa").setLevel(level)


@contextmanager
def _log_level(verbose):
    """Context manager that sets the YASA logger level and restores the previous one on exit."""
    logger = logging.getLogger("yasa")
    old_level = logger.level
    set_log_level(verbose)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def _restore_log_level(func):
    """Decorator that applies the ``verbose`` argument of ``func`` only for the duration of the call.

    Without it, calling a function with ``verbose=True`` would leave the shared "yasa" logger at
    the INFO level for every subsequent call.
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        with _log_level(bound.arguments["verbose"]):
            return func(*args, **kwargs)

    return wrapper


def is_pyriemann_installed():
    """Test if pyRiemann is installed."""
    try:
        import pyriemann  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("pyriemann needs to be installed. Please use `pip install yasa[art]`.")


def is_sleepecg_installed():
    """Test if sleepecg is installed."""
    try:
        import sleepecg  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("sleepecg needs to be installed. Please use `pip install yasa[heart]`.")
