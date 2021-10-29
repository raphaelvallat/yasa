"""Helper functions for YASA (e.g. logger)
"""
import logging


LOGGING_TYPES = dict(DEBUG=logging.DEBUG, INFO=logging.INFO, WARNING=logging.WARNING,
                     ERROR=logging.ERROR, CRITICAL=logging.CRITICAL)


def set_log_level(verbose=None):
    """Convenience function for setting the logging level.

    This function comes from the PySurfer package. See :
    https://github.com/nipy/PySurfer/blob/master/surfer/utils.py

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either
        PROFILER, DEBUG, INFO, WARNING, ERROR, or CRITICAL.
    """
    logger = logging.getLogger('yasa')
    if isinstance(verbose, bool):
        verbose = 'INFO' if verbose else 'WARNING'
    if isinstance(verbose, str):
        if (verbose.upper() in LOGGING_TYPES):
            verbose = verbose.upper()
            verbose = LOGGING_TYPES[verbose]
            logger.setLevel(verbose)
        else:
            raise ValueError("verbose must be in %s" % ', '.join(LOGGING_TYPES))


def is_tensorpac_installed():
    """Test if tensorpac is installed."""
    try:
        import tensorpac  # noqa
    except IOError:  # pragma: no cover
        raise IOError("tensorpac needs to be installed. Please use `pip install tensorpac -U`.")


def is_pyriemann_installed():
    """Test if pyRiemann is installed."""
    try:
        import pyriemann  # noqa
    except IOError:  # pragma: no cover
        raise IOError("pyRiemann needs to be installed. Please use `pip install pyriemann -U`.")
