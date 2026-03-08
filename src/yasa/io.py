"""Helper functions for YASA (e.g. logger)"""

import logging

LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


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
    logger = logging.getLogger("yasa")
    if isinstance(verbose, bool):
        verbose = "INFO" if verbose else "WARNING"
    if isinstance(verbose, str):
        if verbose.upper() in LOGGING_TYPES:
            verbose = verbose.upper()
            verbose = LOGGING_TYPES[verbose]
            logger.setLevel(verbose)
        else:
            raise ValueError("verbose must be in %s" % ", ".join(LOGGING_TYPES))


def is_tensorpac_installed():
    """Test if tensorpac is installed."""
    try:
        import tensorpac  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("tensorpac needs to be installed. Please use `pip install yasa[pac]`.")


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
