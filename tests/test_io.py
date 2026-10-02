"""Test I/O."""

import logging

import pytest

from yasa.io import (
    _log_level,
    _parse_log_level,
    _restore_log_level,
    is_pyriemann_installed,
    is_sleepecg_installed,
    set_log_level,
)

logger = logging.getLogger("yasa")


@pytest.mark.parametrize(
    "verbose, expected",
    [
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("Warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
        (True, logging.INFO),
        (False, logging.WARNING),
        (logging.ERROR, logging.ERROR),
        (5, 5),
        (None, None),
    ],
)
def test_parse_log_level(verbose, expected):
    """Test the conversion of ``verbose`` to a logging level."""
    assert _parse_log_level(verbose) == expected


@pytest.mark.parametrize("verbose", ["WRONG", 1.5, [True]])
def test_parse_log_level_invalid(verbose):
    """Test that an invalid ``verbose`` raises an error."""
    with pytest.raises(ValueError, match="verbose must be"):
        _parse_log_level(verbose)


def test_set_log_level():
    """Test setting the log level, and that None leaves it unchanged."""
    set_log_level("error")
    assert logger.level == logging.ERROR
    set_log_level(None)
    assert logger.level == logging.ERROR
    set_log_level(True)
    assert logger.level == logging.INFO
    with pytest.raises(ValueError):
        set_log_level("WRONG")


def test_log_level_context_manager():
    """Test that the level is restored on exit, including when an error is raised."""
    set_log_level("warning")
    with _log_level("debug"):
        assert logger.level == logging.DEBUG
    assert logger.level == logging.WARNING
    with pytest.raises(RuntimeError), _log_level("error"):
        assert logger.level == logging.ERROR
        raise RuntimeError
    assert logger.level == logging.WARNING


def test_restore_log_level_decorator():
    """Test that ``verbose`` only applies for the duration of the call."""

    @_restore_log_level
    def func(x, verbose=False):
        return x, logger.level

    set_log_level("error")
    assert func(1) == (1, logging.WARNING)  # default value of verbose
    assert func(1, True) == (1, logging.INFO)  # positional
    assert func(1, verbose="debug") == (1, logging.DEBUG)  # keyword
    assert logger.level == logging.ERROR
    assert func.__name__ == "func"


def test_logger_configuration():
    """Test that YASA configures its own logger and not the root logger."""
    assert logger.handlers
    assert logger.propagate
    assert logger.name not in [h.name for h in logging.getLogger().handlers]


def test_logger_handler_defers_to_root(monkeypatch):
    """Test that YASA's handler only prints messages when the root logger has no handler."""
    (handler,) = logger.handlers
    record = logger.makeRecord("yasa", logging.WARNING, __file__, 0, "msg", None, None)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    assert handler.filter(record)
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    assert not handler.filter(record)


def test_logger(caplog):
    """Test that messages are emitted according to the level of the YASA logger."""
    set_log_level("warning")
    logger.info("info")
    logger.warning("warning")
    logger.critical("critical")
    assert [r.levelname for r in caplog.records] == ["WARNING", "CRITICAL"]


def test_dependence():
    """Test dependencies."""
    is_pyriemann_installed()
    is_sleepecg_installed()
