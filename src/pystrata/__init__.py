# The MIT License (MIT)
#
# Copyright (c) 2016-2022 Albert Kottke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging
import sys
from importlib.metadata import version

from . import (
    constitutive,
    curve_fitting,
    motion,
    output,
    propagation,
    site,
    time_integration,
    tools,
    variation,
)
from .units import ureg

__all__ = [
    "constitutive",
    "curve_fitting",
    "motion",
    "output",
    "propagation",
    "site",
    "time_integration",
    "tools",
    "variation",
    "ureg",
    "enable_logging",
    "disable_logging",
]

__author__ = "Albert Kottke"
__copyright__ = "Copyright 2016-2024 Albert Kottke"
__license__ = "MIT"
__title__ = "pyStrata"

# Get version from setuptools-scm
try:
    # First try to get version from setuptools-scm generated file
    from ._version import __version__
except ImportError:
    # Fallback to package metadata
    try:
        from importlib.metadata import version

        __version__ = version("pyStrata")
    except Exception:
        __version__ = "unknown"

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Set up null handler to prevent "No handler found" warnings
# Users must explicitly enable logging to see messages
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# Track the handler added by enable_logging for later removal
_logging_handler: logging.Handler | None = None


def enable_logging(
    level: int | str = logging.INFO,
    handler: logging.Handler | None = None,
    format_string: str | None = None,
) -> None:
    """Enable logging output for pystrata.

    By default, pystrata does not emit any log messages. Call this function
    to enable logging output.

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging level. Can be an integer (e.g., logging.DEBUG) or string
        (e.g., 'DEBUG', 'INFO', 'WARNING').
    handler : logging.Handler, optional
        Custom handler for log output. If None, a StreamHandler writing
        to stderr with a standard format is used.
    format_string : str, optional
        Custom format string for the handler. Ignored if a custom handler
        is provided. Default format includes timestamp, logger name, level,
        and message.

    Examples
    --------
    >>> import pystrata
    >>> pystrata.enable_logging()  # INFO level to stderr
    >>> pystrata.enable_logging('DEBUG')  # DEBUG level
    >>> pystrata.enable_logging(logging.WARNING)  # WARNING level

    To use standard logging configuration instead:
    >>> import logging
    >>> logging.getLogger('pystrata').setLevel(logging.DEBUG)
    >>> logging.getLogger('pystrata').addHandler(logging.StreamHandler())
    """
    global _logging_handler

    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Remove existing handler if present
    if _logging_handler is not None:
        logger.removeHandler(_logging_handler)
        _logging_handler = None

    if handler is None:
        # Create default handler
        handler = logging.StreamHandler(sys.stderr)
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(format_string))

    handler.setLevel(level)
    logger.addHandler(handler)
    _logging_handler = handler


def disable_logging() -> None:
    """Disable logging output for pystrata.

    Removes any handler added by enable_logging() and sets the log level
    to WARNING to minimize overhead.

    Examples
    --------
    >>> import pystrata
    >>> pystrata.enable_logging('DEBUG')
    >>> # ... do some work with logging ...
    >>> pystrata.disable_logging()  # Turn off logging
    """
    global _logging_handler

    logger = logging.getLogger(__name__)

    if _logging_handler is not None:
        logger.removeHandler(_logging_handler)
        _logging_handler = None

    # Reset to WARNING level (effectively silent for normal operation)
    logger.setLevel(logging.WARNING)
