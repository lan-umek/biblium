# -*- coding: utf-8 -*-
"""
Biblium Logging Configuration

Provides centralized logging setup for the entire package.
Replaces scattered print statements with controllable logging.

Usage:
    from biblium.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Missing data in column X")
    logger.debug("Detailed debug info")

Control verbosity:
    import logging
    logging.getLogger("biblium").setLevel(logging.WARNING)  # Only warnings+
    logging.getLogger("biblium").setLevel(logging.DEBUG)    # Everything

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import logging
import sys
from typing import Optional
from pathlib import Path


# =============================================================================
# PACKAGE LOGGER SETUP
# =============================================================================

# Create the main biblium logger
_BIBLIUM_LOGGER = logging.getLogger("biblium")
_BIBLIUM_LOGGER.setLevel(logging.INFO)  # Default level

# Prevent adding multiple handlers if module is reloaded
if not _BIBLIUM_LOGGER.handlers:
    # Console handler with simple format
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.DEBUG)
    
    # Simple format for console (no timestamps by default - cleaner output)
    _console_format = logging.Formatter(
        fmt="[%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    _console_handler.setFormatter(_console_format)
    _BIBLIUM_LOGGER.addHandler(_console_handler)


# =============================================================================
# PUBLIC API
# =============================================================================

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger for a module.
    
    Parameters
    ----------
    name : str
        Module name (usually __name__). If None, returns root biblium logger.
    
    Returns
    -------
    logging.Logger
        Logger instance.
    
    Example
    -------
    >>> from biblium.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting analysis")
    """
    if name is None:
        return _BIBLIUM_LOGGER
    
    # Create child logger under biblium namespace
    if name.startswith("biblium"):
        return logging.getLogger(name)
    else:
        return logging.getLogger(f"biblium.{name}")


def set_level(level: str = "INFO") -> None:
    """
    Set logging level for all biblium loggers.
    
    Parameters
    ----------
    level : str
        Level name: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SILENT"
    
    Example
    -------
    >>> from biblium import logging_config
    >>> logging_config.set_level("WARNING")  # Only warnings and above
    >>> logging_config.set_level("DEBUG")    # Everything
    >>> logging_config.set_level("SILENT")   # Nothing
    """
    level = level.upper()
    
    if level == "SILENT":
        _BIBLIUM_LOGGER.setLevel(logging.CRITICAL + 1)
    else:
        _BIBLIUM_LOGGER.setLevel(getattr(logging, level, logging.INFO))


def enable_debug() -> None:
    """Enable debug level logging."""
    set_level("DEBUG")


def enable_quiet() -> None:
    """Only show warnings and errors."""
    set_level("WARNING")


def enable_silent() -> None:
    """Disable all logging output."""
    set_level("SILENT")


def enable_verbose() -> None:
    """Enable info level (default)."""
    set_level("INFO")


def add_file_handler(
    filepath: str,
    level: str = "DEBUG",
    include_timestamp: bool = True,
) -> None:
    """
    Add file handler to save logs to file.
    
    Parameters
    ----------
    filepath : str
        Path to log file.
    level : str
        Minimum level to log to file.
    include_timestamp : bool
        Whether to include timestamps in file logs.
    
    Example
    -------
    >>> from biblium import logging_config
    >>> logging_config.add_file_handler("biblium.log")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    if include_timestamp:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        fmt = "[%(levelname)s] %(name)s: %(message)s"
    
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    _BIBLIUM_LOGGER.addHandler(file_handler)


def remove_all_handlers() -> None:
    """Remove all handlers (for testing or reconfiguration)."""
    _BIBLIUM_LOGGER.handlers.clear()


def set_format(
    fmt: str = "[%(name)s] %(message)s",
    datefmt: str = "%H:%M:%S",
    include_timestamp: bool = False,
) -> None:
    """
    Set log format for console output.
    
    Parameters
    ----------
    fmt : str
        Log format string.
    datefmt : str
        Date format string.
    include_timestamp : bool
        If True, uses format with timestamp.
    
    Example
    -------
    >>> from biblium import logging_config
    >>> logging_config.set_format(include_timestamp=True)
    """
    if include_timestamp:
        fmt = "%(asctime)s " + fmt
    
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    
    for handler in _BIBLIUM_LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)


# =============================================================================
# CONVENIENCE CLASS FOR BACKWARD COMPATIBILITY
# =============================================================================

class LoggerAdapter:
    """
    Adapter that mimics print() behavior but uses logging.
    
    Use this to easily replace print statements:
        self.log = LoggerAdapter(__name__)
        self.log("Processing...")  # Works like print but uses logging
    """
    
    def __init__(self, name: str = None, level: str = "INFO"):
        self.logger = get_logger(name)
        self.default_level = getattr(logging, level.upper(), logging.INFO)
    
    def __call__(self, message: str, level: str = None) -> None:
        """Log a message (can be called like print)."""
        lvl = getattr(logging, level.upper(), self.default_level) if level else self.default_level
        self.logger.log(lvl, message)
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def ldf(message: str, logger: logging.Logger = None) -> str:
    """
    Log and return message (for existing ldf pattern in codebase).
    
    This allows gradual migration - existing code using ldf() keeps working.
    
    Parameters
    ----------
    message : str
        Message to log and return.
    logger : Logger
        Logger to use. If None, uses root biblium logger.
    
    Returns
    -------
    str
        The same message (for chaining).
    """
    if logger is None:
        logger = _BIBLIUM_LOGGER
    logger.info(message)
    return message


# =============================================================================
# QUICK ACCESS
# =============================================================================

# Pre-configured loggers for main modules
stats_logger = get_logger("biblium.bibstats")
group_logger = get_logger("biblium.bibgroup")
plot_logger = get_logger("biblium.plotbib")
read_logger = get_logger("biblium.readbib")
utils_logger = get_logger("biblium.utilsbib")
addons_logger = get_logger("biblium.addons")
