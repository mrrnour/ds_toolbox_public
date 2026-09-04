"""Logging configuration helper for :mod:`dstoolbox`.

Call :func:`configure_logging` once at the top of a script or notebook to attach
a StreamHandler to the ``dstoolbox`` logger tree; leave it uncalled to let the
host application own logging setup.

Example
-------
>>> from dstoolbox.logging_config import configure_logging
>>> import logging
>>> configure_logging(level=logging.INFO)
>>> import dstoolbox.io_funcs.mssql  # emits INFO/WARNING through the shared handler
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    logger_name: str = "dstoolbox",
) -> logging.Logger:
    """Attach a StreamHandler to the ``dstoolbox`` logger tree.

    Idempotent — calling twice does not add duplicate handlers.

    Parameters
    ----------
    level : int, optional
        Minimum level to emit. Default :data:`logging.INFO`.
    fmt : str, optional
        Formatter string. Default includes timestamp, level, module and message.
    datefmt : str, optional
        Date format for the ``%(asctime)s`` placeholder.
    logger_name : str, optional
        Logger name to configure. Default ``"dstoolbox"`` (i.e. the whole
        package tree).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not any(getattr(h, "_dstoolbox_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handler._dstoolbox_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``dstoolbox`` tree.

    Parameters
    ----------
    name : str, optional
        Module or subsystem name. If ``None``, returns the root ``dstoolbox`` logger.

    Returns
    -------
    logging.Logger
    """
    if name is None:
        return logging.getLogger("dstoolbox")
    if not name.startswith("dstoolbox"):
        name = f"dstoolbox.{name}"
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
