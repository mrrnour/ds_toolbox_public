"""Typed exception hierarchy for :mod:`dstoolbox.utils`."""

from __future__ import annotations


class UtilsError(Exception):
    """Base for :mod:`dstoolbox.utils` runtime errors."""


class DataFrameComparisonError(UtilsError):
    """Raised when :func:`~dstoolbox.utils.dataframes.compare_dataframes_columns` fails."""


class InvalidConfigError(UtilsError):
    """Raised when a config file is missing required keys or has malformed values."""


class OutputFolderError(UtilsError):
    """Raised when the output-folder policy would be violated (e.g. overwrite disallowed)."""


__all__ = [
    "UtilsError",
    "DataFrameComparisonError",
    "InvalidConfigError",
    "OutputFolderError",
]
