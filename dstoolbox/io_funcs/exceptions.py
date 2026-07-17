"""Typed exception hierarchy for :mod:`dstoolbox.io_funcs`.

The DataSource-seam errors (``DataSourceError`` and family) live in
:mod:`dstoolbox.io_funcs.data_sources` for backwards compatibility; this
module re-exports them alongside format/transport-specific errors so
callers can ``from dstoolbox.io_funcs.exceptions import MSSQLError``.
"""

from __future__ import annotations

from .data_sources import (
    DataSourceError,
    UnknownAuthKindError,
    UnknownKindError,
    UnknownTargetError,
)


class IOFuncsError(Exception):
    """Base for :mod:`dstoolbox.io_funcs` runtime errors that are not seam-related."""


class MSSQLError(IOFuncsError):
    """Raised when an MSSQL query, write, or metadata check fails."""


class BlobError(IOFuncsError):
    """Raised when an Azure Blob operation fails or receives an unsupported request."""


class SynapseError(IOFuncsError):
    """Raised when a Synapse query or write fails."""


class DeltaError(IOFuncsError):
    """Raised when a Delta / Databricks table operation fails."""


class PIError(IOFuncsError):
    """Raised when a PI Web API call fails or returns malformed data."""


class OutputSpecError(IOFuncsError):
    """Raised when an ``output_specs`` list is inconsistent with ``df_generator_func`` outputs."""


__all__ = [
    "DataSourceError",
    "UnknownAuthKindError",
    "UnknownKindError",
    "UnknownTargetError",
    "IOFuncsError",
    "MSSQLError",
    "BlobError",
    "SynapseError",
    "DeltaError",
    "PIError",
    "OutputSpecError",
]
