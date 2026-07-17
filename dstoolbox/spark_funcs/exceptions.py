"""Typed exception hierarchy for :mod:`dstoolbox.spark_funcs`."""

from __future__ import annotations


class SparkFuncsError(Exception):
    """Base for :mod:`dstoolbox.spark_funcs` runtime errors."""


class InvalidAggregationError(SparkFuncsError):
    """Raised when an aggregation name is not in the allowed whitelist."""


class InvalidWindowError(SparkFuncsError):
    """Raised when a rolling / tumbling window spec cannot be parsed."""


__all__ = ["SparkFuncsError", "InvalidAggregationError", "InvalidWindowError"]
