"""Typed exception hierarchy for :mod:`dstoolbox.ml_funcs`."""

from __future__ import annotations


class MLFuncsError(Exception):
    """Base for :mod:`dstoolbox.ml_funcs` runtime errors."""


class TrainingError(MLFuncsError):
    """Raised when cross-validated training or comparison fails."""


class TuningError(MLFuncsError):
    """Raised when hyperparameter tuning encounters an invalid configuration."""


class ScoringError(MLFuncsError):
    """Raised when a metric cannot be computed on the supplied predictions."""


class AssumptionsError(MLFuncsError):
    """Raised when a regression-assumption check receives inputs it cannot analyze."""


class BacktestReportError(MLFuncsError):
    """Raised when the ``preds`` frame handed to :class:`BacktestReport` is malformed."""


__all__ = [
    "MLFuncsError",
    "TrainingError",
    "TuningError",
    "ScoringError",
    "AssumptionsError",
    "BacktestReportError",
]
