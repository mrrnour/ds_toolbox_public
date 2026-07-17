"""Optional capability mixins for forecaster adapters.

Mixins are *markers* — they document the contract a model exposes beyond the
plain sklearn ``predict(X)``. The :mod:`dstoolbox.ml_funcs.inspection` module
detects them via ``hasattr``; no metaclass magic.

Use them by multiple inheritance alongside ``BaseEstimator`` / ``RegressorMixin``::

    class MyForecaster(BaseEstimator, RegressorMixin, IntervalMixin):
        def predict_interval(self, X):
            ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IntervalMixin:
    """Forecasters that return parametric or quantile-based prediction intervals."""

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y_lo, y_hi)`` arrays of length ``len(X)``.

        Parameters
        ----------
        X
            Same shape/columns as for :meth:`predict`.
        level
            Two-sided coverage probability. ``0.95`` requests a 95% interval.
        """
        raise NotImplementedError


class ProbabilisticMixin:
    """Forecasters that draw sample paths from the predictive distribution."""

    def predict_samples(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
    ) -> np.ndarray:
        """Return ndarray of shape ``(len(X), n_samples)`` — one column per draw."""
        raise NotImplementedError


class ComponentsMixin:
    """Forecasters that decompose the prediction (trend / seasonality / holiday / ...)."""

    def components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return per-row component contributions.

        Index aligns with ``X``; columns are component names (e.g. ``trend``,
        ``weekly``, ``yearly``, ``holiday``, ``<regressor>``). Rows sum to the
        point forecast.
        """
        raise NotImplementedError
