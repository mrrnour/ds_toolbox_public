"""Naive forecaster baselines — minimal, dependency-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._base import as_datetime_index, horizon_to_cover, infer_freq, train_series


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the training mean for every target date.

    Use this as the floor any "real" model must beat: a negative R^2 on the
    backtest means the model is worse than this baseline.

    Parameters
    ----------
    date_col
        Column in ``X`` holding the per-row timestamp. Used only so the
        adapter has the same call signature as the other forecasters.
    """

    def __init__(self, date_col: str = "ts") -> None:
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y) -> "MeanBaseline":  # noqa: ARG002
        self._mean_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_mean_"])
        return np.full(len(X), self._mean_, dtype=float)


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Repeat the last observed value at the same seasonal offset.

    For target date ``t = t_n + h * freq``, returns
    ``y_history[-season_length + ((h - 1) % season_length)]``.

    Parameters
    ----------
    date_col
        Column in ``X`` holding the per-row timestamp.
    season_length
        Number of frequency steps in one seasonal cycle (7 for weekly
        seasonality on daily data, 12 for monthly-on-monthly, etc.).
    freq
        Pandas frequency string. If ``None``, inferred from ``X[date_col]``
        at fit time; defaults to ``"D"`` if inference fails.
    """

    def __init__(
        self,
        date_col: str = "ts",
        season_length: int = 7,
        freq: str | None = None,
    ) -> None:
        self.date_col = date_col
        self.season_length = season_length
        self.freq = freq

    def fit(self, X: pd.DataFrame, y) -> "SeasonalNaive":
        series = train_series(X, y, self.date_col)
        self._history_ = series.to_numpy()
        self._last_train_ = series.index.max()
        self._freq_ = self.freq or infer_freq(series.index)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_history_", "_last_train_", "_freq_"])
        target = as_datetime_index(X, self.date_col)
        season = self.season_length
        hist = self._history_

        out = np.empty(len(target), dtype=float)
        for i, t in enumerate(target):
            steps = horizon_to_cover(self._last_train_, t, self._freq_)
            if steps <= 0:
                out[i] = hist[-1]
            else:
                out[i] = hist[-season + ((steps - 1) % season)]
        return out
