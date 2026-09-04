"""Shared helpers for forecaster adapters.

These functions exist so every adapter (naive, ARIMA, Silverkite, Darts, ...)
agrees on how to extract the date column, reconstruct the training series,
infer frequency, and translate "predict for these target dates" into a
horizon step count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_datetime_index(X: pd.DataFrame, date_col: str) -> pd.DatetimeIndex:
    """Extract ``X[date_col]`` as a :class:`DatetimeIndex`."""
    if date_col not in X.columns:
        raise KeyError(f"date_col={date_col!r} not in X.columns={list(X.columns)}")
    return pd.DatetimeIndex(pd.to_datetime(X[date_col]))


def train_series(X: pd.DataFrame, y, date_col: str) -> pd.Series:
    """Reindex ``y`` by ``X[date_col]`` and sort chronologically."""
    idx = as_datetime_index(X, date_col)
    name = getattr(y, "name", None) or "y"
    s = pd.Series(np.asarray(y), index=idx, name=name)
    return s.sort_index()


def infer_freq(idx: pd.DatetimeIndex, fallback: str = "D") -> str:
    """Best-effort frequency inference. Returns ``fallback`` if unknown."""
    freq = pd.infer_freq(idx)
    return freq or fallback


def horizon_to_cover(
    last_train: pd.Timestamp,
    target_max: pd.Timestamp,
    freq: str,
) -> int:
    """Number of ``freq`` steps from just after ``last_train`` through ``target_max``.

    Works for any pandas frequency (fixed or calendar-based) — explicitly
    constructs a :func:`pd.date_range` instead of dividing a Timedelta by a
    BaseOffset (which breaks for month-end, business-day, etc.).
    """
    offset = pd.tseries.frequencies.to_offset(freq)
    return len(pd.date_range(last_train + offset, target_max, freq=freq))


def nan_intervals(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` NaN arrays of length ``n`` — for point-only models."""
    nan = np.full(n, np.nan, dtype=float)
    return nan, nan.copy()


# ===== imports preserved from public (needed by extras below) =====
from sklearn.base import BaseEstimator, TransformerMixin

# ===== public-only extensions (preserved on vendor merge) =====


def align_forecast(
    forecast_index: pd.DatetimeIndex,
    forecast_values: np.ndarray,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Pick values from a longer forecast horizon that line up with ``target_index``.

    Used when an adapter forecasts ``horizon`` steps but ``X`` requests a
    subset of those dates. NaN where a target date is missing from the
    forecast (out-of-grid).
    """
    s = pd.Series(forecast_values, index=forecast_index)
    return s.reindex(target_index).to_numpy()


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop named columns from a DataFrame. Pipeline-friendly.

    Lets pure sklearn regressors ignore the date column that forecasters need::

        Pipeline([
            ("drop_date", DropColumns(["ts"])),
            ("scale", StandardScaler()),
            ("model", Ridge()),
        ])
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X, y=None) -> DropColumns:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in X.columns]
        return X.drop(columns=present) if present else X

    def get_feature_names_out(self, input_features=None):
        feats = list(input_features) if input_features is not None else []
        return np.array([f for f in feats if f not in self.columns])
