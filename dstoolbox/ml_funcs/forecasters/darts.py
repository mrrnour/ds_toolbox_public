"""Darts adapters: Theta + NBEATS, wrapped as sklearn regressors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._base_import_warning import optional_import
from ..mixins import IntervalMixin, ProbabilisticMixin
from ._base import as_datetime_index, horizon_to_cover, infer_freq, train_series

_darts_ts = optional_import("darts.timeseries", "DartsTheta/DartsNBEATSSklearn")
_darts_models = optional_import("darts.models", "DartsTheta/DartsNBEATSSklearn")


def _to_darts_series(series: pd.Series, freq: str):
    """Convert a pandas Series into a ``darts.TimeSeries``."""
    return _darts_ts.TimeSeries.from_times_and_values(
        times=series.index,
        values=series.to_numpy(),
        freq=freq,
    )


def _align_darts_forecast(forecast, target: pd.DatetimeIndex) -> np.ndarray:
    """Slice a darts forecast to the requested target dates."""
    s = forecast.pd_series()
    return s.reindex(target).to_numpy()


class DartsThetaSklearn(BaseEstimator, RegressorMixin, IntervalMixin):
    """Darts ``Theta`` (or ``FourTheta``) wrapped as sklearn regressor.

    Implements :class:`IntervalMixin` via sample-based intervals when the
    underlying model is probabilistic (Darts uses ``num_samples > 1``).

    Parameters
    ----------
    date_col, freq
        Same as other adapters.
    season_mode
        ``"multiplicative"`` (default) or ``"additive"``.
    season_length
        Seasonality period for Theta; if ``None``, Darts auto-detects.
    n_samples_for_interval
        Number of sample paths drawn to compute prediction intervals.
    **theta_kwargs
        Forwarded to ``darts.models.Theta``.
    """

    def __init__(
        self,
        date_col: str = "ts",
        freq: str | None = None,
        season_mode: str = "multiplicative",
        season_length: int | None = None,
        n_samples_for_interval: int = 200,
        **theta_kwargs,
    ) -> None:
        self.date_col = date_col
        self.freq = freq
        self.season_mode = season_mode
        self.season_length = season_length
        self.n_samples_for_interval = n_samples_for_interval
        self.theta_kwargs = theta_kwargs

    def fit(self, X: pd.DataFrame, y) -> DartsThetaSklearn:
        series = train_series(X, y, self.date_col)
        self._freq_ = self.freq or infer_freq(series.index)
        self._last_train_ = series.index.max()
        kwargs = dict(self.theta_kwargs)
        if self.season_length is not None:
            kwargs.setdefault("seasonality_period", self.season_length)
        self._model_ = _darts_models.Theta(season_mode=self.season_mode, **kwargs)
        self._model_.fit(_to_darts_series(series, self._freq_))
        return self

    def _horizon(self, X: pd.DataFrame) -> int:
        target = as_datetime_index(X, self.date_col)
        return horizon_to_cover(self._last_train_, target.max(), self._freq_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_model_"])
        target = as_datetime_index(X, self.date_col)
        forecast = self._model_.predict(self._horizon(X))
        return _align_darts_forecast(forecast, target)

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["_model_"])
        target = as_datetime_index(X, self.date_col)
        try:
            forecast = self._model_.predict(
                self._horizon(X),
                num_samples=self.n_samples_for_interval,
            )
            arr = forecast.all_values()  # (T, 1, n_samples)
            lo_q = (1 - level) / 2
            hi_q = 1 - lo_q
            lo_vals = np.quantile(arr[:, 0, :], lo_q, axis=1)
            hi_vals = np.quantile(arr[:, 0, :], hi_q, axis=1)
            idx = forecast.time_index
            lo = pd.Series(lo_vals, index=idx).reindex(target).to_numpy()
            hi = pd.Series(hi_vals, index=idx).reindex(target).to_numpy()
            return lo, hi
        except (ValueError, TypeError):
            n = len(target)
            return np.full(n, np.nan), np.full(n, np.nan)


class DartsNBEATSSklearn(BaseEstimator, RegressorMixin, ProbabilisticMixin):
    """Darts ``NBEATSModel`` wrapped as sklearn regressor.

    Implements :class:`ProbabilisticMixin` (sample paths). Heavy: trains a
    deep neural net per fit; consider :class:`RefitEvery` for backtesting.

    Parameters
    ----------
    date_col, freq
        Same as other adapters.
    input_chunk_length, output_chunk_length
        N-BEATS lookback / forecast block sizes.
    n_epochs
        Training epochs per fit.
    **nbeats_kwargs
        Forwarded to ``darts.models.NBEATSModel``.
    """

    def __init__(
        self,
        date_col: str = "ts",
        freq: str | None = None,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        n_epochs: int = 50,
        **nbeats_kwargs,
    ) -> None:
        self.date_col = date_col
        self.freq = freq
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.nbeats_kwargs = nbeats_kwargs

    def fit(self, X: pd.DataFrame, y) -> DartsNBEATSSklearn:
        series = train_series(X, y, self.date_col)
        self._freq_ = self.freq or infer_freq(series.index)
        self._last_train_ = series.index.max()
        self._model_ = _darts_models.NBEATSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            **self.nbeats_kwargs,
        )
        self._model_.fit(_to_darts_series(series, self._freq_))
        return self

    def _horizon(self, X: pd.DataFrame) -> int:
        target = as_datetime_index(X, self.date_col)
        return horizon_to_cover(self._last_train_, target.max(), self._freq_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_model_"])
        target = as_datetime_index(X, self.date_col)
        forecast = self._model_.predict(self._horizon(X))
        return _align_darts_forecast(forecast, target)

    def predict_samples(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
    ) -> np.ndarray:
        check_is_fitted(self, ["_model_"])
        target = as_datetime_index(X, self.date_col)
        forecast = self._model_.predict(self._horizon(X), num_samples=n_samples)
        arr = forecast.all_values()[:, 0, :]  # (T, n_samples)
        idx = forecast.time_index
        df = pd.DataFrame(arr, index=idx)
        return df.reindex(target).to_numpy()
