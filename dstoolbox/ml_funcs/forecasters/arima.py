"""statsforecast AutoARIMA adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._base_import_warning import optional_import
from ..mixins import IntervalMixin
from ._base import as_datetime_index, horizon_to_cover, infer_freq, train_series

_sf = optional_import("statsforecast", "AutoArimaSklearn")
_sf_models = optional_import("statsforecast.models", "AutoArimaSklearn")


class AutoArimaSklearn(BaseEstimator, RegressorMixin, IntervalMixin):
    """``statsforecast.AutoARIMA`` wrapped as a sklearn regressor.

    Implements :class:`IntervalMixin` — supports parametric prediction
    intervals via the underlying model's ``level=`` argument.

    Parameters
    ----------
    date_col
        Column in ``X`` holding the per-row timestamp.
    season_length
        Seasonal period for AutoARIMA's grid search (e.g. ``7`` for daily
        data with weekly seasonality, ``12`` for monthly).
    freq
        Pandas frequency string. Inferred at fit time if ``None``.
    **arima_kwargs
        Forwarded to ``statsforecast.models.AutoARIMA``.
    """

    def __init__(
        self,
        date_col: str = "ts",
        season_length: int = 1,
        freq: str | None = None,
        **arima_kwargs,
    ) -> None:
        self.date_col = date_col
        self.season_length = season_length
        self.freq = freq
        self.arima_kwargs = arima_kwargs

    def fit(self, X: pd.DataFrame, y) -> AutoArimaSklearn:
        series = train_series(X, y, self.date_col)
        self._freq_ = self.freq or infer_freq(series.index)
        self._last_train_ = series.index.max()

        long_df = pd.DataFrame(
            {
                "unique_id": "y",
                "ds": series.index,
                "y": series.to_numpy(),
            }
        )
        model = _sf_models.AutoARIMA(season_length=self.season_length, **self.arima_kwargs)
        self._sf_ = _sf.StatsForecast(models=[model], freq=self._freq_, n_jobs=1)
        self._sf_.fit(long_df)

        # In-sample fitted values, indexed by training timestamp. Used by
        # ``predict`` when the caller asks for training-window timestamps
        # (e.g. plotting an in-sample overlay alongside the forecast).
        underlying = self._sf_.fitted_[0, 0]
        fitted_arr = None
        # Preferred: official statsforecast API on the fitted model.
        if hasattr(underlying, "predict_in_sample"):
            try:
                in_sample = underlying.predict_in_sample()
                if isinstance(in_sample, dict) and "fitted" in in_sample:
                    fitted_arr = np.asarray(in_sample["fitted"])
                elif hasattr(in_sample, "to_numpy"):
                    fitted_arr = in_sample.to_numpy()
                else:
                    fitted_arr = np.asarray(in_sample)
            except Exception:
                fitted_arr = None
        # Fallback: some statsforecast versions stash fitted values on .model_.
        if fitted_arr is None:
            model_attr = getattr(underlying, "model_", None)
            if isinstance(model_attr, dict) and "fitted" in model_attr:
                fitted_arr = np.asarray(model_attr["fitted"])
        if fitted_arr is not None and len(fitted_arr) == len(series):
            self._fitted_in_sample_ = pd.Series(fitted_arr, index=series.index)
        else:
            self._fitted_in_sample_ = None
        return self

    def _forecast(self, X: pd.DataFrame, level: list[int] | None = None) -> pd.DataFrame:
        check_is_fitted(self, ["_sf_", "_freq_", "_last_train_"])
        target = as_datetime_index(X, self.date_col)
        horizon = horizon_to_cover(self._last_train_, target.max(), self._freq_)
        kwargs = {"h": horizon}
        if level is not None:
            kwargs["level"] = level
        fc = self._sf_.predict(**kwargs).reset_index()
        fc["ds"] = pd.to_datetime(fc["ds"])
        return fc.set_index("ds")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_sf_", "_freq_", "_last_train_"])
        target = as_datetime_index(X, self.date_col)
        if target.max() <= self._last_train_:
            # Pure in-sample request — never call sf.predict (h=0 raises).
            if self._fitted_in_sample_ is None:
                return np.full(len(target), np.nan)
            return self._fitted_in_sample_.reindex(target).to_numpy()
        fc = self._forecast(X)
        return fc["AutoARIMA"].reindex(target).to_numpy()

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = as_datetime_index(X, self.date_col)
        # No parametric in-sample PI; return NaNs so callers can still align.
        if target.max() <= self._last_train_:
            n = len(target)
            return np.full(n, np.nan), np.full(n, np.nan)
        pct = int(round(level * 100))
        fc = self._forecast(X, level=[pct])
        lo = fc[f"AutoARIMA-lo-{pct}"].reindex(target).to_numpy()
        hi = fc[f"AutoARIMA-hi-{pct}"].reindex(target).to_numpy()
        return lo, hi

    def order(self) -> tuple:
        """Fitted ``(p, d, q)(P, D, Q, s)`` of the underlying ARIMA."""
        check_is_fitted(self, ["_sf_"])
        fitted = self._sf_.fitted_[0, 0]
        return getattr(fitted, "model_", {}).get("arma", ())
