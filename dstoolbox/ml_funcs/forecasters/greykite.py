"""Greykite Silverkite (AUTO template) adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .._base_import_warning import optional_import
from ..mixins import ComponentsMixin, IntervalMixin
from ._base import as_datetime_index, horizon_to_cover, infer_freq, train_series

_gk_fc = optional_import("greykite.framework.templates.forecaster", "SilverkiteSklearn")
_gk_cfg = optional_import("greykite.framework.templates.autogen.forecast_config", "SilverkiteSklearn")
_gk_const = optional_import("greykite.common.constants", "SilverkiteSklearn")


class SilverkiteSklearn(BaseEstimator, RegressorMixin, IntervalMixin, ComponentsMixin):
    """Greykite Silverkite AUTO template wrapped as a sklearn regressor.

    Implements :class:`IntervalMixin` (native prediction intervals) and
    :class:`ComponentsMixin` (per-row trend / seasonality / holiday breakdown).

    Parameters
    ----------
    date_col
        Column in ``X`` holding the per-row timestamp.
    freq
        Pandas frequency string. Inferred at fit time if ``None``.
    coverage
        Interval coverage probability (e.g. ``0.95``). Stored at fit and
        reused by :meth:`predict_interval` unless overridden.
    holiday_countries
        Optional list of ISO country codes (e.g. ``["US"]``). When set, the
        AUTO template configures holiday effects with ``holiday_pre_num_days``
        and ``holiday_post_num_days``.
    holiday_pre_num_days, holiday_post_num_days
        Number of days before/after a holiday to model as part of the event.
    **forecast_config_kwargs
        Extra fields forwarded to ``ForecastConfig``.
    """

    def __init__(
        self,
        date_col: str = "ts",
        freq: str | None = None,
        coverage: float = 0.95,
        holiday_countries: list[str] | None = None,
        holiday_pre_num_days: int = 2,
        holiday_post_num_days: int = 2,
        **forecast_config_kwargs,
    ) -> None:
        self.date_col = date_col
        self.freq = freq
        self.coverage = coverage
        self.holiday_countries = holiday_countries
        self.holiday_pre_num_days = holiday_pre_num_days
        self.holiday_post_num_days = holiday_post_num_days
        self.forecast_config_kwargs = forecast_config_kwargs

    def _build_config(self, horizon: int):
        meta = _gk_cfg.MetadataParam(
            time_col=_gk_const.TIME_COL,
            value_col=_gk_const.VALUE_COL,
            freq=self._freq_,
        )
        components = None
        if self.holiday_countries:
            # greykite expects ``events`` as a plain dict; there is no typed
            # EventsConfig class in this release.
            components = _gk_cfg.ModelComponentsParam(events={
                "holiday_lookup_countries": self.holiday_countries,
                "holiday_pre_num_days": self.holiday_pre_num_days,
                "holiday_post_num_days": self.holiday_post_num_days,
            })
        return _gk_cfg.ForecastConfig(
            model_template="SILVERKITE",
            forecast_horizon=horizon,
            coverage=self.coverage,
            metadata_param=meta,
            model_components_param=components,
            **self.forecast_config_kwargs,
        )

    def fit(self, X: pd.DataFrame, y) -> "SilverkiteSklearn":
        series = train_series(X, y, self.date_col)
        self._freq_ = self.freq or infer_freq(series.index)
        self._last_train_ = series.index.max()
        self._train_df_ = pd.DataFrame({
            _gk_const.TIME_COL: series.index,
            _gk_const.VALUE_COL: series.to_numpy(),
        })

        self._forecaster_ = _gk_fc.Forecaster()
        self._result_ = self._forecaster_.run_forecast_config(
            df=self._train_df_,
            config=self._build_config(horizon=1),  # placeholder; we refit at predict
        )
        return self

    def _forecast_df(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["_forecaster_", "_freq_", "_last_train_"])
        target = as_datetime_index(X, self.date_col)
        horizon = horizon_to_cover(self._last_train_, target.max(), self._freq_)
        # ``horizon <= 0`` means the caller wants in-sample fitted values only.
        # ``_result_.forecast.df`` already contains the training rows, so skip
        # the refit and let the reindex below pick them up.
        if horizon > 0 and horizon != self._result_.forecast.forecast_horizon:
            self._result_ = self._forecaster_.run_forecast_config(
                df=self._train_df_,
                config=self._build_config(horizon=horizon),
            )
        df = self._result_.forecast.df.copy()
        df[_gk_const.TIME_COL] = pd.to_datetime(df[_gk_const.TIME_COL])
        return df.set_index(_gk_const.TIME_COL)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        target = as_datetime_index(X, self.date_col)
        df = self._forecast_df(X)
        return df[_gk_const.PREDICTED_COL].reindex(target).to_numpy()

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        if abs(level - self.coverage) > 1e-6:
            self.coverage = level
            self._result_ = self._forecaster_.run_forecast_config(
                df=self._train_df_,
                config=self._build_config(horizon=self._result_.forecast.forecast_horizon),
            )
        target = as_datetime_index(X, self.date_col)
        df = self._forecast_df(X)
        lo = df[_gk_const.PREDICTED_LOWER_COL].reindex(target).to_numpy()
        hi = df[_gk_const.PREDICTED_UPPER_COL].reindex(target).to_numpy()
        return lo, hi

    def components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row trend / seasonality / holiday / regressor contributions."""
        target = as_datetime_index(X, self.date_col)
        comp_df = self._result_.forecast.get_forecast_component_df().copy()
        comp_df[_gk_const.TIME_COL] = pd.to_datetime(comp_df[_gk_const.TIME_COL])
        return comp_df.set_index(_gk_const.TIME_COL).reindex(target)
