"""Lag-feature regressor — wraps any sklearn regressor as a recursive forecaster.

The "scikit-learn way" to do forecasting: turn the time series into a tabular
``(y_{t-1}, y_{t-2}, ..., y_{t-k})`` problem and let any regressor handle it.
At predict time, walk the horizon one step at a time, feeding each new
prediction back as the next lag.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._base import as_datetime_index, horizon_to_cover, infer_freq, train_series


def _build_lag_matrix(series: pd.Series, lags: Sequence[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Build ``(X_lag, y_aligned)`` from a univariate series + lag specification."""
    max_lag = max(lags)
    if len(series) <= max_lag:
        raise ValueError(f"series length {len(series)} too short for max lag {max_lag}")
    feats = {f"lag_{k}": series.shift(k) for k in lags}
    X = pd.DataFrame(feats).iloc[max_lag:].reset_index(drop=True)
    y_aligned = series.iloc[max_lag:].reset_index(drop=True)
    return X, y_aligned


class LagRegressor(BaseEstimator, RegressorMixin):
    """Recursive lag-feature wrapper around any sklearn regressor.

    Parameters
    ----------
    estimator
        Any unfitted sklearn-compatible regressor (``Ridge``, ``XGBRegressor``,
        ``LightGBMRegressor``, ...). Cloned before fitting.
    lags
        Iterable of integer lag steps. Defaults to ``(1, 7, 14, 28)`` —
        captures daily / weekly / biweekly / monthly memory on daily data.
    date_col
        Column in ``X`` holding the per-row timestamp. Required to align the
        training series and to walk the recursive horizon at predict time.
    freq
        Pandas frequency string. Inferred at fit time if ``None``.

    Notes
    -----
    - Predict is *recursive*: ``y_hat(t+1)`` is fed into ``X(t+2)``'s lag-1
      slot, and so on. For long horizons this compounds error — use a
      direct-multi-step strategy or a deeper model in that case.
    - Exogenous regressors are ignored by this wrapper (lags only). Pass them
      through a separate pipeline if you need them.
    """

    def __init__(
        self,
        estimator,
        lags: Sequence[int] = (1, 7, 14, 28),
        date_col: str = "ts",
        freq: str | None = None,
    ) -> None:
        self.estimator = estimator
        self.lags = lags
        self.date_col = date_col
        self.freq = freq

    def fit(self, X: pd.DataFrame, y) -> LagRegressor:
        series = train_series(X, y, self.date_col)
        X_lag, y_aligned = _build_lag_matrix(series, list(self.lags))
        self._estimator_ = clone(self.estimator).fit(X_lag, y_aligned)
        self._history_ = series.to_numpy()
        self._last_train_ = series.index.max()
        self._freq_ = self.freq or infer_freq(series.index)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_estimator_", "_history_", "_last_train_", "_freq_"])
        target = as_datetime_index(X, self.date_col)
        lags = list(self.lags)
        max_steps = max(horizon_to_cover(self._last_train_, t, self._freq_) for t in target)

        history = list(self._history_)
        for _ in range(max_steps):
            feats = np.array([[history[-k] for k in lags]])
            history.append(float(self._estimator_.predict(feats)[0]))

        future = np.array(history[-max_steps:])
        offset = pd.tseries.frequencies.to_offset(self._freq_)
        future_idx = pd.date_range(self._last_train_ + offset, periods=max_steps, freq=self._freq_)
        return pd.Series(future, index=future_idx).reindex(target).to_numpy()
