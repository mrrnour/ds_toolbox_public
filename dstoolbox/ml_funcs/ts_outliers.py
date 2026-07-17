"""Outlier detection & imputation for univariate time series.

Statistical methods only — no ML models. Designed for daily series with
weekly or other low-order seasonality.

Detectors return a boolean Series aligned to the input (``True`` = outlier).
Imputers return a Series with outlier positions replaced.

NaN-handling summary
--------------------

================  ===========  ============================================
detector          NaN-tolerant  notes
================  ===========  ============================================
zscore            yes          NaNs excluded from mean/std; mask is False at NaN
mad               yes          NaNs excluded from median/MAD
iqr               yes          NaNs excluded from quartiles
rolling_zscore    yes          rolling skips NaN via min_periods
rolling_mad       yes          rolling skips NaN via min_periods
stl_resid         no*          STL needs a regular grid w/o NaN; pre-fill first
================  ===========  ============================================

================  ===========  ============================================
imputer           NaN-tolerant  notes
================  ===========  ============================================
linear            yes          pandas.interpolate('linear'); ends ffill/bfill
time              yes          requires DatetimeIndex; calendar-aware
rolling_median    yes          local median over a centered window
seasonal_mean     yes          mean over same season-bucket (e.g. dayofweek)
ffill_bfill       yes          forward then backward fill
stl_recon         no*          STL reconstruction (T+S); pre-fill first
================  ===========  ============================================

\\* STL-based steps require a contiguous numeric series. Either run a
cheap imputer first (``linear`` or ``seasonal_mean``) or call
:func:`replace_outliers` with ``pre_fill='linear'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

DetectMethod = Literal["zscore", "mad", "iqr", "rolling_zscore", "rolling_mad", "stl_resid"]
ImputeMethod = Literal["linear", "time", "rolling_median", "seasonal_mean", "ffill_bfill", "stl_recon"]


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detect_zscore(s: pd.Series, k: float = 3.0) -> pd.Series:
    """Global z-score: ``|x - mean| / std > k``. Non-robust (mean/std are pulled by the outliers themselves)."""
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(False, index=s.index)
    return ((s - mu).abs() / sd > k).fillna(False)


def detect_mad(s: pd.Series, k: float = 3.5) -> pd.Series:
    """Modified z-score using median & MAD (Iglewicz-Hoaglin). Robust to extreme outliers.

    Score = ``0.6745 * (x - median) / MAD``. Flag if ``|score| > k`` (k=3.5 is the textbook cutoff).
    """
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=s.index)
    score = 0.6745 * (s - med) / mad
    return (score.abs() > k).fillna(False)


def detect_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Tukey fences: outside ``[Q1 - k*IQR, Q3 + k*IQR]``. Robust; k=1.5 is the textbook default."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return ((s < lo) | (s > hi)).fillna(False)


def detect_rolling_zscore(s: pd.Series, window: int = 28, k: float = 3.0) -> pd.Series:
    """Local z-score over a centered rolling window. Tracks slow trends / level shifts."""
    roll = s.rolling(window=window, center=True, min_periods=max(3, window // 2))
    mu = roll.mean()
    sd = roll.std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return (z.abs() > k).fillna(False)


def detect_rolling_mad(s: pd.Series, window: int = 28, k: float = 3.5) -> pd.Series:
    """Local MAD-based modified z-score over a centered rolling window. Robust + adaptive."""
    roll = s.rolling(window=window, center=True, min_periods=max(3, window // 2))
    med = roll.median()
    mad = (s - med).abs().rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    score = 0.6745 * (s - med) / mad.replace(0, np.nan)
    return (score.abs() > k).fillna(False)


def detect_stl_resid(s: pd.Series, season_length: int, k: float = 3.5) -> pd.Series:
    """STL-decompose and flag points whose residual exceeds ``k * MAD(residual)``.

    Requires no NaN. If ``s`` has gaps, fill them first (e.g. ``s.interpolate()``).
    """
    from statsmodels.tsa.seasonal import STL

    if s.isna().any():
        raise ValueError("detect_stl_resid: pre-fill NaNs (e.g. s.interpolate()) before calling.")
    res = STL(s, period=season_length, robust=True).fit().resid
    mad = (res - res.median()).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=s.index)
    score = 0.6745 * (res - res.median()) / mad
    return (score.abs() > k).fillna(False)


_DETECTORS = {
    "zscore": detect_zscore,
    "mad": detect_mad,
    "iqr": detect_iqr,
    "rolling_zscore": detect_rolling_zscore,
    "rolling_mad": detect_rolling_mad,
    "stl_resid": detect_stl_resid,
}


def detect_outliers(s: pd.Series, method: DetectMethod = "rolling_mad", **kwargs) -> pd.Series:
    """Dispatch to one of the named detectors. ``**kwargs`` forwarded to the detector."""
    if method not in _DETECTORS:
        raise ValueError(f"Unknown detect method '{method}'. Options: {sorted(_DETECTORS)}")
    return _DETECTORS[method](s, **kwargs)


# ---------------------------------------------------------------------------
# Imputers (replace outliers; also fill incidental NaNs)
# ---------------------------------------------------------------------------

def _mask_to_nan(s: pd.Series, mask: pd.Series) -> pd.Series:
    out = s.astype(float).copy()
    out[mask.fillna(False)] = np.nan
    return out


def impute_linear(s: pd.Series, mask: pd.Series) -> pd.Series:
    """Linear interpolation between neighbours. Endpoints filled by bfill/ffill."""
    return _mask_to_nan(s, mask).interpolate("linear", limit_direction="both")


def impute_time(s: pd.Series, mask: pd.Series) -> pd.Series:
    """Time-aware interpolation. ``s`` must have a DatetimeIndex (or be convertible)."""
    x = _mask_to_nan(s, mask)
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("impute_time requires a DatetimeIndex.")
    return x.interpolate("time", limit_direction="both")


def impute_rolling_median(s: pd.Series, mask: pd.Series, window: int = 7) -> pd.Series:
    """Replace each outlier with the centered rolling median of the original (untainted) values."""
    clean = _mask_to_nan(s, mask)
    med = clean.rolling(window=window, center=True, min_periods=1).median()
    out = clean.copy()
    out[out.isna()] = med[out.isna()]
    return out.interpolate("linear", limit_direction="both")  # safety net for edges


def impute_seasonal_mean(s: pd.Series, mask: pd.Series, season_length: int) -> pd.Series:
    """Replace with mean over same position-in-cycle (e.g. same weekday for season_length=7).

    Position is computed from integer offset in the index, so this works on any index.
    """
    clean = _mask_to_nan(s, mask)
    pos = np.arange(len(clean)) % season_length
    seasonal_mean = pd.Series(clean.values, index=pos).groupby(level=0).mean()
    fill = pd.Series(seasonal_mean.reindex(pos).values, index=clean.index)
    out = clean.copy()
    out[out.isna()] = fill[out.isna()]
    return out.interpolate("linear", limit_direction="both")


def impute_ffill_bfill(s: pd.Series, mask: pd.Series) -> pd.Series:
    """Forward-fill then back-fill. Cheap; biases toward the most recent valid value."""
    return _mask_to_nan(s, mask).ffill().bfill()


def impute_stl_recon(s: pd.Series, mask: pd.Series, season_length: int) -> pd.Series:
    """Replace outliers with the STL trend + seasonal reconstruction ``T_t + S_t``.

    The STL fit needs a contiguous series, so we temporarily fill outliers + NaNs
    with linear interpolation, fit a robust STL, then drop in ``T + S`` at the
    masked positions.
    """
    from statsmodels.tsa.seasonal import STL

    pre = impute_linear(s, mask)
    stl = STL(pre, period=season_length, robust=True).fit()
    recon = stl.trend + stl.seasonal
    out = s.astype(float).copy()
    target = mask.fillna(False) | s.isna()
    out[target] = recon[target]
    return out


_IMPUTERS = {
    "linear": impute_linear,
    "time": impute_time,
    "rolling_median": impute_rolling_median,
    "seasonal_mean": impute_seasonal_mean,
    "ffill_bfill": impute_ffill_bfill,
    "stl_recon": impute_stl_recon,
}


def impute_outliers(s: pd.Series, mask: pd.Series, method: ImputeMethod = "linear", **kwargs) -> pd.Series:
    """Dispatch to one of the named imputers. ``**kwargs`` forwarded to the imputer."""
    if method not in _IMPUTERS:
        raise ValueError(f"Unknown impute method '{method}'. Options: {sorted(_IMPUTERS)}")
    return _IMPUTERS[method](s, mask, **kwargs)


# ---------------------------------------------------------------------------
# End-to-end convenience
# ---------------------------------------------------------------------------

def replace_outliers(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    detect: DetectMethod = "rolling_mad",
    impute: ImputeMethod = "linear",
    detect_kwargs: dict | None = None,
    impute_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Detect outliers in ``value_col`` and return ``(df_cleaned, outlier_mask)``.

    The cleaned frame has ``value_col`` overwritten with imputed values and an
    extra ``f"{value_col}_orig"`` column preserving the raw input. Existing
    NaN gaps in ``value_col`` are filled by the imputer too.

    Parameters
    ----------
    detect, impute : method names — see module docstring.
    detect_kwargs, impute_kwargs : forwarded to the chosen detector / imputer.
    """
    out = df.copy()
    s = pd.Series(
        pd.to_numeric(out[value_col], errors="coerce").values,
        index=pd.to_datetime(out[date_col]).values,
        name=value_col,
    )
    mask = detect_outliers(s, method=detect, **(detect_kwargs or {}))
    clean = impute_outliers(s, mask, method=impute, **(impute_kwargs or {}))

    out[f"{value_col}_orig"] = out[value_col].values
    out[value_col] = clean.values
    mask_aligned = pd.Series(mask.values, index=out.index, name="is_outlier")
    return out, mask_aligned


# ---------------------------------------------------------------------------
# Manual window masking (known-bad periods)
# ---------------------------------------------------------------------------

def _window_bounds(window: Any) -> tuple[Any, Any]:
    """Return ``(start, end)`` from either an object with ``.start``/``.end`` or a 2-tuple."""
    if hasattr(window, "start") and hasattr(window, "end"):
        return window.start, window.end
    start, end = window
    return start, end


def mask_anomalies(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    anomalies: Sequence[Any],
    *,
    fill: ImputeMethod | None = None,
    fill_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``value_col`` set to NaN inside each anomaly window.

    Each entry in ``anomalies`` is either a 2-tuple ``(start, end)`` or any object
    exposing ``.start`` / ``.end`` attributes (e.g. a pydantic ``AnomalyWindow``
    model). Bounds are inclusive on both ends.

    When ``fill`` is ``None`` (default), the masked rows stay NaN. NaN-intolerant
    forecasters (Greykite, Darts, sklearn-lag) will reject the frame; pass one of
    the imputer names from :data:`ImputeMethod` to bridge the gap in the same call:

    - ``"linear"`` / ``"time"`` — pandas interpolation (``"time"`` requires a
      ``DatetimeIndex``, which this function builds internally).
    - ``"ffill_bfill"`` — last-observation-carried-forward then back-fill.
    - ``"rolling_median"`` / ``"seasonal_mean"`` / ``"stl_recon"`` — dispatched
      through :func:`impute_outliers`; forward extra arguments via ``fill_kwargs``
      (e.g. ``fill_kwargs={"season_length": 7}``).

    Mask-only is the right choice if the downstream model tolerates NaN
    (``auto_arima``, ``mean_baseline``); use ``fill="linear"`` for a neutral
    bridge that keeps every forecaster happy.
    """
    if not anomalies:
        return df
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    mask = pd.Series(False, index=out.index)
    for window in anomalies:
        start, end = _window_bounds(window)
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    out.loc[mask, value_col] = float("nan")
    if fill is None or not mask.any():
        return out

    idx = pd.DatetimeIndex(dates.to_numpy())
    s = pd.Series(
        pd.to_numeric(out[value_col], errors="coerce").to_numpy(),
        index=idx,
        name=value_col,
    )
    mask_aligned = pd.Series(mask.to_numpy(), index=idx)
    filled = impute_outliers(s, mask_aligned, method=fill, **(fill_kwargs or {}))
    out[value_col] = filled.to_numpy()
    return out
