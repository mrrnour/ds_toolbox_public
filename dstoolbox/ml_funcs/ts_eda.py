"""Time-series EDA primitives.

Lightweight, dependency-minimal stats for univariate time series:
ACF, Bartlett CI, Ljung-Box, missing-value summary, seasonality aggregates.
Used standalone or as the data layer behind ``ts_plots``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def acf(values, nlags: int = 40) -> np.ndarray:
    """Biased sample autocorrelation function up to ``nlags``.

    NaNs are dropped before computation. Returns length ``nlags + 1`` array
    with ``acf[0] = 1``.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.full(nlags + 1, np.nan)
    x = x - x.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.full(nlags + 1, np.nan)
    n = len(x)
    return np.array([np.dot(x[: n - k], x[k:]) / var for k in range(nlags + 1)])


def acf_confint(n: int, alpha: float = 0.05) -> float:
    """Bartlett 95%-CI half-width for white-noise ACF at sample size ``n``."""
    z = 1.959963984540054 if alpha == 0.05 else 2.5758293035489004
    return z / np.sqrt(n)


def pacf(values, nlags: int = 40, method: str = "ywm") -> np.ndarray:
    """Sample partial autocorrelation function up to ``nlags``.

    Thin wrapper over :func:`statsmodels.tsa.stattools.pacf`. NaNs are dropped
    before computation. Returns length ``nlags + 1`` array with ``pacf[0] = 1``.

    ``method``: ``"ywm"`` (Yule-Walker, biased; default), ``"yw"`` (unbiased),
    ``"ols"``, ``"burg"``, ``"ld"`` (Levinson-Durbin). See statsmodels docs.
    """
    from statsmodels.tsa.stattools import pacf as _pacf

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or nlags >= len(x):
        return np.full(nlags + 1, np.nan)
    try:
        return np.asarray(_pacf(x, nlags=nlags, method=method))
    except Exception:
        return np.full(nlags + 1, np.nan)


def ljung_box(residuals, lags: int = 20) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value.

    Null hypothesis: residuals are independently distributed (white noise).
    Small p-values indicate remaining autocorrelation.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n <= lags + 1:
        return float("nan"), float("nan")
    rho = acf(r, lags)[1:]
    denom = n - np.arange(1, lags + 1)
    q = n * (n + 2) * np.sum((rho**2) / denom)
    p = float(1.0 - chi2.cdf(q, df=lags))
    return float(q), p


def missing_summary(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, float | int]:
    """Per-column missingness + gap stats on the time index."""
    n = len(df)
    n_missing = int(df[value_col].isna().sum())
    runs = (df[value_col].isna() != df[value_col].isna().shift()).cumsum()
    gap_lengths = df.loc[df[value_col].isna()].groupby(runs).size().to_numpy()
    return {
        "rows": n,
        "missing_values": n_missing,
        "missing_pct": (n_missing / n) if n else float("nan"),
        "date_min": df[date_col].min(),
        "date_max": df[date_col].max(),
        "n_gaps": int(len(gap_lengths)),
        "max_gap_len": int(gap_lengths.max()) if len(gap_lengths) else 0,
    }


def seasonal_table(
    df: pd.DataFrame, date_col: str, value_col: str, period: str = "dayofweek"
) -> pd.DataFrame:
    """Group values by a seasonal bucket for boxplots / aggregates.

    ``period``: ``"dayofweek"`` | ``"month"`` | ``"weekofyear"``.
    """
    out = df[[date_col, value_col]].dropna().copy()
    ts = pd.to_datetime(out[date_col]).dt
    if period == "dayofweek":
        out["bucket"] = ts.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        out["bucket"] = pd.Categorical(out["bucket"], categories=order, ordered=True)
    elif period == "month":
        out["bucket"] = ts.month_name()
        order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        out["bucket"] = pd.Categorical(out["bucket"], categories=order, ordered=True)
    elif period == "weekofyear":
        out["bucket"] = ts.isocalendar().week.astype(int)
    else:
        raise ValueError(f"Unknown period: {period}")
    return out


def adf_test(values, regression: str = "c") -> tuple[float, float]:
    """Augmented Dickey-Fuller unit-root test.

    Null hypothesis: the series has a unit root (non-stationary). Small
    p-values reject the null ⇒ the series is stationary.

    ``regression``: ``"c"`` (constant, default), ``"ct"`` (constant + trend),
    ``"ctt"`` (constant + linear + quadratic trend), ``"n"`` (no constant).
    NaNs are dropped before computation.
    """
    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 10:
        return float("nan"), float("nan")
    stat, p, *_ = adfuller(x, regression=regression, autolag="AIC")
    return float(stat), float(p)


def stationarity_report(
    values,
    alpha: float = 0.05,
    max_d: int = 2,
    regression: str = "c",
) -> pd.DataFrame:
    """Run ADF at ``d = 0, 1, ..., max_d`` differences of ``values``.

    Returns one row per differencing order with columns ``d``, ``adf_stat``,
    ``p_value``, ``n_obs``, ``stationary`` (``p_value < alpha``). The first
    row where ``stationary`` is True is the recommended integration order for
    ARIMA. If none is stationary at ``max_d``, the series needs a stronger
    transform (log, seasonal diff, detrending) before ARIMA.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    rows = []
    for d in range(max_d + 1):
        xd = np.diff(x, n=d) if d else x
        stat, p = adf_test(xd, regression=regression)
        rows.append(
            {
                "d": d,
                "adf_stat": stat,
                "p_value": p,
                "n_obs": int(xd.size),
                "stationary": bool(not np.isnan(p) and p < alpha),
            }
        )
    return pd.DataFrame(rows)
