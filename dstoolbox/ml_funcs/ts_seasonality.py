"""Seasonality-period detection primitives.

Lightweight, dependency-minimal helpers for identifying the dominant seasonal
period(s) of a univariate series before fitting a seasonal model (e.g. the
``season_length`` knob of ``AutoArimaSklearn`` / ``SeasonalNaive``).

Four complementary signals are exposed, plus a ranked-summary orchestrator:

* ``periodogram_top_periods``  — Welch periodogram, dominant frequencies.
* ``acf_top_periods``           — peaks of the ACF beyond the Bartlett band.
* ``stl_seasonal_strength``     — Wang-Smith-Hyndman :math:`F_s` for a given
                                  candidate period.
* ``friedman_seasonality_test`` — non-parametric "is there a within-cycle
                                  effect at period :math:`m`?" test.
* ``detect_seasonality``        — combines the four into one ranked table.

These are exploratory primitives, not unit-root tests. To decide *seasonal
differencing order* at a chosen period, use OCSB / Canova-Hansen via
``pmdarima`` or let ``AutoArimaSklearn`` handle it internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import friedmanchisquare

from .ts_eda import acf, acf_confint


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[~np.isnan(x)]


def periodogram_top_periods(
    values,
    *,
    top_k: int = 5,
    min_period: int = 2,
    max_period: int | None = None,
    detrend: str = "linear",
) -> pd.DataFrame:
    """Return the ``top_k`` dominant periods from the Welch periodogram.

    Parameters
    ----------
    values : array-like
        Univariate, evenly-spaced series. NaNs are dropped.
    top_k : int, default 5
        Number of dominant periods to return.
    min_period, max_period : int
        Period bounds in *samples* (e.g. days for daily data). ``max_period``
        defaults to ``n // 2``.
    detrend : str, default ``"linear"``
        Passed to :func:`scipy.signal.periodogram`.

    Returns
    -------
    DataFrame with columns ``period``, ``frequency``, ``power``,
    ``power_ratio`` (share of total in-band power), sorted by ``power`` desc.
    """
    x = _clean(values)
    n = len(x)
    if n < 4:
        return pd.DataFrame(columns=["period", "frequency", "power", "power_ratio"])
    freqs, psd = signal.periodogram(x, fs=1.0, detrend=detrend, scaling="density")
    # drop f=0 (DC component)
    keep = freqs > 0
    freqs, psd = freqs[keep], psd[keep]
    periods = 1.0 / freqs
    hi = max_period if max_period is not None else n // 2
    band = (periods >= min_period) & (periods <= hi)
    freqs, psd, periods = freqs[band], psd[band], periods[band]
    if psd.size == 0:
        return pd.DataFrame(columns=["period", "frequency", "power", "power_ratio"])
    total = psd.sum()
    order = np.argsort(psd)[::-1][:top_k]
    return pd.DataFrame(
        {
            "period": np.round(periods[order], 2),
            "frequency": freqs[order],
            "power": psd[order],
            "power_ratio": psd[order] / total if total > 0 else np.nan,
        }
    ).reset_index(drop=True)


def acf_top_periods(
    values,
    *,
    nlags: int | None = None,
    top_k: int = 5,
    min_lag: int = 2,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return the ``top_k`` lags whose ACF peaks exceed the Bartlett band.

    Uses :func:`scipy.signal.find_peaks` on the sample ACF, keeps peaks whose
    absolute ACF exceeds the white-noise CI half-width at level ``alpha``,
    and ranks by peak ACF value.

    Returns
    -------
    DataFrame with columns ``period`` (the lag), ``acf_value``,
    ``threshold`` (Bartlett half-width).
    """
    x = _clean(values)
    n = len(x)
    if nlags is None:
        nlags = min(n - 1, max(40, n // 3))
    if n <= min_lag + 1:
        return pd.DataFrame(columns=["period", "acf_value", "threshold"])
    rho = acf(x, nlags=nlags)
    thr = acf_confint(n, alpha=alpha)
    peaks, _ = signal.find_peaks(rho[min_lag:], height=thr)
    if peaks.size == 0:
        return pd.DataFrame(columns=["period", "acf_value", "threshold"])
    lags = peaks + min_lag
    order = np.argsort(rho[lags])[::-1][:top_k]
    return pd.DataFrame(
        {
            "period": lags[order].astype(int),
            "acf_value": rho[lags][order],
            "threshold": thr,
        }
    ).reset_index(drop=True)


def stl_seasonal_strength(
    values,
    period: int,
    *,
    robust: bool = True,
) -> float:
    """Wang-Smith-Hyndman seasonal strength :math:`F_s \\in [0, 1]`.

    Defined as :math:`F_s = \\max\\!\\left(0,\\ 1 - \\dfrac{\\mathrm{Var}(R_t)}
    {\\mathrm{Var}(R_t + S_t)}\\right)` where :math:`R_t`, :math:`S_t` are the
    STL remainder and seasonal components at the given period. Values near 0
    indicate no seasonality at ``period``; near 1 indicate strong seasonality.
    """
    from statsmodels.tsa.seasonal import STL

    x = _clean(values)
    if period < 2 or len(x) < 2 * period:
        return float("nan")
    try:
        res = STL(x, period=int(period), robust=robust).fit()
    except Exception:
        return float("nan")
    r, s = np.asarray(res.resid), np.asarray(res.seasonal)
    denom = np.var(r + s)
    if denom <= 0:
        return float("nan")
    return float(max(0.0, 1.0 - np.var(r) / denom))


def stl_decompose(
    values,
    period: int,
    *,
    robust: bool = True,
) -> pd.DataFrame:
    """STL decomposition components as a DataFrame.

    Returns columns ``observed``, ``trend``, ``seasonal``, ``resid`` (one row
    per input observation). Preserves the input index when ``values`` is a
    pandas Series. Requires at least ``2 * period`` observations.
    """
    from statsmodels.tsa.seasonal import STL

    if isinstance(values, pd.Series):
        idx = values.index
        x = np.asarray(values.astype(float).values)
    else:
        idx = None
        x = np.asarray(values, dtype=float)
    if period < 2 or len(x) < 2 * period:
        raise ValueError(
            f"STL needs period >= 2 and at least 2 * period observations "
            f"(got period={period}, n={len(x)})"
        )
    res = STL(x, period=int(period), robust=robust).fit()
    out = pd.DataFrame(
        {
            "observed": x,
            "trend": np.asarray(res.trend),
            "seasonal": np.asarray(res.seasonal),
            "resid": np.asarray(res.resid),
        }
    )
    if idx is not None:
        out.index = idx
    return out


def friedman_seasonality_test(values, period: int) -> tuple[float, float]:
    """Friedman test for a within-cycle effect at ``period``.

    Reshapes the series into complete ``period``-length cycles (trailing
    samples truncated) and runs a Friedman rank test across the ``period``
    positions. Small p-values reject H0 of no seasonality at that period.

    Returns
    -------
    (statistic, p_value); ``(nan, nan)`` if fewer than 2 complete cycles.
    """
    x = _clean(values)
    if period < 2 or len(x) < 2 * period:
        return float("nan"), float("nan")
    n_cycles = len(x) // period
    blocks = x[: n_cycles * period].reshape(n_cycles, period)
    cols = [blocks[:, j] for j in range(period)]
    try:
        stat, p = friedmanchisquare(*cols)
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(p)


def detect_seasonality(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    candidate_periods: list[int] | None = None,
    top_k: int = 5,
    min_period: int = 2,
    max_period: int | None = None,
) -> pd.DataFrame:
    """Rank candidate seasonal periods for a univariate series.

    If ``candidate_periods`` is None, candidates are the union of the
    periodogram top-``top_k`` and ACF top-``top_k`` peaks (rounded to int).
    For each candidate, every signal is computed and the result is sorted by
    a composite rank.

    Returns
    -------
    DataFrame with columns:
    ``period``, ``periodogram_power_ratio``, ``acf_value``,
    ``stl_Fs``, ``friedman_p``, ``rank`` (1 = best).
    """
    s = df[[date_col, value_col]].dropna().sort_values(date_col)[value_col].to_numpy()
    pg = periodogram_top_periods(s, top_k=top_k, min_period=min_period, max_period=max_period)
    ac = acf_top_periods(s, top_k=top_k, min_lag=min_period)

    if candidate_periods is None:
        cand = set()
        cand.update(int(round(p)) for p in pg["period"].tolist())
        cand.update(int(p) for p in ac["period"].tolist())
        candidates = sorted(c for c in cand if c >= min_period)
    else:
        candidates = sorted(set(int(c) for c in candidate_periods))
    if not candidates:
        return pd.DataFrame(
            columns=[
                "period",
                "periodogram_power_ratio",
                "acf_value",
                "stl_Fs",
                "friedman_p",
                "rank",
            ]
        )

    pg_map = {int(round(p)): r for p, r in zip(pg["period"], pg["power_ratio"], strict=False)}
    ac_map = {int(p): v for p, v in zip(ac["period"], ac["acf_value"], strict=False)}

    rows = []
    for m in candidates:
        stat, p_friedman = friedman_seasonality_test(s, m)
        rows.append(
            {
                "period": m,
                "periodogram_power_ratio": pg_map.get(m, np.nan),
                "acf_value": ac_map.get(m, np.nan),
                "stl_Fs": stl_seasonal_strength(s, m),
                "friedman_p": p_friedman,
            }
        )
    out = pd.DataFrame(rows)
    # Composite score: average rank across the four signals (higher is better
    # for power/acf/Fs; lower is better for friedman_p so we invert).
    score = (
        out["periodogram_power_ratio"].rank(ascending=False, na_option="bottom")
        + out["acf_value"].rank(ascending=False, na_option="bottom")
        + out["stl_Fs"].rank(ascending=False, na_option="bottom")
        + out["friedman_p"].rank(ascending=True, na_option="bottom")
    )
    out["rank"] = score.rank(method="min").astype(int)
    return out.sort_values("rank").reset_index(drop=True)
