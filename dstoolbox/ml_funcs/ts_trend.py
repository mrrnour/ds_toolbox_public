"""Package-first non-parametric trend tests.

Thin wrappers over three published Python packages:

* ``pymannkendall`` — 11 MK variants (original, HR, Yue-Wang, PW, TFPW,
  VCTFPW, seasonal, correlated seasonal, partial, regional).
* ``mannkendall`` (MeteoSwiss) — the only Python implementation of 3PW
  blended prewhitening (Collaud Coen et al. 2020, AMT) with Gilbert-style
  analytical CI on Sen's slope.
* ``scipy.stats.theilslopes`` — Theil-Sen with the Kendall-tau analytical
  confidence interval on the slope.

Only three things are implemented locally, because no Python package
exposes them:

1. **Van Belle and Hughes 1984 seasonal-homogeneity χ²** (in
   :func:`seasonal_mk`) — a small missing piece that guards against
   pooling seasons that trend in opposite directions. ``pymannkendall``
   returns pooled seasonal MK but not the homogeneity statistic.
2. **Full VBH χ² decomposition across sites × seasons** (in
   :func:`regional_homogeneity`) — ``pymannkendall.regional_test`` yields
   only the pooled regional trend, not the season / station / interaction
   breakdown.
3. **Paired pre/post Sen-slope percentile bootstrap** (in
   :func:`paired_slope_test_boot`) — kept for arms with strong residual
   autocorrelation, where the analytical CI on ``theilslopes`` is too
   tight.

The default paired test (:func:`paired_slope_test`) is fully analytical
and delegates to ``scipy.stats.theilslopes`` + ``pymannkendall``.

Install once with::

    pip install pymannkendall mannkendall

``scipy``, ``numpy`` and ``pandas`` are assumed already available.

Third-party licenses
--------------------
All runtime dependencies are permissive and safe for commercial /
closed-source use:

* ``pymannkendall`` — MIT License (© Md. Manjurul Hussain Shourov)
* ``mannkendall`` — BSD 3-Clause License (© MeteoSwiss)
* ``scipy`` — BSD 3-Clause License (© SciPy Developers)
* ``numpy`` — BSD 3-Clause License (© NumPy Developers)
* ``pandas`` — BSD 3-Clause License (© AQR Capital Management, Lambda
  Foundry, PyData Development Team, and Open source contributors)

MIT and BSD 3-Clause both require the upstream copyright notice and
license text to be preserved when the code is redistributed. BSD
3-Clause additionally forbids using the project or contributor names in
promotional / marketing materials without written permission. Neither
license imposes any obligation on the license of the combined work, so
this module can be used inside proprietary code. See
``THIRD_PARTY_LICENSES.md`` in this directory for the compliance
summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, theilslopes


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendResult:
    """Single-series MK trend test result.

    ``slope_ci`` is ``None`` when the underlying package does not return a
    CI on the Sen slope (all ``pymannkendall`` tests). Use :func:`sen_slope_ci`
    or :func:`mk_3pw` to obtain a CI alongside the p-value.
    """

    method: str
    n: int
    slope: float
    slope_ci: tuple[float, float] | None
    p: float
    z: float
    tau: float

    def summary(self) -> str:
        ci = (f"CI [{self.slope_ci[0]:+.4f}, {self.slope_ci[1]:+.4f}]"
              if self.slope_ci is not None else "CI n/a")
        return (
            f"[{self.method}] n={self.n}  slope={self.slope:+.4f}  {ci}  "
            f"z={self.z:+.3f}  tau={self.tau:+.3f}  p={self.p:.4f}"
        )


@dataclass(frozen=True)
class SeasonalTrendResult:
    """Pooled seasonal MK + Van Belle-Hughes homogeneity guard."""

    n: int
    n_seasons: int
    slope: float
    p: float
    z: float
    tau: float
    homogeneity_Q: float
    homogeneity_p: float
    per_season: pd.DataFrame

    def pool_ok(self, alpha: float = 0.05) -> bool:
        """True when the seasons agree well enough to trust the pooled p."""
        return bool(np.isfinite(self.homogeneity_p) and self.homogeneity_p > alpha)

    def summary(self) -> str:
        verdict = "pool OK" if self.pool_ok() else "POOL UNSAFE — read per_season"
        return (
            f"n={self.n} across {self.n_seasons} seasons  "
            f"slope={self.slope:+.4f}  z={self.z:+.3f}  p={self.p:.4f}\n"
            f"  homogeneity Q={self.homogeneity_Q:.2f}  p={self.homogeneity_p:.4f}  ({verdict})"
        )


@dataclass(frozen=True)
class VbhDecomposition:
    """Van Belle-Hughes χ² decomposition of a seasonal MK statistic.

    Splits the pooled seasonal χ² into a 1-df trend component (all seasons
    move together) and a (k-1)-df heterogeneity component (seasons cancel).
    See Hipel & McLeod (1994) §8.4 and Van Belle & Hughes (1984).

    Attributes
    ----------
    per_season : pd.DataFrame
        One row per season with columns
        ``season, label, n, S, var_S, Z``.
    S_total, var_total : float
        Sum of Kendall S and Var(S) over seasons with ``var_S > 0``.
    chi2_trend, p_trend, df_trend : float, float, int
        Trend component ``(ΣS)² / ΣVar(S)`` and its p-value; ``df=1``.
    chi2_het, p_het, df_het : float, float, int
        Heterogeneity component ``χ²_total − χ²_trend`` and its p-value;
        ``df = k-1`` where ``k`` is the number of non-degenerate seasons.
    k : int
        Number of seasons contributing to the χ² (``var_S > 0``).
    """

    per_season: pd.DataFrame
    S_total: float
    var_total: float
    chi2_trend: float
    p_trend: float
    df_trend: int
    chi2_het: float
    p_het: float
    df_het: int
    k: int

    def is_inhomogeneous(self, alpha: float = 0.05) -> bool:
        """True when ``p_het < alpha`` (seasons disagree)."""
        return bool(np.isfinite(self.p_het) and self.p_het < alpha)

    def summary(self, alpha: float = 0.05) -> str:
        verdict = "inhomogeneous" if self.is_inhomogeneous(alpha) else "homogeneous"
        return (
            f"ΣS={self.S_total:+.0f}  "
            f"χ²_trend={self.chi2_trend:.2f} (p={self.p_trend:.3f}, df={self.df_trend})  "
            f"χ²_het={self.chi2_het:.2f} (p={self.p_het:.3f}, df={self.df_het})  "
            f"→ {verdict}"
        )


@dataclass(frozen=True)
class PairedSlopeResult:
    """Pre/post Sen-slope comparison."""

    slope_pre: float
    slope_post: float
    slope_ci_pre: tuple[float, float]
    slope_ci_post: tuple[float, float]
    delta_slope: float
    delta_ci: tuple[float, float]
    pct_rate_change: float | None
    pct_ci: tuple[float, float] | None
    pval_pre: float
    pval_post: float
    n_pre: int
    n_post: int
    method: str
    homogeneity_p_pre: float | None = None
    homogeneity_p_post: float | None = None

    @property
    def pct_unstable(self) -> bool:
        """True when the ratio is anchored on a near-zero ``slope_pre``."""
        if self.pct_ci is None or self.pct_rate_change is None:
            return False
        lo, hi = self.pct_ci
        if hi - lo > 2.0:
            return True
        half = max(abs(self.pct_rate_change - lo), abs(hi - self.pct_rate_change))
        return half > 2.0 * max(abs(self.pct_rate_change), 1e-12)

    def summary(self) -> str:
        if self.pct_rate_change is None:
            pct_line = "rate change= n/a  (slope_pre = 0)"
        elif self.pct_unstable:
            pct_line = (
                f"rate change= unstable (|slope_pre|={abs(self.slope_pre):.4f} too small "
                f"to anchor a ratio — use Δslope instead)"
            )
        else:
            pct = f"{self.pct_rate_change * 100:+.0f}%"
            assert self.pct_ci is not None
            pct_ci = f"[{self.pct_ci[0] * 100:+.0f}%, {self.pct_ci[1] * 100:+.0f}%]"
            pct_line = f"rate change= {pct}  CI {pct_ci}"
        return (
            f"slope_pre  = {self.slope_pre:+.4f}/step   "
            f"CI [{self.slope_ci_pre[0]:+.4f}, {self.slope_ci_pre[1]:+.4f}]  "
            f"(n={self.n_pre}, p={self.pval_pre:.4f})\n"
            f"slope_post = {self.slope_post:+.4f}/step  "
            f"CI [{self.slope_ci_post[0]:+.4f}, {self.slope_ci_post[1]:+.4f}]  "
            f"(n={self.n_post}, p={self.pval_post:.4f})\n"
            f"Δslope     = {self.delta_slope:+.4f}/step  "
            f"CI [{self.delta_ci[0]:+.4f}, {self.delta_ci[1]:+.4f}]  ({self.method})\n"
            f"{pct_line}"
        )


@dataclass(frozen=True)
class RegionalHomogeneityResult:
    """Van Belle-Hughes 1984 χ² decomposition across sites (× seasons)."""

    n_sites: int
    n_seasons: int
    chi_total: float
    chi_trend: float
    chi_homogeneity: float
    chi_season: float
    chi_station: float
    chi_station_season: float
    p_total: float
    p_trend: float
    p_homogeneity: float
    p_season: float
    p_station: float
    p_station_season: float
    overall_trend_meaningful: bool
    alpha: float
    per_group: pd.DataFrame

    def summary(self) -> str:
        if self.n_seasons <= 1:
            verdict = "homogeneous — pooling OK" if self.p_homogeneity > self.alpha else "HETEROGENEOUS — do not pool"
            return (
                f"1-way homogeneity across {self.n_sites} sites\n"
                f"  chi_homogeneity = {self.chi_homogeneity:.2f}   "
                f"p = {self.p_homogeneity:.4f}  ({verdict})"
            )
        verdict = "MEANINGFUL" if self.overall_trend_meaningful else "NOT MEANINGFUL"
        return (
            f"2-way homogeneity ({self.n_sites} sites × {self.n_seasons} seasons)\n"
            f"  chi_total          = {self.chi_total:.2f}    p = {self.p_total:.4f}\n"
            f"  chi_trend          = {self.chi_trend:.2f}    p = {self.p_trend:.4f}\n"
            f"  chi_homogeneity    = {self.chi_homogeneity:.2f}    p = {self.p_homogeneity:.4f}\n"
            f"  chi_season         = {self.chi_season:.2f}    p = {self.p_season:.4f}\n"
            f"  chi_station        = {self.chi_station:.2f}    p = {self.p_station:.4f}\n"
            f"  chi_station_season = {self.chi_station_season:.2f}    p = {self.p_station_season:.4f}\n"
            f"  overall regional trend: {verdict}"
        )


# ---------------------------------------------------------------------------
# Single-series MK — pymannkendall wrappers
# ---------------------------------------------------------------------------


def _clean(y: Sequence[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return arr[np.isfinite(arr)]


def _to_trend_result(r, method: str, n: int) -> TrendResult:
    """Convert a pymannkendall named-tuple to :class:`TrendResult`."""
    return TrendResult(
        method=method,
        n=n,
        slope=float(r.slope),
        slope_ci=None,
        p=float(r.p),
        z=float(r.z),
        tau=float(r.Tau),
    )


def mk_original(y: Sequence[float], alpha: float = 0.05) -> TrendResult:
    """Plain Mann-Kendall (no autocorrelation correction).

    Delegates to ``pymannkendall.original_test``.
    """
    import pymannkendall as pmk
    y = _clean(y)
    return _to_trend_result(pmk.original_test(y, alpha=alpha), "original", len(y))


def mk_hamed_rao(y: Sequence[float], alpha: float = 0.05, lag: int | None = 1) -> TrendResult:
    """Hamed and Rao 1998 variance-corrected MK.

    Delegates to ``pymannkendall.hamed_rao_modification_test``. Default
    ``lag=1`` avoids the package's default of using every lag up to
    ``n − 1``, which over-corrects on short arms; pass ``lag=None`` for
    the package default when the series is long and stationary.
    """
    import pymannkendall as pmk
    y = _clean(y)
    return _to_trend_result(
        pmk.hamed_rao_modification_test(y, alpha=alpha, lag=lag),
        "hamed_rao", len(y),
    )


def mk_yue_wang(y: Sequence[float], alpha: float = 0.05, lag: int | None = 1) -> TrendResult:
    """Yue and Wang 2004 modified MK.

    Delegates to ``pymannkendall.yue_wang_modification_test``.
    """
    import pymannkendall as pmk
    y = _clean(y)
    return _to_trend_result(
        pmk.yue_wang_modification_test(y, alpha=alpha, lag=lag),
        "yue_wang", len(y),
    )


def mk_tfpw(y: Sequence[float], alpha: float = 0.05) -> TrendResult:
    """Yue et al. 2002 trend-free pre-whitened MK.

    Delegates to ``pymannkendall.trend_free_pre_whitening_modification_test``.
    """
    import pymannkendall as pmk
    y = _clean(y)
    return _to_trend_result(
        pmk.trend_free_pre_whitening_modification_test(y, alpha=alpha),
        "tfpw", len(y),
    )


def mk_pw(y: Sequence[float], alpha: float = 0.05) -> TrendResult:
    """Kulkarni-von Storch 1995 pre-whitened MK.

    Delegates to ``pymannkendall.pre_whitening_modification_test``.
    """
    import pymannkendall as pmk
    y = _clean(y)
    return _to_trend_result(
        pmk.pre_whitening_modification_test(y, alpha=alpha),
        "pw", len(y),
    )


def mk_3pw(
    dts: Sequence,
    y: Sequence[float],
    resolution: float = 1.0,
    alpha_mk: float = 95.0,
    alpha_cl: float = 90.0,
) -> TrendResult:
    """3PW blended-prewhitening MK (Collaud Coen et al. 2020, AMT).

    Delegates to ``mannkendall.mk_temp_aggr``. This is the only Python
    implementation of 3PW blending (PW + TFPW-Y for significance, VCTFPW
    for slope) and it returns a Gilbert-style analytical CI on the Sen
    slope — the reason to prefer it over ``pymannkendall`` when a CI on
    the slope itself is needed.

    Parameters
    ----------
    dts : array-like of datetime
        Timestamps in time order.
    y : array-like
        Values.
    resolution : float, default 1.0
        Instrument resolution — controls tie detection only.
    alpha_mk : float, default 95.0
        Significance level for the MK test (percent).
    alpha_cl : float, default 90.0
        Confidence level for the Sen slope CI (percent).

    Notes
    -----
    ``mannkendall`` returns the slope in **units per year** by convention.
    Divide by 365.25 (or the appropriate cycle length) to convert to a
    per-step slope comparable with :func:`sen_slope_ci`.
    """
    import mannkendall as _mk

    arr = np.asarray(y, dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    # ``mannkendall.mk_stats.s_test`` explicitly rejects numpy datetime64 and
    # anything else that is not a ``datetime.datetime`` instance
    # ("Ouch ! I need proper datetime.datetime entities !"). Coerce via
    # pandas so the wrapper is agnostic to the caller's timestamp type.
    dts_native = pd.to_datetime(np.asarray(dts))[mask].to_pydatetime()
    dts_native = np.asarray(dts_native, dtype=object)

    # ``mannkendall.mk_stats.s_test`` computes S by grouping observations by
    # calendar year — it was designed for multi-year climate series. When the
    # data spans a single year, S is always 0, p collapses to 1.0, and the
    # Gilbert CI degenerates. Refuse loudly instead of returning garbage.
    years = {t.year for t in dts_native}
    if len(years) < 2:
        raise ValueError(
            "mk_3pw requires timestamps that span at least two calendar years "
            "(mannkendall.s_test groups by year). Got a single year "
            f"({sorted(years)[0]}). For sub-year daily data use "
            "paired_slope_test_boot instead — see ts_trend.paired_slope_test_ar1 "
            "for an auto-picker."
        )

    out = _mk.mk_temp_aggr([dts_native], [arr], resolution=float(resolution),
                           alpha_mk=alpha_mk, alpha_cl=alpha_cl)
    # Yearly aggregate is the last key; for a single-season call it equals
    # the per-season entry, but reading it uniformly is safer.
    entry = out[max(out.keys())]
    return TrendResult(
        method="3pw",
        n=len(arr),
        slope=float(entry["slope"]),
        slope_ci=(float(entry["lcl"]), float(entry["ucl"])),
        p=float(entry["p"]),
        z=float("nan"),  # mannkendall does not return Z separately
        tau=float("nan"),
    )


def sen_slope_ci(
    y: Sequence[float],
    confidence_level: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Sen slope + analytical Kendall-tau CI via ``scipy.stats.theilslopes``.

    Assumes independence within the series. For an autocorrelation-aware
    CI on the slope, use :func:`mk_3pw`.
    """
    arr = _clean(y)
    slope, _, lo, hi = theilslopes(arr, np.arange(len(arr)), alpha=confidence_level)
    return float(slope), (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Small diagnostics — used by the intervention workflow (§6 of the paper)
# ---------------------------------------------------------------------------


def deseason(y: Sequence[float], period: int) -> np.ndarray:
    """Subtract the per-phase mean from ``y``.

    For daily data with weekly seasonality use ``period=7``. Missing values
    are ignored when computing each phase mean, then filled back into the
    output as ``nan``. Point-wise seasonality removal — not aggregation.
    """
    arr = np.asarray(y, dtype=float)
    period = int(period)
    phase = np.arange(len(arr)) % period
    out = arr.copy()
    for k in range(period):
        mask = phase == k
        chunk = arr[mask]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[mask] = chunk - float(finite.mean())
    return out


def lag1_acf(y: Sequence[float]) -> float:
    """Return the lag-1 autocorrelation of ``y`` (NaN-safe).

    Convention matches ``statsmodels.tsa.stattools.acf(y, nlags=1)[1]``:
    biased estimator, mean-centered, divided by the zero-lag variance.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    z = arr - arr.mean()
    denom = float((z * z).sum())
    if denom == 0.0:
        return float("nan")
    return float((z[1:] * z[:-1]).sum() / denom)


# ---------------------------------------------------------------------------
# Seasonal MK + Van Belle-Hughes homogeneity χ² (local: not in packages)
# ---------------------------------------------------------------------------


def seasonal_mk(
    y: Sequence[float],
    period: int,
    alpha: float = 0.05,
    hr_lag: int | None = 1,
) -> SeasonalTrendResult:
    """Seasonal MK (Hirsch 1982) with Van Belle-Hughes 1984 homogeneity.

    Pooled seasonal test comes from ``pymannkendall.seasonal_test``. The
    homogeneity χ² is added on top because no Python package exposes it:

    .. math::

        Q = \\sum_s Z_s^2 - \\frac{(\\sum_s Z_s)^2}{k}
        \\quad\\sim\\quad \\chi^2_{k-1}

    A small ``homogeneity_p`` signals that the seasons disagree in
    direction (e.g. weekdays up, weekends down); in that case the pooled
    p-value should not be quoted — read ``per_season`` instead.
    Per-season stats come from ``pymannkendall.hamed_rao_modification_test``.

    Parameters
    ----------
    y : array-like
        Series in time order.
    period : int
        Seasonal cycle length (7 for daily-weekly, 12 for monthly).
    alpha : float, default 0.05
        Significance level; only used for the h flag inside pymannkendall.
    hr_lag : int or None, default 1
        ``lag`` passed to the per-season HR test.
    """
    import pymannkendall as pmk
    y = _clean(y)
    period = int(period)

    pooled = pmk.seasonal_test(y, period=period, alpha=alpha)

    rows: list[dict] = []
    z_list: list[float] = []
    for s in range(period):
        y_s = y[s::period]
        if len(y_s) < 4 or np.unique(y_s).size == 1:
            continue
        r = pmk.hamed_rao_modification_test(y_s, alpha=alpha, lag=hr_lag)
        rows.append({
            "season": s,
            "n": len(y_s),
            "z": float(r.z),
            "p": float(r.p),
            "slope": float(r.slope),
            "tau": float(r.Tau),
        })
        z_list.append(float(r.z))

    z_arr = np.asarray(z_list, dtype=float)
    z_arr = z_arr[np.isfinite(z_arr)]
    k = len(z_arr)
    if k > 1:
        Q = float((z_arr ** 2).sum() - (z_arr.sum() ** 2) / k)
        Q = max(Q, 0.0)
        homog_p = float(chi2.sf(Q, df=k - 1))
    else:
        Q = float("nan")
        homog_p = float("nan")

    return SeasonalTrendResult(
        n=len(y),
        n_seasons=k,
        slope=float(pooled.slope),
        p=float(pooled.p),
        z=float(pooled.z),
        tau=float(pooled.Tau),
        homogeneity_Q=Q,
        homogeneity_p=homog_p,
        per_season=pd.DataFrame(rows),
    )


def _kendall_s_and_var(y: np.ndarray) -> tuple[float, float, int]:
    """Return ``(S, Var(S), n)`` for one series without tie correction.

    Uses the classic Kendall S = Σ sgn(y_j − y_i) over ``i<j`` and
    Var(S) = n(n−1)(2n+5)/18 (no ties). Returns zeros when ``n < 4``.
    """
    y = y[np.isfinite(y)]
    n = int(len(y))
    if n < 4:
        return 0.0, 0.0, n
    i, j = np.triu_indices(n, k=1)
    s = float(np.sign(y[j] - y[i]).sum())
    var = n * (n - 1) * (2 * n + 5) / 18.0
    return s, float(var), n


def vbh_chi2_decomposition(
    y: Sequence[float],
    period: int,
    labels: Sequence[str] | None = None,
) -> VbhDecomposition:
    """Van Belle-Hughes χ² trend / heterogeneity decomposition (paper §8.4).

    Splits the pooled seasonal MK χ² into

    * ``χ²_trend = (ΣS)² / ΣVar(S)``  (df=1) — all seasons trending together;
    * ``χ²_het   = Σ S²/Var(S) − χ²_trend``  (df=k-1) — seasons cancelling.

    Uses raw Kendall S per season with no tie or autocorrelation
    correction, matching the illustration in Van Belle & Hughes (1984).

    Parameters
    ----------
    y : array-like
        Series in time order (NaNs dropped).
    period : int
        Seasonal cycle length (7 for daily-weekly, 12 for monthly).
    labels : sequence of str, optional
        Human-readable season labels of length ``period``. Defaults to
        ``["s0", "s1", …]``.

    Returns
    -------
    VbhDecomposition
        Per-season table plus the trend / heterogeneity χ² split.

    Example
    -------
    >>> d = vbh_chi2_decomposition(y, period=7, labels=day_labels)
    >>> d.is_inhomogeneous(alpha=0.05)
    True
    """
    arr = _clean(y)
    period = int(period)
    if labels is None:
        labels = [f"s{i}" for i in range(period)]
    if len(labels) != period:
        raise ValueError(
            f"labels must have length period={period}, got {len(labels)}"
        )

    rows: list[dict] = []
    for s in range(period):
        S, V, n = _kendall_s_and_var(arr[s::period])
        Z = S / np.sqrt(V) if V > 0 else 0.0
        rows.append({
            "season": s,
            "label": labels[s],
            "n": n,
            "S": S,
            "var_S": V,
            "Z": Z,
        })

    per_season = pd.DataFrame(rows)
    contributing = per_season["var_S"] > 0
    S_total = float(per_season.loc[contributing, "S"].sum())
    var_total = float(per_season.loc[contributing, "var_S"].sum())
    chi2_total = float(
        (per_season.loc[contributing, "S"] ** 2
         / per_season.loc[contributing, "var_S"]).sum()
    )
    chi2_trend = (S_total ** 2) / var_total if var_total > 0 else 0.0
    chi2_het = max(chi2_total - chi2_trend, 0.0)
    k = int(contributing.sum())
    df_trend = 1
    df_het = max(k - 1, 1)
    p_trend = float(chi2.sf(chi2_trend, df=df_trend)) if var_total > 0 else float("nan")
    p_het   = float(chi2.sf(chi2_het,   df=df_het))   if k > 1 else float("nan")

    return VbhDecomposition(
        per_season=per_season,
        S_total=S_total,
        var_total=var_total,
        chi2_trend=chi2_trend, p_trend=p_trend, df_trend=df_trend,
        chi2_het=chi2_het,     p_het=p_het,     df_het=df_het,
        k=k,
    )


def correlated_seasonal_mk(y: Sequence[float], period: int, alpha: float = 0.05) -> TrendResult:
    """Hipel 1994 seasonal MK with cross-day covariance.

    Delegates to ``pymannkendall.correlated_seasonal_test`` and normalises
    the raw NamedTuple into a :class:`TrendResult` so callers can use
    ``.summary()`` uniformly. Use it when adjacent days share information
    (typical for daily product metrics with weekly seasonality) and you
    want the pooled test to account for cross-season covariance.
    """
    import pymannkendall as pmk
    arr = _clean(y)
    res = pmk.correlated_seasonal_test(arr, period=int(period), alpha=alpha)
    return TrendResult(
        method="correlated_seasonal",
        n=len(arr),
        slope=float(res.slope),
        slope_ci=None,
        p=float(res.p),
        z=float(res.z),
        tau=float(res.Tau),
    )


def partial_mk(y: Sequence[float], covariate: Sequence[float], alpha: float = 0.05) -> TrendResult:
    """Libiseller-Grimvall 2002 partial MK with one covariate.

    Delegates to ``pymannkendall.partial_test`` and normalises the raw
    NamedTuple into a :class:`TrendResult`. Use to control for a known
    confounder (e.g. traffic on a different segment) before quoting a
    trend on ``y``.
    """
    import pymannkendall as pmk
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(covariate, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr = y_arr[mask]
    res = pmk.partial_test(np.column_stack([y_arr, x_arr[mask]]), alpha=alpha)
    return TrendResult(
        method="partial",
        n=len(y_arr),
        slope=float(res.slope),
        slope_ci=None,
        p=float(res.p),
        z=float(res.z),
        tau=float(res.Tau),
    )


# ---------------------------------------------------------------------------
# Paired pre/post comparison — analytical (default) and bootstrap fallback
# ---------------------------------------------------------------------------


def paired_slope_test(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    confidence_level: float = 0.95,
    freq: int | None = None,
    hr_lag: int | None = 1,
) -> PairedSlopeResult:
    """Pre/post Sen-slope comparison with per-arm CIs and an analytical CI on Δ.

    Fully package-based:

    * per-arm slope + CI: ``scipy.stats.theilslopes`` (Kendall-tau CI),
      surfaced as ``slope_ci_pre`` / ``slope_ci_post``.
    * per-arm p-value: ``pymannkendall.hamed_rao_modification_test``, or
      ``pymannkendall.seasonal_test`` when ``freq`` is given.
    * Δslope CI: ``half = √(h_pre² + h_post²)`` combining two independent
      per-arm CIs.
    * rate-change CI: delta-method propagation through
      ``slope_post / slope_pre``.

    Trade-off: ``theilslopes`` treats each arm as independent, so on
    strongly autocorrelated arms this CI will be too tight. The p-value
    still corrects for autocorrelation via HR. When the CI itself needs
    to widen for autocorrelation, use :func:`paired_slope_test_boot`.
    """
    import pymannkendall as pmk

    y_pre = _clean(y_pre)
    y_post = _clean(y_post)
    n_pre, n_post = len(y_pre), len(y_post)

    slope_pre, _, lo_pre, hi_pre = theilslopes(
        y_pre, np.arange(n_pre), alpha=confidence_level,
    )
    slope_post, _, lo_post, hi_post = theilslopes(
        y_post, np.arange(n_post), alpha=confidence_level,
    )

    alpha = 1.0 - confidence_level
    if freq is None:
        pval_pre = float(pmk.hamed_rao_modification_test(y_pre, alpha=alpha, lag=hr_lag).p)
        pval_post = float(pmk.hamed_rao_modification_test(y_post, alpha=alpha, lag=hr_lag).p)
        hp_pre: float | None = None
        hp_post: float | None = None
    else:
        # Route through seasonal_mk so the Van Belle-Hughes homogeneity guard
        # (implemented locally: no package provides it) travels alongside p.
        smk_pre = seasonal_mk(y_pre, period=int(freq), alpha=alpha, hr_lag=hr_lag)
        smk_post = seasonal_mk(y_post, period=int(freq), alpha=alpha, hr_lag=hr_lag)
        pval_pre = smk_pre.p
        pval_post = smk_post.p
        hp_pre = smk_pre.homogeneity_p
        hp_post = smk_post.homogeneity_p

    half_pre = 0.5 * float(hi_pre - lo_pre)
    half_post = 0.5 * float(hi_post - lo_post)
    half_delta = float(np.hypot(half_pre, half_post))
    delta = float(slope_post - slope_pre)
    delta_ci = (delta - half_delta, delta + half_delta)

    # Rate-change CI via delta method:
    #   Var(post/pre) ≈ Var(post)/pre² + post²·Var(pre)/pre⁴.
    if slope_pre != 0:
        z_alpha = float(norm.ppf(0.5 * (1 + confidence_level)))
        sd_pre = half_pre / z_alpha if z_alpha > 0 else float("nan")
        sd_post = half_post / z_alpha if z_alpha > 0 else float("nan")
        var_ratio = (sd_post / slope_pre) ** 2 + (slope_post * sd_pre / slope_pre ** 2) ** 2
        half_ratio = z_alpha * float(np.sqrt(var_ratio))
        pct_change: float | None = float(slope_post / slope_pre - 1.0)
        pct_ci: tuple[float, float] | None = (pct_change - half_ratio, pct_change + half_ratio)
    else:
        pct_change = None
        pct_ci = None

    return PairedSlopeResult(
        slope_pre=float(slope_pre),
        slope_post=float(slope_post),
        slope_ci_pre=(float(lo_pre), float(hi_pre)),
        slope_ci_post=(float(lo_post), float(hi_post)),
        delta_slope=delta,
        delta_ci=delta_ci,
        pct_rate_change=pct_change,
        pct_ci=pct_ci,
        pval_pre=pval_pre,
        pval_post=pval_post,
        n_pre=n_pre,
        n_post=n_post,
        method="analytical",
        homogeneity_p_pre=hp_pre,
        homogeneity_p_post=hp_post,
    )


def paired_slope_test_boot(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    confidence_level: float = 0.95,
    freq: int | None = None,
    n_boot: int = 2000,
    seed: int | None = 0,
    progress: bool = True,
) -> PairedSlopeResult:
    """Bootstrap-CI variant of :func:`paired_slope_test`.

    Local implementation because no Python package provides a paired
    Sen-slope percentile bootstrap. Same interface, same output type;
    only the CIs differ. Sen slope points come from
    ``scipy.stats.theilslopes`` (single-value form). Per-arm p-values
    come from ``pymannkendall.hamed_rao_modification_test`` or
    ``pymannkendall.seasonal_test``.

    Use this over :func:`paired_slope_test` when arm autocorrelation
    should also widen the CI on Δslope. Set ``progress=False`` to
    suppress the tqdm progress bar over the bootstrap loop.
    """
    import pymannkendall as pmk
    from tqdm.auto import tqdm

    y_pre = _clean(y_pre)
    y_post = _clean(y_post)
    n_pre, n_post = len(y_pre), len(y_post)

    slope_pre = float(theilslopes(y_pre, np.arange(n_pre)).slope)
    slope_post = float(theilslopes(y_post, np.arange(n_post)).slope)

    alpha = 1.0 - confidence_level
    if freq is None:
        pval_pre = float(pmk.hamed_rao_modification_test(y_pre, alpha=alpha, lag=1).p)
        pval_post = float(pmk.hamed_rao_modification_test(y_post, alpha=alpha, lag=1).p)
        hp_pre: float | None = None
        hp_post: float | None = None
    else:
        smk_pre = seasonal_mk(y_pre, period=int(freq), alpha=alpha)
        smk_post = seasonal_mk(y_post, period=int(freq), alpha=alpha)
        pval_pre = smk_pre.p
        pval_post = smk_post.p
        hp_pre = smk_pre.homogeneity_p
        hp_post = smk_post.homogeneity_p

    rng = np.random.default_rng(seed)
    slopes_pre = np.empty(n_boot)
    slopes_post = np.empty(n_boot)
    delta = np.empty(n_boot)
    pct = np.empty(n_boot)
    x_pre = np.arange(n_pre)
    x_post = np.arange(n_post)
    iterator = tqdm(
        range(n_boot),
        desc="bootstrap Δslope",
        disable=not progress,
        leave=False,
    )
    for b in iterator:
        idx_pre = np.sort(rng.integers(0, n_pre, n_pre))
        idx_post = np.sort(rng.integers(0, n_post, n_post))
        sp = float(theilslopes(y_pre[idx_pre], x_pre).slope)
        sq = float(theilslopes(y_post[idx_post], x_post).slope)
        slopes_pre[b] = sp
        slopes_post[b] = sq
        delta[b] = sq - sp
        pct[b] = (sq / sp - 1.0) if sp != 0 else np.nan

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    delta_ci = (float(np.percentile(delta, lo_q)), float(np.percentile(delta, hi_q)))
    slope_ci_pre = (
        float(np.percentile(slopes_pre, lo_q)),
        float(np.percentile(slopes_pre, hi_q)),
    )
    slope_ci_post = (
        float(np.percentile(slopes_post, lo_q)),
        float(np.percentile(slopes_post, hi_q)),
    )
    finite_pct = pct[np.isfinite(pct)]
    if finite_pct.size and slope_pre != 0:
        pct_change: float | None = float(slope_post / slope_pre - 1.0)
        pct_ci: tuple[float, float] | None = (
            float(np.percentile(finite_pct, lo_q)),
            float(np.percentile(finite_pct, hi_q)),
        )
    else:
        pct_change = None
        pct_ci = None

    return PairedSlopeResult(
        slope_pre=slope_pre,
        slope_post=slope_post,
        slope_ci_pre=slope_ci_pre,
        slope_ci_post=slope_ci_post,
        delta_slope=slope_post - slope_pre,
        delta_ci=delta_ci,
        pct_rate_change=pct_change,
        pct_ci=pct_ci,
        pval_pre=pval_pre,
        pval_post=pval_post,
        n_pre=n_pre,
        n_post=n_post,
        method=f"bootstrap(n={n_boot})",
        homogeneity_p_pre=hp_pre,
        homogeneity_p_post=hp_post,
    )


def paired_slope_test_3pw(
    dts_pre: Sequence,
    y_pre: Sequence[float],
    dts_post: Sequence,
    y_post: Sequence[float],
    period: int = 7,
    resolution: float = 1.0,
    confidence_level: float = 0.95,
) -> PairedSlopeResult:
    """Heavy-ACF pre/post pipeline using 3PW prewhitening per arm.

    Deseasons each arm point-wise, calls :func:`mk_3pw` on the residual to
    get an autocorrelation-aware Sen slope + Gilbert CI + p-value, then
    combines the two per-arm CIs into a Δ CI via half-widths in quadrature
    (same rule as :func:`paired_slope_test`). Runs the Van Belle-Hughes
    homogeneity guard from :func:`seasonal_mk` on the *raw* arms in
    parallel — the guard verdict is independent of prewhitening.

    Parameters
    ----------
    dts_pre, dts_post : array-like of datetime
        Timestamps for the pre / post arms.
    y_pre, y_post : array-like of float
        Values for the pre / post arms.
    period : int, default 7
        Seasonal cycle length; ``7`` for daily data with a weekly cycle.
    resolution : float, default 1.0
        Instrument resolution passed to ``mannkendall.mk_temp_aggr``.
    confidence_level : float, default 0.95
        Confidence level for the Gilbert CI on each arm.
    """
    y_pre_arr = np.asarray(y_pre, dtype=float)
    y_post_arr = np.asarray(y_post, dtype=float)

    resid_pre = deseason(y_pre_arr, period=period)
    resid_post = deseason(y_post_arr, period=period)

    alpha_mk = 100.0 * confidence_level
    alpha_cl = 100.0 * confidence_level
    pre = mk_3pw(dts_pre, resid_pre, resolution=resolution,
                 alpha_mk=alpha_mk, alpha_cl=alpha_cl)
    post = mk_3pw(dts_post, resid_post, resolution=resolution,
                  alpha_mk=alpha_mk, alpha_cl=alpha_cl)
    if pre.slope_ci is None or post.slope_ci is None:
        raise RuntimeError("mk_3pw returned no CI; cannot form Δ CI in quadrature")

    # ``mannkendall.mk_temp_aggr`` returns slopes in units per year. Convert
    # to units per step (matches :func:`paired_slope_test`) using the median
    # timestamp gap so the two pipelines return comparable numbers.
    dts_all = pd.to_datetime(np.concatenate([np.asarray(dts_pre), np.asarray(dts_post)]))
    step_seconds = float(np.median(np.diff(dts_all.values).astype("timedelta64[s]").astype(float)))
    year_seconds = 365.25 * 86400.0
    per_step_factor = step_seconds / year_seconds
    slope_pre = pre.slope * per_step_factor
    slope_post = post.slope * per_step_factor
    ci_pre = (pre.slope_ci[0] * per_step_factor, pre.slope_ci[1] * per_step_factor)
    ci_post = (post.slope_ci[0] * per_step_factor, post.slope_ci[1] * per_step_factor)

    half_pre = 0.5 * (ci_pre[1] - ci_pre[0])
    half_post = 0.5 * (ci_post[1] - ci_post[0])
    half_delta = float(np.hypot(half_pre, half_post))
    delta = float(slope_post - slope_pre)
    delta_ci = (delta - half_delta, delta + half_delta)

    if slope_pre != 0:
        z_alpha = float(norm.ppf(0.5 * (1 + confidence_level)))
        sd_pre = half_pre / z_alpha if z_alpha > 0 else float("nan")
        sd_post = half_post / z_alpha if z_alpha > 0 else float("nan")
        var_ratio = (sd_post / slope_pre) ** 2 + (slope_post * sd_pre / slope_pre ** 2) ** 2
        half_ratio = z_alpha * float(np.sqrt(var_ratio))
        pct_change: float | None = float(slope_post / slope_pre - 1.0)
        pct_ci: tuple[float, float] | None = (pct_change - half_ratio, pct_change + half_ratio)
    else:
        pct_change = None
        pct_ci = None

    # Homogeneity guard on the raw arms — same convention as paired_slope_test.
    alpha = 1.0 - confidence_level
    smk_pre = seasonal_mk(y_pre_arr, period=int(period), alpha=alpha)
    smk_post = seasonal_mk(y_post_arr, period=int(period), alpha=alpha)

    return PairedSlopeResult(
        slope_pre=float(slope_pre),
        slope_post=float(slope_post),
        slope_ci_pre=(float(ci_pre[0]), float(ci_pre[1])),
        slope_ci_post=(float(ci_post[0]), float(ci_post[1])),
        delta_slope=delta,
        delta_ci=delta_ci,
        pct_rate_change=pct_change,
        pct_ci=pct_ci,
        pval_pre=float(pre.p),
        pval_post=float(post.p),
        n_pre=int(pre.n),
        n_post=int(post.n),
        method="3pw",
        homogeneity_p_pre=smk_pre.homogeneity_p,
        homogeneity_p_post=smk_post.homogeneity_p,
    )


def paired_slope_test_ar1(
    dts_pre: Sequence,
    y_pre: Sequence[float],
    dts_post: Sequence,
    y_post: Sequence[float],
    period: int = 7,
    resolution: float = 1.0,
    confidence_level: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = 0,
    progress: bool = True,
) -> PairedSlopeResult:
    """Auto-picking heavy-ACF paired slope test.

    Picks :func:`paired_slope_test_3pw` when **both arms** span at least two
    calendar years (the regime the ``mannkendall`` package was designed for)
    and :func:`paired_slope_test_boot` otherwise. Both return the same
    :class:`PairedSlopeResult` type so the caller code does not branch on
    ``method``. ``progress`` is forwarded to the bootstrap branch only
    (3PW has no inner loop worth showing).

    Use this when the notebook's ACF diagnostic classifies the arms as
    heavy-ACF but you do not want to hand-pick between prewhitening and a
    percentile bootstrap based on whether each arm spans multiple years.
    """
    dts_pre_ts = pd.to_datetime(np.asarray(dts_pre))
    dts_post_ts = pd.to_datetime(np.asarray(dts_post))
    years_pre = {ts.year for ts in dts_pre_ts.to_pydatetime()}
    years_post = {ts.year for ts in dts_post_ts.to_pydatetime()}
    if len(years_pre) >= 2 and len(years_post) >= 2:
        return paired_slope_test_3pw(
            dts_pre, y_pre, dts_post, y_post,
            period=period, resolution=resolution, confidence_level=confidence_level,
        )
    return paired_slope_test_boot(
        y_pre, y_post,
        confidence_level=confidence_level, freq=period,
        n_boot=n_boot, seed=seed, progress=progress,
    )


def per_day_delta_slopes(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    period: int = 7,
    labels: Sequence[str] | None = None,
    min_obs: int = 3,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Per-phase Δslope = slope_post − slope_pre with per-arm Sen CIs.

    Used as the fallback when the Van Belle-Hughes homogeneity guard fires
    inside :func:`paired_slope_test` or :func:`paired_slope_test_3pw` — the
    pooled Δ hides sign cancellation across phases, so we quote one Δ per
    day-of-week. Uses :func:`sen_slope_ci` (theilslopes, iid) inside each
    phase; treat this as a diagnostic table, not an autocorrelation-aware
    estimate.

    Returns a DataFrame with columns
    ``phase, label, n_pre, n_post, slope_pre, slope_pre_ci_lo, slope_pre_ci_hi,
    slope_post, slope_post_ci_lo, slope_post_ci_hi, delta``.
    """
    y_pre_arr = np.asarray(y_pre, dtype=float)
    y_post_arr = np.asarray(y_post, dtype=float)
    period = int(period)
    phase_pre = np.arange(len(y_pre_arr)) % period
    phase_post = np.arange(len(y_post_arr)) % period

    if labels is None:
        labels = [f"phase_{k}" for k in range(period)]
    if len(labels) != period:
        raise ValueError(f"labels must have length {period}, got {len(labels)}")

    rows: list[dict] = []
    for k, name in enumerate(labels):
        yp = y_pre_arr[phase_pre == k]
        yq = y_post_arr[phase_post == k]
        yp = yp[np.isfinite(yp)]
        yq = yq[np.isfinite(yq)]
        if len(yp) < min_obs or len(yq) < min_obs:
            continue
        sp, (sp_lo, sp_hi) = sen_slope_ci(yp, confidence_level=confidence_level)
        sq, (sq_lo, sq_hi) = sen_slope_ci(yq, confidence_level=confidence_level)
        rows.append({
            "phase": k,
            "label": name,
            "n_pre": len(yp),
            "n_post": len(yq),
            "slope_pre": sp,
            "slope_pre_ci_lo": sp_lo,
            "slope_pre_ci_hi": sp_hi,
            "slope_post": sq,
            "slope_post_ci_lo": sq_lo,
            "slope_post_ci_hi": sq_hi,
            "delta": sq - sp,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch drivers (pandas + package delegation)
# ---------------------------------------------------------------------------


def aggregate_by_period(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: Sequence[str] | None = None,
    period: str = "ME",
    agg: str = "median",
) -> pd.DataFrame:
    """Aggregate ``value_col`` to a coarser calendar period per group.

    Pure pandas — no MK involved. Passes a ``pd.Grouper(freq=period)`` to
    :meth:`pandas.DataFrame.groupby`. Kept here for convenience; if you
    have a preferred aggregation utility elsewhere, use that.
    """
    if value_col not in df.columns or date_col not in df.columns:
        raise KeyError("value_col / date_col not in df")
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    group_cols = list(group_cols or [])
    keys = group_cols + [pd.Grouper(key=date_col, freq=period)] if group_cols else [pd.Grouper(key=date_col, freq=period)]
    out = (
        work.groupby(keys, dropna=False)[value_col]
        .agg([("value", agg), ("n_obs", "count")])
        .reset_index()
        .rename(columns={"value": value_col})
    )
    return out[out["n_obs"] > 0].reset_index(drop=True)


def mk_by_group(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_cols: Sequence[str],
    period: int | None = None,
    method: str = "hamed_rao",
    alpha: float = 0.05,
    min_obs: int = 4,
) -> pd.DataFrame:
    """Batch MK per group via pymannkendall.

    ``method`` picks the pymannkendall variant: one of ``"original"``,
    ``"hamed_rao"``, ``"yue_wang"``, ``"tfpw"``, ``"pw"``. When ``period``
    is given, :func:`seasonal_mk` runs instead and the homogeneity χ²
    p-value is added to the output.
    """
    import pymannkendall as pmk

    fns = {
        "original": pmk.original_test,
        "hamed_rao": pmk.hamed_rao_modification_test,
        "yue_wang": pmk.yue_wang_modification_test,
        "tfpw": pmk.trend_free_pre_whitening_modification_test,
        "pw": pmk.pre_whitening_modification_test,
    }
    if period is None and method not in fns:
        raise ValueError(f"method must be one of {list(fns)} when period is None")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    rows: list[dict] = []
    for keys, g in work.groupby(list(group_cols), dropna=False, sort=False):
        y = g.sort_values(date_col)[value_col].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < min_obs or np.unique(y).size == 1:
            continue
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        if period is None:
            r = fns[method](y, alpha=alpha)
            rows.append({
                **key_dict,
                "n": len(y),
                "slope": float(r.slope),
                "z": float(r.z),
                "p": float(r.p),
                "tau": float(r.Tau),
                "method": method,
                "verdict": r.trend if r.p <= alpha else "no trend",
            })
        else:
            r = seasonal_mk(y, period=period, alpha=alpha)
            rows.append({
                **key_dict,
                "n": r.n,
                "n_seasons": r.n_seasons,
                "slope": r.slope,
                "z": r.z,
                "p": r.p,
                "tau": r.tau,
                "homogeneity_p": r.homogeneity_p,
                "pool_ok": r.pool_ok(alpha=alpha),
                "method": f"seasonal (period={period})",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Van Belle-Hughes χ² decomposition across sites × seasons (local: no pkg)
# ---------------------------------------------------------------------------


def regional_homogeneity(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    site_col: str,
    season_col: str | None = None,
    freq: int | None = None,
    ci: float = 0.95,
    min_obs: int = 4,
    hr_lag: int | None = 1,
) -> RegionalHomogeneityResult:
    """Van Belle-Hughes 1984 χ² decomposition across sites (× seasons).

    Local implementation because ``pymannkendall.regional_test`` only
    reports the pooled trend, not the season / station / interaction
    breakdown. Per-cell MK Z comes from
    ``pymannkendall.hamed_rao_modification_test``.

    Interpretation:

    * ``p_homogeneity > alpha`` → sites (and seasons) agree — pooled trend
      is defensible.
    * ``p_season < alpha`` or ``p_station < alpha`` → structured
      disagreement across seasons or sites — quote per-cell trends.
    * ``p_station_season < alpha`` → interaction dominates; regional
      pooling is not defensible.
    """
    import pymannkendall as pmk

    if site_col not in df.columns:
        raise KeyError(f"site_col {site_col!r} not in df")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    two_way = season_col is not None or freq is not None
    if two_way and season_col is None:
        assert freq is not None
        work = work.sort_values([site_col, date_col])
        work["_season"] = work.groupby(site_col).cumcount() % int(freq)
        season_col = "_season"

    alpha = 1.0 - ci

    def _cell_z(vals: np.ndarray) -> float | None:
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_obs or np.unique(vals).size == 1:
            return None
        r = pmk.hamed_rao_modification_test(vals, alpha=alpha, lag=hr_lag)
        return float(r.z)

    nan = float("nan")

    if two_way:
        assert season_col is not None
        cells: list[dict] = []
        for (site, season), g in work.groupby([site_col, season_col], dropna=False, sort=True):
            z = _cell_z(g.sort_values(date_col)[value_col].to_numpy(dtype=float))
            if z is None or not np.isfinite(z):
                continue
            cells.append({"site": site, "season": season, "Z": z})
        per_group = pd.DataFrame(cells)
        if per_group.empty:
            return RegionalHomogeneityResult(
                0, 0, nan, nan, nan, nan, nan, nan,
                nan, nan, nan, nan, nan, nan,
                False, alpha, per_group,
            )
        z_all = per_group["Z"].to_numpy(dtype=float)
        z_site = per_group.groupby("site")["Z"].mean().to_numpy()
        z_season = per_group.groupby("season")["Z"].mean().to_numpy()
        z_grand = float(z_all.mean())
        M = int(per_group["site"].nunique())
        K = int(per_group["season"].nunique())

        chi_total = float((z_all ** 2).sum())
        chi_trend = float(M * K * z_grand ** 2)
        chi_homog = chi_total - chi_trend
        chi_season = float(M * (z_season ** 2).sum()) - chi_trend
        chi_station = float(K * (z_site ** 2).sum()) - chi_trend
        chi_ss = max(chi_homog - chi_season - chi_station, 0.0)

        p_total = float(chi2.sf(max(chi_total, 0.0), df=M * K))
        p_trend = float(chi2.sf(max(chi_trend, 0.0), df=1))
        p_homog = float(chi2.sf(max(chi_homog, 0.0), df=max(M * K - 1, 1)))
        p_season = float(chi2.sf(max(chi_season, 0.0), df=max(K - 1, 1))) if K > 1 else nan
        p_station = float(chi2.sf(max(chi_station, 0.0), df=max(M - 1, 1))) if M > 1 else nan
        p_ss = float(chi2.sf(chi_ss, df=max((K - 1) * (M - 1), 1))) if (K > 1 and M > 1) else nan

        season_sig = np.isfinite(p_season) and p_season <= alpha
        station_sig = np.isfinite(p_station) and p_station <= alpha
        ss_sig = np.isfinite(p_ss) and p_ss <= alpha
        meaningful = not ((season_sig and station_sig) or ss_sig)

        return RegionalHomogeneityResult(
            n_sites=M, n_seasons=K,
            chi_total=chi_total, chi_trend=chi_trend, chi_homogeneity=chi_homog,
            chi_season=chi_season, chi_station=chi_station, chi_station_season=chi_ss,
            p_total=p_total, p_trend=p_trend, p_homogeneity=p_homog,
            p_season=p_season, p_station=p_station, p_station_season=p_ss,
            overall_trend_meaningful=bool(meaningful), alpha=alpha,
            per_group=per_group,
        )

    # 1-way (site only).
    rows: list[dict] = []
    for site, g in work.groupby(site_col, dropna=False, sort=True):
        z = _cell_z(g.sort_values(date_col)[value_col].to_numpy(dtype=float))
        if z is None or not np.isfinite(z):
            continue
        rows.append({"site": site, "Z": z})
    per_group = pd.DataFrame(rows)
    M = len(per_group)
    if M < 2:
        return RegionalHomogeneityResult(
            M, 1, nan, nan, nan, nan, nan, nan,
            nan, nan, nan, nan, nan, nan,
            False, alpha, per_group,
        )
    z = per_group["Z"].to_numpy(dtype=float)
    chi_homog = max(float((z ** 2).sum() - M * z.mean() ** 2), 0.0)
    p_homog = float(chi2.sf(chi_homog, df=M - 1))
    return RegionalHomogeneityResult(
        n_sites=M, n_seasons=1,
        chi_total=nan, chi_trend=nan, chi_homogeneity=chi_homog,
        chi_season=nan, chi_station=nan, chi_station_season=nan,
        p_total=nan, p_trend=nan, p_homogeneity=p_homog,
        p_season=nan, p_station=nan, p_station_season=nan,
        overall_trend_meaningful=bool(p_homog > alpha),
        alpha=alpha, per_group=per_group,
    )


__all__ = [
    "TrendResult",
    "SeasonalTrendResult",
    "VbhDecomposition",
    "PairedSlopeResult",
    "RegionalHomogeneityResult",
    # single-series wrappers
    "mk_original",
    "mk_hamed_rao",
    "mk_yue_wang",
    "mk_tfpw",
    "mk_pw",
    "mk_3pw",
    "sen_slope_ci",
    # diagnostics
    "deseason",
    "lag1_acf",
    # seasonal / partial / correlated
    "seasonal_mk",
    "vbh_chi2_decomposition",
    "correlated_seasonal_mk",
    "partial_mk",
    # paired
    "paired_slope_test",
    "paired_slope_test_boot",
    "paired_slope_test_3pw",
    "paired_slope_test_ar1",
    "per_day_delta_slopes",
    # batch
    "aggregate_by_period",
    "mk_by_group",
    # regional
    "regional_homogeneity",
]
