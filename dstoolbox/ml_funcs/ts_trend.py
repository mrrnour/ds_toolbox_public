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
3. **Paired pre/post Sen-slope moving-block bootstrap with pluggable
   MK p-value** (in :func:`mk_delta_mbb`) — the single unified
   paired test. Slope always comes from ``theilslopes`` on raw ``y``;
   CI always from a moving-block percentile bootstrap; the MK variant
   feeding the p-value is chosen via ``mk_method``.

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
from typing import Literal, Sequence

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
class MKAdaptiveCoreArmResult:
    """Per-arm adaptive MK — headline building block.

    Product of :func:`mk_adaptive_core_arm`: one seasonality gate + §4.2 AR-ladder
    call followed by the picked MK variant on one arm. Slope point always
    comes from ``theilslopes(y_arm).slope`` on raw ``y``; the closed-form
    MK p-value comes from the picked variant.

    ``reason`` is a short human-readable explanation of *why* the ladder
    picked this variant (ρ₁ vs cutoff, span vs 2-yr gate, forced override).
    ``dts_span_years`` is the arm's calendar span, used by the 3PW gate;
    ``None`` when ``dts`` was not supplied.
    """

    n: int
    slope: float
    pvalue: float
    mk_method: str
    deseasoned: bool
    rho1: float
    period: int | None
    reason: str = ""
    dts_span_years: float | None = None

    def summary(self) -> str:
        span_txt = "n/a" if self.dts_span_years is None else f"{self.dts_span_years:.2f}yr"
        # Header keeps only the fields NOT re-explained by the season→/ladder→
        # lines below. mk_method and substrate are redundant — they are the
        # outputs the two decision lines narrate.
        head = (
            f"slope={self.slope:+.4f}/step  p={self.pvalue:.4f}  "
            f"(n={self.n}, ρ1={self.rho1:+.3f}, span={span_txt})"
        )
        # Explicit narration of the two §4 decisions:
        #   1. seasonality gate  → deseasoned vs raw substrate
        #   2. §4.2 AR ladder    → MK variant picked (Plain / Hamed–Rao / 3PW)
        if self.deseasoned:
            head += f"\n         season→ period={self.period} given → deseasoned before ρ₁ + MK"
        else:
            head += "\n         season→ no period given → raw substrate (no deseasoning)"
        if self.reason:
            head += f"\n         ladder→ {self.reason}"
        return head


@dataclass(frozen=True)
class MKAdaptiveCoreResult:
    """Headline result — adaptive MK per arm plus derived Δ point.

    Product of :func:`mk_adaptive_core`. All fields are closed-form
    (no bootstrap). ``delta_slope`` is the point difference
    ``slope_post − slope_pre``; there is no Δ CI or Δ p here — use
    :func:`mk_adaptive_mbb` (or :func:`mk_delta_mbb`) for those.
    """

    pre: MKAdaptiveCoreArmResult
    post: MKAdaptiveCoreArmResult
    slope_pre: float
    slope_post: float
    pval_pre: float
    pval_post: float
    delta_slope: float
    pct_rate_change: float | None
    n_pre: int
    n_post: int

    @property
    def pct_unstable(self) -> bool:
        """Ratio anchor is unreliable — ``|slope_pre|`` is small vs ``|slope_post|``.

        No CI is available at this stage, so we use a simple magnitude check:
        if the pre-arm slope is < 10 % of the post-arm slope in absolute
        value (or is non-finite), the ratio blows up and the % rate change
        should not be quoted. Use ``delta_slope`` instead, or run
        :func:`mk_adaptive_mbb` for a CI-backed ratio.
        """
        if self.pct_rate_change is None:
            return False
        if not (np.isfinite(self.slope_pre) and np.isfinite(self.slope_post)):
            return True
        return abs(self.slope_pre) < 0.10 * abs(self.slope_post)

    def summary(self) -> str:
        # % rate change line — guard against flat-pre-slope ratio blow-up
        if self.pct_rate_change is None:
            pct_line = "  rate change   = n/a  (slope_pre = 0)"
        elif self.pct_unstable:
            pct_line = (
                f"  rate change   = unstable "
                f"(slope_pre={self.slope_pre:+.4f} ≪ slope_post={self.slope_post:+.4f} in magnitude; "
                f"ratio anchor unreliable — use Δslope instead)"
            )
        else:
            pct_line = f"  rate change   = {self.pct_rate_change * 100:+.0f}%  (= slope_post / slope_pre − 1)"

        # Per-arm significance verdict at α = 0.05 (matches MKAdaptiveCoreArmResult default)
        def _verdict(p: float) -> str:
            if not np.isfinite(p):
                return "n/a"
            return "reject H₀ (trend detected)" if p < 0.05 else "fail to reject H₀ (no evidence of trend)"

        return (
            f"[HEADLINE — adaptive MK per arm]\n"
            f"  pre  : {self.pre.summary()}\n"
            f"  post : {self.post.summary()}\n"
            f"  Δslope        = {self.delta_slope:+.4f}/step  "
            f"(point estimate only — no CI or p on Δ; run mk_adaptive_mbb for those)\n"
            f"{pct_line}\n"
            f"  [interpretation]\n"
            f"    slope        — Sen (Theil–Sen) slope, units of y per time step (raw y, not deseasoned).\n"
            f"    p_pre/p_post — H₀: no monotonic trend within that arm (Sen slope = 0).\n"
            f"                   Small p ⇒ evidence of a trend; sign is given by slope.\n"
            f"    ρ₁           — lag-1 autocorrelation of the *deseasoned* residual for that arm\n"
            f"                   (measures short-memory serial dependence after removing the weekly cycle).\n"
            f"                   |ρ₁| ≤ acf_cutoff → Plain MK;  |ρ₁| > cutoff → Hamed–Rao (or 3PW if ≥ 2 yrs).\n"
            f"    n            — arm length (number of daily observations after cleaning).\n"
            f"    span         — arm calendar span in years (used by the 2-yr gate for 3PW eligibility).\n"
            f"    season→      — seasonality-gate decision: 'deseasoned' when a period was given, 'raw' otherwise.\n"
            f"                   The Sen slope itself is always fit on raw y.\n"
            f"    ladder→      — §4.2 AR-ladder decision: which MK variant was picked and why.\n"
            f"  [verdicts at α=0.05]\n"
            f"    pre  → p={self.pval_pre:.4f}: {_verdict(self.pval_pre)}\n"
            f"    post → p={self.pval_post:.4f}: {_verdict(self.pval_post)}\n"
            f"    Δslope is a *point comparison* of the two per-arm Sen slopes; whether\n"
            f"    it is significantly different from 0 requires Track 2 (paired MBB)."
        )


@dataclass(frozen=True)
class VbhBranchResult:
    """Optional VBH homogeneity add-on — per-arm χ² diagnostic.

    Product of :func:`mk_vbh`. Wraps two per-arm
    :class:`SeasonalTrendResult` calls and exposes the pair of
    homogeneity p-values used to decide whether the pooled headline
    hides Simpson-style per-phase cancellation.
    """

    pre: SeasonalTrendResult
    post: SeasonalTrendResult
    homogeneity_p_pre: float
    homogeneity_p_post: float
    alpha: float

    @property
    def is_inhomogeneous(self) -> bool:
        return bool(
            (np.isfinite(self.homogeneity_p_pre) and self.homogeneity_p_pre < self.alpha)
            or (np.isfinite(self.homogeneity_p_post) and self.homogeneity_p_post < self.alpha)
        )

    def summary(self) -> str:
        verdict = (
            "INHOMOGENEOUS — report per-phase Δ_g"
            if self.is_inhomogeneous
            else "homogeneous — pooled Δslope OK"
        )
        return (
            f"[VBH χ² homogeneity]\n"
            f"  pre  homogeneity_p = {self.homogeneity_p_pre:.4f}\n"
            f"  post homogeneity_p = {self.homogeneity_p_post:.4f}\n"
            f"  → {verdict}"
        )


@dataclass(frozen=True)
class MbbDeltaResult:
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
    # ── Δ interpretation (auto-computed by :func:`mk_delta_mbb`) ─────────
    # For ``ci_method='mbb'``: ``delta_pvalue`` is the two-sided tail-area
    # of the paired-MBB ``{Δ_b}`` array at Δ = 0 (Efron 1979), read off
    # the same array as ``delta_ci``; ``delta_power`` uses the empirical
    # SD of ``{Δ_b}`` under a normal approximation.
    # For ``ci_method='gilbert'``: both are computed via Wald inversion
    # of ``delta_ci`` under a normal approximation (no bootstrap array).
    alpha: float | None = None
    delta_significant: bool | None = None
    delta_pvalue: float | None = None
    delta_pvalue_text: str | None = None
    delta_power: float | None = None
    delta_power_text: str | None = None
    direction: str | None = None
    meaning: str | None = None

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
                f"rate change= unstable (slope_pre={self.slope_pre:+.4f} too small in magnitude "
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


def sen_fit_line(y: Sequence[float], slope_per_step: float) -> np.ndarray:
    """Return a centered Sen-style fit line for plotting.

    Uses a robust intercept anchored at the median index/value pair so the
    fitted line overlays noisy time-series data in a visually stable way.
    """
    arr = np.asarray(y, dtype=float)
    step = np.arange(len(arr), dtype=float)
    med_y = float(np.nanmedian(arr))
    med_x = float(np.nanmedian(step))
    return med_y + float(slope_per_step) * (step - med_x)


def _bootstrap_tail_area(delta_boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value at Δ = 0 (Efron 1979).

    ``p = 2 · min(mean(Δ_b ≤ 0), mean(Δ_b ≥ 0))``, clipped to ``[0, 1]``.
    Returns ``nan`` when the array holds no finite entries.
    """
    finite = delta_boot[np.isfinite(delta_boot)]
    if finite.size == 0:
        return float("nan")
    p_left = float(np.mean(finite <= 0.0))
    p_right = float(np.mean(finite >= 0.0))
    return float(min(1.0, 2.0 * min(p_left, p_right)))


def _wald_tail_area(delta_slope: float, half_ci: float, confidence_level: float) -> float:
    """Two-sided Wald-inversion p-value from a Δ point + CI half-width."""
    z_alpha = float(norm.ppf(0.5 * (1.0 + confidence_level)))
    if half_ci <= 0 or z_alpha <= 0:
        return float("nan")
    sd_delta = half_ci / z_alpha
    z_delta = float(delta_slope / sd_delta)
    return float(2.0 * (1.0 - norm.cdf(abs(z_delta))))


def _compute_delta_interpretation(
    delta_slope: float,
    delta_ci: tuple[float, float],
    confidence_level: float,
    delta_boot: np.ndarray | None = None,
) -> dict:
    """Δ p-value + power + display strings for :class:`MbbDeltaResult`.

    Two dispatch modes, aligned with the chart 6 workflow:

    * **MBB path** — pass ``delta_boot`` (the paired-MBB ``{Δ_b}`` array).
      ``delta_pvalue`` is the two-sided **tail-area** of that array at
      ``Δ = 0`` (Efron 1979, ``p = 2 · min(mean(Δ_b ≤ 0), mean(Δ_b ≥ 0))``);
      no Wald inversion, no independent normal assumption. ``delta_power``
      is computed from the empirical standard deviation of ``Δ_b`` under
      a normal approximation ``Δ ~ N(delta_slope, sd(Δ_b))``.
    * **Gilbert / no-bootstrap path** — call with ``delta_boot=None``.
      ``delta_pvalue`` and ``delta_power`` are both derived from the
      half-width of ``delta_ci`` via Wald inversion under a normal
      approximation ``sd = half-width / z_{1-α/2}``. This is only used
      when there is no bootstrap array to read from (e.g. the closed-form
      Gilbert CI branch inside :func:`mk_delta_mbb`).

    Args:
        delta_slope: The point Δ (``slope_post − slope_pre``).
        delta_ci: The Δ confidence interval (any construction).
        confidence_level: Nominal CL used to build ``delta_ci``.
        delta_boot: Optional paired-MBB ``{Δ_b}`` array. When given,
            drives the tail-area p-value and the empirical power calc.
    """
    p_floor = 1e-16
    alpha = 1.0 - confidence_level
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    half_delta = 0.5 * (delta_ci[1] - delta_ci[0])
    delta_sig = not (delta_ci[0] <= 0 <= delta_ci[1])

    if delta_boot is not None:
        boot_arr = np.asarray(delta_boot, dtype=float)
        delta_pval = _bootstrap_tail_area(boot_arr)
        finite = boot_arr[np.isfinite(boot_arr)]
        sd_delta = float(finite.std(ddof=1)) if finite.size >= 2 else 0.0
    else:
        delta_pval = _wald_tail_area(delta_slope, half_delta, confidence_level)
        sd_delta = (half_delta / float(norm.ppf(0.5 * (1.0 + confidence_level)))
                    if half_delta > 0 else 0.0)

    if sd_delta > 0 and np.isfinite(sd_delta):
        effect = abs(float(delta_slope) / sd_delta)
        delta_power = float(norm.cdf(-z_crit - effect) + (1.0 - norm.cdf(z_crit - effect)))
    else:
        delta_power = float("nan")

    if not np.isfinite(delta_pval):
        p_txt = "n/a"
    elif delta_pval < p_floor:
        p_txt = f"<{p_floor:.0e}"
    else:
        p_txt = f"{delta_pval:.3g}"

    power_txt = "n/a" if not np.isfinite(delta_power) else f"{delta_power:.1%}"

    direction = "increase" if delta_slope > 0 else "decrease" if delta_slope < 0 else "no change"
    if delta_sig:
        meaning = f"Estimated post-vs-pre trend changed significantly ({direction})."
    else:
        meaning = "Estimated post-vs-pre trend change is not statistically distinguishable from zero."

    return dict(
        alpha=alpha,
        delta_significant=bool(delta_sig),
        delta_pvalue=delta_pval,
        delta_pvalue_text=p_txt,
        delta_power=delta_power,
        delta_power_text=power_txt,
        direction=direction,
        meaning=meaning,
    )


def pre_post_sen_figure(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    date_col: str,
    value_col: str,
    y_pre: Sequence[float],
    y_post: Sequence[float],
    result: MbbDeltaResult,
    intervention_date: str | pd.Timestamp,
    experiment: str,
    confidence_level: float,
):
    """Build a pre/post scatter figure with per-arm Sen fit overlays.

    This keeps notebook plotting logic thin and consistent across experiments.
    Reads interpretation fields (``delta_significant``, ``direction``,
    ``delta_pvalue_text``) directly off ``result``, which
    :func:`mk_delta_mbb` populates automatically.
    """
    import plotly.graph_objects as go

    sig_flag = "significant" if result.delta_significant else "not significant"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=y_pre, mode="markers", name="pre points",
        marker=dict(color="#1f77b4", size=5, opacity=0.6),
        hovertemplate="date=%{x}<br>pre=%{y:.4g}<extra></extra>",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=y_post, mode="markers", name="post points",
        marker=dict(color="#d62728", size=5, opacity=0.6),
        hovertemplate="date=%{x}<br>post=%{y:.4g}<extra></extra>",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=sen_fit_line(y_pre, result.slope_pre),
        mode="lines", name=f"Sen pre ({result.slope_pre:+.4g}/day)",
        line=dict(color="#1f77b4", width=3),
        hovertemplate="date=%{x}<br>Sen pre=%{y:.4g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=sen_fit_line(y_post, result.slope_post),
        mode="lines", name=f"Sen post ({result.slope_post:+.4g}/day)",
        line=dict(color="#d62728", width=3),
        hovertemplate="date=%{x}<br>Sen post=%{y:.4g}<extra></extra>",
    ))
    _sig_sym = "✓" if result.delta_significant else "✗"
    _dir_arrow = "↑" if result.direction == "increase" else "↓" if result.direction == "decrease" else "→"
    _footer = (
        f"Mann-Kendall ({result.method})  ·  "
        f"95% CI [{result.delta_ci[0]:+.4g}, {result.delta_ci[1]:+.4g}]  ·  "
        f"p-value = {result.delta_pvalue_text}  {_sig_sym} {sig_flag}"
    )
    fig.add_vline(x=pd.Timestamp(intervention_date).isoformat(), line_dash="dash", line_color="black")
    fig.add_annotation(
        text=_footer,
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=11, color="#555555"),
        xanchor="center",
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{experiment}</b>  —  Δ slope = {result.delta_slope:+.4g} {_dir_arrow}",
            font=dict(size=16),
        ),
        xaxis_title="date",
        yaxis_title=value_col,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=80),
    )
    return fig


def mk_result_line(
    result: "TrendResult | SeasonalTrendResult",
    arm_label: str,
    alpha: float = 0.05,
) -> str:
    """Return a single ``arm_label → slope · direction · z · p · sig`` line.

    Works for both :class:`TrendResult` (non-seasonal / Hamed-Rao) and
    :class:`SeasonalTrendResult` (Hirsch seasonal). When a seasonal result
    is passed, the homogeneity-p and pool verdict are appended.

    Args:
        result: A MK result dataclass produced by this module.
        arm_label: Short label to prefix (e.g. ``"pre"``, ``"post"``).
        alpha: Significance threshold used for the ✓/✗ marker.
    """
    slope = float(result.slope)
    direction = "↑" if slope > 0 else "↓" if slope < 0 else "→"
    sig = "✓ significant" if result.p < alpha else "✗ not significant"
    core = (
        f"{arm_label:>5} → slope={slope:+.6g}  {direction}  "
        f"z={result.z:+.3f}  p={result.p:.4g}  {sig}"
    )
    tau = getattr(result, "tau", None)
    if tau is not None:
        core = core.replace(f"z={result.z:+.3f}", f"z={result.z:+.3f}  tau={tau:+.3f}")
    if isinstance(result, SeasonalTrendResult):
        pool = "pool OK" if result.pool_ok(alpha) else "⚠ POOL UNSAFE"
        core += f"  homog_p={result.homogeneity_p:.4g}  ({pool})"
    return core


def mk_figure(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    date_col: str,
    value_col: str,
    y_pre: Sequence[float],
    y_post: Sequence[float],
    mk_pre: "TrendResult | SeasonalTrendResult",
    mk_post: "TrendResult | SeasonalTrendResult",
    intervention_date: str | pd.Timestamp,
    experiment: str,
    alpha: float = 0.05,
    method_label: str | None = None,
):
    """Pre/post scatter with per-arm MK Sen fit overlays.

    Method-agnostic: works with any MK result exposing ``.slope`` and ``.p``
    — :class:`TrendResult` (non-seasonal / Hamed-Rao / 3PW) or
    :class:`SeasonalTrendResult` (Hirsch seasonal). No bootstrap; the Sen
    overlay is a deterministic median-anchored projection of ``.slope``.
    Footer annotation shows p-value + significance flag for each arm.

    Args:
        df_pre / df_post: DataFrames with the ``date_col`` column.
        date_col / value_col: Column names.
        y_pre / y_post: Numeric series aligned with ``df_pre / df_post``.
        mk_pre / mk_post: MK result dataclasses per arm. Must both be the
            same variant so the label stays consistent.
        intervention_date: Intervention timestamp; drawn as a dashed vertical line.
        experiment: Title label.
        alpha: Significance threshold used for the ✓/✗ marker.
        method_label: Override for the method name shown in the title /
            legend / annotation (e.g. ``"Hamed-Rao MK"``). When ``None``,
            auto-derived: ``"Seasonal MK"`` for :class:`SeasonalTrendResult`
            inputs and ``"MK"`` otherwise (or taken from
            ``mk_pre.method`` when available).
    """
    import plotly.graph_objects as go

    if method_label is None:
        if isinstance(mk_pre, SeasonalTrendResult):
            method_label = "Seasonal MK"
        else:
            method_label = getattr(mk_pre, "method", None) or "MK"

    sig_pre = "✓" if mk_pre.p < alpha else "✗"
    sig_post = "✓" if mk_post.p < alpha else "✗"
    dir_pre = "↑" if mk_pre.slope > 0 else "↓" if mk_pre.slope < 0 else "→"
    dir_post = "↑" if mk_post.slope > 0 else "↓" if mk_post.slope < 0 else "→"
    delta_slope = float(mk_post.slope) - float(mk_pre.slope)
    dir_delta = "↑" if delta_slope > 0 else "↓" if delta_slope < 0 else "→"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=y_pre, mode="markers", name="pre",
        marker=dict(color="#1f77b4", size=5, opacity=0.6),
        hovertemplate="date=%{x}<br>pre=%{y:.4g}<extra></extra>",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=y_post, mode="markers", name="post",
        marker=dict(color="#d62728", size=5, opacity=0.6),
        hovertemplate="date=%{x}<br>post=%{y:.4g}<extra></extra>",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=sen_fit_line(y_pre, mk_pre.slope),
        mode="lines", name=f"{method_label} pre",
        line=dict(color="#1f77b4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=sen_fit_line(y_post, mk_post.slope),
        mode="lines", name=f"{method_label} post",
        line=dict(color="#d62728", width=3),
    ))
    fig.add_vline(x=pd.Timestamp(intervention_date).isoformat(), line_dash="dash", line_color="black")
    fig.add_annotation(
        text=(
            f"pre slope={mk_pre.slope:+.4g} {dir_pre} (p={mk_pre.p:.3g} {sig_pre})  ·  "
            f"post slope={mk_post.slope:+.4g} {dir_post} (p={mk_post.p:.3g} {sig_post})"
        ),
        xref="paper", yref="paper", x=0.5, y=-0.18,
        showarrow=False, font=dict(size=11, color="#555555"), xanchor="center",
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{experiment}</b>  —  {method_label}  |  "
                f"Δ slope = {delta_slope:+.4g} {dir_delta}"
            ),
            font=dict(size=15),
        ),
        xaxis_title="date", yaxis_title=value_col, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=80),
    )
    return fig


def seasonal_mk_figure(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    date_col: str,
    value_col: str,
    y_pre: Sequence[float],
    y_post: Sequence[float],
    smk_pre: SeasonalTrendResult,
    smk_post: SeasonalTrendResult,
    intervention_date: str | pd.Timestamp,
    experiment: str,
    alpha: float = 0.05,
):
    """Backward-compat wrapper around :func:`mk_figure`.

    Kept so existing notebooks that call ``seasonal_mk_figure(...)``
    with ``smk_pre=`` / ``smk_post=`` keep working. New code should call
    :func:`mk_figure` directly — it accepts any MK result type.
    """
    return mk_figure(
        df_pre=df_pre,
        df_post=df_post,
        date_col=date_col,
        value_col=value_col,
        y_pre=y_pre,
        y_post=y_post,
        mk_pre=smk_pre,
        mk_post=smk_post,
        intervention_date=intervention_date,
        experiment=experiment,
        alpha=alpha,
        method_label="Seasonal MK",
    )


def _significance_stars(p: float) -> str:
    """Return APA-style significance markers for a p-value.

    ``***`` for p < 0.001, ``**`` for p < 0.01, ``*`` for p < 0.05,
    ``ns`` otherwise. Returns ``"n/a"`` when ``p`` is not finite.
    """
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mk_adaptive_sen_figure(
    df_pre: pd.DataFrame,
    df_post: pd.DataFrame,
    date_col: str,
    value_col: str,
    y_pre: Sequence[float],
    y_post: Sequence[float],
    result: "MKAdaptiveCoreResult | MbbDeltaResult",
    intervention_date: str | pd.Timestamp,
    experiment: str,
    *,
    period: int | None = None,
    show_deseasoned: bool = True,
    ci_level: float = 0.95,
):
    """Pre/post scatter + per-arm Sen fit — Track 1 (core) or Track 2 (MBB).

    ``result`` accepts either flavour and the figure adapts accordingly:

    * :class:`MKAdaptiveCoreResult` (Track 1, :func:`mk_adaptive_core`) —
      point Sen fits per arm, subtitle quotes ``Δ-slope`` as a point
      estimate, legend entries carry the per-arm MK variant picked by
      §4's AR ladder (``[hamed_rao]``, ``[3pw]``, …).
    * :class:`MbbDeltaResult` (Track 2, :func:`mk_adaptive_mbb`) — same
      point Sen fits (identical by construction: both flavours take Sen
      on raw ``y``) plus per-arm slope-CI ribbons from ``slope_ci_pre`` /
      ``slope_ci_post`` and a subtitle that quotes ``delta_ci`` /
      ``delta_pvalue``. MBB uses one MK variant for both arms so the
      legend labels are identical (parsed from ``mbb.method``).

    Legend / subtitle wording leads with the word "slope" (e.g. ``pre Sen
    slope: +0.079``, ``Δ-slope = +0.174``) so the numbers cannot be mistaken
    for a per-day *level*. No '/day' suffix — slope already implies
    per-step.

    APA-style significance stars (``***`` / ``**`` / ``*`` / ``ns``)
    decorate the per-arm p-values. When ``show_deseasoned`` is true and
    ``period`` is given, the deseasoned residuals are overlaid as ``x``
    markers on the same axis (shifted back to the raw scale by adding the
    arm mean) so the reader can eyeball the trend after weekly cycle
    removal.

    Args:
        df_pre / df_post: DataFrames holding ``date_col``.
        date_col / value_col: Column names.
        y_pre / y_post: Numeric series aligned with ``df_pre / df_post``.
        result: Either :class:`MKAdaptiveCoreResult` (§4 view) or
            :class:`MbbDeltaResult` (§8 / Track 2 view). The figure picks
            its subtitle and whether to draw CI ribbons based on which
            type is passed.
        intervention_date: Intervention timestamp; drawn as a dashed vertical line.
        experiment: Title label.
        period: Seasonal cycle length used for :func:`deseason`. Required
            when ``show_deseasoned=True``; ignored otherwise.
        show_deseasoned: Overlay per-arm deseasoned point clouds. Set
            ``False`` to hide the ``x`` markers when the arms are non
            seasonal or when the plot needs to stay uncluttered.
        ci_level: Confidence level used in the subtitle text (label only;
            ignored when ``result`` is a :class:`MKAdaptiveCoreResult`).

    Returns:
        A ``plotly.graph_objects.Figure``.

    Notes:
        Uses ``add_shape`` + ``add_annotation`` for the intervention line rather
        than ``add_vline(annotation_text=...)``. The latter trips a plotly
        bug that internally does ``sum([x0, x1])`` on the timestamp inputs,
        which fails when ``intervention_date`` is a ``pandas.Timestamp``.
    """
    import plotly.graph_objects as go
    import re as _re

    # Route on the result flavour. Track 2 (MBB) view carries CIs and a Δ
    # p-value; Track 1 (core) is point-only. Both flavours have identical
    # slope_pre / slope_post by design (Sen on raw y).
    if isinstance(result, MbbDeltaResult):
        _mbb: MbbDeltaResult | None = result
        slope_pre_val = result.slope_pre
        slope_post_val = result.slope_post
        pval_pre_val = result.pval_pre
        pval_post_val = result.pval_post
        delta_val = result.delta_slope
        # MBB uses one variant for both arms; parse it out of
        # e.g. "mbb(mk=hamed_rao, bl=7, n=2000)".
        _m = _re.search(r"mk=([^,\)]+)", result.method)
        mk_label_pre = mk_label_post = _m.group(1) if _m else result.method
    else:
        _mbb = None
        slope_pre_val = result.slope_pre
        slope_post_val = result.slope_post
        pval_pre_val = result.pval_pre
        pval_post_val = result.pval_post
        delta_val = result.delta_slope
        mk_label_pre = result.pre.mk_method
        mk_label_post = result.post.mk_method

    y_pre_arr = np.asarray(y_pre, dtype=float)
    y_post_arr = np.asarray(y_post, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=y_pre_arr, mode="markers", name="pre (raw)",
        marker=dict(color="#1f77b4", size=5, opacity=0.35),
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=y_post_arr, mode="markers", name="post (raw)",
        marker=dict(color="#d62728", size=5, opacity=0.35),
    ))

    if show_deseasoned:
        if period is None:
            raise ValueError("period is required when show_deseasoned=True")
        # `deseason` centers the residual at 0; re-add the arm mean so the
        # markers sit alongside the raw scatter rather than at y ≈ 0.
        y_pre_ds = deseason(y_pre_arr, period=period) + float(np.nanmean(y_pre_arr))
        y_post_ds = deseason(y_post_arr, period=period) + float(np.nanmean(y_post_arr))
        fig.add_trace(go.Scatter(
            x=df_pre[date_col], y=y_pre_ds, mode="markers", name="pre (deseasoned)",
            marker=dict(color="#1f77b4", size=6, symbol="x", opacity=0.9),
        ))
        fig.add_trace(go.Scatter(
            x=df_post[date_col], y=y_post_ds, mode="markers", name="post (deseasoned)",
            marker=dict(color="#d62728", size=6, symbol="x", opacity=0.9),
        ))

    # Optional slope-CI ribbons (drawn under the Sen lines so the point
    # fit stays visually on top). Uses fill='tonexty' between the low- and
    # high-slope median-anchored lines from mbb.slope_ci_pre / _post.
    if _mbb is not None:
        for x_dates, y_arm, ci, colour in (
            (df_pre[date_col], y_pre_arr, _mbb.slope_ci_pre, "31, 119, 180"),   # pre
            (df_post[date_col], y_post_arr, _mbb.slope_ci_post, "214, 39, 40"),  # post
        ):
            lo_line = sen_fit_line(y_arm, float(ci[0]))
            hi_line = sen_fit_line(y_arm, float(ci[1]))
            fig.add_trace(go.Scatter(
                x=x_dates, y=lo_line, mode="lines",
                line=dict(color=f"rgba({colour},0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=x_dates, y=hi_line, mode="lines", fill="tonexty",
                line=dict(color=f"rgba({colour},0)"),
                fillcolor=f"rgba({colour},0.18)",
                name=f"slope CI [{ci[0]:+.3g}, {ci[1]:+.3g}]",
                hoverinfo="skip",
            ))

    star_pre = _significance_stars(pval_pre_val)
    star_post = _significance_stars(pval_post_val)
    # Legend/subtitle labels lead with the word "slope" (not just "Sen:").
    # We do NOT append a '/day' unit — slope already implies per-step, and
    # the redundant '/day' invites confusion with a per-day *level*.
    fig.add_trace(go.Scatter(
        x=df_pre[date_col], y=sen_fit_line(y_pre_arr, slope_pre_val), mode="lines",
        name=(
            f"pre  Sen slope: {slope_pre_val:+.4g}  "
            f"p={pval_pre_val:.3g} {star_pre}  [{mk_label_pre}]"
        ),
        line=dict(color="#1f77b4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=df_post[date_col], y=sen_fit_line(y_post_arr, slope_post_val), mode="lines",
        name=(
            f"post Sen slope: {slope_post_val:+.4g}  "
            f"p={pval_post_val:.3g} {star_post}  [{mk_label_post}]"
        ),
        line=dict(color="#d62728", width=3),
    ))

    # Intervention line — see docstring: `add_vline(annotation_text=...)` breaks on
    # pandas Timestamp inputs, so use add_shape + add_annotation.
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=intervention_date, x1=intervention_date, y0=0, y1=1,
        line=dict(color="black", dash="dash"),
    )
    fig.add_annotation(
        x=intervention_date, y=1.0, xref="x", yref="paper",
        text="intervention", showarrow=False, yanchor="bottom",
    )

    if _mbb is not None:
        _delta_sig = not (_mbb.delta_ci[0] <= 0 <= _mbb.delta_ci[1])
        _sig_flag = "significant" if _delta_sig else "not significant"
        pv_txt = _mbb.delta_pvalue_text or (
            f"{_mbb.delta_pvalue:.3g}" if _mbb.delta_pvalue is not None else "n/a"
        )
        subtitle = (
            f"Δ-slope = {delta_val:+.4g}  "
            f"CI [{_mbb.delta_ci[0]:+.4g}, {_mbb.delta_ci[1]:+.4g}]  "
            f"p = {pv_txt}  ({_sig_flag} at {ci_level:.0%})"
        )
    else:
        subtitle = f"Δ-slope = {delta_val:+.4g} — point estimate"

    fig.update_layout(
        title=f"{experiment} — Sen-slope trend<br><sub>{subtitle}</sub>",
        yaxis_title=value_col,
        legend=dict(yanchor="top", y=-0.15, xanchor="left", x=0, orientation="h"),
    )
    return fig


@dataclass(frozen=True)
class MKPowerCurve:
    """Return type of :func:`mk_power_curve`.

    Holds the closed-form MK p-value swept over sample size ``n`` at a
    fixed Kendall ``tau_obs``, plus the observed point ``(n_obs, p_obs)``
    and the smallest ``n`` on the grid whose two-sided MK p-value falls at
    or below ``alpha`` (``None`` when no grid point crosses ``alpha``).
    """

    tau_obs: float
    n_obs: int
    p_obs: float
    n_grid: np.ndarray
    p_grid: np.ndarray
    alpha: float
    n_needed: int | None


def mk_asymptotic_pvalue(tau: float, n: int) -> float:
    """Two-sided MK asymptotic p-value from Kendall ``tau`` and sample size ``n``.

    Uses the standard normal approximation to Mann-Kendall's ``S`` statistic:

    .. math::

        S \\approx \\tau \\cdot n(n-1)/2, \\quad
        \\mathrm{Var}(S) \\approx n(n-1)(2n+5)/18, \\quad
        Z = S / \\sqrt{\\mathrm{Var}(S)}.

    Returns ``nan`` when ``n < 4`` or ``tau`` is non-finite (MK is
    undefined under those conditions).
    """
    if n < 4 or not np.isfinite(tau):
        return float("nan")
    S = float(tau) * n * (n - 1) / 2.0
    var_S = n * (n - 1) * (2 * n + 5) / 18.0
    z = S / np.sqrt(var_S)
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def mk_power_curve(
    y: Sequence[float],
    *,
    alpha: float = 0.05,
    n_range: tuple[int, int] = (5, 400),
) -> MKPowerCurve:
    """MK power curve — fix ``tau`` at the observed value, sweep ``n``.

    Computes Kendall's ``tau`` on the finite entries of ``y`` (NaN-safe),
    then evaluates :func:`mk_asymptotic_pvalue` across the integer grid
    ``[n_range[0], n_range[1]]``. Answers the reader's question
    "if the true monotone strength is this ``tau``, how many observations
    would I need for MK to reject H₀ at level ``alpha``?" — the crossing
    of the ``p(n)`` curve with ``alpha`` is stored on
    :attr:`MKPowerCurve.n_needed`.

    Args:
        y: Numeric series (NaNs are dropped before Kendall's ``tau``).
        alpha: Significance threshold used to compute ``n_needed``.
        n_range: Inclusive ``(n_min, n_max)`` bounds for the sweep grid.
    """
    from scipy.stats import kendalltau

    arr = np.asarray(y, dtype=float)
    mask = np.isfinite(arr)
    y_clean = arr[mask]
    x_clean = np.arange(len(arr))[mask]
    n_obs = int(mask.sum())
    tau_obs, _ = kendalltau(x_clean, y_clean)
    tau_obs = float(tau_obs) if np.isfinite(tau_obs) else float("nan")
    p_obs = mk_asymptotic_pvalue(tau_obs, n_obs)

    n_min, n_max = int(n_range[0]), int(n_range[1])
    n_grid = np.arange(n_min, n_max + 1)
    p_grid = np.array([mk_asymptotic_pvalue(tau_obs, int(n)) for n in n_grid])
    crossing = np.isfinite(p_grid) & (p_grid <= alpha)
    n_needed = int(n_grid[crossing][0]) if crossing.any() else None

    return MKPowerCurve(
        tau_obs=tau_obs,
        n_obs=n_obs,
        p_obs=p_obs,
        n_grid=n_grid,
        p_grid=p_grid,
        alpha=float(alpha),
        n_needed=n_needed,
    )


def mk_power_curve_figure(
    y: Sequence[float],
    *,
    alpha: float = 0.05,
    n_range: tuple[int, int] = (5, 400),
    sample_ns: Sequence[int] = (50, 100, 200, 400),
    arm_label: str = "post",
):
    """One-picture explanation of MK power vs sample size at fixed ``tau``.

    Runs :func:`mk_power_curve` on ``y``, then renders the closed-form
    MK ``p(n)`` curve with:

    * bare dots at ``sample_ns`` (no text labels),
    * a red dashed line at ``p = alpha``,
    * a red ✕ at ``(n_obs, p_obs)`` labelled "you are here",
    * a green ★ at ``n_needed`` labelled with ``(n, α)`` when the curve
      crosses ``alpha``.

    Only the two highlight points (``n_obs`` and ``n_needed``) carry
    text annotations to keep the plot readable.

    Useful for stakeholder-facing "why isn't this p-value < 0.05?"
    conversations: the plot makes the ``p \\propto 1/\\sqrt{n}`` scaling
    of the MK test visible at a glance.

    Args:
        y: Numeric series (NaNs are dropped before Kendall's ``tau``).
        alpha: Significance threshold; drawn as a dashed reference line.
        n_range: Sweep grid bounds ``(n_min, n_max)``.
        sample_ns: Extra ``n`` values to annotate on the curve.
        arm_label: Short label used in the title (e.g. ``"pre"``, ``"post"``).

    Returns:
        Tuple of ``(plotly.graph_objects.Figure, MKPowerCurve)``.
    """
    import plotly.graph_objects as go

    curve = mk_power_curve(y, alpha=alpha, n_range=n_range)

    # Reference markers on the curve (no text labels) — only n_obs and
    # n_needed get annotated below, everything else is a bare dot.
    n_min, n_max = int(n_range[0]), int(n_range[1])
    highlight_ns = {int(curve.n_obs)}
    if curve.n_needed is not None:
        highlight_ns.add(int(curve.n_needed))
    marker_ns = sorted({int(n) for n in sample_ns if n_min <= n <= n_max}
                       - highlight_ns)
    marker_ps = [mk_asymptotic_pvalue(curve.tau_obs, n) for n in marker_ns]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.n_grid, y=curve.p_grid, mode="lines",
        name=f"MK p(n) at fixed τ = {curve.tau_obs:+.3f}",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=marker_ns, y=marker_ps, mode="markers",
        marker=dict(color="#1f77b4", size=7, symbol="circle"),
        name="sample n → p", showlegend=False,
    ))
    fig.add_hline(
        y=alpha, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"α = {alpha:g}", annotation_position="top right",
        annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=[curve.n_obs], y=[curve.p_obs], mode="markers+text",
        marker=dict(color="#d62728", size=13, symbol="x"),
        text=[f"n={curve.n_obs}<br>p={curve.p_obs:.3f}"],
        textposition="top right", textfont=dict(size=10, color="#d62728"),
        name=f"you are here (n={curve.n_obs}, p={curve.p_obs:.3f})",
    ))
    if curve.n_needed is not None:
        fig.add_trace(go.Scatter(
            x=[curve.n_needed], y=[alpha], mode="markers+text",
            marker=dict(color="#2ca02c", size=13, symbol="star"),
            text=[f"n={curve.n_needed}<br>p={alpha:g}"],
            textposition="top right", textfont=dict(size=10, color="#2ca02c"),
            name=f"n needed to reject α={alpha:g} (n≈{curve.n_needed})",
        ))

    p_max_finite = float(np.nanmax(curve.p_grid)) if np.isfinite(curve.p_grid).any() else 1.0
    y_top = min(1.0, max(0.5, p_max_finite * 1.05))
    need_txt = f"need ~{curve.n_needed} points." if curve.n_needed else "grid did not cross α."
    fig.update_layout(
        title=dict(
            text=(
                f"{arm_label}-arm MK power curve  |  observed τ = {curve.tau_obs:+.3f}<br>"
                f"<sub>fix τ, sweep n → p-value falls as √n. "
                f"At n = {curve.n_obs} we cannot reject H₀; {need_txt}</sub>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="hypothetical sample size n"),
        yaxis_title="MK two-sided p-value",
        yaxis=dict(range=[0, y_top]),
        font=dict(size=11),
        legend=dict(yanchor="top", y=-0.20, xanchor="left", x=0, orientation="h",
                    font=dict(size=10)),
    )
    return fig, curve


@dataclass(frozen=True)
class MKPowerOfTest:
    """Return type of :func:`mk_power_of_test`.

    Companion to :class:`MKPowerCurve`: fixes Kendall ``tau_obs`` and sweeps
    ``n`` to give MK *power* (probability of rejecting H₀ at level
    ``alpha``) instead of the p-value. Stores the observed ``power_now``
    at ``n_obs`` plus the smallest ``n`` on the grid whose power reaches
    ``target_power`` (``None`` when no grid point clears the target).
    """

    tau_obs: float
    n_obs: int
    power_now: float
    n_grid: np.ndarray
    power_grid: np.ndarray
    alpha: float
    target_power: float
    n_needed_power: int | None


def mk_asymptotic_power(tau: float, n: int, alpha: float = 0.05) -> float:
    """Two-sided asymptotic MK power at fixed Kendall ``tau`` and ``n``.

    Uses the closed-form Gaussian approximation to Mann-Kendall's ``Z``
    under the alternative :math:`\\tau = \\tau`:

    .. math::

        \\mu_Z = \\tau \\cdot \\sqrt{\\tfrac{9\\,n(n-1)}{2(2n+5)}}, \\quad
        \\text{power} =
        \\Phi(\\mu_Z - z_{1-\\alpha/2}) +
        \\Phi(-\\mu_Z - z_{1-\\alpha/2}).

    Returns ``nan`` when ``n < 4`` or ``tau`` is non-finite (MK is
    undefined under those conditions).
    """
    if n < 4 or not np.isfinite(tau):
        return float("nan")
    mu_z = float(tau) * np.sqrt(9.0 * n * (n - 1) / (2.0 * (2 * n + 5)))
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    return float(norm.cdf(mu_z - z_crit) + norm.cdf(-mu_z - z_crit))


def mk_power_of_test(
    curve: MKPowerCurve,
    *,
    target_power: float = 0.80,
) -> MKPowerOfTest:
    """MK power-of-test sweep — mirror of :func:`mk_power_curve` for power.

    Takes an existing :class:`MKPowerCurve` (so ``tau_obs``, ``n_obs``,
    ``alpha`` and ``n_grid`` are already fixed) and evaluates
    :func:`mk_asymptotic_power` across the same grid. Answers the reader's
    question "at ``tau = tau_obs``, how many observations would I need for
    MK to reject H₀ at least ``target_power`` fraction of the time?" — the
    crossing is stored on :attr:`MKPowerOfTest.n_needed_power`.

    Args:
        curve: :class:`MKPowerCurve` produced by :func:`mk_power_curve`.
        target_power: Power threshold used to compute ``n_needed_power``
            (0.80 is the conventional target).
    """
    power_grid = np.array(
        [mk_asymptotic_power(curve.tau_obs, int(n), alpha=curve.alpha)
         for n in curve.n_grid]
    )
    power_now = mk_asymptotic_power(curve.tau_obs, curve.n_obs, alpha=curve.alpha)
    hit = np.isfinite(power_grid) & (power_grid >= target_power)
    n_needed_power = int(curve.n_grid[hit][0]) if hit.any() else None

    return MKPowerOfTest(
        tau_obs=curve.tau_obs,
        n_obs=curve.n_obs,
        power_now=power_now,
        n_grid=curve.n_grid,
        power_grid=power_grid,
        alpha=curve.alpha,
        target_power=float(target_power),
        n_needed_power=n_needed_power,
    )


def mk_power_of_test_figure(
    curve: MKPowerCurve,
    *,
    target_power: float = 0.80,
    arm_label: str = "post",
):
    """One-picture explanation of MK power vs sample size at fixed ``tau``.

    Companion to :func:`mk_power_curve_figure`. Runs :func:`mk_power_of_test`
    on ``curve``, then renders the closed-form ``power(n)`` curve with:

    * a grey dotted line at ``power = alpha`` (chance-level rejection),
    * a red dashed line at ``power = target_power``,
    * a red ✕ at ``(n_obs, power_now)`` labelled "you are here",
    * a green ★ at ``n_needed_power`` when the curve reaches
      ``target_power``.

    Useful for stakeholder-facing "we should have collected more data"
    conversations: shows the sample size needed to detect the observed
    trend at conventional power.

    Args:
        curve: :class:`MKPowerCurve` produced by :func:`mk_power_curve`.
        target_power: Power threshold; drawn as a dashed reference line.
        arm_label: Short label used in the title (e.g. ``"pre"``, ``"post"``).

    Returns:
        Tuple of ``(plotly.graph_objects.Figure, MKPowerOfTest)``.
    """
    import plotly.graph_objects as go

    power = mk_power_of_test(curve, target_power=target_power)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=power.n_grid, y=power.power_grid, mode="lines",
        name=f"power(n) at fixed τ = {power.tau_obs:+.3f}",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_hline(
        y=power.alpha, line=dict(color="grey", dash="dot"),
        annotation_text=f"α = {power.alpha:g} (chance)",
        annotation_position="bottom right", annotation_font_size=10,
    )
    fig.add_hline(
        y=power.target_power, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"target = {power.target_power:.2f}",
        annotation_position="top right", annotation_font_size=10,
    )
    fig.add_trace(go.Scatter(
        x=[power.n_obs], y=[power.power_now], mode="markers+text",
        marker=dict(color="#d62728", size=13, symbol="x"),
        text=[f"n={power.n_obs}<br>power={power.power_now:.2f}"],
        textposition="top right", textfont=dict(size=10, color="#d62728"),
        name=f"you are here (n={power.n_obs}, power={power.power_now:.2f})",
    ))
    if power.n_needed_power is not None:
        fig.add_trace(go.Scatter(
            x=[power.n_needed_power], y=[power.target_power],
            mode="markers+text",
            marker=dict(color="#2ca02c", size=13, symbol="star"),
            text=[f"n={power.n_needed_power}<br>power={power.target_power:.2f}"],
            textposition="top left", textfont=dict(size=10, color="#2ca02c"),
            name=f"n needed for power ≥ {power.target_power:.2f} "
                 f"(n≈{power.n_needed_power})",
        ))

    need_txt = (
        f"need ~{power.n_needed_power} points for power ≥ {power.target_power:.2f}."
        if power.n_needed_power is not None
        else f"grid did not reach power {power.target_power:.2f}."
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{arm_label}-arm power of test  |  observed τ = {power.tau_obs:+.3f}<br>"
                f"<sub>fix τ, sweep n → power(n) = P(reject H₀ | τ = τ_obs). "
                f"At n = {power.n_obs} power is {power.power_now:.2f}; {need_txt}</sub>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="hypothetical sample size n"),
        yaxis=dict(title="MK two-sided power", range=[0, 1.02]),
        font=dict(size=11),
        legend=dict(yanchor="top", y=-0.20, xanchor="left", x=0, orientation="h",
                    font=dict(size=10)),
    )
    return fig, power


# ---------------------------------------------------------------------------
# MK dual-axis p+power view and A/B t-test vs MK overlays
# ---------------------------------------------------------------------------

# Palette convention within a family: distinct shades per ROLE (line /
# observed marker / n-needed marker) so a shared legend disambiguates
# axis AND role even when several traces sit on the same axis.
_MK_P_PALETTE = {"line": "#1f77b4", "obs": "#0d3d66", "n_needed": "#17becf"}      # blues
_MK_POWER_PALETTE = {"line": "#ff7f0e", "obs": "#8c3d00", "n_needed": "#ffbb78"}  # oranges


def _add_pvalue_traces(
    fig, curve: MKPowerCurve, *, secondary_y: bool | None = None,
) -> None:
    """Add MK p(n) line + observed X + optional n-needed star (blue family).

    ``secondary_y`` is forwarded to ``fig.add_trace`` only when set, so the
    helper works both on a :func:`plotly.subplots.make_subplots` figure with
    a secondary y-axis and on a plain :class:`plotly.graph_objects.Figure`.
    """
    import plotly.graph_objects as go
    pal = _MK_P_PALETTE
    kw = {} if secondary_y is None else {"secondary_y": secondary_y}
    fig.add_trace(go.Scatter(
        x=curve.n_grid, y=curve.p_grid, mode="lines",
        name=f"[p] MK p(n) at τ = {curve.tau_obs:+.3f}",
        line=dict(color=pal["line"], width=2), legendgroup="mk_p",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=[curve.n_obs], y=[curve.p_obs], mode="markers+text",
        marker=dict(color=pal["obs"], size=13, symbol="x"),
        text=[f"obs<br>n={curve.n_obs}, p={curve.p_obs:.3f}"],
        textposition="top right", textfont=dict(size=10, color=pal["obs"]),
        name=f"[p] observed (n={curve.n_obs}, p={curve.p_obs:.3f})",
        legendgroup="mk_p",
    ), **kw)
    if curve.n_needed is not None:
        fig.add_trace(go.Scatter(
            x=[curve.n_needed], y=[curve.alpha], mode="markers+text",
            marker=dict(color=pal["n_needed"], size=13, symbol="star"),
            text=[f"n≈{curve.n_needed}"],
            textposition="top left", textfont=dict(size=10, color=pal["n_needed"]),
            name=f"[p] n needed for α ≤ {curve.alpha:g} (n≈{curve.n_needed})",
            legendgroup="mk_p",
        ), **kw)


def _add_power_traces(
    fig, power: MKPowerOfTest, *, secondary_y: bool | None = None,
) -> None:
    """Add MK power(n) line + observed X + optional n-needed star (orange family).

    ``secondary_y`` is forwarded to ``fig.add_trace`` only when set, so the
    helper works both on a :func:`plotly.subplots.make_subplots` figure with
    a secondary y-axis and on a plain :class:`plotly.graph_objects.Figure`.
    """
    import plotly.graph_objects as go
    pal = _MK_POWER_PALETTE
    kw = {} if secondary_y is None else {"secondary_y": secondary_y}
    fig.add_trace(go.Scatter(
        x=power.n_grid, y=power.power_grid, mode="lines",
        name=f"[power] MK power(n) at τ = {power.tau_obs:+.3f}",
        line=dict(color=pal["line"], width=2), legendgroup="mk_power",
    ), **kw)
    fig.add_trace(go.Scatter(
        x=[power.n_obs], y=[power.power_now], mode="markers+text",
        marker=dict(color=pal["obs"], size=13, symbol="x"),
        text=[f"obs<br>n={power.n_obs}, power={power.power_now:.2f}"],
        textposition="bottom right", textfont=dict(size=10, color=pal["obs"]),
        name=f"[power] observed (n={power.n_obs}, power={power.power_now:.2f})",
        legendgroup="mk_power",
    ), **kw)
    if power.n_needed_power is not None:
        fig.add_trace(go.Scatter(
            x=[power.n_needed_power], y=[power.target_power], mode="markers+text",
            marker=dict(color=pal["n_needed"], size=13, symbol="star"),
            text=[f"n≈{power.n_needed_power}"],
            textposition="top left", textfont=dict(size=10, color=pal["n_needed"]),
            name=(f"[power] n needed for power ≥ {power.target_power:.2f} "
                  f"(n≈{power.n_needed_power})"),
            legendgroup="mk_power",
        ), **kw)


def _apply_dual_axis_layout(
    fig, curve: MKPowerCurve, power: MKPowerOfTest, arm_label: str,
) -> None:
    """Axis titles/ranges/colours + top-level layout for the dual-axis figure."""
    p_max = float(np.nanmax(curve.p_grid)) if np.isfinite(curve.p_grid).any() else 1.0
    y_top = min(1.0, max(0.5, p_max * 1.05))
    fig.add_hline(
        y=curve.alpha, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"α = {curve.alpha:g} (p)",
        annotation_position="top left", annotation_font_size=10,
        secondary_y=False,
    )
    fig.add_hline(
        y=power.target_power, line=dict(color="#2ca02c", dash="dash"),
        annotation_text=f"target power = {power.target_power:.2f}",
        annotation_position="top right", annotation_font_size=10,
        secondary_y=True,
    )
    fig.update_xaxes(title_text="hypothetical sample size n")
    fig.update_yaxes(
        title_text="MK two-sided p-value", range=[0, y_top],
        color=_MK_P_PALETTE["line"], secondary_y=False,
    )
    fig.update_yaxes(
        title_text="MK two-sided power", range=[0, 1.02],
        color=_MK_POWER_PALETTE["line"], secondary_y=True,
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{arm_label}-arm MK — p-value (left, blue) &amp; "
                f"power (right, orange) vs n  |  "
                f"observed τ = {curve.tau_obs:+.3f}, n = {curve.n_obs}<br>"
                f"<sub>fix τ, sweep n → p-value ↓ as √n; power ↑ toward 1.</sub>"
            ),
            font=dict(size=13),
        ),
        font=dict(size=11), height=560,
        legend=dict(yanchor="top", y=-0.18, xanchor="left", x=0,
                    orientation="h", font=dict(size=10)),
    )


def _apply_pvalue_only_layout(
    fig, curve: MKPowerCurve, arm_label: str,
) -> None:
    """Single-axis layout for the p-value-only variant of the dual figure."""
    p_max = float(np.nanmax(curve.p_grid)) if np.isfinite(curve.p_grid).any() else 1.0
    y_top = min(1.0, max(0.5, p_max * 1.05))
    fig.add_hline(
        y=curve.alpha, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"α = {curve.alpha:g}",
        annotation_position="top left", annotation_font_size=10,
    )
    fig.update_xaxes(title_text="hypothetical sample size n")
    fig.update_yaxes(
        title_text="MK two-sided p-value", range=[0, y_top],
        color=_MK_P_PALETTE["line"],
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{arm_label}-arm MK — p-value vs n  |  "
                f"observed τ = {curve.tau_obs:+.3f}, n = {curve.n_obs}<br>"
                f"<sub>fix τ, sweep n → p-value ↓ as √n.</sub>"
            ),
            font=dict(size=13),
        ),
        font=dict(size=11), height=560,
        legend=dict(yanchor="top", y=-0.18, xanchor="left", x=0,
                    orientation="h", font=dict(size=10)),
    )


def _apply_power_only_layout(
    fig, curve: MKPowerCurve, power: MKPowerOfTest, arm_label: str,
) -> None:
    """Single-axis layout for the power-only variant of the dual figure."""
    fig.add_hline(
        y=curve.alpha, line=dict(color="grey", dash="dot"),
        annotation_text=f"α = {curve.alpha:g} (chance)",
        annotation_position="bottom right", annotation_font_size=10,
    )
    fig.add_hline(
        y=power.target_power, line=dict(color="#2ca02c", dash="dash"),
        annotation_text=f"target power = {power.target_power:.2f}",
        annotation_position="top right", annotation_font_size=10,
    )
    fig.update_xaxes(title_text="hypothetical sample size n")
    fig.update_yaxes(
        title_text="MK two-sided power", range=[0, 1.02],
        color=_MK_POWER_PALETTE["line"],
    )
    fig.update_layout(
        title=dict(
            text=(
                f"{arm_label}-arm MK — power vs n  |  "
                f"observed τ = {power.tau_obs:+.3f}, n = {power.n_obs}<br>"
                f"<sub>fix τ, sweep n → power ↑ toward 1.</sub>"
            ),
            font=dict(size=13),
        ),
        font=dict(size=11), height=560,
        legend=dict(yanchor="top", y=-0.18, xanchor="left", x=0,
                    orientation="h", font=dict(size=10)),
    )


def mk_pvalue_power_dual_axis_figure(
    curve: MKPowerCurve,
    *,
    target_power: float = 0.80,
    arm_label: str = "post",
    show: str = "both",
):
    """MK p-value and MK power vs hypothetical sample size.

    Consolidates :func:`mk_power_curve_figure` and
    :func:`mk_power_of_test_figure` into a single entry point. ``show``
    picks the layout:

    * ``"both"`` (default) — dual-y-axis panel: p-value on the left
      (blue family), power on the right (orange family), sharing the
      ``n`` axis. Reference lines: α on the left (red dashed),
      ``target_power`` on the right (green dashed).
    * ``"pvalue"`` — single-axis figure with only the p-value family
      (blue traces + α reference line).
    * ``"power"`` — single-axis figure with only the power family
      (orange traces + α/target reference lines). Equivalent to the
      standalone :func:`mk_power_of_test_figure`.

    Args:
        curve: :class:`MKPowerCurve` produced by :func:`mk_power_curve`.
        target_power: Threshold used for the power "n needed for
            power ≥ target" marker.
        arm_label: Short label used in the title (e.g. ``"pre"``, ``"post"``).
        show: One of ``"both"`` / ``"pvalue"`` / ``"power"``. Default
            ``"both"`` preserves the historical dual-axis behaviour.

    Returns:
        Tuple of ``(plotly.graph_objects.Figure, MKPowerOfTest)``. The
        second element is the :class:`MKPowerOfTest` that was computed
        internally, so callers do not need to re-run
        :func:`mk_power_of_test` (even when ``show="pvalue"``).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if show not in ("both", "pvalue", "power"):
        raise ValueError(
            f"show must be one of 'both', 'pvalue', 'power'; got {show!r}"
        )

    power = mk_power_of_test(curve, target_power=target_power)
    if show == "both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        _add_pvalue_traces(fig, curve, secondary_y=False)
        _add_power_traces(fig, power, secondary_y=True)
        _apply_dual_axis_layout(fig, curve, power, arm_label)
    elif show == "pvalue":
        fig = go.Figure()
        _add_pvalue_traces(fig, curve)
        _apply_pvalue_only_layout(fig, curve, arm_label)
    else:  # show == "power"
        fig = go.Figure()
        _add_power_traces(fig, power)
        _apply_power_only_layout(fig, curve, power, arm_label)
    return fig, power


@dataclass(frozen=True)
class AbTTestSweep:
    """Two-sample t-test companion to :class:`MKPowerCurve` / :class:`MKPowerOfTest`.

    Fixes the observed Cohen's ``d`` (mean_post − mean_pre over pooled
    SD) and sweeps balanced sample size ``n`` per arm over the MK power
    curve's ``n_grid``, so A/B and MK p-values (and powers) can be
    plotted on shared axes. The observed p-value comes from Welch's
    two-sample t (unequal variances allowed); the analytical sweep uses
    pooled-t via :class:`statsmodels.stats.power.TTestIndPower`. ``n_obs``
    is the smaller of the two arms (the balanced-t effective n).
    """

    d_obs: float
    n_obs: int
    p_obs: float
    power_now: float
    n_grid: np.ndarray
    p_grid: np.ndarray
    power_grid: np.ndarray
    alpha: float
    target_power: float
    n_needed_alpha: int | None
    n_needed_power: int | None
    n_pre: int
    n_post: int


@dataclass(frozen=True)
class MkVsTTestFigures:
    """Return type of :func:`mk_vs_ttest_figures`.

    Bundles the two shared-axis overlays (A/B t-test vs Mann-Kendall) and
    the :class:`AbTTestSweep` summary numbers, so the caller can render
    the figures AND print a headline table without a second call.
    """

    fig_pvalue: object   # plotly.graph_objects.Figure
    fig_power: object    # plotly.graph_objects.Figure
    ab: AbTTestSweep


def _pooled_sd(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int) -> float:
    """Pooled SD for the two-sample equal-variance t-test (NaN-safe)."""
    va = float(np.nanvar(a, ddof=1))
    vb = float(np.nanvar(b, ddof=1))
    return float(np.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2)))


def _cohens_d(pre: np.ndarray, post: np.ndarray, pooled_sd: float) -> float:
    """Cohen's d = (mean_post − mean_pre) / pooled_sd; NaN when SD is 0."""
    if pooled_sd <= 0:
        return float("nan")
    return float(np.nanmean(post) - np.nanmean(pre)) / pooled_sd


def _ttest_pvalue_sweep(abs_d: float, n_grid: np.ndarray) -> np.ndarray:
    """Two-sided balanced-t p-values across ``n``: t = d·√(n/2), df = 2n−2."""
    from scipy.stats import t as student_t
    if not np.isfinite(abs_d) or abs_d <= 0:
        return np.full(n_grid.shape, float("nan"))
    t_vals = abs_d * np.sqrt(n_grid / 2.0)
    df_vals = 2 * n_grid - 2
    return np.array([
        float(2.0 * (1.0 - student_t.cdf(t, df=df)))
        for t, df in zip(t_vals, df_vals)
    ])


def _ttest_power_at(abs_d: float, n: int, *, alpha: float) -> float:
    """Two-sided balanced-t power at a single ``n`` via TTestIndPower."""
    from statsmodels.stats.power import TTestIndPower
    if not np.isfinite(abs_d) or abs_d <= 0 or n < 2:
        return float("nan")
    return float(TTestIndPower().solve_power(
        effect_size=abs_d, nobs1=int(n), alpha=alpha, ratio=1.0,
        alternative="two-sided",
    ))


def _ttest_power_sweep(
    abs_d: float, n_grid: np.ndarray, *, alpha: float,
) -> np.ndarray:
    """Analytic two-sided balanced-t power across ``n`` via TTestIndPower."""
    if not np.isfinite(abs_d) or abs_d <= 0:
        return np.full(n_grid.shape, float("nan"))
    return np.array([_ttest_power_at(abs_d, int(n), alpha=alpha) for n in n_grid])


def _first_crossing(
    grid: np.ndarray, values: np.ndarray, threshold: float, *, mode: str,
) -> int | None:
    """Smallest grid point where ``values`` cross ``threshold``.

    ``mode='le'`` searches for ``values <= threshold`` (used for p ≤ α);
    ``mode='ge'`` searches for ``values >= threshold`` (used for power ≥
    target). Returns ``None`` when no grid point satisfies the condition.
    """
    if mode == "le":
        hit = np.isfinite(values) & (values <= threshold)
    elif mode == "ge":
        hit = np.isfinite(values) & (values >= threshold)
    else:
        raise ValueError(f"mode must be 'le' or 'ge', got {mode!r}")
    return int(grid[hit][0]) if hit.any() else None


def ab_ttest_sweep(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    n_grid: np.ndarray,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> AbTTestSweep:
    """Welch t-test on the daily metric + fixed-d balanced-n sweep.

    Computes Cohen's ``d`` on pre vs post (pooled SD), the observed
    Welch p-value, and the analytic p-value + power curves at that fixed
    ``d`` across the supplied ``n_grid`` (typically
    :attr:`MKPowerCurve.n_grid` so the A/B and MK curves share an x-axis).

    Args:
        y_pre: Daily metric on the pre-intervention arm.
        y_post: Daily metric on the post-intervention arm.
        n_grid: Integer sample-size grid to sweep (per arm, balanced design).
        alpha: Significance threshold for the p-value sweep (drives
            ``n_needed_alpha``).
        target_power: Power threshold (drives ``n_needed_power``).

    Returns:
        :class:`AbTTestSweep` with observed d / p / power and the swept
        curves.

    Raises:
        ValueError: If either arm has fewer than 2 finite observations.
    """
    from scipy.stats import ttest_ind

    pre_arr = np.asarray(y_pre, dtype=float)
    post_arr = np.asarray(y_post, dtype=float)
    n_pre = int(np.isfinite(pre_arr).sum())
    n_post = int(np.isfinite(post_arr).sum())
    if n_pre < 2 or n_post < 2:
        raise ValueError(
            f"ab_ttest_sweep needs ≥2 finite points per arm; "
            f"got n_pre={n_pre}, n_post={n_post}."
        )
    pooled_sd = _pooled_sd(pre_arr, post_arr, n_pre, n_post)
    d_obs = _cohens_d(pre_arr, post_arr, pooled_sd)
    _, p_obs = ttest_ind(post_arr, pre_arr, equal_var=False, nan_policy="omit")
    n_obs = min(n_pre, n_post)

    abs_d = abs(d_obs) if np.isfinite(d_obs) else float("nan")
    grid = np.asarray(n_grid, dtype=int)
    p_grid = _ttest_pvalue_sweep(abs_d, grid)
    power_grid = _ttest_power_sweep(abs_d, grid, alpha=alpha)
    power_now = _ttest_power_at(abs_d, n_obs, alpha=alpha)

    return AbTTestSweep(
        d_obs=d_obs, n_obs=n_obs, p_obs=float(p_obs), power_now=power_now,
        n_grid=grid, p_grid=p_grid, power_grid=power_grid,
        alpha=float(alpha), target_power=float(target_power),
        n_needed_alpha=_first_crossing(grid, p_grid, alpha, mode="le"),
        n_needed_power=_first_crossing(grid, power_grid, target_power, mode="ge"),
        n_pre=n_pre, n_post=n_post,
    )


def _add_observed_marker(
    fig, x: float, y: float, *,
    color: str, label: str, position: str = "top right",
) -> None:
    """Add an X marker with text at (x, y) — 'you are here' style."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        marker=dict(color=color, size=13, symbol="x"),
        text=[label], textposition=position,
        textfont=dict(size=10, color=color),
        showlegend=False,
    ))


def _add_n_needed_marker(
    fig, x: float, y: float, *,
    color: str, label: str, position: str = "top left",
) -> None:
    """Add a star marker with text at the (n_needed, threshold) crossing."""
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        marker=dict(color=color, size=12, symbol="star"),
        text=[label], textposition=position,
        textfont=dict(size=10, color=color),
        showlegend=False,
    ))


def _apply_overlay_layout(fig, *, title: str, y_title: str, y_range) -> None:
    """Common layout for shared-axis A/B vs MK overlays."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="hypothetical sample size n"),
        yaxis=dict(title=y_title, range=list(y_range)),
        font=dict(size=11),
        legend=dict(yanchor="top", y=-0.15, xanchor="left", x=0,
                    orientation="h", font=dict(size=10)),
    )


def _ab_vs_mk_pvalue_figure(
    ab: AbTTestSweep, curve: MKPowerCurve, *, experiment: str,
):
    """Shared-axis p-value overlay (A/B blue, MK orange)."""
    import plotly.graph_objects as go
    ab_pal = _MK_P_PALETTE
    mk_pal = _MK_POWER_PALETTE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ab.n_grid, y=ab.p_grid, mode="lines",
        name=f"A/B t-test  (d = {ab.d_obs:+.3f})",
        line=dict(color=ab_pal["line"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=curve.n_grid, y=curve.p_grid, mode="lines",
        name=f"Mann–Kendall  (τ = {curve.tau_obs:+.3f})",
        line=dict(color=mk_pal["line"], width=2),
    ))
    fig.add_hline(
        y=ab.alpha, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"α = {ab.alpha:g}",
        annotation_position="top right", annotation_font_size=10,
    )
    _add_observed_marker(
        fig, ab.n_obs, ab.p_obs, color=ab_pal["obs"],
        label=f"A/B obs<br>n={ab.n_obs}, p={ab.p_obs:.3f}",
    )
    _add_observed_marker(
        fig, curve.n_obs, curve.p_obs, color=mk_pal["obs"],
        label=f"MK obs<br>n={curve.n_obs}, p={curve.p_obs:.3f}",
        position="bottom right",
    )
    if ab.n_needed_alpha is not None:
        _add_n_needed_marker(
            fig, ab.n_needed_alpha, ab.alpha, color=ab_pal["n_needed"],
            label=f"A/B n≈{ab.n_needed_alpha}",
        )
    if curve.n_needed is not None:
        _add_n_needed_marker(
            fig, curve.n_needed, ab.alpha, color=mk_pal["n_needed"],
            label=f"MK n≈{curve.n_needed}", position="top right",
        )
    p_max = float(np.nanmax(np.concatenate([ab.p_grid, curve.p_grid])))
    y_top = min(1.0, max(0.5, p_max * 1.05))
    _apply_overlay_layout(
        fig,
        title=(
            f"{experiment} — p-value vs n:  A/B t-test vs Mann–Kendall<br>"
            f"<sub>fix observed effect size (d = {ab.d_obs:+.3f}, "
            f"τ = {curve.tau_obs:+.3f}), sweep n → shared axes, two curves.</sub>"
        ),
        y_title="two-sided p-value", y_range=(0, y_top),
    )
    return fig


def _ab_vs_mk_power_figure(
    ab: AbTTestSweep, power: MKPowerOfTest, *, experiment: str,
):
    """Shared-axis power overlay (A/B blue, MK orange)."""
    import plotly.graph_objects as go
    ab_pal = _MK_P_PALETTE
    mk_pal = _MK_POWER_PALETTE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ab.n_grid, y=ab.power_grid, mode="lines",
        name=f"A/B t-test  (d = {ab.d_obs:+.3f})",
        line=dict(color=ab_pal["line"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=power.n_grid, y=power.power_grid, mode="lines",
        name=f"Mann–Kendall  (τ = {power.tau_obs:+.3f})",
        line=dict(color=mk_pal["line"], width=2),
    ))
    fig.add_hline(
        y=ab.alpha, line=dict(color="grey", dash="dot"),
        annotation_text=f"α = {ab.alpha:g} (chance)",
        annotation_position="bottom right", annotation_font_size=10,
    )
    fig.add_hline(
        y=ab.target_power, line=dict(color="#d62728", dash="dash"),
        annotation_text=f"target = {ab.target_power:.2f}",
        annotation_position="top right", annotation_font_size=10,
    )
    _add_observed_marker(
        fig, ab.n_obs, ab.power_now, color=ab_pal["obs"],
        label=f"A/B obs<br>n={ab.n_obs}, power={ab.power_now:.2f}",
    )
    _add_observed_marker(
        fig, power.n_obs, power.power_now, color=mk_pal["obs"],
        label=f"MK obs<br>n={power.n_obs}, power={power.power_now:.2f}",
        position="bottom right",
    )
    if ab.n_needed_power is not None:
        _add_n_needed_marker(
            fig, ab.n_needed_power, ab.target_power, color=ab_pal["n_needed"],
            label=f"A/B n≈{ab.n_needed_power}",
        )
    if power.n_needed_power is not None:
        _add_n_needed_marker(
            fig, power.n_needed_power, ab.target_power, color=mk_pal["n_needed"],
            label=f"MK n≈{power.n_needed_power}", position="top right",
        )
    _apply_overlay_layout(
        fig,
        title=(
            f"{experiment} — power vs n:  A/B t-test vs Mann–Kendall<br>"
            f"<sub>fix observed effect size (d = {ab.d_obs:+.3f}, "
            f"τ = {power.tau_obs:+.3f}), sweep n → shared axes, two curves.</sub>"
        ),
        y_title="two-sided power", y_range=(0, 1.02),
    )
    return fig


def mk_vs_ttest_figures(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    curve: MKPowerCurve,
    *,
    target_power: float = 0.80,
    experiment: str = "",
) -> MkVsTTestFigures:
    """A/B t-test vs Mann-Kendall — p-value + power overlays on shared axes.

    Two figures on the same hypothetical-n axis:

    1. **p-value overlay** — Welch t-test p-value curve (blue) and MK
       p-value curve (orange), α reference line, "you are here" markers
       and n-needed-for-α stars for each track.
    2. **power overlay** — analytic t-test power (blue) and MK power
       (orange), α + ``target_power`` reference lines, observed markers
       and n-needed-for-target stars.

    Both curves fix the observed effect size (Cohen's ``d`` for A/B,
    Kendall's ``τ`` for MK) and sweep the balanced-design sample size
    over ``curve.n_grid``, so the x-axes are directly comparable. The
    A/B "you are here" marker uses the smaller arm's n (balanced-t's
    effective n).

    Args:
        y_pre: Daily metric on the pre-intervention arm.
        y_post: Daily metric on the post-intervention arm.
        curve: :class:`MKPowerCurve` produced by :func:`mk_power_curve`
            (its ``alpha`` is reused for the A/B sweep so both share the
            same α reference line).
        target_power: Power threshold used by the right axis of the
            power overlay.
        experiment: Short label injected into both figure titles.

    Returns:
        :class:`MkVsTTestFigures` bundling both figures and the
        :class:`AbTTestSweep` numbers.
    """
    power = mk_power_of_test(curve, target_power=target_power)
    ab = ab_ttest_sweep(
        y_pre, y_post, curve.n_grid,
        alpha=curve.alpha, target_power=target_power,
    )
    fig_pvalue = _ab_vs_mk_pvalue_figure(ab, curve, experiment=experiment)
    fig_power = _ab_vs_mk_power_figure(ab, power, experiment=experiment)
    return MkVsTTestFigures(fig_pvalue=fig_pvalue, fig_power=fig_power, ab=ab)


def intervention_summary_row(
    experiment: str,
    intervention_date: str | pd.Timestamp,
    result: MbbDeltaResult,
    *,
    regime: AcfRegime | None = None,
    boot: MbbDeltaResult | None = None,
    homogeneity_p_pre: float | None = None,
    homogeneity_p_post: float | None = None,
    guard_fires: bool | None = None,
    delta_pvalue: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Flatten a paired MK analysis into a single dict (one CSV row).

    All fields with no available value are recorded as ``NaN`` / ``None``
    so downstream ``pd.DataFrame`` concatenation stays column-aligned.

    Args:
        experiment: Identifier used as the row's primary key.
        intervention_date: Intervention date (formatted as ISO string).
        result: Main :class:`MbbDeltaResult` from :func:`mk_delta_mbb`.
        regime: Optional ACF diagnostic (adds ``heavy_acf``, ``rho_pre``,
            ``rho_post``).
        boot: Optional cross-check :class:`MbbDeltaResult` (adds
            ``boot_*`` fields). Reserved for future variant sweeps; the
            unified :func:`mk_delta_mbb` already returns bootstrap CIs.
        homogeneity_p_pre / homogeneity_p_post: Seasonal-MK VBH p-values.
        guard_fires: Boolean flag from the VBH guard.
        delta_pvalue: Optional Δ p-value (defaults to ``result.delta_pvalue``
            which :func:`mk_delta_mbb` populates automatically).
        extra: Any additional key/value pairs to merge in last (wins on
            conflict).

    Returns:
        Ordered ``dict`` suitable for ``pd.DataFrame([row]).to_csv(...)``.
    """
    row: dict = {
        "experiment": experiment,
        "intervention_date": pd.Timestamp(intervention_date).date().isoformat(),
        "n_pre": int(result.n_pre),
        "n_post": int(result.n_post),
        "pipeline": result.method,
        "heavy_acf": bool(regime.heavy_acf) if regime is not None else None,
        "rho_pre": float(regime.rho_pre) if regime is not None else float("nan"),
        "rho_post": float(regime.rho_post) if regime is not None else float("nan"),
        "slope_pre": float(result.slope_pre),
        "slope_pre_ci_lo": float(result.slope_ci_pre[0]),
        "slope_pre_ci_hi": float(result.slope_ci_pre[1]),
        "slope_post": float(result.slope_post),
        "slope_post_ci_lo": float(result.slope_ci_post[0]),
        "slope_post_ci_hi": float(result.slope_ci_post[1]),
        "delta_slope": float(result.delta_slope),
        "delta_ci_lo": float(result.delta_ci[0]),
        "delta_ci_hi": float(result.delta_ci[1]),
        "delta_significant": bool(not (result.delta_ci[0] <= 0 <= result.delta_ci[1])),
        "delta_pval": float(delta_pvalue) if delta_pvalue is not None else float("nan"),
        "pval_pre": float(result.pval_pre),
        "pval_post": float(result.pval_post),
        "homogeneity_p_pre": (
            float(homogeneity_p_pre) if homogeneity_p_pre is not None else float("nan")
        ),
        "homogeneity_p_post": (
            float(homogeneity_p_post) if homogeneity_p_post is not None else float("nan")
        ),
        "guard_fires": bool(guard_fires) if guard_fires is not None else None,
        "boot_delta": float(boot.delta_slope) if boot is not None else float("nan"),
        "boot_ci_lo": float(boot.delta_ci[0]) if boot is not None else float("nan"),
        "boot_ci_hi": float(boot.delta_ci[1]) if boot is not None else float("nan"),
    }
    if extra:
        row.update(extra)
    return row


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
            f"({sorted(years)[0]}). For sub-year daily data call "
            "mk_delta_mbb(mk_method='hamed_rao') instead."
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


def tfpw_y(y: Sequence[float]) -> tuple[np.ndarray, float]:
    """Yue-Wang trend-free prewhitening (TFPW-Y).

    Steps (Yue et al. 2002, HP; Yue and Wang 2002, WRR):

    1. Estimate the Sen slope with the original Mann-Kendall test.
    2. Detrend: subtract ``slope * t`` from the series.
    3. Estimate lag-1 AR: ``rho1 = corr(detrended[:-1], detrended[1:])``.
    4. Whiten: ``w[i] = detrended[i] - rho1 * detrended[i-1]``.
    5. Re-add the trend so the whitened series carries the same slope.

    Returns ``(whitened_series, rho1_used)``. The whitened series is one
    shorter than ``y`` (first sample consumed by AR(1) subtraction).

    Args:
        y: One-dimensional numeric series (missing values are dropped).

    Returns:
        Tuple ``(whitened, rho1)`` with the trend-preserving whitened array
        and the lag-1 coefficient that was removed.

    Example:
        >>> w, rho = tfpw_y([1.0, 2.1, 2.9, 4.2, 5.0])
        >>> w.shape[0] == 4  # one sample shorter than input
        True
    """
    import pymannkendall as _pmk

    arr = _clean(y)
    if arr.size < 3:
        return np.asarray(arr[1:], dtype=float), float("nan")
    slope = float(_pmk.original_test(arr).slope)
    t = np.arange(arr.size, dtype=float)
    detrended = arr - slope * t
    rho1 = float(np.corrcoef(detrended[:-1], detrended[1:])[0, 1])
    whitened = detrended[1:] - rho1 * detrended[:-1] + slope * t[1:]
    return whitened, rho1


@dataclass(frozen=True)
class AcfRegime:
    """Paired pre/post lag-1 ACF diagnostic + pipeline-dispatch verdict."""

    rho_pre: float
    rho_post: float
    heavy_acf: bool
    acf_cutoff: float
    pipeline_label: str

    @property
    def max_abs_rho(self) -> float:
        return float(max(abs(self.rho_pre), abs(self.rho_post)))

    def summary(self) -> str:
        return (
            f"lag-1 ACF (deseasoned): pre={self.rho_pre:+.3f}  "
            f"post={self.rho_post:+.3f}  (cutoff=±{self.acf_cutoff:.2f})\n"
            f"→ pipeline: {self.pipeline_label}"
        )


def classify_acf_regime(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    period: int,
    acf_cutoff: float = 0.20,
) -> AcfRegime:
    """Deseason both arms, measure lag-1 ACF, dispatch a paired-test pipeline.

    Deseasons each arm via :func:`deseason` at the given period, computes
    :func:`lag1_acf` on the residuals, then returns an :class:`AcfRegime`
    that says whether the workflow should use the analytical (light-ACF)
    or AR-aware (heavy-ACF) paired test.

    Args:
        y_pre: Pre-intervention series.
        y_post: Post-intervention series.
        period: Seasonal period (7 for daily data with weekly seasonality).
        acf_cutoff: Absolute lag-1 ACF above which the heavy path is used.

    Returns:
        :class:`AcfRegime` with ``rho_pre``, ``rho_post``, ``heavy_acf``
        and a human-readable ``pipeline_label``.
    """
    resid_pre = deseason(y_pre, period=period)
    resid_post = deseason(y_post, period=period)
    rho_pre = lag1_acf(resid_pre)
    rho_post = lag1_acf(resid_post)
    heavy = max(abs(rho_pre), abs(rho_post)) > acf_cutoff
    label = (
        "heavy-ACF → mk_delta_mbb(mk_method='3pw' or 'hamed_rao') + MBB CI"
        if heavy
        else "light-ACF → mk_delta_mbb(mk_method='seasonal') + MBB CI"
    )
    return AcfRegime(
        rho_pre=float(rho_pre),
        rho_post=float(rho_post),
        heavy_acf=bool(heavy),
        acf_cutoff=float(acf_cutoff),
        pipeline_label=label,
    )


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
# Paired pre/post comparison — unified moving-block bootstrap
# ---------------------------------------------------------------------------


_MK_METHODS = ("original", "hamed_rao", "yue_wang", "tfpw", "pw", "seasonal", "3pw")


def _mbb_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block-bootstrap resample of ``range(n)``.

    Draws overlapping blocks of length ``block_length`` uniformly at random
    from the ``n - block_length + 1`` possible start positions and stitches
    them together (last block truncated so total length is exactly ``n``).
    Preserves short-range dependence up to ``block_length``.

    ``block_length ≤ 1`` degenerates to iid resampling. Indices are returned
    sorted so callers can safely pair them with ``np.arange(n)`` as the
    ``x`` axis of ``theilslopes``.
    """
    if block_length <= 1 or n <= 1:
        return np.sort(rng.integers(0, n, n))
    bl = min(block_length, n)
    n_blocks = int(np.ceil(n / bl))
    max_start = n - bl
    starts = rng.integers(0, max_start + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
    return np.sort(idx)


def _mk_pvalue(
    y: np.ndarray,
    *,
    mk_method: str,
    alpha: float,
    period: int | None,
    dts: np.ndarray | None,
    mk_kwargs: dict,
) -> tuple[float, float | None, TrendResult | None]:
    """Return ``(p_value, homogeneity_p_or_None, mk3pw_result_or_None)``.

    ``homogeneity_p`` is populated only when ``mk_method`` is ``"seasonal"``
    or ``"3pw"`` (both run :func:`seasonal_mk` on the raw arm to obtain the
    Van Belle-Hughes χ² guard). All other methods return ``None`` for it.

    ``mk3pw_result`` carries the full ``TrendResult`` from :func:`mk_3pw`
    (slope + Gilbert-style CI + p) when ``mk_method='3pw'``, and ``None``
    otherwise. Consumers that want ``ci_method='gilbert'`` read the CI
    from here without paying for a second 3PW call.
    """
    import pymannkendall as pmk

    if mk_method == "original":
        return float(pmk.original_test(y, alpha=alpha).p), None, None
    if mk_method == "hamed_rao":
        return float(
            pmk.hamed_rao_modification_test(y, alpha=alpha, lag=mk_kwargs.get("hr_lag", 1)).p
        ), None, None
    if mk_method == "yue_wang":
        return float(
            pmk.yue_wang_modification_test(y, alpha=alpha, lag=mk_kwargs.get("hr_lag", 1)).p
        ), None, None
    if mk_method == "tfpw":
        return float(pmk.trend_free_pre_whitening_modification_test(y, alpha=alpha).p), None, None
    if mk_method == "pw":
        return float(pmk.pre_whitening_modification_test(y, alpha=alpha).p), None, None
    if mk_method == "seasonal":
        if period is None:
            raise ValueError("mk_method='seasonal' requires period=<int>")
        smk = seasonal_mk(
            y, period=int(period), alpha=alpha, hr_lag=mk_kwargs.get("hr_lag", 1),
        )
        return float(smk.p), float(smk.homogeneity_p), None
    if mk_method == "3pw":
        if period is None:
            raise ValueError("mk_method='3pw' requires period=<int>")
        if dts is None:
            raise ValueError("mk_method='3pw' requires dts_pre and dts_post")
        # 3PW is defined on a deseasoned residual (Collaud Coen et al. 2020).
        resid = deseason(y, period=int(period))
        r = mk_3pw(
            dts, resid,
            resolution=mk_kwargs.get("resolution", 1.0),
            alpha_mk=100.0 * (1.0 - alpha),
            alpha_cl=100.0 * (1.0 - alpha),
        )
        # Homogeneity guard runs on raw y — independent of prewhitening.
        smk = seasonal_mk(y, period=int(period), alpha=alpha)
        return float(r.p), float(smk.homogeneity_p), r
    raise ValueError(
        f"unknown mk_method={mk_method!r}; expected one of {list(_MK_METHODS)}"
    )


# ---------------------------------------------------------------------------
# §6 workflow — per-arm adaptive MK + three branch wrappers
# ---------------------------------------------------------------------------


def _dts_span_years(dts: Sequence) -> float:
    """Return the calendar-year span of a timestamp array (max − min, in years)."""
    if dts is None:
        return 0.0
    arr = np.asarray(dts, dtype="datetime64[ns]")
    if arr.size < 2:
        return 0.0
    span_s = float((arr.max() - arr.min()).astype("timedelta64[s]").astype(np.int64))
    return span_s / (365.25 * 86400.0)


def mk_adaptive_core_arm(
    y: Sequence[float],
    *,
    period: int | None = None,
    dts: Sequence | None = None,
    acf_cutoff: float = 0.20,
    force_variant: str | None = None,
    alpha: float = 0.05,
    mk_kwargs: dict | None = None,
) -> MKAdaptiveCoreArmResult:
    """Per-arm adaptive MK (chart 4.3 Path A): seasonality gate → §4.2 AR ladder → MK variant.

    Runs a single arm through the paper's variant-selection pipeline and
    returns the Sen slope point (from raw ``y``) and the closed-form MK
    p-value from the variant §4 picked.

    The selection rules match §4.3 Path A:

    * If ``period`` is given, deseason the arm (subtract per-phase mean)
      to obtain the residual substrate; otherwise the raw ``y`` is the
      substrate.
    * Measure the lag-1 ACF ``ρ_1`` of the substrate.
    * If ``|ρ_1| ≤ acf_cutoff`` → **plain MK** (``mk_method='original'``)
      on the substrate.
    * Else if ``dts`` is given and spans ≥ 2 full years → **3PW**
      (``mk_method='3pw'``); the 3PW call deseasons internally, so it is
      passed the raw ``y`` (not the substrate) to avoid double-deseasoning.
    * Else → **Hamed–Rao** (``mk_method='hamed_rao'``) on the substrate.

    ``force_variant`` overrides the ladder and is useful for smoke tests
    and for pinning the variant when the ACF-based choice is known to be
    marginal.

    The point slope is always ``theilslopes(y_arm).slope`` on raw ``y``
    (never the substrate) — Sen's slope is consistent under seasonality
    and AR(1), and stakeholders expect units of "change per step in raw y".
    """
    y_arr = _clean(y)
    n = int(y_arr.size)
    slope = float(theilslopes(y_arr, np.arange(n)).slope) if n >= 2 else float("nan")

    # ── Seasonality gate ───────────────────────────────────────────────────
    period_int = int(period) if period is not None else None
    deseasoned = period_int is not None and period_int > 1
    if deseasoned and period_int is not None:
        substrate = deseason(y_arr, period=period_int)
        substrate = substrate[np.isfinite(substrate)]
    else:
        substrate = y_arr

    # ── §4.2 AR ladder on substrate ────────────────────────────────────────
    rho = lag1_acf(substrate)
    rho_f = float(rho) if np.isfinite(rho) else 0.0
    span_yr = _dts_span_years(dts) if dts is not None else None
    if force_variant is not None:
        variant = force_variant
        reason = f"forced_variant={force_variant!r} (ladder bypassed)"
    elif abs(rho_f) <= acf_cutoff:
        variant = "original"
        reason = (
            f"ρ₁={rho_f:+.3f} → |ρ₁| ≤ acf_cutoff={acf_cutoff:.2f}, no AR(1) detected → Plain MK"
        )
    else:
        has_two_years = span_yr is not None and span_yr >= 2.0
        if has_two_years:
            variant = "3pw"
            reason = (
                f"AR(1) detected (ρ₁={rho_f:+.3f}, |ρ₁| > {acf_cutoff:.2f}) "
                f"and span={span_yr:.2f}yr ≥ 2yr → 3PW"
            )
        else:
            variant = "hamed_rao"
            span_txt = f"span={span_yr:.2f}yr" if span_yr is not None else "span=n/a (no dts)"
            reason = (
                f"AR(1) detected (ρ₁={rho_f:+.3f}, |ρ₁| > {acf_cutoff:.2f}) "
                f"but {span_txt} < 2yr → Hamed–Rao"
            )

    # ── Feed the variant its expected substrate ────────────────────────────
    # 3PW deseasons internally, so it must receive raw y. Plain / HR /
    # Yue–Wang / TFPW / PW all operate on whatever we hand them, so they
    # get the deseasoned residual (or raw y when no seasonality gate).
    dts_arr = None if dts is None else np.asarray(dts)
    if variant == "3pw":
        p, _, _ = _mk_pvalue(
            y_arr, mk_method=variant, alpha=alpha,
            period=period_int,
            dts=dts_arr, mk_kwargs=dict(mk_kwargs or {}),
        )
    else:
        p, _, _ = _mk_pvalue(
            substrate, mk_method=variant, alpha=alpha,
            period=period_int,
            dts=dts_arr, mk_kwargs=dict(mk_kwargs or {}),
        )
    return MKAdaptiveCoreArmResult(
        n=n,
        slope=slope,
        pvalue=float(p),
        mk_method=variant,
        deseasoned=deseasoned,
        rho1=rho_f,
        period=period_int,
        reason=reason,
        dts_span_years=span_yr,
    )


def mk_adaptive_core(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    *,
    period: int | None = None,
    dts_pre: Sequence | None = None,
    dts_post: Sequence | None = None,
    acf_cutoff: float = 0.20,
    force_variant: str | None = None,
    alpha: float = 0.05,
    mk_kwargs: dict | None = None,
) -> MKAdaptiveCoreResult:
    """Headline branch (§6) — adaptive MK on each arm + derived Δ point.

    Runs :func:`mk_adaptive_core_arm` independently on ``y_pre`` and ``y_post``
    (each arm may end up with a different MK variant if its AR regime
    differs) and returns the five required headline numbers:
    ``slope_pre``, ``slope_post``, ``pval_pre``, ``pval_post``, and
    ``delta_slope = slope_post − slope_pre``.

    No bootstrap. For a CI or p-value on Δ, run :func:`mk_adaptive_mbb`
    (or :func:`mk_delta_mbb` directly) as the optional add-on.
    """
    pre = mk_adaptive_core_arm(
        y_pre, period=period, dts=dts_pre, acf_cutoff=acf_cutoff,
        force_variant=force_variant, alpha=alpha, mk_kwargs=mk_kwargs,
    )
    post = mk_adaptive_core_arm(
        y_post, period=period, dts=dts_post, acf_cutoff=acf_cutoff,
        force_variant=force_variant, alpha=alpha, mk_kwargs=mk_kwargs,
    )
    delta = float(post.slope - pre.slope)
    if pre.slope != 0 and np.isfinite(pre.slope) and np.isfinite(post.slope):
        pct: float | None = float(post.slope / pre.slope - 1.0)
    else:
        pct = None
    return MKAdaptiveCoreResult(
        pre=pre,
        post=post,
        slope_pre=pre.slope,
        slope_post=post.slope,
        pval_pre=pre.pvalue,
        pval_post=post.pvalue,
        delta_slope=delta,
        pct_rate_change=pct,
        n_pre=pre.n,
        n_post=post.n,
    )


def mk_vbh(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    *,
    period: int,
    alpha: float = 0.05,
    hr_lag: int | None = 1,
) -> VbhBranchResult:
    """Optional VBH homogeneity branch (§6.1) — per-arm Seasonal MK + VBH χ².

    Runs :func:`seasonal_mk` (which computes Van Belle–Hughes χ²
    homogeneity as a byproduct) on each arm and exposes the pair of
    per-arm homogeneity p-values on a small result container. Use
    :attr:`VbhBranchResult.is_inhomogeneous` to decide whether to fall
    back to per-phase Δ reporting.
    """
    pre_smk = seasonal_mk(y_pre, period=int(period), alpha=alpha, hr_lag=hr_lag)
    post_smk = seasonal_mk(y_post, period=int(period), alpha=alpha, hr_lag=hr_lag)
    return VbhBranchResult(
        pre=pre_smk,
        post=post_smk,
        homogeneity_p_pre=float(pre_smk.homogeneity_p),
        homogeneity_p_post=float(post_smk.homogeneity_p),
        alpha=float(alpha),
    )


def mk_adaptive_mbb(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    *,
    mk_method: str = "hamed_rao",
    period: int | None = None,
    dts_pre: Sequence | None = None,
    dts_post: Sequence | None = None,
    confidence_level: float = 0.95,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int | None = 0,
    mk_kwargs: dict | None = None,
    progress: bool = True,
) -> "MbbDeltaResult":
    """Optional MBB branch (§6.1) — paired moving-block bootstrap for Δ inference.

    Thin alias for :func:`mk_delta_mbb` with ``ci_method='mbb'``. Produces
    ``delta_ci`` and ``delta_pvalue`` (Δ inference) plus the byproduct
    per-arm slope CIs ``slope_ci_pre`` / ``slope_ci_post``. Use when a CI
    or p-value on Δ is required on top of the headline point comparison
    from :func:`mk_adaptive_core`.
    """
    return mk_delta_mbb(
        y_pre, y_post,
        mk_method=mk_method,
        ci_method="mbb",
        period=period,
        dts_pre=dts_pre,
        dts_post=dts_post,
        confidence_level=confidence_level,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
        mk_kwargs=mk_kwargs,
        progress=progress,
    )


def mk_delta_mbb(
    y_pre: Sequence[float],
    y_post: Sequence[float],
    *,
    mk_method: str = "hamed_rao",
    ci_method: Literal["mbb", "gilbert"] = "mbb",
    period: int | None = None,
    dts_pre: Sequence | None = None,
    dts_post: Sequence | None = None,
    confidence_level: float = 0.95,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int | None = 0,
    mk_kwargs: dict | None = None,
    progress: bool = True,
) -> MbbDeltaResult:
    """Pre/post Sen-slope comparison — pluggable CI and MK p-value.

    Design invariants:

    * **Slope point** for each arm is ``theilslopes(y).slope`` on the raw
      series. Sen's slope is consistent under autocorrelation — AR(1) noise
      inflates the *variance* of the estimator, not its expectation, so no
      prewhitening is needed for the point estimate.
    * **Per-arm slope CI and Δ CI** come from the machinery selected by
      ``ci_method``:

      - ``"mbb"`` (default) — moving-block percentile bootstrap of
        ``theilslopes`` on **raw ``y``** with ``block_length = period`` by
        default. The block preserves short-range dependence automatically.
        Applicable with any ``mk_method``.
      - ``"gilbert"`` — reuse the Gilbert-style analytical CI already
        computed inside ``mannkendall.mk_temp_aggr``: half-widths from
        each arm's Gilbert CI are combined in quadrature (independence
        across disjoint pre/post arms) to form the Δ CI, and each per-arm
        CI is recentered on the raw-``y`` Sen slope so the reported
        centre matches ``slope_pre`` / ``slope_post``. Requires
        ``mk_method='3pw'``. Skips the bootstrap loop entirely.

    * **Δ p-value** (``delta_pvalue``) depends on the CI machinery:

      - ``ci_method='mbb'`` — two-sided **tail-area** of the paired-MBB
        ``{Δ_b}`` array at ``Δ = 0`` (Efron 1979), read off the same
        array that produced ``delta_ci``. No Wald inversion, no second
        loop.
      - ``ci_method='gilbert'`` — Wald inversion of ``delta_ci`` under a
        normal approximation (no bootstrap array to read from).

    * **Per-arm p-value** is dispatched by ``mk_method``. One of
      ``"original"``, ``"hamed_rao"`` (default), ``"yue_wang"``,
      ``"tfpw"``, ``"pw"``, ``"seasonal"``, ``"3pw"``. Method-specific
      extras (``hr_lag`` for HR / YW, ``resolution`` for 3PW) travel
      through ``mk_kwargs``.
    * **Homogeneity p-value** (Van Belle-Hughes χ²) is populated only when
      ``mk_method`` is ``"seasonal"`` or ``"3pw"``.

    Prewhitening is the caller's responsibility. To use HR / seasonal on an
    already-whitened series, pass ``mk_method="original"`` with the whitened
    ``y``. Slope and CI never operate on whitened input.

    Parameters
    ----------
    y_pre, y_post : array-like of float
        Values for the pre and post arms.
    mk_method : {'original', 'hamed_rao', 'yue_wang', 'tfpw', 'pw', 'seasonal', '3pw'}
        Which MK variant supplies the per-arm p-value. Default ``'hamed_rao'``.
    ci_method : {'mbb', 'gilbert'}, default ``'mbb'``
        Which CI machinery supplies ``slope_ci_*`` and ``delta_ci``.
        ``'gilbert'`` is only valid together with ``mk_method='3pw'`` and
        makes the CI internally consistent with the 3PW p-value (both come
        from ``mannkendall.mk_temp_aggr``); ``'mbb'`` is method-agnostic.
    period : int, optional
        Seasonal cycle length; required by ``mk_method`` in
        ``{'seasonal', '3pw'}``. Also used as the default ``block_length``.
    dts_pre, dts_post : array-like of datetime, optional
        Timestamps for the pre / post arms; required only when
        ``mk_method='3pw'`` (``mannkendall.mk_temp_aggr`` groups by year).
    confidence_level : float, default 0.95
        Percentile level for the bootstrap CIs (``ci_method='mbb'``) or
        the Gilbert CI level passed through to ``mk_temp_aggr``
        (``ci_method='gilbert'``).
    n_boot : int, default 2000
        Number of bootstrap resamples. Ignored when ``ci_method='gilbert'``.
    block_length : int, optional
        Moving-block length. Defaults to ``period`` (or ``1`` when ``period``
        is ``None`` — degenerates to iid resampling). Clamped to
        ``min(n_pre, n_post)``. Ignored when ``ci_method='gilbert'``.
    seed : int or None, default 0
        Seed for :class:`numpy.random.default_rng`. Ignored when
        ``ci_method='gilbert'``.
    mk_kwargs : dict, optional
        Extra kwargs forwarded to the ``mk_method`` handler.
    progress : bool, default True
        Show a tqdm bar over the bootstrap loop. Ignored when
        ``ci_method='gilbert'``.
    """
    from tqdm.auto import tqdm

    if mk_method not in _MK_METHODS:
        raise ValueError(
            f"unknown mk_method={mk_method!r}; expected one of {list(_MK_METHODS)}"
        )
    if ci_method not in ("mbb", "gilbert"):
        raise ValueError(
            f"unknown ci_method={ci_method!r}; expected 'mbb' or 'gilbert'"
        )
    if ci_method == "gilbert" and mk_method != "3pw":
        raise ValueError(
            "ci_method='gilbert' requires mk_method='3pw' — the Gilbert bounds "
            "are only produced by mannkendall.mk_temp_aggr. Use ci_method='mbb' "
            f"with mk_method={mk_method!r}."
        )
    mk_kwargs = dict(mk_kwargs or {})

    y_pre_arr = _clean(y_pre)
    y_post_arr = _clean(y_post)
    n_pre, n_post = len(y_pre_arr), len(y_post_arr)

    slope_pre = float(theilslopes(y_pre_arr, np.arange(n_pre)).slope)
    slope_post = float(theilslopes(y_post_arr, np.arange(n_post)).slope)

    alpha = 1.0 - confidence_level
    dts_pre_arr = None if dts_pre is None else np.asarray(dts_pre)
    dts_post_arr = None if dts_post is None else np.asarray(dts_post)
    pval_pre, hp_pre, mk3pw_pre = _mk_pvalue(
        y_pre_arr, mk_method=mk_method, alpha=alpha, period=period,
        dts=dts_pre_arr, mk_kwargs=mk_kwargs,
    )
    pval_post, hp_post, mk3pw_post = _mk_pvalue(
        y_post_arr, mk_method=mk_method, alpha=alpha, period=period,
        dts=dts_post_arr, mk_kwargs=mk_kwargs,
    )

    # ── Fast path: Gilbert CI (no bootstrap) ────────────────────────────────
    if ci_method == "gilbert":
        # _mk_pvalue guarantees mk3pw_* are populated when we reach here
        # (mk_method='3pw' enforced above).
        assert mk3pw_pre is not None and mk3pw_post is not None
        # mannkendall.mk_temp_aggr returns slope + Gilbert CI in **per-year**
        # units by convention (independent of the ``resolution`` kwarg, which
        # only controls tie detection). theilslopes on the raw daily arrays
        # returns per-step slopes. Convert Gilbert half-widths from per-year
        # to per-step using the median timestep of each arm.
        def _step_years(dts_arr: np.ndarray) -> float:
            dts64 = np.asarray(dts_arr, dtype="datetime64[ns]")
            deltas = np.diff(dts64).astype("timedelta64[s]").astype(np.float64)
            median_step_s = float(np.median(deltas))
            return median_step_s / (365.25 * 86400.0)

        step_pre = _step_years(dts_pre_arr)
        step_post = _step_years(dts_post_arr)
        hw_pre = 0.5 * (mk3pw_pre.slope_ci[1] - mk3pw_pre.slope_ci[0]) * step_pre
        hw_post = 0.5 * (mk3pw_post.slope_ci[1] - mk3pw_post.slope_ci[0]) * step_post
        # Recenter Gilbert half-widths on the raw-y Sen slope so
        # slope_ci_pre truly brackets slope_pre. mannkendall's Sen slope is
        # computed on the VCTFPW-whitened deseasoned residual; the two Sen
        # estimators track each other on the daily arms this pipeline
        # targets, but they are not identical, so we quote widths not raw
        # bounds.
        slope_ci_pre = (slope_pre - hw_pre, slope_pre + hw_pre)
        slope_ci_post = (slope_post - hw_post, slope_post + hw_post)
        hw_delta = float(np.hypot(hw_pre, hw_post))
        delta = slope_post - slope_pre
        delta_ci = (delta - hw_delta, delta + hw_delta)
        # Percentage change via first-order propagation. Skip when either
        # half-width is not finite or slope_pre is zero.
        if slope_pre != 0 and np.isfinite(hw_pre) and np.isfinite(hw_post):
            pct_change: float | None = float(slope_post / slope_pre - 1.0)
            hw_pct = float(np.hypot(hw_post / slope_pre, slope_post * hw_pre / slope_pre**2))
            pct_ci: tuple[float, float] | None = (pct_change - hw_pct, pct_change + hw_pct)
        else:
            pct_change = None
            pct_ci = None
        interp = _compute_delta_interpretation(float(delta), delta_ci, confidence_level)
        return MbbDeltaResult(
            slope_pre=slope_pre,
            slope_post=slope_post,
            slope_ci_pre=slope_ci_pre,
            slope_ci_post=slope_ci_post,
            delta_slope=float(delta),
            delta_ci=delta_ci,
            pct_rate_change=pct_change,
            pct_ci=pct_ci,
            pval_pre=pval_pre,
            pval_post=pval_post,
            n_pre=n_pre,
            n_post=n_post,
            method=f"gilbert(mk={mk_method}, cl={int(round(100 * confidence_level))}%)",
            homogeneity_p_pre=hp_pre,
            homogeneity_p_post=hp_post,
            **interp,
        )

    # ── Moving-block bootstrap on Δ = slope_post − slope_pre ────────────────
    bl_requested = block_length if block_length is not None else (period or 1)
    bl = max(1, min(int(bl_requested), min(n_pre, n_post)))
    rng = np.random.default_rng(seed)
    slopes_pre_b = np.empty(n_boot)
    slopes_post_b = np.empty(n_boot)
    delta_b = np.empty(n_boot)
    pct_b = np.empty(n_boot)
    x_pre = np.arange(n_pre)
    x_post = np.arange(n_post)
    iterator = tqdm(
        range(n_boot),
        desc=f"mbb Δslope (bl={bl})",
        disable=not progress,
        leave=False,
    )
    for b in iterator:
        idx_pre = _mbb_indices(n_pre, bl, rng)
        idx_post = _mbb_indices(n_post, bl, rng)
        sp = float(theilslopes(y_pre_arr[idx_pre], x_pre).slope)
        sq = float(theilslopes(y_post_arr[idx_post], x_post).slope)
        slopes_pre_b[b] = sp
        slopes_post_b[b] = sq
        delta_b[b] = sq - sp
        pct_b[b] = (sq / sp - 1.0) if sp != 0 else np.nan

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    slope_ci_pre = (
        float(np.percentile(slopes_pre_b, lo_q)),
        float(np.percentile(slopes_pre_b, hi_q)),
    )
    slope_ci_post = (
        float(np.percentile(slopes_post_b, lo_q)),
        float(np.percentile(slopes_post_b, hi_q)),
    )
    delta_ci = (
        float(np.percentile(delta_b, lo_q)),
        float(np.percentile(delta_b, hi_q)),
    )
    finite_pct = pct_b[np.isfinite(pct_b)]
    if finite_pct.size and slope_pre != 0:
        pct_change: float | None = float(slope_post / slope_pre - 1.0)
        pct_ci: tuple[float, float] | None = (
            float(np.percentile(finite_pct, lo_q)),
            float(np.percentile(finite_pct, hi_q)),
        )
    else:
        pct_change = None
        pct_ci = None

    delta_point = float(slope_post - slope_pre)
    interp = _compute_delta_interpretation(
        delta_point, delta_ci, confidence_level, delta_boot=delta_b,
    )
    return MbbDeltaResult(
        slope_pre=slope_pre,
        slope_post=slope_post,
        slope_ci_pre=slope_ci_pre,
        slope_ci_post=slope_ci_post,
        delta_slope=delta_point,
        delta_ci=delta_ci,
        pct_rate_change=pct_change,
        pct_ci=pct_ci,
        pval_pre=pval_pre,
        pval_post=pval_post,
        n_pre=n_pre,
        n_post=n_post,
        method=f"mbb(mk={mk_method}, bl={bl}, n={n_boot})",
        homogeneity_p_pre=hp_pre,
        homogeneity_p_post=hp_post,
        **interp,
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
    inside :func:`mk_delta_mbb` — the pooled Δ hides sign cancellation
    across phases, so we quote one Δ per day-of-week. Uses
    :func:`sen_slope_ci` (theilslopes, iid) inside each phase; treat this
    as a diagnostic table, not an autocorrelation-aware estimate.

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
    "MbbDeltaResult",
    "RegionalHomogeneityResult",
    "AcfRegime",
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
    "tfpw_y",
    "classify_acf_regime",
    "mk_asymptotic_pvalue",
    "mk_asymptotic_power",
    "mk_power_curve",
    "MKPowerCurve",
    "mk_power_of_test",
    "MKPowerOfTest",
    # seasonal / partial / correlated
    "seasonal_mk",
    "vbh_chi2_decomposition",
    "correlated_seasonal_mk",
    "partial_mk",
    # paired
    "mk_delta_mbb",
    "per_day_delta_slopes",
    "sen_fit_line",
    "mk_result_line",
    "pre_post_sen_figure",
    "mk_figure",
    "seasonal_mk_figure",
    "mk_adaptive_sen_figure",
    "mk_power_curve_figure",
    "mk_power_of_test_figure",
    "intervention_summary_row",
    # batch
    "aggregate_by_period",
    "mk_by_group",
    # regional
    "regional_homogeneity",
]
