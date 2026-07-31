"""Bayesian two-sample estimation (BEST + ROPE).

Thin wrapper around Kruschke's *Bayesian Estimation Supersedes the T-Test*
(BEST, Kruschke 2013) implemented in PyMC, plus a Region-of-Practical-
Equivalence (ROPE) decision rule for translating the posterior of the
mean difference into a categorical verdict.

The module is cfg-free — inputs are primitive ``numpy`` arrays / ``pandas``
Series, outputs are ``@dataclass(frozen=True)`` results and DataFrames.
It is general-purpose: any two continuous samples, not TS-specific. The
pre/post application in ``oldest-newest-prepost`` is one caller among
many.

Heavy dependencies (``pymc``, ``arviz``) are guarded via the shared
``optional_import`` helper — importing this module without them succeeds;
calling a function without them raises with a
``pip install 'dstoolbox[bayes]'`` hint.

References
----------
- Kruschke, J. K. (2013). Bayesian estimation supersedes the t-test.
  *Journal of Experimental Psychology: General*, 142(2), 573.
- https://www.pymc.io/projects/examples/en/latest/case_studies/BEST.html
- https://best.readthedocs.io — R package producing the canonical report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._base_import_warning import optional_import

_pm = optional_import("pymc", "stat_bayes")
_az = optional_import("arviz", "stat_bayes")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BestResult:
    """Output of a single BEST fit.

    Attributes
    ----------
    trace
        The full :class:`arviz.InferenceData` object (posterior + sample_stats).
    posterior_delta
        1-D flattened posterior draws of ``delta = mu_post - mu_pre``.
    posterior_mean_delta
        Posterior mean of ``delta``.
    hdi
        ``(low, high)`` highest-density interval on ``delta`` at ``hdi_prob``.
    n_pre, n_post
        Sample sizes fed to the fit.
    prior_spec
        Which prior set was used (``"kruschke"`` or ``"weakly_informative"``).
    """

    trace: object  # arviz.InferenceData (avoid hard type import)
    posterior_delta: np.ndarray
    posterior_mean_delta: float
    hdi: tuple[float, float]
    n_pre: int
    n_post: int
    prior_spec: str


@dataclass(frozen=True)
class RopeDecision:
    """Verdict from applying a ROPE to a posterior of the mean difference.

    ``decision`` is one of:

    - ``"meaningful_positive"`` — ``P(delta > rope_high) >= threshold``.
    - ``"meaningful_negative"`` — ``P(delta < rope_low)  >= threshold``.
    - ``"equivalent"``           — ``P(delta in ROPE)    >= threshold``.
    - ``"inconclusive"``         — none of the above.
    """

    rope_low: float
    rope_high: float
    prob_gt_high: float
    prob_in_rope: float
    prob_lt_low: float
    decision: str


@dataclass(frozen=True)
class BetaBinomialResult:
    """Output of a single Beta-Binomial two-sample fit.

    The canonical conjugate model for comparing two conversion rates:
    ``p ~ Beta(alpha, beta)`` prior, ``k ~ Binomial(n, p)`` likelihood,
    posterior ``Beta(alpha + k, beta + n - k)``. See
    https://en.wikipedia.org/wiki/Bayes_estimator §"Estimating p in a
    binomial distribution".

    Attributes
    ----------
    trace
        Full :class:`arviz.InferenceData` object (posterior + sample_stats).
    posterior_p_pre, posterior_p_post
        1-D flattened posterior draws of the rate parameters.
    posterior_delta
        1-D flattened posterior draws of ``delta = p_post - p_pre``.
    posterior_mean_delta
        Posterior mean of ``delta``.
    hdi
        ``(low, high)`` highest-density interval on ``delta`` at ``hdi_prob``.
    successes_pre, trials_pre, successes_post, trials_post
        The four numbers fed to the fit.
    prior_spec
        Which prior set was used (``"uniform"`` = Beta(1,1) or
        ``"jeffreys"`` = Beta(0.5, 0.5)).
    """

    trace: object  # arviz.InferenceData
    posterior_p_pre: np.ndarray
    posterior_p_post: np.ndarray
    posterior_delta: np.ndarray
    posterior_mean_delta: float
    hdi: tuple[float, float]
    successes_pre: int
    trials_pre: int
    successes_post: int
    trials_post: int
    prior_spec: str

    @property
    def rate_pre(self) -> float:
        """Empirical pre-rate ``successes_pre / trials_pre``."""
        return self.successes_pre / self.trials_pre

    @property
    def rate_post(self) -> float:
        """Empirical post-rate ``successes_post / trials_post``."""
        return self.successes_post / self.trials_post


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def _kruschke_priors(pooled_mean: float, pooled_sd: float) -> dict[str, object]:
    """Canonical Kruschke (2013) priors — very wide, weakly informed by data."""
    return {
        "mu_prior_mean": pooled_mean,
        "mu_prior_sd": pooled_sd * 1000.0,
        "sigma_low": pooled_sd / 1000.0,
        "sigma_high": pooled_sd * 1000.0,
        "nu_mean": 29.0,
    }


def _weakly_informative_priors(pooled_mean: float, pooled_sd: float) -> dict[str, object]:
    """Tighter, weakly-informative alternative for prior-sensitivity checks."""
    return {
        "mu_prior_mean": pooled_mean,
        "mu_prior_sd": pooled_sd * 3.0,
        "sigma_scale": pooled_sd * 2.0,
        "nu_alpha": 2.0,
        "nu_beta": 0.1,
    }


def _build_model(y_pre: np.ndarray, y_post: np.ndarray, prior: str):
    """Return a ``pymc.Model`` implementing BEST with the requested prior set."""
    pooled_mean = float(np.mean(np.concatenate([y_pre, y_post])))
    pooled_sd = float(np.std(np.concatenate([y_pre, y_post]), ddof=1))
    if pooled_sd == 0.0:
        raise ValueError("pooled_sd is zero; cannot fit BEST on constant data.")

    with _pm.Model() as model:  # type: ignore[union-attr]
        if prior == "kruschke":
            p = _kruschke_priors(pooled_mean, pooled_sd)
            mu_pre = _pm.Normal("mu_pre", mu=p["mu_prior_mean"], sigma=p["mu_prior_sd"])
            mu_post = _pm.Normal("mu_post", mu=p["mu_prior_mean"], sigma=p["mu_prior_sd"])
            sigma_pre = _pm.Uniform("sigma_pre", lower=p["sigma_low"], upper=p["sigma_high"])
            sigma_post = _pm.Uniform("sigma_post", lower=p["sigma_low"], upper=p["sigma_high"])
            nu_minus_one = _pm.Exponential("nu_minus_one", 1.0 / p["nu_mean"])
            nu = _pm.Deterministic("nu", nu_minus_one + 1.0)
        elif prior == "weakly_informative":
            p = _weakly_informative_priors(pooled_mean, pooled_sd)
            mu_pre = _pm.Normal("mu_pre", mu=p["mu_prior_mean"], sigma=p["mu_prior_sd"])
            mu_post = _pm.Normal("mu_post", mu=p["mu_prior_mean"], sigma=p["mu_prior_sd"])
            sigma_pre = _pm.HalfNormal("sigma_pre", sigma=p["sigma_scale"])
            sigma_post = _pm.HalfNormal("sigma_post", sigma=p["sigma_scale"])
            nu = _pm.Gamma("nu", alpha=p["nu_alpha"], beta=p["nu_beta"])
        else:
            raise ValueError(
                f"Unknown prior spec: {prior!r}. Use 'kruschke' or 'weakly_informative'."
            )

        _pm.StudentT("y_pre", nu=nu, mu=mu_pre, sigma=sigma_pre, observed=y_pre)
        _pm.StudentT("y_post", nu=nu, mu=mu_post, sigma=sigma_post, observed=y_post)
        _pm.Deterministic("delta", mu_post - mu_pre)
        _pm.Deterministic("delta_sigma", sigma_post - sigma_pre)
        pooled_sigma = (sigma_pre + sigma_post) / 2.0
        _pm.Deterministic("effect_size", (mu_post - mu_pre) / pooled_sigma)

    return model


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def best_two_sample(
    y_pre,
    y_post,
    *,
    prior: str = "kruschke",
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int | None = None,
    progressbar: bool = False,
) -> BestResult:
    """Fit BEST (Kruschke 2013) to two continuous samples.

    Parameters
    ----------
    y_pre, y_post
        1-D numeric arrays of the two samples. Do NOT need to be the same length.
    prior
        ``"kruschke"`` (canonical, very wide) or ``"weakly_informative"``.
    hdi_prob
        Coverage of the returned HDI on ``delta``.
    draws, tune, chains, target_accept
        Forwarded to :func:`pymc.sample`.
    random_seed
        Seed for reproducibility. Fixes both PyMC and NumPy RNG paths.
    progressbar
        Show PyMC's sampling progressbar. Defaults to off (notebook-friendly).

    Returns
    -------
    BestResult
        Dataclass with the trace and the summarised posterior on ``delta``.
    """
    y_pre_arr = np.asarray(y_pre, dtype=float).ravel()
    y_post_arr = np.asarray(y_post, dtype=float).ravel()
    if y_pre_arr.size < 2 or y_post_arr.size < 2:
        raise ValueError(
            f"Need >=2 observations per group; got n_pre={y_pre_arr.size}, "
            f"n_post={y_post_arr.size}."
        )

    model = _build_model(y_pre_arr, y_post_arr, prior)
    with model:
        trace = _pm.sample(  # type: ignore[union-attr]
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    delta_samples = np.asarray(trace.posterior["delta"]).ravel()  # type: ignore[attr-defined]
    hdi_arr = _az.hdi(delta_samples, hdi_prob=hdi_prob)  # type: ignore[union-attr]
    hdi_arr = np.asarray(hdi_arr).ravel()
    return BestResult(
        trace=trace,
        posterior_delta=delta_samples,
        posterior_mean_delta=float(delta_samples.mean()),
        hdi=(float(hdi_arr[0]), float(hdi_arr[1])),
        n_pre=int(y_pre_arr.size),
        n_post=int(y_post_arr.size),
        prior_spec=prior,
    )


# ---------------------------------------------------------------------------
# Beta-Binomial (canonical two-proportion model)
# ---------------------------------------------------------------------------
#
# Primary tool when the outcome is a binary trial (converted / not converted)
# rather than a continuous measurement. Matches:
# - Wikipedia §"Estimating p in a binomial distribution" — conjugate Beta prior.
# - Kruschke (2015) *Doing Bayesian Data Analysis* Ch. 8 (two-proportion model).
# - Every A/B testing library (VWO, Google Analytics, PyMC A/B examples).
#
# Advantages over feeding daily aggregates to BEST:
#   1. Posterior is on the *rate* difference p_post - p_pre — the causal target
#      of a reranker / editorial change, not a count that confounds rate and
#      traffic volume.
#   2. Sample-size weighting is automatic (a heavier-traffic day contributes
#      proportionally more information).
#   3. Correct likelihood family for a binary outcome; no CLT approximation.

_BB_PRIOR_ALPHA_BETA: dict[str, tuple[float, float]] = {
    "uniform":  (1.0, 1.0),   # Beta(1, 1) — Bayes-Laplace flat prior
    "jeffreys": (0.5, 0.5),   # Beta(1/2, 1/2) — Jeffreys reference prior
}


def _build_beta_binomial_model(
    successes_pre: int,
    trials_pre: int,
    successes_post: int,
    trials_post: int,
    prior: str,
):
    """Return a ``pymc.Model`` implementing the two-proportion Beta-Binomial."""
    try:
        alpha, beta = _BB_PRIOR_ALPHA_BETA[prior]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prior spec: {prior!r}. Use 'uniform' or 'jeffreys'."
        ) from exc

    with _pm.Model() as model:  # type: ignore[union-attr]
        p_pre = _pm.Beta("p_pre", alpha=alpha, beta=beta)
        p_post = _pm.Beta("p_post", alpha=alpha, beta=beta)
        _pm.Binomial("obs_pre",  n=trials_pre,  p=p_pre,  observed=successes_pre)
        _pm.Binomial("obs_post", n=trials_post, p=p_post, observed=successes_post)
        _pm.Deterministic("delta", p_post - p_pre)
        _pm.Deterministic("rel_lift", (p_post - p_pre) / p_pre)

    return model


def beta_binomial_two_sample(
    successes_pre,
    trials_pre,
    successes_post,
    trials_post,
    *,
    prior: str = "uniform",
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    progressbar: bool = False,
) -> BetaBinomialResult:
    """Fit the two-proportion Beta-Binomial model on totals.

    Parameters
    ----------
    successes_pre, trials_pre
        Total converting searches / total searches in the pre-window.
    successes_post, trials_post
        Same for the post-window.
    prior
        ``"uniform"`` (Beta(1, 1), Bayes-Laplace) or ``"jeffreys"``
        (Beta(0.5, 0.5), reference prior).
    hdi_prob
        Coverage of the returned HDI on ``delta = p_post - p_pre``.
    draws, tune, chains, target_accept, random_seed, progressbar
        Forwarded to :func:`pymc.sample`. Conjugate posteriors sample cheaply.

    Returns
    -------
    BetaBinomialResult
    """
    s_pre = int(successes_pre)
    n_pre = int(trials_pre)
    s_post = int(successes_post)
    n_post = int(trials_post)
    for name, s, n in (("pre", s_pre, n_pre), ("post", s_post, n_post)):
        if n <= 0:
            raise ValueError(f"trials_{name} must be positive; got {n}.")
        if s < 0 or s > n:
            raise ValueError(
                f"successes_{name} must be in [0, trials_{name}]; got "
                f"successes={s}, trials={n}."
            )

    model = _build_beta_binomial_model(s_pre, n_pre, s_post, n_post, prior)
    with model:
        trace = _pm.sample(  # type: ignore[union-attr]
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    delta_samples = np.asarray(trace.posterior["delta"]).ravel()          # type: ignore[attr-defined]
    p_pre_samples = np.asarray(trace.posterior["p_pre"]).ravel()          # type: ignore[attr-defined]
    p_post_samples = np.asarray(trace.posterior["p_post"]).ravel()        # type: ignore[attr-defined]
    hdi_arr = _az.hdi(delta_samples, hdi_prob=hdi_prob)                   # type: ignore[union-attr]
    hdi_arr = np.asarray(hdi_arr).ravel()
    return BetaBinomialResult(
        trace=trace,
        posterior_p_pre=p_pre_samples,
        posterior_p_post=p_post_samples,
        posterior_delta=delta_samples,
        posterior_mean_delta=float(delta_samples.mean()),
        hdi=(float(hdi_arr[0]), float(hdi_arr[1])),
        successes_pre=s_pre,
        trials_pre=n_pre,
        successes_post=s_post,
        trials_post=n_post,
        prior_spec=prior,
    )


# ---------------------------------------------------------------------------
# ROPE decision
# ---------------------------------------------------------------------------

def _classify(
    prob_gt_high: float,
    prob_in_rope: float,
    prob_lt_low: float,
    *,
    threshold: float = 0.95,
) -> str:
    """Bucket a posterior into one of the four decision labels."""
    if prob_gt_high >= threshold:
        return "meaningful_positive"
    if prob_lt_low >= threshold:
        return "meaningful_negative"
    if prob_in_rope >= threshold:
        return "equivalent"
    return "inconclusive"


def rope_decision(
    posterior_delta,
    *,
    rope_low: float,
    rope_high: float,
    threshold: float = 0.95,
) -> RopeDecision:
    """Apply a ROPE decision rule to a posterior of the mean difference.

    Parameters
    ----------
    posterior_delta
        1-D array of posterior draws of ``delta = mu_post - mu_pre``.
    rope_low, rope_high
        ROPE bounds. ``rope_low < rope_high`` required.
    threshold
        Posterior mass needed in a region to trigger a meaningful/equivalent
        verdict. Kruschke's default is ``0.95``.

    Returns
    -------
    RopeDecision
    """
    if not rope_low < rope_high:
        raise ValueError(f"rope_low ({rope_low}) must be < rope_high ({rope_high}).")
    samples = np.asarray(posterior_delta, dtype=float).ravel()
    prob_gt_high = float((samples > rope_high).mean())
    prob_lt_low = float((samples < rope_low).mean())
    prob_in_rope = float(((samples >= rope_low) & (samples <= rope_high)).mean())
    return RopeDecision(
        rope_low=float(rope_low),
        rope_high=float(rope_high),
        prob_gt_high=prob_gt_high,
        prob_in_rope=prob_in_rope,
        prob_lt_low=prob_lt_low,
        decision=_classify(prob_gt_high, prob_in_rope, prob_lt_low, threshold=threshold),
    )


def rope_comparison_table(
    posterior_delta,
    *,
    ropes: dict[str, tuple[float | None, float | None]],
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Apply several ROPEs to the same posterior; return one row per ROPE.

    ROPEs with ``(None, None)`` bounds are emitted as an ``"undefined"`` row
    so callers can carry a TODO business threshold through the pipeline
    without a branch.
    """
    rows = []
    for name, bounds in ropes.items():
        lo, hi = bounds
        if lo is None or hi is None:
            rows.append({
                "rope_low": np.nan,
                "rope_high": np.nan,
                "prob_gt_high": np.nan,
                "prob_in_rope": np.nan,
                "prob_lt_low": np.nan,
                "decision": "undefined",
            })
            continue
        dec = rope_decision(
            posterior_delta, rope_low=lo, rope_high=hi, threshold=threshold,
        )
        rows.append({
            "rope_low": dec.rope_low,
            "rope_high": dec.rope_high,
            "prob_gt_high": dec.prob_gt_high,
            "prob_in_rope": dec.prob_in_rope,
            "prob_lt_low": dec.prob_lt_low,
            "decision": dec.decision,
        })
    return pd.DataFrame(rows, index=pd.Index(list(ropes.keys()), name="rope"))


# ---------------------------------------------------------------------------
# Prior sensitivity
# ---------------------------------------------------------------------------

def prior_sensitivity(
    y_pre,
    y_post,
    *,
    priors: tuple[str, ...] = ("kruschke", "weakly_informative"),
    hdi_prob: float = 0.95,
    **sample_kwargs,
) -> tuple[dict[str, BestResult], pd.DataFrame]:
    """Fit BEST under multiple prior specs; return per-fit results + shift table.

    The shift table quantifies how far each alternative's ``delta`` posterior
    median moves relative to the *first* prior in ``priors`` (the primary
    spec).
    """
    if not priors:
        raise ValueError("`priors` must contain at least one prior spec.")
    results: dict[str, BestResult] = {}
    for name in priors:
        results[name] = best_two_sample(
            y_pre, y_post, prior=name, hdi_prob=hdi_prob, **sample_kwargs,
        )
    primary_name = priors[0]
    primary_mean = results[primary_name].posterior_mean_delta
    rows = []
    for name, res in results.items():
        rows.append({
            "prior": name,
            "mean_delta": res.posterior_mean_delta,
            "hdi_low": res.hdi[0],
            "hdi_high": res.hdi[1],
            "shift_from_primary": res.posterior_mean_delta - primary_mean,
        })
    return results, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
#
# Bayesian plots live here (not in ``intervention_plots``) because they wrap
# ``arviz.plot_posterior`` (matplotlib) whereas ``intervention_plots`` is a
# plotly module. Mixing backends in one file would be a smell. All plots
# are guarded by the same ``[bayes]`` extras that gate the fit primitives.


def _save_fig(fig: Figure, out_path: Path | str | None) -> None:
    if out_path is None:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=150)


def plot_kruschke_report(
    result: BestResult,
    *,
    y_pre,
    y_post,
    out_path: Path | str | None = None,
) -> Figure:
    """Render the canonical Kruschke 9-panel report for a BEST fit.

    Panels: posteriors of ``mu_pre, mu_post, sigma_pre, sigma_post, nu,
    delta, delta_sigma, effect_size`` plus a data + posterior-mean overlay.
    """
    trace = result.trace
    var_names = [
        "mu_pre", "mu_post", "sigma_pre", "sigma_post",
        "nu", "delta", "delta_sigma", "effect_size",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes_flat = axes.ravel()
    for ax, var in zip(axes_flat[:8], var_names):
        _az.plot_posterior(trace, var_names=[var], ax=ax)  # type: ignore[union-attr]
        ax.set_title(var)

    # Overlay: raw data histograms + posterior-mean μ per group
    ax_data = axes_flat[8]
    y_pre_arr = np.asarray(y_pre, dtype=float).ravel()
    y_post_arr = np.asarray(y_post, dtype=float).ravel()
    ax_data.hist(y_pre_arr, bins=20, alpha=0.5, label=f"pre (n={result.n_pre})", color="#4c72b0")
    ax_data.hist(y_post_arr, bins=20, alpha=0.5, label=f"post (n={result.n_post})", color="#dd8452")
    mu_pre_mean = float(np.asarray(trace.posterior["mu_pre"]).mean())  # type: ignore[attr-defined]
    mu_post_mean = float(np.asarray(trace.posterior["mu_post"]).mean())  # type: ignore[attr-defined]
    ax_data.axvline(mu_pre_mean, color="#4c72b0", linestyle="--", label=f"μ_pre ≈ {mu_pre_mean:.2f}")
    ax_data.axvline(mu_post_mean, color="#dd8452", linestyle="--", label=f"μ_post ≈ {mu_post_mean:.2f}")
    ax_data.set_title("Data + posterior means")
    ax_data.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"BEST — {result.prior_spec} priors  |  n_pre={result.n_pre}, n_post={result.n_post}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_fig(fig, out_path)
    return fig


def plot_rope_decision(
    posterior_delta,
    *,
    rope: tuple[float, float],
    ref_val: float = 0.0,
    hdi_prob: float = 0.95,
    out_path: Path | str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Thin wrapper over ``az.plot_posterior(..., rope=..., ref_val=...)``."""
    samples = np.asarray(posterior_delta, dtype=float).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure  # type: ignore[assignment]
    _az.plot_posterior(  # type: ignore[union-attr]
        samples, rope=list(rope), ref_val=ref_val, hdi_prob=hdi_prob, ax=ax,
    )
    ax.set_title(f"Posterior of δ  |  ROPE = [{rope[0]:.3g}, {rope[1]:.3g}]")
    _save_fig(fig, out_path)
    return fig


def plot_prior_sensitivity(
    results: dict[str, BestResult],
    *,
    out_path: Path | str | None = None,
) -> Figure:
    """Overlay δ posterior densities from multiple prior specs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res in results.items():
        _az.plot_kde(  # type: ignore[union-attr]
            res.posterior_delta,
            label=f"{name}  (mean={res.posterior_mean_delta:.3g})",
            ax=ax,
        )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("δ = μ_post − μ_pre")
    ax.set_ylabel("posterior density")
    ax.set_title("Prior sensitivity — δ posterior across priors")
    ax.legend(loc="best", fontsize=9)
    _save_fig(fig, out_path)
    return fig


def plot_beta_binomial_report(
    result: BetaBinomialResult,
    *,
    rope: tuple[float, float] | None = None,
    out_path: Path | str | None = None,
) -> Figure:
    """Render a 4-panel report for a Beta-Binomial two-proportion fit.

    Panels
    ------
    (0, 0) Posterior of ``p_pre``.
    (0, 1) Posterior of ``p_post``.
    (1, 0) Posterior of ``delta = p_post - p_pre`` with optional ROPE overlay.
    (1, 1) Posterior of ``rel_lift = (p_post - p_pre) / p_pre`` (percent lift).
    """
    trace = result.trace
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _az.plot_posterior(trace, var_names=["p_pre"],  ax=axes[0, 0])   # type: ignore[union-attr]
    axes[0, 0].set_title(f"p_pre  (empirical = {result.rate_pre:.4f})")

    _az.plot_posterior(trace, var_names=["p_post"], ax=axes[0, 1])   # type: ignore[union-attr]
    axes[0, 1].set_title(f"p_post  (empirical = {result.rate_post:.4f})")

    _plot_kw: dict[str, object] = {"var_names": ["delta"], "ax": axes[1, 0], "ref_val": 0.0}
    if rope is not None:
        _plot_kw["rope"] = list(rope)
    _az.plot_posterior(trace, **_plot_kw)                            # type: ignore[union-attr]
    axes[1, 0].set_title("δ = p_post − p_pre")

    _az.plot_posterior(trace, var_names=["rel_lift"], ax=axes[1, 1], ref_val=0.0)  # type: ignore[union-attr]
    axes[1, 1].set_title("relative lift = δ / p_pre")

    fig.suptitle(
        f"Beta-Binomial — {result.prior_spec} prior  |  "
        f"pre: {result.successes_pre}/{result.trials_pre},  "
        f"post: {result.successes_post}/{result.trials_post}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_path)
    return fig
