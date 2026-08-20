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

from collections.abc import Sequence
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
        Name of the prior used: ``"uniform"`` = Beta(1,1),
        ``"jeffreys"`` = Beta(0.5, 0.5), or the ``name`` of a custom
        :class:`BetaPrior`.
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


@dataclass(frozen=True)
class BetaPrior:
    """A named ``Beta(alpha, beta)`` prior on a rate parameter.

    Carries the label alongside the two shape parameters so plots and
    result tables can report *which* prior produced a posterior rather
    than echoing raw numbers.

    Attributes
    ----------
    name
        Label surfaced in ``BetaBinomialResult.prior_spec``, plot titles
        and shift tables (e.g. ``"informative"``).
    alpha, beta
        Shape parameters. Both must be strictly positive.
    """

    name: str
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BetaPrior.name must be a non-empty string.")
        if self.alpha <= 0:
            raise ValueError(f"BetaPrior.alpha must be > 0; got {self.alpha}.")
        if self.beta <= 0:
            raise ValueError(f"BetaPrior.beta must be > 0; got {self.beta}.")

    @property
    def weight(self) -> float:
        """Total pseudo-observations the prior contributes (``alpha + beta``)."""
        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        """Prior mean rate ``alpha / (alpha + beta)``."""
        return self.alpha / (self.alpha + self.beta)


def beta_prior_from_baseline(
    baseline_rate: float,
    weight: float,
    *,
    name: str,
) -> BetaPrior:
    """Build a Beta prior centred on a historical rate with a chosen weight.

    Splits ``weight`` pseudo-observations across successes and failures so
    the prior mean equals ``baseline_rate``. Raising ``weight`` makes the
    prior harder for new data to move — the standard way to encode a
    skeptical "nothing changed" belief in an A/B or pre/post analysis.

    Parameters
    ----------
    baseline_rate
        Historical conversion rate, strictly between 0 and 1.
    weight
        Pseudo-observations the prior is worth. ``100`` reads as "my
        history is worth 100 extra visitors"; ``500`` is five times as
        stubborn.
    name
        Label carried into results and plots.

    Returns
    -------
    BetaPrior

    Example
    -------
    >>> beta_prior_from_baseline(0.05, 100, name="weakly_informative")
    BetaPrior(name='weakly_informative', alpha=5.0, beta=95.0)
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(
            f"baseline_rate must be in (0, 1); got {baseline_rate}."
        )
    if weight <= 0:
        raise ValueError(f"weight must be > 0; got {weight}.")
    return BetaPrior(
        name=name,
        alpha=float(baseline_rate) * float(weight),
        beta=(1.0 - float(baseline_rate)) * float(weight),
    )


def _resolve_beta_prior(prior: str | BetaPrior) -> BetaPrior:
    """Normalise a prior spec to a :class:`BetaPrior`."""
    if isinstance(prior, BetaPrior):
        return prior
    try:
        alpha, beta = _BB_PRIOR_ALPHA_BETA[prior]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown prior spec: {prior!r}. Use one of "
            f"{sorted(_BB_PRIOR_ALPHA_BETA)} or a BetaPrior instance."
        ) from exc
    return BetaPrior(name=prior, alpha=alpha, beta=beta)


def _build_beta_binomial_model(
    successes_pre: int,
    trials_pre: int,
    successes_post: int,
    trials_post: int,
    prior: str | BetaPrior,
):
    """Return a ``pymc.Model`` implementing the two-proportion Beta-Binomial."""
    spec = _resolve_beta_prior(prior)
    alpha, beta = spec.alpha, spec.beta

    with _pm.Model() as model:  # type: ignore[union-attr]
        p_pre = _pm.Beta("p_pre", alpha=alpha, beta=beta)
        p_post = _pm.Beta("p_post", alpha=alpha, beta=beta)
        _pm.Binomial("obs_pre",  n=trials_pre,  p=p_pre,  observed=successes_pre)
        _pm.Binomial("obs_post", n=trials_post, p=p_post, observed=successes_post)
        _pm.Deterministic("delta", p_post - p_pre)
        _pm.Deterministic("rel_lift", (p_post - p_pre) / p_pre)

    return model


def _build_beta_bernoulli_model(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    prior: str | BetaPrior,
):
    """Return a ``pymc.Model`` implementing the flat Beta-Bernoulli model on raw obs.

    Each element of ``y_pre`` / ``y_post`` is a single binary (0/1) observation.
    Mathematically equivalent to :func:`_build_beta_binomial_model` on the same
    data, but takes raw row-level arrays instead of aggregated counts.
    """
    spec = _resolve_beta_prior(prior)
    alpha, beta = spec.alpha, spec.beta

    with _pm.Model() as model:  # type: ignore[union-attr]
        p_pre  = _pm.Beta("p_pre",  alpha=alpha, beta=beta)
        p_post = _pm.Beta("p_post", alpha=alpha, beta=beta)
        _pm.Bernoulli("obs_pre",  p=p_pre,  observed=y_pre)
        _pm.Bernoulli("obs_post", p=p_post, observed=y_post)
        _pm.Deterministic("delta",    p_post - p_pre)
        _pm.Deterministic("rel_lift", (p_post - p_pre) / p_pre)

    return model



def beta_bernoulli_two_sample(
    y_pre,
    y_post,
    *,
    prior: str | BetaPrior = "jeffreys",
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    progressbar: bool = False,
) -> BetaBinomialResult:
    """Fit the flat Beta-Bernoulli model on raw binary observations.

    Each element of ``y_pre`` / ``y_post`` is a single 0/1 event (e.g. one
    search row).  No user-level aggregation is performed.  Produces the same
    posterior as :func:`beta_binomial_two_sample` on the same data because
    ``Binomial(n, p)`` is the sufficient statistic of ``n`` i.i.d.
    ``Bernoulli(p)`` draws — useful as a baseline that skips the aggregation
    step and avoids any Design Effect / deduplication choices.

    Parameters
    ----------
    y_pre, y_post
        1-D integer arrays of 0/1 binary values (one element per row).
    prior
        ``"jeffreys"`` (Beta(0.5, 0.5), default), ``"uniform"``
        (Beta(1, 1)), or a :class:`BetaPrior` — see
        :func:`beta_prior_from_baseline`.
    hdi_prob
        Coverage of the returned HDI on ``delta = p_post - p_pre``.
    draws, tune, chains, target_accept, random_seed, progressbar
        Forwarded to :func:`pymc.sample`.

    Returns
    -------
    BetaBinomialResult
        Same result container as :func:`beta_binomial_two_sample`.
    """
    y_pre  = np.asarray(y_pre,  dtype=int)
    y_post = np.asarray(y_post, dtype=int)
    for name, arr in (("pre", y_pre), ("post", y_post)):
        if arr.ndim != 1:
            raise ValueError(f"y_{name} must be 1-D; got shape {arr.shape}.")
        if len(arr) == 0:
            raise ValueError(f"y_{name} is empty.")
        if not np.isin(arr, [0, 1]).all():
            raise ValueError(f"y_{name} must contain only 0/1 values.")

    model = _build_beta_bernoulli_model(y_pre, y_post, prior)
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

    delta_samples  = np.asarray(trace.posterior["delta"]).ravel()    # type: ignore[attr-defined]
    p_pre_samples  = np.asarray(trace.posterior["p_pre"]).ravel()    # type: ignore[attr-defined]
    p_post_samples = np.asarray(trace.posterior["p_post"]).ravel()   # type: ignore[attr-defined]
    hdi_arr = _az.hdi(delta_samples, hdi_prob=hdi_prob)              # type: ignore[union-attr]
    hdi_arr = np.asarray(hdi_arr).ravel()
    return BetaBinomialResult(
        trace=trace,
        posterior_p_pre=p_pre_samples,
        posterior_p_post=p_post_samples,
        posterior_delta=delta_samples,
        posterior_mean_delta=float(delta_samples.mean()),
        hdi=(float(hdi_arr[0]), float(hdi_arr[1])),
        successes_pre=int(y_pre.sum()),
        trials_pre=len(y_pre),
        successes_post=int(y_post.sum()),
        trials_post=len(y_post),
        prior_spec=_resolve_beta_prior(prior).name,
    )


def beta_binomial_two_sample(
    successes_pre,
    trials_pre,
    successes_post,
    trials_post,
    *,
    prior: str | BetaPrior = "uniform",
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
        ``"uniform"`` (Beta(1, 1), Bayes-Laplace), ``"jeffreys"``
        (Beta(0.5, 0.5), reference prior), or a :class:`BetaPrior` built
        by :func:`beta_prior_from_baseline`. A custom prior applies to
        ``p_pre`` and ``p_post`` alike.
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
        prior_spec=_resolve_beta_prior(prior).name,
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


def beta_binomial_prior_sensitivity(
    successes_pre,
    trials_pre,
    successes_post,
    trials_post,
    *,
    priors: Sequence[str | BetaPrior] = ("uniform", "jeffreys"),
    hdi_prob: float = 0.95,
    **sample_kwargs,
) -> tuple[dict[str, BetaBinomialResult], pd.DataFrame]:
    """Fit Beta-Binomial under multiple prior specs; return per-fit results + shift table.

    The Beta-Binomial counterpart of :func:`prior_sensitivity`.  Feeds the
    same four count inputs to :func:`beta_binomial_two_sample` under each
    prior, then returns a shift table whose ``shift_from_primary`` column
    measures how far each alternative posterior mean moves relative to the
    *first* prior in ``priors``.

    Parameters
    ----------
    successes_pre, trials_pre, successes_post, trials_post
        Total converting searches / total searches per window — the same
        inputs as :func:`beta_binomial_two_sample`.
    priors
        Ordered sequence of prior specs: the names accepted by
        :func:`beta_binomial_two_sample` (``"uniform"``, ``"jeffreys"``)
        and/or :class:`BetaPrior` instances. The first element is the
        primary spec; all others are compared to it. Names must be unique.
    hdi_prob
        HDI coverage forwarded to every fit.
    **sample_kwargs
        Forwarded to :func:`pymc.sample` (``draws``, ``tune``, ``chains``,
        ``target_accept``, ``random_seed``, ``progressbar``).

    Returns
    -------
    results : dict[str, BetaBinomialResult]
        One fitted result per prior name, in the order given.
    shift_table : pd.DataFrame
        Columns: ``prior``, ``mean_delta``, ``hdi_low``, ``hdi_high``,
        ``shift_from_primary``.  Row order matches ``priors``.
    """
    if not priors:
        raise ValueError("`priors` must contain at least one prior spec.")
    specs = [_resolve_beta_prior(p) for p in priors]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Prior names must be unique; got {names}.")

    results: dict[str, BetaBinomialResult] = {}
    for spec in specs:
        results[spec.name] = beta_binomial_two_sample(
            successes_pre, trials_pre, successes_post, trials_post,
            prior=spec, hdi_prob=hdi_prob, **sample_kwargs,
        )
    primary_mean = results[names[0]].posterior_mean_delta
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


#: Least interval agreement with the reference that still reads as "the prior
#: did not move the answer". Below it the verdict degrades to PRIOR_SENSITIVE.
DEFAULT_MIN_OVERLAP = 0.60

#: Largest move of the posterior mean, in reference-HDI widths, that still
#: reads as "the prior did not move the answer".
DEFAULT_MAX_SHIFT_FRAC = 0.25

_VERDICT_RANK = {"PRIOR_ROBUST": 0, "PRIOR_SENSITIVE": 1, "PRIOR_DRIVEN": 2}

#: Columns :func:`prior_overlap_table` produces. Dropped from the input first,
#: so re-grading a table that was already graded stays idempotent.
_OVERLAP_COLUMNS = (
    "is_primary", "hdi_overlap", "shift_hdi_frac", "direction",
    "direction_flip", "row_verdict",
)


def _direction(
    prob_gt_zero: float | None,
    hdi_low: float,
    hdi_high: float,
    *,
    prob_threshold: float,
) -> str:
    """Label the conclusion a single fit supports: positive, negative or unclear.

    Uses ``P(delta > 0)`` when the sweep carried it, and falls back to whether
    the HDI clears zero.
    """
    if prob_gt_zero is not None and not np.isnan(prob_gt_zero):
        if prob_gt_zero >= prob_threshold:
            return "positive"
        if prob_gt_zero <= 1.0 - prob_threshold:
            return "negative"
        return "unclear"
    if hdi_low > 0:
        return "positive"
    if hdi_high < 0:
        return "negative"
    return "unclear"


def prior_overlap_table(
    shift_table: pd.DataFrame,
    *,
    primary: str | None = None,
    prob_threshold: float = 0.95,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_shift_frac: float = DEFAULT_MAX_SHIFT_FRAC,
) -> pd.DataFrame:
    """Score every prior against the reference prior, one row at a time.

    Three questions, each from the small-sample workflow this implements:
    how much of the interval survives the change of prior (Larson et al.
    2023, overlapping PPIs), how far the posterior mean travelled relative
    to the width of the reference interval, and whether the conclusion the
    fit supports changed at all.

    Parameters
    ----------
    shift_table
        Output of :func:`beta_binomial_prior_sensitivity` or
        :func:`prior_sensitivity` — needs ``prior``, ``hdi_low``,
        ``hdi_high``. ``mean_delta`` and ``prob_delta_gt_0`` are used when
        present.
    primary
        Prior every other row is compared against. Defaults to the first row.
    prob_threshold
        Posterior mass on one side of zero needed before a fit counts as
        supporting a direction.
    min_overlap, max_shift_frac
        Cuts applied to ``hdi_overlap`` and ``shift_hdi_frac`` when grading
        each row.

    Returns
    -------
    pd.DataFrame
        ``shift_table`` plus ``is_primary``, ``hdi_overlap``,
        ``shift_hdi_frac``, ``direction``, ``direction_flip`` and
        ``row_verdict``.

        ``hdi_overlap`` divides the shared length by the *narrower* of the
        two intervals, so an interval that sits wholly inside the reference
        scores 1.0 however much tighter it is. A prior that only sharpens
        the estimate is not a prior that changed the answer.
    """
    required = {"prior", "hdi_low", "hdi_high"}
    missing = required - set(shift_table.columns)
    if missing:
        raise ValueError(f"shift_table is missing columns: {sorted(missing)}.")
    if shift_table.empty:
        raise ValueError("shift_table is empty.")

    names = [str(p) for p in shift_table["prior"]]
    primary_name = primary if primary is not None else names[0]
    if primary_name not in names:
        raise ValueError(
            f"primary prior {primary_name!r} is not in shift_table; have {names}."
        )

    ref = shift_table.iloc[names.index(primary_name)]
    ref_low, ref_high = float(ref["hdi_low"]), float(ref["hdi_high"])
    ref_width = ref_high - ref_low
    ref_mean = float(ref["mean_delta"]) if "mean_delta" in shift_table else float("nan")
    ref_direction = _direction(
        float(ref["prob_delta_gt_0"]) if "prob_delta_gt_0" in shift_table else None,
        ref_low, ref_high, prob_threshold=prob_threshold,
    )

    rows = []
    for name, (_, row) in zip(names, shift_table.iterrows()):
        low, high = float(row["hdi_low"]), float(row["hdi_high"])
        shared = min(ref_high, high) - max(ref_low, low)
        narrower = min(ref_width, high - low)
        overlap = 1.0 if narrower <= 0 else max(0.0, shared) / narrower

        shift_frac = float("nan")
        if "mean_delta" in shift_table and ref_width > 0:
            shift_frac = abs(float(row["mean_delta"]) - ref_mean) / ref_width

        direction = _direction(
            float(row["prob_delta_gt_0"]) if "prob_delta_gt_0" in shift_table else None,
            low, high, prob_threshold=prob_threshold,
        )
        flip = direction != ref_direction

        if shared < 0:
            verdict = "PRIOR_DRIVEN"
        elif overlap < min_overlap or flip or (
            not np.isnan(shift_frac) and shift_frac > max_shift_frac
        ):
            verdict = "PRIOR_SENSITIVE"
        else:
            verdict = "PRIOR_ROBUST"

        rows.append({
            "is_primary": name == primary_name,
            "hdi_overlap": overlap,
            "shift_hdi_frac": shift_frac,
            "direction": direction,
            "direction_flip": flip,
            "row_verdict": "PRIOR_ROBUST" if name == primary_name else verdict,
        })

    return pd.concat(
        [
            shift_table.reset_index(drop=True).drop(
                columns=list(_OVERLAP_COLUMNS), errors="ignore",
            ),
            pd.DataFrame(rows),
        ],
        axis=1,
    )


def prior_sensitivity_verdict(
    shift_table: pd.DataFrame,
    *,
    primary: str | None = None,
    prob_threshold: float = 0.95,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_shift_frac: float = DEFAULT_MAX_SHIFT_FRAC,
) -> str:
    """Judge how much the choice of prior moved the answer.

    Larson et al. (2023) set the outer test: posterior intervals that
    separate across prior specs mean the prior, not the data, is deciding.
    Between "separated" and "identical" sits a band the tutorial reads by
    eye — how far the means and intervals travelled — which this function
    grades explicitly rather than collapsing to a pass/fail.

    Parameters
    ----------
    shift_table
        Output of :func:`beta_binomial_prior_sensitivity` or
        :func:`prior_sensitivity` — needs ``prior``, ``hdi_low``,
        ``hdi_high``. ``mean_delta`` and ``prob_delta_gt_0`` sharpen the
        grade when present.
    primary
        Prior name to compare every other row against. Defaults to the
        first row.
    prob_threshold, min_overlap, max_shift_frac
        Forwarded to :func:`prior_overlap_table`.

    Returns
    -------
    str
        The worst row grade:

        * ``"PRIOR_ROBUST"``    — every interval overlaps the reference by at
          least ``min_overlap``, no mean moved more than ``max_shift_frac``
          reference widths, and every prior supports the same conclusion.
        * ``"PRIOR_SENSITIVE"`` — intervals still overlap, but the estimate
          or the conclusion shifts with the prior. Report the range, not a
          single number.
        * ``"PRIOR_DRIVEN"``    — at least one interval is disjoint from the
          reference. The sample cannot outvote the prior.

        The verdict speaks only about the influence of the prior. An answer
        can be ``PRIOR_ROBUST`` and still be inconclusive, since an interval
        that stays put across priors may straddle zero under all of them.

    Example
    -------
    >>> import pandas as pd
    >>> table = pd.DataFrame({
    ...     "prior": ["noninformative", "informative"],
    ...     "hdi_low": [-0.01, -0.02], "hdi_high": [0.03, 0.01],
    ... })
    >>> prior_sensitivity_verdict(table)
    'PRIOR_ROBUST'
    """
    graded = prior_overlap_table(
        shift_table, primary=primary, prob_threshold=prob_threshold,
        min_overlap=min_overlap, max_shift_frac=max_shift_frac,
    )
    return max(graded["row_verdict"], key=_VERDICT_RANK.__getitem__)


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


def _posterior_mode(samples: np.ndarray) -> float:
    """KDE-based mode estimate — robust across all scipy versions."""
    from scipy.stats import gaussian_kde
    arr = np.asarray(samples, dtype=float).ravel()
    kde = gaussian_kde(arr)
    xs = np.linspace(arr.min(), arr.max(), 1000)
    return float(xs[np.argmax(kde(xs))])


def _plot_posterior_predictive(
    ax,
    y: np.ndarray,
    mu_samples: np.ndarray,
    sigma_samples: np.ndarray,
    nu_samples: np.ndarray,
    *,
    n_label: str,
    n_curves: int = 30,
) -> None:
    """Draw data histogram + posterior-predictive Student-t curves (BEST website style).

    Histogram bars use ``bar_color`` (group colour); predictive curves are
    drawn in light blue, matching the canonical BEST website layout.
    """
    from scipy import stats as _stats

    curve_color = "#7bafd4"   # light blue — same as BEST website for all groups
    hist_color  = "#c44e52"   # red — BEST website uses same red for both groups

    n_bins = max(5, min(len(y), 20))
    x_lo = y.mean() - 4 * y.std()
    x_hi = y.mean() + 4 * y.std()
    xs = np.linspace(x_lo, x_hi, 300)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(mu_samples), size=min(n_curves, len(mu_samples)), replace=False)
    for i in idx:
        pdf = _stats.t.pdf(xs, df=nu_samples[i], loc=mu_samples[i], scale=sigma_samples[i])
        ax.plot(xs, pdf, color=curve_color, alpha=0.15, linewidth=0.8, zorder=1)
    pdf_mean = _stats.t.pdf(
        xs,
        df=float(nu_samples.mean()),
        loc=float(mu_samples.mean()),
        scale=float(sigma_samples.mean()),
    )
    ax.plot(xs, pdf_mean, color=curve_color, linewidth=2.0, zorder=2)
    # Draw histogram on top so bars are always visible
    ax.hist(
        y, bins=n_bins, density=True,
        alpha=0.85, color=hist_color, edgecolor="white", linewidth=0.4,
        zorder=3,
    )
    ax.annotate(n_label, xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=9)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Probability")


def plot_kruschke_report(
    result: BestResult,
    *,
    y_pre,
    y_post,
    out_path: Path | str | None = None,
) -> Figure:
    """Render the canonical Kruschke 10-panel report for a BEST fit.

    Layout (5 rows × 2 cols) mirrors the `best` package website:

    ┌─────────────────────┬──────────────────────────────────────┐
    │ μ_post posterior    │ post data + posterior predictive     │
    │ μ_pre  posterior    │ pre  data + posterior predictive     │
    │ σ_post posterior    │ Difference of means                  │
    │ σ_pre  posterior    │ Difference of std devs               │
    │ ν (normality)       │ Effect size                          │
    └─────────────────────┴──────────────────────────────────────┘
    """
    trace = result.trace
    post = trace.posterior  # type: ignore[attr-defined]

    def _flat(var: str) -> np.ndarray:
        return np.asarray(post[var]).ravel()

    mu_post_s    = _flat("mu_post")
    mu_pre_s     = _flat("mu_pre")
    sigma_post_s = _flat("sigma_post")
    sigma_pre_s  = _flat("sigma_pre")
    nu_s         = _flat("nu")

    y_pre_arr  = np.asarray(y_pre,  dtype=float).ravel()
    y_post_arr = np.asarray(y_post, dtype=float).ravel()

    fig, axes = plt.subplots(5, 2, figsize=(12, 18))

    # Row 0 — μ_post | post data + posterior predictive
    _az.plot_posterior(trace, var_names=["mu_post"], ax=axes[0, 0],  # type: ignore[union-attr]
                       point_estimate="mean")
    axes[0, 0].set_title("Study group mean", pad=8)
    _plot_posterior_predictive(
        axes[0, 1], y_post_arr, mu_post_s, sigma_post_s, nu_s,
        n_label=f"N = {result.n_post}",
    )
    axes[0, 1].set_title("Study group data with post. pred.", pad=8)

    # Row 1 — μ_pre | pre data + posterior predictive
    _az.plot_posterior(trace, var_names=["mu_pre"], ax=axes[1, 0],  # type: ignore[union-attr]
                       point_estimate="mean")
    axes[1, 0].set_title("Control group mean", pad=8)
    _plot_posterior_predictive(
        axes[1, 1], y_pre_arr, mu_pre_s, sigma_pre_s, nu_s,
        n_label=f"N = {result.n_pre}",
    )
    axes[1, 1].set_title("Control group data with post. pred.", pad=8)

    # Row 2 — σ_post | Difference of means
    _az.plot_posterior(trace, var_names=["sigma_post"], ax=axes[2, 0],  # type: ignore[union-attr]
                       point_estimate="mode")
    axes[2, 0].set_title("Study group std. dev.", pad=8)
    _az.plot_posterior(trace, var_names=["delta"], ax=axes[2, 1],  # type: ignore[union-attr]
                       point_estimate="mean", ref_val=0)
    axes[2, 1].set_title("Difference of means", pad=8)

    # Row 3 — σ_pre | Difference of std devs
    _az.plot_posterior(trace, var_names=["sigma_pre"], ax=axes[3, 0],  # type: ignore[union-attr]
                       point_estimate="mode")
    axes[3, 0].set_title("Control group std. dev.", pad=8)
    _az.plot_posterior(trace, var_names=["delta_sigma"], ax=axes[3, 1],  # type: ignore[union-attr]
                       point_estimate="mode", ref_val=0)
    axes[3, 1].set_title("Difference of std. dev.s", pad=8)

    # Row 4 — ν (normality) | Effect size
    _az.plot_posterior(trace, var_names=["nu"], ax=axes[4, 0],  # type: ignore[union-attr]
                       point_estimate="mode")
    axes[4, 0].set_title("Normality", pad=8)
    _az.plot_posterior(trace, var_names=["effect_size"], ax=axes[4, 1],  # type: ignore[union-attr]
                       point_estimate="mode", ref_val=0)
    axes[4, 1].set_title("Effect size", pad=8)
    axes[4, 1].set_xlabel(r"$(\mu_1 - \mu_2)\,/\,\sqrt{(\sigma_1^2 + \sigma_2^2)\,/\,2}$")

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
    results: dict[str, BestResult | BetaBinomialResult],
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
