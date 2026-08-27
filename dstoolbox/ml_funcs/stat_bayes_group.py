"""Compare the conversion rate of two groups with a hierarchical Beta-Binomial.

One question, one answer: *does the rate differ between two groups of
units?* Give it each group's per-unit counts and an equivalence band; get
back a verdict.

    baseline = GroupCounts.from_events(control_df, label="control")
    variant  = GroupCounts.from_events(treated_df, label="treated")
    effect   = fit_group_comparison(baseline, variant, rope_pct_coef=0.10)
    print(effect.decision)      # 'meaningful_positive' | ... | 'inconclusive'

The model does not know what separates the groups. A/B arms, two cohorts,
two countries and two date windows are all the same problem once the data
is counted, and keeping the split out of the model is what makes it reusable.

Splitting by date
-----------------
Pre/post is the common special case, so :class:`PrePostWindow` and
:func:`split_by_window` are provided to turn four dates into two groups:

    window = PrePostWindow("2026-05-29", "2026-07-01", "2026-07-02", "2026-08-04")
    effect = fit_group_comparison(*split_by_window(events, window))

The estimand
------------
``mu`` is the population mean of the *unit-level* rate distribution: one
user, one vote, whatever their search volume. That is deliberately **not**
the pooled rate ``sum(k_i)/sum(n_i)`` that dashboards report — pooled is
per-*search*, so heavy users dominate it. The two answer different
questions and can disagree in size or sign. Say which one you are quoting.

Groups must be comparably composed
----------------------------------
Because ``mu`` averages over whoever is in the group, its value depends on
the *composition* of that group, not only on the behaviour in it.
Conversion rises monotonically with per-unit volume — measured at 5.3% for
one-search users up to 14.2% for 100+ — so any rule that admits more
one-off units to one side drags that side's ``mu`` down.

Date windows are where this bites hardest, because a longer window
mechanically sweeps in a longer tail of one-off users. Comparing a long pre
against a short post manufactures an effect out of the mismatch alone, so
:class:`PrePostWindow` **rejects unequal windows** unless you pass
``allow_unequal=True`` and accept the artefact. When you build groups by
some other rule, that check is yours to make.

``pymc`` / ``arviz`` are optional — ``pip install 'dstoolbox[bayes]'``.

References
----------
- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed., §5.1, §5.3.
- Kruschke (2015). *Doing Bayesian Data Analysis*, 2nd ed., ch. 9, ch. 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._base_import_warning import optional_import
from .stat_bayes import BetaPrior, RopeDecision, rope_decision
from .stat_bayes_hier import (
    DEFAULT_KAPPA_PRIOR,
    HierBetaBinomialFit,
    hier_beta_binomial_fit,
    verdict_without_rope,
)

_az = optional_import("arviz", "stat_bayes_group")

__all__ = [
    "GroupCounts",
    "GroupEffect",
    "PrePostWindow",
    "aggregate_counts",
    "fit_group_comparison",
    "fit_prepost",
    "rope_from_baseline",
    "split_by_window",
]


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrePostWindow:
    """Four explicit dates. Both bounds of both windows are inclusive.

    Parameters
    ----------
    pre_start, pre_end, post_start, post_end
        Anything :func:`pandas.Timestamp` accepts. Must satisfy
        ``pre_start <= pre_end < post_start <= post_end``.
    allow_unequal
        Permit ``n_days_pre != n_days_post``. Off by default, and leaving it
        off is the point — see the module docstring. Turn it on only when
        you have decided the composition artefact is acceptable for the
        question you are asking, and say so where the result is reported.

    Example
    -------
    >>> w = PrePostWindow("2026-05-29", "2026-07-01", "2026-07-02", "2026-08-04")
    >>> w.n_days_pre, w.n_days_post
    (34, 34)
    """

    pre_start: pd.Timestamp
    pre_end: pd.Timestamp
    post_start: pd.Timestamp
    post_end: pd.Timestamp
    allow_unequal: bool = False

    def __post_init__(self) -> None:
        for name in ("pre_start", "pre_end", "post_start", "post_end"):
            object.__setattr__(self, name, pd.Timestamp(getattr(self, name)).normalize())

        if self.pre_start > self.pre_end:
            raise ValueError(
                f"pre_start ({self.pre_start.date()}) is after "
                f"pre_end ({self.pre_end.date()})."
            )
        if self.post_start > self.post_end:
            raise ValueError(
                f"post_start ({self.post_start.date()}) is after "
                f"post_end ({self.post_end.date()})."
            )
        if self.pre_end >= self.post_start:
            raise ValueError(
                f"windows overlap: pre_end ({self.pre_end.date()}) must be "
                f"before post_start ({self.post_start.date()})."
            )
        if not self.allow_unequal and self.n_days_pre != self.n_days_post:
            raise ValueError(
                f"pre window is {self.n_days_pre} days, post is "
                f"{self.n_days_post}. `mu` is an unweighted mean over units, "
                "so it drifts with window length: conversion runs 5.3% for "
                "one-search users up to 14.2% for 100+, and the longer "
                "window sweeps in the one-off tail. An unequal comparison "
                "measures that composition shift, not the intervention. "
                "Match the lengths, or pass allow_unequal=True and report "
                "the caveat."
            )

    @property
    def n_days_pre(self) -> int:
        """Inclusive day count of the pre window."""
        return (self.pre_end - self.pre_start).days + 1

    @property
    def n_days_post(self) -> int:
        """Inclusive day count of the post window."""
        return (self.post_end - self.post_start).days + 1

    def pre_mask(self, dates) -> np.ndarray:
        """Boolean mask selecting rows inside the pre window."""
        d = pd.to_datetime(dates)
        return ((d >= self.pre_start) & (d <= self.pre_end)).to_numpy()

    def post_mask(self, dates) -> np.ndarray:
        """Boolean mask selecting rows inside the post window."""
        d = pd.to_datetime(dates)
        return ((d >= self.post_start) & (d <= self.post_end)).to_numpy()

    def __str__(self) -> str:
        return (
            f"pre [{self.pre_start.date()} .. {self.pre_end.date()}] "
            f"({self.n_days_pre}d) vs "
            f"post [{self.post_start.date()} .. {self.post_end.date()}] "
            f"({self.n_days_post}d)"
        )


# ---------------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupCounts:
    """One group's per-unit ``(trials, successes)``, ready to fit.

    This is the model's whole view of a group. How the units were selected
    — an experiment arm, a date window, a country — is the caller's
    business and is deliberately not recorded here beyond ``label``.

    Parameters
    ----------
    label
        Name for reports and plots, e.g. ``"control"`` or ``"pre"``.
    trials, successes
        One entry per unit, same length, ``0 <= successes <= trials``.

    Example
    -------
    >>> g = GroupCounts("control", [10, 4, 7], [1, 0, 3])
    >>> g.n_units, g.trials_total, g.pooled_rate
    (3, 21, 0.19047619047619047)
    """

    label: str
    trials: np.ndarray
    successes: np.ndarray

    def __post_init__(self) -> None:
        trials = np.asarray(self.trials, dtype=int)
        successes = np.asarray(self.successes, dtype=int)
        if trials.shape != successes.shape:
            raise ValueError(
                f"group {self.label!r}: trials has {trials.shape} entries but "
                f"successes has {successes.shape}; they must line up per unit."
            )
        if trials.size == 0:
            raise ValueError(f"group {self.label!r} is empty.")
        if (trials <= 0).any():
            raise ValueError(f"group {self.label!r}: every unit needs at least one trial.")
        if ((successes < 0) | (successes > trials)).any():
            raise ValueError(
                f"group {self.label!r}: successes must lie in [0, trials] for every unit."
            )
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "successes", successes)

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        *,
        label: str,
        mask=None,
        unit_col: str = "user_id",
        metric_col: str = "convert",
    ) -> GroupCounts:
        """Build from event-level rows, one row per trial.

        ``metric_col`` must be 0/1 per event. ``mask`` selects the group's
        rows; leave it out to use every row.
        """
        counts = aggregate_counts(
            events,
            slice(None) if mask is None else mask,
            unit_col=unit_col,
            metric_col=metric_col,
        )
        return cls(label, counts["trials"].to_numpy(), counts["successes"].to_numpy())

    @property
    def n_units(self) -> int:
        """Number of units contributing to this group."""
        return int(self.trials.size)

    @property
    def trials_total(self) -> int:
        """Total trials summed over units."""
        return int(self.trials.sum())

    @property
    def successes_total(self) -> int:
        """Total successes summed over units."""
        return int(self.successes.sum())

    @property
    def pooled_rate(self) -> float:
        """Per-*trial* rate. Context only — the model estimates per-*unit* ``mu``."""
        return self.successes_total / self.trials_total


def split_by_window(
    events: pd.DataFrame,
    window: PrePostWindow,
    *,
    unit_col: str = "user_id",
    metric_col: str = "convert",
    date_col: str = "datepart",
) -> tuple[GroupCounts, GroupCounts]:
    """Turn event rows plus four dates into ``(pre, post)`` groups.

    The pre/post special case, kept out of the model. Rows outside both
    windows are dropped.

    Raises
    ------
    ValueError
        If a required column is missing, or either window selects no rows.
    """
    missing = {unit_col, metric_col, date_col} - set(events.columns)
    if missing:
        raise ValueError(f"events is missing column(s): {sorted(missing)}.")

    dates = events[date_col]
    shared = dict(unit_col=unit_col, metric_col=metric_col)
    return (
        GroupCounts.from_events(events, label="pre", mask=window.pre_mask(dates), **shared),
        GroupCounts.from_events(events, label="post", mask=window.post_mask(dates), **shared),
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupEffect:
    """Posterior on ``delta = mu_variant - mu_baseline``, plus its verdict.

    ``hdi_low`` / ``hdi_high`` are the narrowest interval holding
    ``hdi_prob`` of the posterior, computed by :func:`arviz.hdi` — the same
    interval the rest of :mod:`dstoolbox.ml_funcs.stat_bayes` reports, and
    the one the plots draw.

    ``rope`` is ``None`` when no equivalence band was supplied, in which
    case ``decision`` is a direction-only verdict — ``"positive"``,
    ``"negative"`` or ``"inconclusive"`` — and never ``"equivalent"``,
    since nothing defines what counts as small.

    ``window`` is set only when the groups came from :func:`split_by_window`.
    """

    baseline_label: str
    variant_label: str
    metric: str
    prior_spec: str
    window: PrePostWindow | None

    n_units_baseline: int
    n_units_variant: int
    n_trials_baseline: int
    n_trials_variant: int
    n_successes_baseline: int
    n_successes_variant: int

    mu_baseline_mean: float
    mu_variant_mean: float
    mu_per_unit_baseline_mean: float
    mu_per_unit_variant_mean: float
    delta_mean: float
    delta_median: float
    hdi_low: float
    hdi_high: float
    hdi_prob: float
    prob_gt_zero: float
    rel_lift: float

    rope: RopeDecision | None
    decision: str

    diagnostics: pd.DataFrame
    rhat_max: float
    ess_min: float
    divergences: int

    fit_baseline: HierBetaBinomialFit = field(repr=False)
    fit_variant: HierBetaBinomialFit = field(repr=False)
    delta_samples: np.ndarray = field(repr=False)

    @property
    def hdi_width(self) -> float:
        """Width of the credible interval on the delta."""
        return self.hdi_high - self.hdi_low

    @property
    def pooled_rate_baseline(self) -> float:
        """Empirical pooled rate over the pre window, ``successes/trials``.

        Context only, and the closest raw analogue of ``mu_baseline_mean``:
        both weight by trial. The modelled one shrinks light units toward the
        population mean, so the two differ but should stay in the same
        neighbourhood. ``mu_per_unit_baseline_mean`` can sit well away from
        both when heavy and light units convert at different rates.
        """
        return self.n_successes_baseline / self.n_trials_baseline

    @property
    def pooled_rate_variant(self) -> float:
        """Empirical pooled rate over the post window."""
        return self.n_successes_variant / self.n_trials_variant

    @property
    def converged(self) -> bool:
        """Standard convergence gate: R-hat < 1.01, ESS > 400, no divergences."""
        return self.rhat_max < 1.01 and self.ess_min > 400 and self.divergences == 0

    def to_row(self) -> dict:
        """Flatten to one record for a summary table."""
        row = {
            "metric": self.metric,
            "model": "hierarchical_beta_binomial",
            "prior": self.prior_spec,
            "baseline": self.baseline_label,
            "variant": self.variant_label,
            "n_units_baseline": self.n_units_baseline,
            "n_units_variant": self.n_units_variant,
            "n_trials_baseline": self.n_trials_baseline,
            "n_trials_variant": self.n_trials_variant,
            "n_successes_baseline": self.n_successes_baseline,
            "n_successes_variant": self.n_successes_variant,
            "mu_baseline_mean": self.mu_baseline_mean,
            "mu_variant_mean": self.mu_variant_mean,
            "mu_per_unit_baseline_mean": self.mu_per_unit_baseline_mean,
            "mu_per_unit_variant_mean": self.mu_per_unit_variant_mean,
            "delta_mean": self.delta_mean,
            "delta_median": self.delta_median,
            "delta_hdi_low": self.hdi_low,
            "delta_hdi_high": self.hdi_high,
            "rel_lift_mean": self.rel_lift,
            "prob_delta_gt_0": self.prob_gt_zero,
            "decision": self.decision,
            "rhat_max": self.rhat_max,
            "ess_min": self.ess_min,
            "divergences": self.divergences,
        }
        if self.window is not None:
            row.update({
                "pre_start": self.window.pre_start.date().isoformat(),
                "pre_end": self.window.pre_end.date().isoformat(),
                "post_start": self.window.post_start.date().isoformat(),
                "post_end": self.window.post_end.date().isoformat(),
                "n_days_pre": self.window.n_days_pre,
                "n_days_post": self.window.n_days_post,
            })
        if self.rope is not None:
            row.update({
                "rope_low": self.rope.rope_low,
                "rope_high": self.rope.rope_high,
                "prob_gt_rope": self.rope.prob_gt_high,
                "prob_in_rope": self.rope.prob_in_rope,
                "prob_lt_rope": self.rope.prob_lt_low,
            })
        return row

    def __str__(self) -> str:
        band = (
            f"ROPE ±{self.rope.rope_high:.4%}"
            if self.rope is not None
            else "no ROPE"
        )
        header = (
            str(self.window) if self.window is not None
            else f"{self.baseline_label} vs {self.variant_label}"
        )
        return (
            f"{header}\n"
            f"mu  {self.mu_baseline_mean:.4%} -> {self.mu_variant_mean:.4%}   "
            f"delta {self.delta_mean:+.4%} "
            f"[{self.hdi_low:+.4%}, {self.hdi_high:+.4%}]\n"
            f"per-unit  {self.mu_per_unit_baseline_mean:.4%} -> "
            f"{self.mu_per_unit_variant_mean:.4%}\n"
            f"P(delta>0) = {self.prob_gt_zero:.3f}   {band}   "
            f"verdict: {self.decision}"
        )


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def aggregate_counts(
    events: pd.DataFrame,
    mask,
    *,
    unit_col: str = "user_id",
    metric_col: str = "convert",
) -> pd.DataFrame:
    """Collapse masked event rows to one ``(trials, successes)`` row per unit.

    ``metric_col`` must be 0/1 per event. Rows outside ``mask`` are dropped
    before the groupby, so the model never sees them.
    """
    sub = events.loc[mask, [unit_col, metric_col]]
    if sub.empty:
        raise ValueError("the mask selected no rows — check it against the data.")
    return (
        sub.assign(**{metric_col: sub[metric_col].astype(int)})
        .groupby(unit_col)[metric_col]
        .agg(trials="count", successes="sum")
        .reset_index(drop=True)
    )


def rope_from_baseline(baseline_mean: float, pct_coef: float) -> tuple[float, float]:
    """Symmetric equivalence band as a fraction of the baseline group's rate.

    Scaling by the baseline keeps "10% of where we started" meaning the same
    thing whether the rate is 2% or 20%, which an absolute band in
    percentage points does not.
    """
    if pct_coef <= 0:
        raise ValueError(f"pct_coef must be > 0; got {pct_coef}.")
    half = pct_coef * baseline_mean
    return -half, half


def _paired_delta(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """``post - pre``, paired by position after truncating to equal length.

    The periods are fitted independently, so no draw of ``post`` corresponds
    to any particular draw of ``pre``. Pairing by position is arbitrary but
    harmless: the difference of two independent samples has the right
    distribution however they are matched up.
    """
    n = min(len(pre), len(post))
    return np.asarray(post)[:n] - np.asarray(pre)[:n]


def _merge_diagnostics(
    fit_baseline: HierBetaBinomialFit,
    fit_variant: HierBetaBinomialFit,
) -> tuple[pd.DataFrame, float, float, int]:
    """Worst-case convergence view across both fits.

    A delta is only as trustworthy as the less-converged of the two chains
    behind it, so the scalars are worst-over-both. ``ess_min`` spans
    ``ess_bulk`` *and* ``ess_tail`` because the interval bounds are tail
    quantities — a healthy bulk ESS alone does not license them.
    """
    table = pd.concat([
        fit_baseline.diagnostics.rename(index=lambda v: f"pre:{v}"),
        fit_variant.diagnostics.rename(index=lambda v: f"post:{v}"),
    ])
    return (
        table,
        float(table["r_hat"].max()),
        float(table[["ess_bulk", "ess_tail"]].min().min()),
        fit_baseline.divergences + fit_variant.divergences,
    )


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

def fit_group_comparison(
    baseline: GroupCounts,
    variant: GroupCounts,
    *,
    metric: str = "convert",
    window: PrePostWindow | None = None,
    prior: str | BetaPrior = "uniform",
    kappa_prior: tuple[float, float] = DEFAULT_KAPPA_PRIOR,
    rope_pct_coef: float | None = None,
    credibility_threshold: float = 0.95,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
    progressbar: bool = False,
) -> GroupEffect:
    """Fit both groups and adjudicate ``delta = mu_variant - mu_baseline``.

    ``mu`` here is the **trial-weighted** population rate, so the verdict
    answers "did successes per trial move". The unit-averaged rate is
    reported too, as ``mu_per_unit_baseline_mean`` /
    ``mu_per_unit_variant_mean``. Check them: when heavy and light units
    convert at different rates and the mix shifts between groups, the two
    estimands can move in opposite directions, and that disagreement is a
    finding rather than a bug.

    Parameters
    ----------
    baseline, variant
        The two groups. ``baseline`` anchors the ROPE and the relative
        lift, so put the status quo there.
    metric
        Recorded on the result so a summary table says what was measured.
    window
        Set by :func:`fit_prepost` when the groups came from dates. Carried
        through to the output; the model itself ignores it.
    prior
        Prior on ``mu``: ``"uniform"``, ``"jeffreys"``, or a
        :class:`~dstoolbox.ml_funcs.stat_bayes.BetaPrior`. Applied to
        *both* groups, so a baseline-anchored prior does not tilt the delta.
    rope_pct_coef
        Half-width of the equivalence band as a fraction of the baseline
        rate — ``0.10`` means "±10% of baseline is not worth acting on".
        ``None`` gives a direction-only verdict.
    credibility_threshold
        Posterior mass a region needs to win the verdict.
    hdi_prob
        Mass of the reported credible interval.
    random_seed
        The baseline fit uses this, the variant fit uses ``random_seed + 1``.
        Two independent chains on the same seed would be perfectly
        correlated and shrink the delta's spread.

    Returns
    -------
    GroupEffect

    Example
    -------
    >>> effect = fit_group_comparison(          # doctest: +SKIP
    ...     GroupCounts.from_events(control, label="control"),
    ...     GroupCounts.from_events(treated, label="treated"),
    ...     rope_pct_coef=0.10,
    ... )
    """
    shared = dict(
        prior=prior,
        kappa_prior=kappa_prior,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        progressbar=progressbar,
    )
    fit_baseline = hier_beta_binomial_fit(
        baseline.trials, baseline.successes, random_seed=random_seed, **shared,
    )
    fit_variant = hier_beta_binomial_fit(
        variant.trials, variant.successes, random_seed=random_seed + 1, **shared,
    )

    # Trial-weighted, so the headline answers "did conversions per trial
    # move" — the same question the pooled rate asks. The unit-averaged
    # rate rides along in `mu_per_unit_*`; the two can disagree in sign
    # when the heavy/light mix shifts between windows.
    delta = _paired_delta(
        fit_baseline.mu_weighted_samples, fit_variant.mu_weighted_samples
    )
    hdi_low, hdi_high = np.asarray(_az.hdi(delta, hdi_prob=hdi_prob)).ravel()
    prob_gt_zero = float((delta > 0).mean())
    mu_baseline_mean = fit_baseline.mu_weighted_mean

    if rope_pct_coef is None:
        rope = None
        decision = verdict_without_rope(prob_gt_zero, credibility_threshold)
    else:
        lo, hi = rope_from_baseline(mu_baseline_mean, rope_pct_coef)
        rope = rope_decision(
            delta, rope_low=lo, rope_high=hi, threshold=credibility_threshold
        )
        decision = rope.decision

    diagnostics, rhat_max, ess_min, divergences = _merge_diagnostics(
        fit_baseline, fit_variant
    )

    return GroupEffect(
        baseline_label=baseline.label,
        variant_label=variant.label,
        metric=metric,
        prior_spec=fit_baseline.prior_spec,
        window=window,
        n_units_baseline=fit_baseline.n_units,
        n_units_variant=fit_variant.n_units,
        n_trials_baseline=fit_baseline.trials,
        n_trials_variant=fit_variant.trials,
        n_successes_baseline=fit_baseline.successes,
        n_successes_variant=fit_variant.successes,
        mu_baseline_mean=mu_baseline_mean,
        mu_variant_mean=fit_variant.mu_weighted_mean,
        mu_per_unit_baseline_mean=fit_baseline.mu_mean,
        mu_per_unit_variant_mean=fit_variant.mu_mean,
        delta_mean=float(delta.mean()),
        delta_median=float(np.median(delta)),
        hdi_low=float(hdi_low),
        hdi_high=float(hdi_high),
        hdi_prob=hdi_prob,
        prob_gt_zero=prob_gt_zero,
        rel_lift=float(delta.mean() / mu_baseline_mean) if mu_baseline_mean else 0.0,
        rope=rope,
        decision=decision,
        diagnostics=diagnostics,
        rhat_max=rhat_max,
        ess_min=ess_min,
        divergences=divergences,
        fit_baseline=fit_baseline,
        fit_variant=fit_variant,
        delta_samples=delta,
    )


def fit_prepost(
    events: pd.DataFrame,
    window: PrePostWindow,
    *,
    unit_col: str = "user_id",
    metric_col: str = "convert",
    date_col: str = "datepart",
    **kwargs,
) -> GroupEffect:
    """:func:`split_by_window` then :func:`fit_group_comparison`.

    The pre/post path in one call, for callers whose groups are two date
    windows over the same population. ``**kwargs`` are passed straight
    through to :func:`fit_group_comparison`.

    Raises
    ------
    ValueError
        If the windows are unequal length (see :class:`PrePostWindow`), or
        either window selects no rows.
    """
    pre, post = split_by_window(
        events, window,
        unit_col=unit_col, metric_col=metric_col, date_col=date_col,
    )
    return fit_group_comparison(
        pre, post, metric=metric_col, window=window, **kwargs
    )
