"""Compare the conversion rate of two groups with a hierarchical Beta-Binomial.

One question, one answer: *does the rate differ between two groups of
units?* Give it each group's per-unit counts and an equivalence band; get
back a verdict.

    control = GroupCounts.from_events(control_df, label="control")
    treatment  = GroupCounts.from_events(treated_df, label="treated")
    effect   = fit_group_comparison(control, treatment, rope_pct_coef=0.10)
    print(effect.decision)      # 'positive' | 'negative' | 'equivalent' | 'inconclusive'

Three equivalence bands are available and can be asked for together, since
"bigger than noise", "big relative to where we started" and "big enough to
act on" are three different questions:

    effect = fit_group_comparison(
        control, treatment,
        rope_stat_coef=0.1,          # +/-0.1 * SE of the control group
        rope_pct_coef=0.10,          # +/-10% of the control rate
        rope_biz=(-0.005, 0.005),    # a threshold somebody signed off on
    )
    {b: d.decision for b, d in effect.ropes.items() if d}

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
against a short post manufactures an effect out of the mismatch alone.
Matching the window lengths — and, for any other split rule, checking that
the groups are comparably composed — is the caller's job.

``pymc`` / ``arviz`` are optional — ``pip install 'dstoolbox[bayes]'``.

References
----------
- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed., §5.1, §5.3.
- Kruschke (2015). *Doing Bayesian Data Analysis*, 2nd ed., ch. 9, ch. 12.
"""

from __future__ import annotations

import warnings
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
    "ROPE_BANDS",
    "aggregate_counts",
    "fit_group_comparison",
    "fit_prepost",
    "rope_from_control",
    "rope_from_control_se",
    "split_by_window",
]

#: The named equivalence bands every fit reports on, in report order.
ROPE_BANDS: tuple[str, ...] = ("stat", "pct", "biz")


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

    def __post_init__(self) -> None:
        for name in ("pre_start", "pre_end", "post_start", "post_end"):
            object.__setattr__(self, name, pd.Timestamp(getattr(self, name)).normalize())

        if self.pre_start > self.pre_end:
            raise ValueError(
                f"pre_start ({self.pre_start.date()}) is after " f"pre_end ({self.pre_end.date()})."
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
    >>> g.n_users, g.events_total, g.pooled_rate
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
    def n_users(self) -> int:
        """Number of units contributing to this group."""
        return int(self.trials.size)

    @property
    def events_total(self) -> int:
        """Total trials summed over units."""
        return int(self.trials.sum())

    @property
    def successes_total(self) -> int:
        """Total successes summed over units."""
        return int(self.successes.sum())

    @property
    def pooled_rate(self) -> float:
        """Per-*trial* rate. Context only — the model estimates per-*unit* ``mu``."""
        return self.successes_total / self.events_total


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
    """Posterior on ``delta = mu_treatment - mu_control``, plus its verdict.

    ``mu`` is the model's own population parameter, so ``estimate``,
    ``lcl`` / ``ucl`` and ``prob_gt_zero`` are read directly off the
    sampled posterior. Nothing is reconstructed or reweighted after the fit.
    ``mu`` is *unit*-level — one unit, one vote — so it will not track a
    trial-weighted pooled rate, and need not. ``pooled_rate_control`` /
    ``pooled_rate_treatment`` carry the empirical trial-weighted rates when you
    want to see how far the two questions diverge.

    ``lcl`` / ``ucl`` are the narrowest interval holding
    ``hdi_prob`` of the posterior, computed by :func:`arviz.hdi` — the same
    interval the rest of :mod:`dstoolbox.ml_funcs.stat_bayes` reports, and
    the one the plots draw.

    ``rope`` is ``None`` when no equivalence band was supplied, in which
    case ``decision`` is a direction-only verdict — ``"positive"``,
    ``"negative"`` or ``"inconclusive"`` — and never ``"equivalent"``,
    since nothing defines what counts as small.

    ``ropes`` holds all three named bands — ``"stat"``, ``"pct"``, ``"biz"``
    — with ``None`` for any that was not requested. They are three different
    questions about the same posterior, so they can and do disagree:
    ``stat`` asks whether the shift beats the control's own noise, ``pct``
    whether it is large relative to where we started, ``biz`` whether it
    clears a threshold somebody signed off on. ``decision`` is the verdict of
    the primary band (the first of ``stat``, ``pct``, ``biz`` that was
    supplied), kept as the single headline answer.

    ``window`` is set only when the groups came from :func:`split_by_window`.
    """

    control_label: str
    treatment_label: str
    metric: str
    prior_spec: str
    window: PrePostWindow | None

    n_users_control: int
    n_users_treatment: int
    n_events_control: int
    n_events_treatment: int
    n_successes_control: int
    n_successes_treatment: int

    mu_control_mean: float
    mu_treatment_mean: float
    estimate: float
    delta_median: float
    lcl: float
    ucl: float
    hdi_prob: float
    prob_gt_zero: float
    rel_lift: float

    rope: RopeDecision | None
    decision: str
    ropes: dict[str, RopeDecision | None]

    diagnostics: pd.DataFrame
    rhat_max: float
    ess_min: float
    divergences: int

    fit_control: HierBetaBinomialFit = field(repr=False)
    fit_treatment: HierBetaBinomialFit = field(repr=False)
    delta_samples: np.ndarray = field(repr=False)

    @property
    def ci_width(self) -> float:
        """Width of the credible interval on the delta."""
        return self.ucl - self.lcl

    @property
    def pooled_rate_control(self) -> float:
        """Empirical pooled rate over the pre window, ``successes/trials``.

        Context only, and a different question from the one the verdict
        answers: this weights by *trial*, so heavy units dominate it, while
        ``mu_control_mean`` weights by *unit*. The two can sit well apart
        when heavy and light units convert at different rates.
        """
        return self.n_successes_control / self.n_events_control

    @property
    def pooled_rate_treatment(self) -> float:
        """Empirical pooled rate over the post window."""
        return self.n_successes_treatment / self.n_events_treatment

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
            "control": self.control_label,
            "treatment": self.treatment_label,
            "n_users_control": self.n_users_control,
            "n_users_treatment": self.n_users_treatment,
            "n_events_control": self.n_events_control,
            "n_events_treatment": self.n_events_treatment,
            "n_successes_control": self.n_successes_control,
            "n_successes_treatment": self.n_successes_treatment,
            "estimand": "mu",
            "mu_control_mean": self.mu_control_mean,
            "mu_treatment_mean": self.mu_treatment_mean,
            "estimate": self.estimate,
            "delta_median": self.delta_median,
            "lcl": self.lcl,
            "ucl": self.ucl,
            "rel_lift_mean": self.rel_lift,
            "prob_delta_gt_0": self.prob_gt_zero,
            "decision": self.decision,
            "rhat_max": self.rhat_max,
            "ess_min": self.ess_min,
            "divergences": self.divergences,
        }
        if self.window is not None:
            row.update(
                {
                    "pre_start": self.window.pre_start.date().isoformat(),
                    "pre_end": self.window.pre_end.date().isoformat(),
                    "post_start": self.window.post_start.date().isoformat(),
                    "post_end": self.window.post_end.date().isoformat(),
                    "n_days_pre": self.window.n_days_pre,
                    "n_days_post": self.window.n_days_post,
                }
            )
        if self.rope is not None:
            row.update(
                {
                    "rope_low": self.rope.rope_low,
                    "rope_high": self.rope.rope_high,
                    "prob_gt_rope": self.rope.prob_gt_high,
                    "prob_in_rope": self.rope.prob_in_rope,
                    "prob_lt_rope": self.rope.prob_lt_low,
                }
            )
        # Every band gets a column whether or not it was requested, so a
        # summary table over several fits keeps a stable schema.
        for band in ROPE_BANDS:
            dec = self.ropes.get(band)
            row[f"decision_{band}"] = None if dec is None else dec.decision
            row[f"rope_{band}_low"] = np.nan if dec is None else dec.rope_low
            row[f"rope_{band}_high"] = np.nan if dec is None else dec.rope_high
        return row

    def __str__(self) -> str:
        header = (
            str(self.window)
            if self.window is not None
            else f"{self.control_label} vs {self.treatment_label}"
        )
        lines = [
            header,
            f"mu  {self.mu_control_mean:.4%} -> {self.mu_treatment_mean:.4%}   "
            f"delta {self.estimate:+.4%} "
            f"[{self.lcl:+.4%}, {self.ucl:+.4%}]",
            f"pooled (context)  {self.pooled_rate_control:.4%} -> "
            f"{self.pooled_rate_treatment:.4%}",
            f"P(delta>0) = {self.prob_gt_zero:.3f}",
        ]
        # One line per band that was asked for. They interrogate the same
        # posterior with different questions, so they are allowed to disagree.
        asked = [(b, self.ropes[b]) for b in ROPE_BANDS if self.ropes.get(b)]
        if asked:
            for band, dec in asked:
                lines.append(
                    f"  {band:<4} ROPE [{dec.rope_low:+.4%}, "
                    f"{dec.rope_high:+.4%}]   verdict: {dec.decision}"
                )
        else:
            lines.append(f"  no ROPE   verdict: {self.decision}")
        return "\n".join(lines)


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


def rope_from_control(control_mean: float, pct_coef: float) -> tuple[float, float]:
    """Symmetric equivalence band as a fraction of the control group's rate.

    Scaling by the control keeps "10% of where we started" meaning the same
    thing whether the rate is 2% or 20%, which an absolute band in
    percentage points does not.
    """
    if pct_coef <= 0:
        raise ValueError(f"pct_coef must be > 0; got {pct_coef}.")
    half = pct_coef * control_mean
    return -half, half


def rope_from_control_se(
    control_mean: float,
    n_users_control: int,
    stat_coef: float,
) -> tuple[float, float]:
    """Symmetric equivalence band as a multiple of the control's standard error.

    ``half = stat_coef * sqrt(mu_c * (1 - mu_c) / n_users_c)``, the Cohen-like
    "effect too small to care about" band: ``stat_coef=0.1`` is a tenth of the
    noise the control group alone carries.

    Unlike :func:`rope_from_control` this is **sample-size aware** — the band
    shrinks as the control grows, so a study with the resolution to detect a
    small shift is allowed to call it meaningful, while a small one is not.
    That is also its hazard: the band is not a fixed statement of what matters
    to the business, so it can only ever answer "is this bigger than noise?".
    Use ``biz`` for "is this worth acting on?".

    Both inputs come from the *control* group only, so the band is fixed
    before the delta is looked at.
    """
    if stat_coef <= 0:
        raise ValueError(f"stat_coef must be > 0; got {stat_coef}.")
    if n_users_control <= 0:
        raise ValueError(f"n_users_control must be > 0; got {n_users_control}.")
    half = stat_coef * float(np.sqrt(control_mean * (1.0 - control_mean) / n_users_control))
    if half <= 0:
        raise ValueError(
            "the control rate is 0 or 1, so its standard error is 0 and the "
            "band collapses to a point; use rope_pct_coef or rope_biz instead."
        )
    return -half, half


def _band_bounds(
    *,
    control_mean: float,
    n_users_control: int,
    rope_stat_coef: float | None,
    rope_pct_coef: float | None,
    rope_biz: tuple[float, float] | None,
) -> dict[str, tuple[float, float] | None]:
    """Resolve the three named bands to bounds, ``None`` where not requested."""
    return {
        "stat": (
            rope_from_control_se(control_mean, n_users_control, rope_stat_coef)
            if rope_stat_coef is not None
            else None
        ),
        "pct": (
            rope_from_control(control_mean, rope_pct_coef) if rope_pct_coef is not None else None
        ),
        "biz": rope_biz,
    }


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
    fit_control: HierBetaBinomialFit,
    fit_treatment: HierBetaBinomialFit,
) -> tuple[pd.DataFrame, float, float, int]:
    """Worst-case convergence view across both fits.

    A delta is only as trustworthy as the less-converged of the two chains
    behind it, so the scalars are worst-over-both. ``ess_min`` spans
    ``ess_bulk`` *and* ``ess_tail`` because the interval bounds are tail
    quantities — a healthy bulk ESS alone does not license them.
    """
    table = pd.concat(
        [
            fit_control.diagnostics.rename(index=lambda v: f"pre:{v}"),
            fit_treatment.diagnostics.rename(index=lambda v: f"post:{v}"),
        ]
    )
    return (
        table,
        float(table["r_hat"].max()),
        float(table[["ess_bulk", "ess_tail"]].min().min()),
        fit_control.divergences + fit_treatment.divergences,
    )


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def fit_group_comparison(
    control: GroupCounts,
    treatment: GroupCounts,
    *,
    metric: str = "convert",
    window: PrePostWindow | None = None,
    prior: str | BetaPrior = "uniform",
    kappa_prior: tuple[float, float] = DEFAULT_KAPPA_PRIOR,
    rope_stat_coef: float | None = None,
    rope_pct_coef: float | None = None,
    rope_biz: tuple[float, float] | None = None,
    credibility_threshold: float = 0.95,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
    min_users: int = 0,
    progressbar: bool = False,
) -> GroupEffect | None:
    """Fit both groups and adjudicate ``delta = mu_treatment - mu_control``.

    ``mu`` is the hierarchical model's population parameter, so the effect,
    its interval and its verdict come straight off the sampled posterior.
    There is no post-fit reconstruction step in the verdict path.

    ``mu`` is a **unit-level** rate — one unit, one vote, whatever its trial
    count. It is therefore not the trial-weighted pooled rate, and the two
    can move in opposite directions when heavy and light units convert at
    different rates and the mix shifts between groups. That is a difference
    of question, not an error. ``pooled_rate_control`` /
    ``pooled_rate_treatment`` carry the empirical trial-weighted rates so the
    divergence stays visible.

    Parameters
    ----------
    control, treatment
        The two groups. ``control`` anchors the ROPE and the relative
        lift, so put the status quo there.
    metric
        Recorded on the result so a summary table says what was measured.
    window
        Set by :func:`fit_prepost` when the groups came from dates. Carried
        through to the output; the model itself ignores it.
    prior
        Prior on ``mu``: ``"uniform"``, ``"jeffreys"``, or a
        :class:`~dstoolbox.ml_funcs.stat_bayes.BetaPrior`. Applied to
        *both* groups, so a control-anchored prior does not tilt the delta.
    rope_pct_coef
        Half-width of the equivalence band as a fraction of the control
        rate — ``0.10`` means "±10% of control is not worth acting on".
    rope_stat_coef
        Half-width of the equivalence band as a multiple of the control's
        standard error — ``0.1`` is the Cohen-like "smaller than noise"
        band. Sample-size aware, unlike ``rope_pct_coef``.
    rope_biz
        Explicit ``(low, high)`` band in the units of ``delta``, for a
        threshold somebody has actually signed off on.

        All three are adjudicated independently and reported side by side on
        ``ropes`` / ``to_row()``; leave one out and its verdict is ``None``.
        Leave out all three for a direction-only verdict. ``decision`` is the
        first supplied band's verdict, in the order ``stat``, ``pct``,
        ``biz``.
    credibility_threshold
        Posterior mass a region needs to win the verdict.
    hdi_prob
        Mass of the reported credible interval.
    min_users
        Refuse to fit if either group has fewer units than this. A
        hierarchical model needs several units to learn ``kappa`` from; below
        a handful the concentration is set by the prior and the delta's
        interval is not worth quoting. Warns and returns ``None`` rather than
        handing back a confident-looking number.
    random_seed
        The control fit uses this, the treatment fit uses ``random_seed + 1``.
        Two independent chains on the same seed would be perfectly
        correlated and shrink the delta's spread.

    Returns
    -------
    GroupEffect or None
        ``None`` when the ``min_users`` guard fired.

    Raises
    ------
    ValueError
        If ``hdi_prob`` or ``credibility_threshold`` is outside ``(0, 1)``,
        or a ROPE coefficient is not positive. Checked before sampling —
        :func:`arviz.hdi` would otherwise reject ``hdi_prob`` only after
        both models had been fitted.

    Example
    -------
    >>> effect = fit_group_comparison(          # doctest: +SKIP
    ...     GroupCounts.from_events(control, label="control"),
    ...     GroupCounts.from_events(treated, label="treated"),
    ...     rope_pct_coef=0.10,
    ... )
    """
    if not 0.0 < hdi_prob < 1.0:
        raise ValueError(f"hdi_prob must be in (0, 1); got {hdi_prob}.")
    if not 0.0 < credibility_threshold < 1.0:
        raise ValueError(f"credibility_threshold must be in (0, 1); got {credibility_threshold}.")
    if rope_pct_coef is not None and rope_pct_coef <= 0:
        raise ValueError(f"rope_pct_coef must be > 0; got {rope_pct_coef}.")
    if rope_stat_coef is not None and rope_stat_coef <= 0:
        raise ValueError(f"rope_stat_coef must be > 0; got {rope_stat_coef}.")
    if rope_biz is not None and not rope_biz[0] < rope_biz[1]:
        raise ValueError(f"rope_biz must satisfy low < high; got {rope_biz}.")

    # Guard before sampling: nothing downstream can rescue a fit whose kappa
    # is pure prior, and returning None makes the caller confront that.
    for group in (control, treatment):
        if group.n_users < min_users:
            warnings.warn(
                f"group {group.label!r} has {group.n_users} unit(s), below "
                f"min_users={min_users}; skipping the fit and returning None.",
                stacklevel=2,
            )
            return None

    shared = dict(
        prior=prior,
        kappa_prior=kappa_prior,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        progressbar=progressbar,
    )
    fit_control = hier_beta_binomial_fit(
        control.trials,
        control.successes,
        random_seed=random_seed,
        **shared,
    )
    fit_treatment = hier_beta_binomial_fit(
        treatment.trials,
        treatment.successes,
        random_seed=random_seed + 1,
        **shared,
    )

    # The verdict is on the model's own parameter. `delta` is a difference of
    # sampled posteriors, so its mean, HDI and P(>0) are read off directly —
    # nothing is reconstructed or reweighted after the fit.
    delta = _paired_delta(fit_control.mu_samples, fit_treatment.mu_samples)
    lcl, ucl = np.asarray(_az.hdi(delta, hdi_prob=hdi_prob)).ravel()
    prob_gt_zero = float((delta > 0).mean())
    mu_control_mean = fit_control.mu_mean

    # Every requested band is adjudicated against the same posterior; they
    # answer different questions and are allowed to disagree. The headline
    # `decision` takes the first one supplied.
    bounds = _band_bounds(
        control_mean=mu_control_mean,
        n_users_control=control.n_users,
        rope_stat_coef=rope_stat_coef,
        rope_pct_coef=rope_pct_coef,
        rope_biz=rope_biz,
    )
    ropes: dict[str, RopeDecision | None] = {
        band: (
            None
            if b is None
            else rope_decision(
                delta,
                rope_low=b[0],
                rope_high=b[1],
                threshold=credibility_threshold,
            )
        )
        for band, b in bounds.items()
    }
    primary = next((ropes[b] for b in ROPE_BANDS if ropes[b] is not None), None)
    if primary is None:
        decision = verdict_without_rope(prob_gt_zero, credibility_threshold)
    else:
        decision = primary.decision

    diagnostics, rhat_max, ess_min, divergences = _merge_diagnostics(fit_control, fit_treatment)

    return GroupEffect(
        control_label=control.label,
        treatment_label=treatment.label,
        metric=metric,
        prior_spec=fit_control.prior_spec,
        window=window,
        n_users_control=fit_control.n_users,
        n_users_treatment=fit_treatment.n_users,
        n_events_control=fit_control.trials,
        n_events_treatment=fit_treatment.trials,
        n_successes_control=fit_control.successes,
        n_successes_treatment=fit_treatment.successes,
        mu_control_mean=mu_control_mean,
        mu_treatment_mean=fit_treatment.mu_mean,
        estimate=float(delta.mean()),
        delta_median=float(np.median(delta)),
        lcl=float(lcl),
        ucl=float(ucl),
        hdi_prob=hdi_prob,
        prob_gt_zero=prob_gt_zero,
        rel_lift=float(delta.mean() / mu_control_mean) if mu_control_mean else 0.0,
        rope=primary,
        decision=decision,
        ropes=ropes,
        diagnostics=diagnostics,
        rhat_max=rhat_max,
        ess_min=ess_min,
        divergences=divergences,
        fit_control=fit_control,
        fit_treatment=fit_treatment,
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
) -> GroupEffect | None:
    """:func:`split_by_window` then :func:`fit_group_comparison`.

    The pre/post path in one call, for callers whose groups are two date
    windows over the same population. ``**kwargs`` are passed straight
    through to :func:`fit_group_comparison`, and ``None`` comes back if its
    ``min_users`` guard fires.

    Raises
    ------
    ValueError
        If a required column is missing, or either window selects no rows.
    """
    pre, post = split_by_window(
        events,
        window,
        unit_col=unit_col,
        metric_col=metric_col,
        date_col=date_col,
    )
    return fit_group_comparison(pre, post, metric=metric_col, window=window, **kwargs)
