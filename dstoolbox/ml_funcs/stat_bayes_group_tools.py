"""Extras built on :func:`~dstoolbox.ml_funcs.stat_bayes_group.fit_prepost`.

Two questions you ask *after* you have an answer:

**Did the prior decide this?** :func:`prior_sensitivity_groups` refits the
same windows under several priors and grades the spread with the existing
:func:`~dstoolbox.ml_funcs.stat_bayes.prior_sensitivity_verdict`.

**When did it become visible?** :func:`sequential_scan` refits over a
sequence of windows so you can see the verdict settle — or fail to.

Both are wrappers. Neither reimplements the model, and neither is needed to
get a verdict; reach for them when a reviewer asks how load-bearing a choice
was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stat_bayes import BetaPrior, _resolve_beta_prior, prior_sensitivity_verdict
from .stat_bayes_group import GroupEffect, PrePostWindow, fit_prepost

__all__ = [
    "DEFAULT_PRIORS",
    "matched_sequential_windows",
    "prior_forest_rows",
    "prior_sensitivity_groups",
    "prior_shape_table",
    "sequential_scan",
]

#: A reference and two neighbours either side of it. ``uniform`` is the
#: reference — Beta(1,1) adds one notional success and one failure, which
#: at these sample sizes is nothing. ``jeffreys`` (Beta(0.5,0.5)) leans on
#: the extremes, ``weak`` (Beta(2,2)) leans on the middle; if the answer
#: survives both, the prior is not driving it.
DEFAULT_PRIORS = ("uniform", "jeffreys", BetaPrior("weak", 2.0, 2.0))


def prior_sensitivity_groups(
    events: pd.DataFrame,
    window: PrePostWindow,
    *,
    priors=DEFAULT_PRIORS,
    primary: str | None = None,
    **fit_kwargs,
) -> tuple[pd.DataFrame, str]:
    """Refit the same windows under several priors and grade the spread.

    The prior is applied to *both* periods, so a prior that pulls ``mu``
    toward some control pulls both ends of the delta and largely cancels.
    That is why a well-powered pre/post is usually ``PRIOR_ROBUST`` — and
    why it is worth checking rather than assuming.

    Parameters
    ----------
    events, window
        As :func:`~dstoolbox.ml_funcs.stat_bayes_group.fit_prepost`.
    priors
        Prior names or :class:`~dstoolbox.ml_funcs.stat_bayes.BetaPrior`
        objects. The first is the reference unless ``primary`` says
        otherwise.
    primary
        Prior every other row is compared against.
    **fit_kwargs
        Forwarded to ``fit_prepost``. ``prior`` is not accepted here — that
        is what ``priors`` is for.

    Returns
    -------
    table : pd.DataFrame
        One row per prior. Columns are named ``lcl`` / ``ucl`` /
        ``mean_delta`` / ``prob_delta_gt_0`` / ``shift_from_primary``
        because that is the contract
        :func:`~dstoolbox.ml_funcs.stat_bayes.prior_overlap_table` reads,
        and they carry the same bounds as ``GroupEffect.lcl`` /
        ``GroupEffect.ucl``.
    verdict : str
        ``"PRIOR_ROBUST"``, ``"PRIOR_SENSITIVE"`` or ``"PRIOR_DRIVEN"``.

    Raises
    ------
    ValueError
        If ``priors`` is empty, ``prior`` was passed in ``fit_kwargs``, two
        priors share a name, or ``primary`` names a prior that was not swept.
    """
    if "prior" in fit_kwargs:
        raise ValueError("pass the priors to sweep via `priors`, not `prior`.")
    priors = list(priors)
    if not priors:
        raise ValueError("`priors` is empty: nothing to compare.")
    # Every downstream grade keys rows on `prior`, so duplicate names would
    # silently compare a row against the wrong reference. Check before the
    # fits, not after — these are minutes of sampling apiece.
    names = [_resolve_beta_prior(p).name for p in priors]
    if len(set(names)) != len(names):
        raise ValueError(f"Prior names must be unique; got {names}.")
    reference = primary if primary is not None else names[0]
    if reference not in names:
        raise ValueError(
            f"primary prior {reference!r} is not among the priors swept; "
            f"have {names}."
        )

    rows = []
    for prior in priors:
        effect = fit_prepost(events, window, prior=prior, **fit_kwargs)
        rows.append({
            "prior": effect.prior_spec,
            "mean_delta": effect.estimate,
            "lcl": effect.lcl,
            "ucl": effect.ucl,
            "prob_delta_gt_0": effect.prob_gt_zero,
            "decision": effect.decision,
        })

    table = pd.DataFrame(rows)
    primary_mean = float(table.loc[names.index(reference), "mean_delta"])
    table["shift_from_primary"] = table["mean_delta"] - primary_mean
    return table, prior_sensitivity_verdict(table, primary=primary)


def matched_sequential_windows(
    intervention_date,
    *,
    n_periods: int,
    period_days: int = 7,
) -> list[PrePostWindow]:
    """Windows that grow on *both* sides of an intervention, one period at a time.

    Period ``k`` compares the ``k`` periods after ``intervention_date``
    against the ``k`` periods before it. Both sides grow together on
    purpose: holding the pre-period fixed while the post-period expands
    would make the two windows unequal, and ``mu`` moves with window length
    on composition alone — the scan would then show a trend that is nothing
    but the growing mismatch. See
    :class:`~dstoolbox.ml_funcs.stat_bayes_group.PrePostWindow`.

    Parameters
    ----------
    intervention_date
        First day of the post-period. The pre-period ends the day before.
    n_periods
        How many windows to generate. Window ``k`` covers ``k *
        period_days`` on each side.
    period_days
        Days per period. 7 gives a weekly scan.

    Returns
    -------
    list[PrePostWindow]
        Length ``n_periods``, shortest first.

    Example
    -------
    >>> windows = matched_sequential_windows("2026-07-02", n_periods=2)
    >>> [(w.n_days_pre, w.n_days_post) for w in windows]
    [(7, 7), (14, 14)]

    Raises
    ------
    ValueError
        If ``n_periods`` or ``period_days`` is not positive.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}.")
    if period_days < 1:
        raise ValueError(f"period_days must be >= 1, got {period_days}.")

    post_start = pd.Timestamp(intervention_date).normalize()
    pre_end = post_start - pd.Timedelta(days=1)

    return [
        PrePostWindow(
            pre_end - pd.Timedelta(days=k * period_days - 1),
            pre_end,
            post_start,
            post_start + pd.Timedelta(days=k * period_days - 1),
        )
        for k in range(1, n_periods + 1)
    ]


def sequential_scan(
    events: pd.DataFrame,
    windows: list[PrePostWindow],
    **fit_kwargs,
) -> tuple[pd.DataFrame, list[GroupEffect]]:
    """Fit a sequence of windows and tabulate how the verdict evolves.

    This is a *description*, not a stopping rule. Every window is a fresh
    look at accumulating data, so reading the table as "stop when it goes
    significant" inflates the false-positive rate the same way peeking at a
    frequentist test does. Use it to see whether an effect arrived
    abruptly, drifted in, or never settled.

    Parameters
    ----------
    events
        Event-level rows covering every window.
    windows
        Fitted in the order given. :func:`matched_sequential_windows`
        builds the usual sequence.
    **fit_kwargs
        Forwarded to :func:`~dstoolbox.ml_funcs.stat_bayes_group.fit_prepost`.

    Returns
    -------
    table : pd.DataFrame
        One row per window — ``GroupEffect.to_row()`` plus a ``period``
        index starting at 1.
    effects : list[GroupEffect]
        The fits themselves, for plotting.

    Raises
    ------
    ValueError
        If ``windows`` is empty.
    """
    if not windows:
        raise ValueError("`windows` is empty: nothing to scan.")

    effects = [fit_prepost(events, w, **fit_kwargs) for w in windows]
    table = pd.DataFrame(
        [{"period": i, **e.to_row()} for i, e in enumerate(effects, start=1)]
    )
    return table, effects


def prior_shape_table(priors) -> pd.DataFrame:
    """One row per prior: its shape, its weight and its mean.

    Report this next to a sensitivity table so a reader can see *what* was
    varied, not just that something was. A sweep whose priors turn out to
    be near-identical proves nothing, and only their shapes reveal that.

    Parameters
    ----------
    priors
        :class:`~dstoolbox.ml_funcs.stat_bayes.BetaPrior` objects.

    Returns
    -------
    pd.DataFrame
        Columns ``prior``, ``alpha``, ``beta``, ``prior_weight``,
        ``prior_mean``. Merge onto a sensitivity table on ``prior``.
    """
    return pd.DataFrame([{
        "prior": p.name,
        "alpha": p.alpha,
        "beta": p.beta,
        "prior_weight": p.weight,
        "prior_mean": p.mean,
    } for p in priors])


def prior_forest_rows(per_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten several groups' sensitivity tables into plot-ready forest rows.

    One row per (group, prior), ordered so the first group lands at the top
    of the axis. Effect columns are restated in percentage points, which is
    how they are usually quoted.

    Parameters
    ----------
    per_group
        ``{label: table}`` in display order, where each table came from
        :func:`prior_sensitivity_groups` — so it carries ``prior``,
        ``mean_delta``, ``lcl`` and ``ucl``.

    Returns
    -------
    pd.DataFrame
        Columns ``group``, ``prior``, ``mean_pp``, ``lcl_pp``,
        ``ucl_pp``, ``y``.

    Raises
    ------
    ValueError
        If ``per_group`` is empty.
    """
    if not per_group:
        raise ValueError("cannot build a forest plot with no groups.")

    rows = [
        {
            "group": label,
            "prior": row["prior"],
            "mean_pp": float(row["mean_delta"]) * 100.0,
            "lcl_pp": float(row["lcl"]) * 100.0,
            "ucl_pp": float(row["ucl"]) * 100.0,
        }
        for label, table in per_group.items()
        for _, row in table.iterrows()
    ]
    out = pd.DataFrame(rows)
    out["y"] = np.arange(len(out) - 1, -1, -1)
    return out
