"""Figures for :mod:`dstoolbox.ml_funcs.stat_bayes_group`.

Every function takes a :class:`~dstoolbox.ml_funcs.stat_bayes_group.GroupEffect`
(or a list of them) and returns a :class:`matplotlib.figure.Figure`. None of
them write files, read config or decide anything — the caller picks the path
and the format, so the same figure can go to a report, a notebook or an SVG
without a second code path.

    fig = plot_effect(effect)
    fig.savefig("arm.png", dpi=150, bbox_inches="tight")

    fig = plot_summary([(label, effect), ...])
    fig.savefig("summary.svg")          # vector, no hand-written coordinates

``matplotlib`` is required; ``arviz`` is needed only by
:func:`plot_convergence` and :func:`plot_forest`, which read the sampler
traces. Install both with ``pip install 'dstoolbox[bayes]'``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .stat_bayes_group import GroupEffect

__all__ = [
    "plot_convergence",
    "plot_effect",
    "plot_forest",
    "plot_prior_forest",
    "plot_summary",
]


_PRE_COLOR = "#1565c0"
_POST_COLOR = "#6a1b9a"
_DELTA_COLOR = "#2e7d32"

# One lookup for every verdict the model can return. Both decision rules
# share this vocabulary, so the label alone cannot say whether a band was
# consulted and these strings do not claim it did. Callers that know the
# band is set can make the stronger claim in their own prose; this module
# cannot.
_VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "positive":     ("#2e7d32", "Improvement"),
    "negative":     ("#c62828", "Regression"),
    "equivalent":   ("#f57f17", "Practically equivalent"),
    "inconclusive": ("#555555", "Inconclusive"),
}

# The user-level ``theta`` is marginalised out of the model, so ``mu`` and
# ``kappa`` are the whole posterior — and the only things a trace plot can
# show.
_DIAG_VARS = ["mu", "kappa"]


def verdict_style(decision: str) -> tuple[str, str]:
    """Colour and human-readable label for a verdict. Unknown ones read grey."""
    return _VERDICT_STYLE.get(decision, ("#555555", decision))


def _density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    from scipy import stats  # noqa: PLC0415 (import cost only when plotting)

    return stats.gaussian_kde(samples)(grid)


def _annotate(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x, y, text,
        transform=ax.transAxes, va="top", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
    )


# ---------------------------------------------------------------------------
# The result of one fit
# ---------------------------------------------------------------------------

def plot_effect(effect: GroupEffect, *, title: str | None = None) -> Figure:
    """Two panels: the two population rates, and the delta against its band.

    The left panel is the pair of ``mu`` posteriors with their means; the
    right is the posterior on ``delta = mu_post - mu_pre`` with the credible
    interval, the ROPE if there is one, and the verdict in the title.

    Parameters
    ----------
    effect
        A fitted result.
    title
        Suptitle. Defaults to the metric name and the two windows.

    Returns
    -------
    matplotlib.figure.Figure
        Not saved and not closed — that is the caller's call.
    """
    fig, (ax_mu, ax_delta) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        title or f"{effect.metric}  ·  {effect.window}",
        fontsize=12, fontweight="bold",
    )

    # ── Left: mu_pre vs mu_post ──────────────────────────────────────────
    # Same estimand the delta panel adjudicates. Annotating this density
    # with any other rate would put two quantities in one axes.
    mu_pre, mu_post = effect.fit_control.mu_samples, effect.fit_treatment.mu_samples
    lo = min(mu_pre.min(), mu_post.min())
    hi = max(mu_pre.max(), mu_post.max())
    pad = 0.15 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    for samples, color, label, mean in (
        (mu_pre, _PRE_COLOR, "pre", effect.mu_control_mean),
        (mu_post, _POST_COLOR, "post", effect.mu_treatment_mean),
    ):
        ax_mu.fill_between(
            grid, _density(samples, grid),
            alpha=0.35, color=color, label=f"{label}  μ={mean:.3%}",
        )
        ax_mu.axvline(mean, color=color, lw=1.5, ls="--")

    ax_mu.set_xlabel("Population conversion rate μ  (per unit)")
    ax_mu.set_ylabel("Density")
    ax_mu.set_title("Posterior: μ_pre vs μ_post")
    ax_mu.legend(fontsize=8)
    _annotate(
        ax_mu, 0.03, 0.97,
        f"users  pre={effect.n_users_control:,}  post={effect.n_users_treatment:,}\n"
        f"events pre={effect.n_events_control:,}  post={effect.n_events_treatment:,}\n"
        f"pooled {effect.pooled_rate_control:.3%} → {effect.pooled_rate_treatment:.3%}",
    )

    # ── Right: delta, its interval, and the band ─────────────────────────
    delta = effect.delta_samples
    span = delta.max() - delta.min()
    grid_d = np.linspace(delta.min() - 0.15 * span, delta.max() + 0.15 * span, 400)

    ax_delta.fill_between(grid_d, _density(delta, grid_d), alpha=0.45, color=_DELTA_COLOR)
    ax_delta.axvspan(
        effect.lcl, effect.ucl,
        alpha=0.15, color=_DELTA_COLOR,
        label=f"{effect.hdi_prob:.0%} HDI",
    )
    ax_delta.axvline(0, color="black", lw=0.8, ls=":")
    ax_delta.axvline(
        effect.estimate, color=_DELTA_COLOR, lw=1.5, ls="--",
        label=f"mean = {effect.estimate:+.3%}",
    )

    lines = [f"P(δ>0) = {effect.prob_gt_zero:.3f}"]
    if effect.rope is not None:
        ax_delta.axvspan(
            effect.rope.rope_low, effect.rope.rope_high,
            alpha=0.12, color="gray", label="ROPE",
        )
        lines += [
            f"P(δ>ROPE) = {effect.rope.prob_gt_high:.3f}",
            f"P(δ∈ROPE) = {effect.rope.prob_in_rope:.3f}",
            f"P(δ<ROPE) = {effect.rope.prob_lt_low:.3f}",
        ]
    lines.append(f"HDI [{effect.lcl:+.3%}, {effect.ucl:+.3%}]")

    color, label = verdict_style(effect.decision)
    ax_delta.set_title(
        f"Posterior δ = μ_post − μ_pre\n{label}", color=color, fontweight="bold",
    )
    ax_delta.set_xlabel("δ  (absolute rate difference)")
    ax_delta.set_ylabel("Density")
    ax_delta.legend(fontsize=8)
    _annotate(ax_delta, 0.03, 0.97, "\n".join(lines))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sampler diagnostics
# ---------------------------------------------------------------------------

def plot_convergence(effect: GroupEffect, *, label: str = "") -> dict[str, Figure]:
    """Trace, rank and autocorrelation figures for the pre and post fits.

    Returned together and keyed by kind (``"trace"``, ``"rank"``,
    ``"autocorr"``) because reading any one of them alone is what lets a
    badly-mixed chain through. Trace gives the density/trace pair, rank
    replaces Gelman-Rubin and Geweke visuals for NUTS, and autocorrelation
    shows whether the draws are effectively independent.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
    """
    import arviz as az  # noqa: PLC0415 (heavy; import deferred)

    traces = {"pre": effect.fit_control.trace, "post": effect.fit_treatment.trace}
    n = len(traces)
    header = (
        f"{label or effect.metric}  "
        f"(R̂ max={effect.rhat_max:.4f}, ESS min={effect.ess_min:.0f}, "
        f"divergences={effect.divergences})"
    )

    fig_trace, axes = plt.subplots(2 * n, 2, figsize=(12, 5 * n), squeeze=False)
    for row, (period, trace) in enumerate(traces.items()):
        az.plot_trace(trace, var_names=_DIAG_VARS, axes=axes[row * 2:row * 2 + 2, :])
        for ax, var in zip(axes[row * 2], _DIAG_VARS):
            ax.set_title(f"{period}: {var}")
    fig_trace.suptitle(f"MCMC trace — {header}", fontsize=12)
    fig_trace.tight_layout()

    figs = {"trace": fig_trace}
    for kind, plotter, caption in (
        ("rank", az.plot_rank, "uniform bars ⇒ chains agree"),
        ("autocorr", az.plot_autocorr, "bars should collapse to zero"),
    ):
        fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)
        for row, (period, trace) in enumerate(traces.items()):
            kwargs = {"combined": True} if kind == "autocorr" else {}
            plotter(trace, var_names=_DIAG_VARS, ax=axes[row], **kwargs)
            for ax, var in zip(axes[row], _DIAG_VARS):
                ax.set_title(f"{period}: {var}")
        fig.suptitle(f"{kind.capitalize()} — {header}  ({caption})", fontsize=12)
        fig.tight_layout()
        figs[kind] = fig

    return figs


def plot_forest(effect: GroupEffect, *, label: str = "") -> Figure:
    """Side-by-side interval plot of the two population rates."""
    import arviz as az  # noqa: PLC0415 (heavy; import deferred)

    az.plot_forest(
        {
            "mu_pre": effect.fit_control.mu_samples.reshape(1, -1),
            "mu_post": effect.fit_treatment.mu_samples.reshape(1, -1),
        },
        combined=True,
        hdi_prob=effect.hdi_prob,
    )
    fig = plt.gcf()
    fig.suptitle(
        f"μ_pre vs μ_post — {label or effect.metric}  "
        f"({effect.hdi_prob:.0%} HDI)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Many fits at once
# ---------------------------------------------------------------------------

def _summary_view(item: GroupEffect | Mapping[str, Any]) -> dict[str, Any]:
    """Normalise a fit or a ``to_row()`` mapping to the five things drawn.

    The summary chart is the one plot routinely rebuilt from saved CSVs
    rather than from live fits, so requiring a ``GroupEffect`` would mean
    refitting a model purely to redraw a figure.
    """
    if isinstance(item, GroupEffect):
        rope = None if item.rope is None else (item.rope.rope_low, item.rope.rope_high)
        return {
            "decision": item.decision,
            "estimate": item.estimate,
            "lcl": item.lcl,
            "ucl": item.ucl,
            "rope": rope,
        }

    has_rope = item.get("rope_low") is not None and item.get("rope_high") is not None
    return {
        "decision": str(item["decision"]),
        "estimate": float(item["estimate"]),
        "lcl": float(item["lcl"]),
        "ucl": float(item["ucl"]),
        "rope": (float(item["rope_low"]), float(item["rope_high"])) if has_rope else None,
    }


def plot_summary(
    effects: list[tuple[str, GroupEffect | Mapping[str, Any]]],
    *,
    title: str = "Effect by group",
) -> Figure:
    """One row per arm: the delta, its interval, its band and its verdict.

    Takes any number of arms and sizes itself from the data — the row count
    sets the height and the widest interval sets the x-limits, so nothing
    here is tuned to a particular experiment. Save it as ``.svg`` for a
    report or ``.png`` for a notebook; it is the same figure either way.

    Parameters
    ----------
    effects
        ``(label, item)`` pairs, drawn top to bottom in the order given.
        Each ``item`` is either a :class:`GroupEffect` or a mapping in the
        shape :meth:`GroupEffect.to_row` produces, so a saved ``summary.csv``
        row can be replotted without refitting.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``effects`` is empty.
    KeyError
        If a mapping is missing ``decision``, ``estimate``,
        ``lcl`` or ``ucl``.
    """
    if not effects:
        raise ValueError("nothing to plot: `effects` is empty.")

    views = [(label, _summary_view(item)) for label, item in effects]

    n = len(views)
    fig, ax = plt.subplots(figsize=(10, 1.1 * n + 1.8))
    positions = np.arange(n)[::-1]  # first entry at the top

    for y, (_label, view) in zip(positions, views):
        color, verdict = verdict_style(view["decision"])

        if view["rope"] is not None:
            rope_low, rope_high = view["rope"]
            ax.barh(
                y, rope_high - rope_low,
                left=rope_low, height=0.7,
                color="gray", alpha=0.15, zorder=1,
            )
        ax.plot(
            [view["lcl"], view["ucl"]], [y, y],
            color=color, lw=2.5, solid_capstyle="round", zorder=2,
        )
        ax.scatter(view["estimate"], y, color=color, s=60, zorder=3)
        ax.annotate(
            f"{view['estimate']:+.2%}   {verdict}",
            xy=(view["ucl"], y), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8.5, color=color,
        )

    ax.axvline(0, color="black", lw=0.9, ls=":", zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([label for label, _ in views], fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("δ = μ_treatment − μ_control  (absolute rate difference)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1%}")

    # Leave room on the right for the longest annotation.
    left = min(v["lcl"] for _, v in views)
    right = max(v["ucl"] for _, v in views)
    span = right - left or abs(right) or 1.0
    ax.set_xlim(left - 0.08 * span, right + 0.55 * span)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    return fig


def plot_prior_forest(
    rows: pd.DataFrame,
    verdicts: dict[str, str] | None = None,
    *,
    title: str = "Prior sensitivity — δ and credible interval by group and prior",
    xlabel: str = "δ = μ_treatment − μ_control  (percentage points)",
):
    """Draw one dot-and-interval per (group, prior), grouped by group.

    A forest plot rather than overlaid densities: groups often sit on very
    different effect scales, and shared-axis KDEs would flatten the small
    ones into spikes.

    Parameters
    ----------
    rows
        Output of
        :func:`~dstoolbox.ml_funcs.stat_bayes_group_tools.prior_forest_rows`.
    verdicts
        ``{group: verdict}`` annotated beside each block, e.g. the
        ``"PRIOR_ROBUST"`` grade. Omit to label with the group name alone.
    title, xlabel
        Axis text, overridable when the estimand is not a conversion delta.

    Returns
    -------
    matplotlib.figure.Figure
    """


    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    prior_order = list(dict.fromkeys(rows["prior"]))
    colour = {name: palette[i % len(palette)] for i, name in enumerate(prior_order)}
    verdicts = verdicts or {}

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 2.5))
    for _, row in rows.iterrows():
        c = colour[row["prior"]]
        ax.plot(
            [row["lcl_pp"], row["ucl_pp"]], [row["y"], row["y"]],
            color=c, linewidth=2.5, solid_capstyle="butt",
        )
        ax.plot(row["mean_pp"], row["y"], "o", color=c, markersize=7)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(rows["y"])
    ax.set_yticklabels(list(rows["prior"]))
    ax.set_xlabel(xlabel)

    # Group label + verdict beside each block, with a rule between blocks.
    for label, block in rows.groupby("group", sort=False):
        mid = float(block["y"].mean())
        ax.text(
            1.02, mid, f"{label}\n{verdicts.get(label, '')}".rstrip(),
            transform=ax.get_yaxis_transform(), va="center", fontsize=9,
        )
        boundary = float(block["y"].min()) - 0.5
        if boundary > rows["y"].min() - 0.5:
            ax.axhline(boundary, color="lightgray", linewidth=1)

    ax.set_title(title)
    ax.margins(y=0.02)
    return fig
