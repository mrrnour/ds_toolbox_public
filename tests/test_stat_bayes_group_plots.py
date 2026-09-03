"""Tests for the pre/post figures.

Figures are checked for structure — panel counts, row counts, axis limits —
not appearance. One module-scoped fit is shared by everything that needs a
real result, since sampling dominates the runtime.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dstoolbox.ml_funcs.stat_bayes_group import PrePostWindow, fit_prepost  # noqa: E402
from dstoolbox.ml_funcs.stat_bayes_group_plots import (  # noqa: E402
    plot_convergence,
    plot_effect,
    plot_forest,
    plot_summary,
    verdict_style,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

FIT_KW = {"draws": 300, "tune": 300, "chains": 2, "random_seed": 0}


def _events(rate_pre: float, rate_post: float, *, n_users: int = 150, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for start, end, rate in (
        ("2026-01-01", "2026-01-10", rate_pre),
        ("2026-01-11", "2026-01-20", rate_post),
    ):
        days = pd.date_range(start, end)
        for user in range(n_users):
            for day in rng.choice(days, size=rng.integers(1, 4), replace=False):
                rows.append((user, day, int(rng.random() < rate)))
    return pd.DataFrame(rows, columns=["user_id", "datepart", "convert"])


@pytest.fixture(scope="module")
def effect():
    pytest.importorskip("pymc")
    window = PrePostWindow("2026-01-01", "2026-01-10", "2026-01-11", "2026-01-20")
    return fit_prepost(_events(0.10, 0.25), window, rope_pct_coef=0.10, **FIT_KW)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Verdict styling — the single lookup every figure shares
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "decision",
    ["positive", "negative", "equivalent", "inconclusive"],
)
def test_every_verdict_has_a_colour_and_a_label(decision):
    color, label = verdict_style(decision)
    assert color.startswith("#")
    assert label and label != decision


def test_unknown_verdict_degrades_instead_of_raising():
    color, label = verdict_style("something_new")
    assert color == "#555555"
    assert label == "something_new"


# --------------------------------------------------------------------------- #
# plot_effect
# --------------------------------------------------------------------------- #

def test_effect_figure_has_two_panels(effect):
    fig = plot_effect(effect)
    assert len(fig.axes) == 2


def test_effect_figure_titles_carry_the_verdict(effect):
    fig = plot_effect(effect)
    _, expected = verdict_style(effect.decision)
    assert expected in fig.axes[1].get_title()


def test_effect_figure_accepts_a_custom_title(effect):
    fig = plot_effect(effect, title="Arm A")
    assert fig._suptitle.get_text() == "Arm A"


def test_effect_figure_draws_the_band_only_when_there_is_one(effect):
    with_band = len(plot_effect(effect).axes[1].patches)

    plain = fit_prepost(
        _events(0.10, 0.25), effect.window, **FIT_KW
    )
    without_band = len(plot_effect(plain).axes[1].patches)

    assert with_band > without_band


# --------------------------------------------------------------------------- #
# ArviZ-backed diagnostics
# --------------------------------------------------------------------------- #

def test_convergence_returns_all_three_diagnostics(effect):
    pytest.importorskip("arviz")
    figs = plot_convergence(effect)
    assert set(figs) == {"trace", "rank", "autocorr"}
    # Two periods x two parameters, so four panels in every grid.
    assert all(len(fig.axes) >= 4 for fig in figs.values())


def test_convergence_header_reports_the_diagnostics(effect):
    pytest.importorskip("arviz")
    header = plot_convergence(effect, label="arm A")["trace"]._suptitle.get_text()
    assert "arm A" in header
    assert f"divergences={effect.divergences}" in header


def test_forest_plots_both_periods(effect):
    pytest.importorskip("arviz")
    fig = plot_forest(effect, label="arm A")
    assert "arm A" in fig._suptitle.get_text()
    assert fig.axes


# --------------------------------------------------------------------------- #
# plot_summary — sized from the data, any number of arms
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_arms", [1, 2, 5])
def test_summary_takes_any_number_of_arms(effect, n_arms):
    fig = plot_summary([(f"arm {i}", effect) for i in range(n_arms)])
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == n_arms
    assert fig.get_figheight() > 1.1 * n_arms


def test_summary_keeps_the_given_order_top_down(effect):
    fig = plot_summary([("first", effect), ("second", effect)])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    ticks = list(ax.get_yticks())
    assert labels == ["first", "second"]
    assert ticks[0] > ticks[1]  # 'first' sits higher on the axis


def test_summary_limits_come_from_the_intervals(effect):
    fig = plot_summary([("arm", effect)])
    lo, hi = fig.axes[0].get_xlim()
    assert lo < effect.lcl
    assert hi > effect.ucl


def test_summary_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty"):
        plot_summary([])


def test_summary_replots_a_saved_row_without_refitting(effect):
    """The pipeline redraws this chart from summary.csv, not from a live fit."""
    row = effect.to_row()
    fig = plot_summary([("from_csv", row)])
    lo, hi = fig.axes[0].get_xlim()
    assert lo < effect.lcl
    assert hi > effect.ucl


def test_summary_row_and_effect_agree(effect):
    from_effect = plot_summary([("a", effect)]).axes[0].get_xlim()
    from_row = plot_summary([("a", effect.to_row())]).axes[0].get_xlim()
    assert from_effect == pytest.approx(from_row)
