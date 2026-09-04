"""Tests for the two-group extras: prior sweeps, sequential scans, report frames.

Window generation and frame shaping need no fitting and are checked
exhaustively; the two fitting wrappers get one small fit each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dstoolbox.ml_funcs.stat_bayes import BetaPrior
from dstoolbox.ml_funcs.stat_bayes_group import PrePostWindow
from dstoolbox.ml_funcs.stat_bayes_group_tools import (
    matched_sequential_windows,
    prior_forest_rows,
    prior_sensitivity_groups,
    prior_shape_table,
    sequential_scan,
)

FIT_KW = {"draws": 300, "tune": 300, "chains": 2, "random_seed": 0}


def _events(rate_pre: float, rate_post: float, *, n_users: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for start, end, rate in (
        ("2026-01-01", "2026-01-28", rate_pre),
        ("2026-01-29", "2026-02-25", rate_post),
    ):
        days = pd.date_range(start, end)
        for user in range(n_users):
            for day in rng.choice(days, size=rng.integers(1, 4), replace=False):
                rows.append((user, day, int(rng.random() < rate)))
    return pd.DataFrame(rows, columns=["user_id", "datepart", "convert"])


# --------------------------------------------------------------------------- #
# Window generation — the equal-length invariant
# --------------------------------------------------------------------------- #


def test_windows_grow_on_both_sides_together():
    windows = matched_sequential_windows("2026-07-02", n_periods=3)
    assert [(w.n_days_pre, w.n_days_post) for w in windows] == [(7, 7), (14, 14), (21, 21)]


def test_windows_meet_at_the_intervention_without_overlapping():
    for w in matched_sequential_windows("2026-07-02", n_periods=3):
        assert w.post_start == pd.Timestamp("2026-07-02")
        assert w.pre_end == pd.Timestamp("2026-07-01")


def test_period_length_is_configurable():
    (w,) = matched_sequential_windows("2026-07-02", n_periods=1, period_days=30)
    assert w.n_days_pre == 30
    assert w.pre_start == pd.Timestamp("2026-06-02")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_periods": 0}, "n_periods"),
        ({"n_periods": 2, "period_days": 0}, "period_days"),
    ],
)
def test_window_generation_rejects_nonpositive_counts(kwargs, match):
    with pytest.raises(ValueError, match=match):
        matched_sequential_windows("2026-07-02", **kwargs)


# --------------------------------------------------------------------------- #
# Prior sensitivity
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sensitivity():
    pytest.importorskip("pymc")
    window = PrePostWindow("2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25")
    return prior_sensitivity_groups(_events(0.10, 0.25), window, **FIT_KW)


def test_sensitivity_returns_one_row_per_prior(sensitivity):
    table, _ = sensitivity
    assert len(table) == 3
    assert table["prior"].nunique() == 3


def test_sensitivity_table_matches_the_overlap_contract(sensitivity):
    table, _ = sensitivity
    assert {"prior", "lcl", "ucl", "mean_delta", "prob_delta_gt_0", "shift_from_primary"} <= set(
        table.columns
    )


def test_sensitivity_measures_the_shift_against_the_first_prior(sensitivity):
    table, _ = sensitivity
    assert table.iloc[0]["shift_from_primary"] == 0.0
    expected = table["mean_delta"] - table.iloc[0]["mean_delta"]
    pd.testing.assert_series_equal(
        table["shift_from_primary"],
        expected,
        check_names=False,
    )


def test_sensitivity_rejects_duplicate_prior_names():
    pytest.importorskip("pymc")
    window = PrePostWindow("2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25")
    with pytest.raises(ValueError, match="unique"):
        prior_sensitivity_groups(
            _events(0.10, 0.25),
            window,
            priors=("uniform", BetaPrior("uniform", 2.0, 2.0)),
            **FIT_KW,
        )


def test_sensitivity_rejects_an_unswept_primary():
    pytest.importorskip("pymc")
    window = PrePostWindow("2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25")
    with pytest.raises(ValueError, match="not among the priors swept"):
        prior_sensitivity_groups(
            _events(0.10, 0.25),
            window,
            priors=("uniform", "jeffreys"),
            primary="kruschke",
            **FIT_KW,
        )


def test_a_clear_effect_is_robust_to_the_prior(sensitivity):
    _, verdict = sensitivity
    assert verdict == "PRIOR_ROBUST"


def test_sensitivity_rejects_a_conflicting_prior_argument():
    window = PrePostWindow("2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25")
    with pytest.raises(ValueError, match="not `prior`"):
        prior_sensitivity_groups(pd.DataFrame(), window, prior="uniform")


def test_sensitivity_rejects_an_empty_prior_list():
    window = PrePostWindow("2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25")
    with pytest.raises(ValueError, match="empty"):
        prior_sensitivity_groups(pd.DataFrame(), window, priors=[])


# --------------------------------------------------------------------------- #
# Sequential scan
# --------------------------------------------------------------------------- #


def test_scan_returns_a_row_per_window():
    pytest.importorskip("pymc")
    windows = matched_sequential_windows("2026-01-29", n_periods=2, period_days=14)
    table, effects = sequential_scan(_events(0.10, 0.25), windows, **FIT_KW)

    assert list(table["period"]) == [1, 2]
    assert len(effects) == 2
    assert (table["n_days_pre"] == table["n_days_post"]).all()


def test_scan_rejects_an_empty_window_list():
    with pytest.raises(ValueError, match="empty"):
        sequential_scan(pd.DataFrame(), [])


# --------------------------------------------------------------------------- #
# prior_forest_rows — flattening several groups into one plot-ready frame
# --------------------------------------------------------------------------- #


def _group_table(means: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prior": ["noninformative", "weakly_informative", "informative"],
            "mean_delta": means,
            "lcl": [m - 0.01 for m in means],
            "ucl": [m + 0.01 for m in means],
        }
    )


def test_forest_rows_keeps_group_then_prior_order():
    rows = prior_forest_rows(
        {
            "video_newest": _group_table([0.025, 0.024, 0.020]),
            "stills_oldest": _group_table([0.002, 0.002, 0.002]),
        }
    )
    assert list(rows["group"])[:3] == ["video_newest"] * 3
    assert list(rows["prior"])[:3] == [
        "noninformative",
        "weakly_informative",
        "informative",
    ]
    assert list(rows["group"])[3:] == ["stills_oldest"] * 3


def test_forest_rows_stacks_the_first_group_at_the_top():
    """Row order is display order: y descends so the caller reads top-down."""
    rows = prior_forest_rows(
        {
            "video_newest": _group_table([0.025, 0.024, 0.020]),
            "stills_oldest": _group_table([0.002, 0.002, 0.002]),
        }
    )
    assert rows["y"].iloc[0] > rows["y"].iloc[-1]
    assert rows["y"].is_monotonic_decreasing


def test_forest_rows_converts_to_percentage_points():
    rows = prior_forest_rows({"video_newest": _group_table([0.025, 0.024, 0.020])})
    assert rows["mean_pp"].iloc[0] == pytest.approx(2.5)
    assert rows["lcl_pp"].iloc[0] == pytest.approx(1.5)
    assert rows["ucl_pp"].iloc[0] == pytest.approx(3.5)


def test_forest_rows_rejects_empty_input():
    with pytest.raises(ValueError, match="no groups"):
        prior_forest_rows({})


# --------------------------------------------------------------------------- #
# prior_shape_table — what was actually varied
# --------------------------------------------------------------------------- #


def test_shape_table_reports_one_row_per_prior_in_order():
    priors = [BetaPrior("weak", 2.0, 2.0), BetaPrior("strong", 20.0, 80.0)]
    table = prior_shape_table(priors)
    assert list(table["prior"]) == ["weak", "strong"]
    assert list(table["alpha"]) == [2.0, 20.0]
    assert table["prior_weight"].iloc[1] == pytest.approx(100.0)
    assert table["prior_mean"].iloc[1] == pytest.approx(0.2)
