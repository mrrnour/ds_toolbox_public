"""Tests for the pre/post Beta-Binomial workflow.

Fits are deliberately small (few draws, 2 chains, fixed seed) — these check
wiring, validation and invariants, not convergence. The window validation
tests need no fit at all and run instantly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dstoolbox.ml_funcs.stat_bayes_group import (
    ROPE_BANDS,
    PrePostWindow,
    aggregate_counts,
    fit_prepost,
    rope_from_control,
)

FIT_KW = {"draws": 300, "tune": 300, "chains": 2, "random_seed": 0}

PRE = ("2026-01-01", "2026-01-10")
POST = ("2026-01-11", "2026-01-20")


def _window(**kw) -> PrePostWindow:
    return PrePostWindow(*PRE, *POST, **kw)


def _events(
    rate_pre: float,
    rate_post: float,
    *,
    n_users: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Event-level frame: every user active on every day of both windows."""
    rng = np.random.default_rng(seed)
    rows = []
    for start, end, rate in (
        (*PRE, rate_pre),
        (*POST, rate_post),
    ):
        days = pd.date_range(start, end)
        for user in range(n_users):
            # Volume varies across users so the unit structure is real.
            for day in rng.choice(days, size=rng.integers(1, 4), replace=False):
                rows.append((user, day, int(rng.random() < rate)))
    return pd.DataFrame(rows, columns=["user_id", "datepart", "convert"])


# --------------------------------------------------------------------------- #
# Window validation — the guardrail, no fitting involved
# --------------------------------------------------------------------------- #


def test_window_reports_inclusive_day_counts():
    w = _window()
    assert (w.n_days_pre, w.n_days_post) == (10, 10)


def test_window_allows_unequal_lengths():
    w = PrePostWindow("2026-01-01", "2026-01-20", "2026-01-21", "2026-01-30")
    assert w.n_days_pre == 20
    assert w.n_days_post == 10


@pytest.mark.parametrize(
    ("dates", "match"),
    [
        (("2026-01-10", "2026-01-01", "2026-01-11", "2026-01-20"), "pre_start"),
        (("2026-01-01", "2026-01-10", "2026-01-20", "2026-01-11"), "post_start"),
        (("2026-01-01", "2026-01-15", "2026-01-11", "2026-01-25"), "overlap"),
    ],
)
def test_window_rejects_incoherent_dates(dates, match):
    with pytest.raises(ValueError, match=match):
        PrePostWindow(*dates)


def test_window_boundaries_are_inclusive_and_disjoint():
    w = _window()
    dates = pd.Series(pd.date_range("2025-12-30", "2026-01-22"))
    pre, post = w.pre_mask(dates), w.post_mask(dates)

    assert not (pre & post).any()
    assert dates[pre].min() == pd.Timestamp(PRE[0])
    assert dates[pre].max() == pd.Timestamp(PRE[1])
    assert dates[post].min() == pd.Timestamp(POST[0])
    assert dates[post].max() == pd.Timestamp(POST[1])


# --------------------------------------------------------------------------- #
# Aggregation and the band
# --------------------------------------------------------------------------- #


def test_aggregate_counts_collapses_to_one_row_per_unit():
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "convert": [1, 0, 1, 0, 0],
        }
    )
    counts = aggregate_counts(events, np.ones(5, dtype=bool))

    assert len(counts) == 2
    assert counts["trials"].tolist() == [3, 2]
    assert counts["successes"].tolist() == [2, 0]


def test_aggregate_counts_rejects_an_empty_window():
    events = pd.DataFrame({"user_id": [1], "convert": [1]})
    with pytest.raises(ValueError, match="selected no rows"):
        aggregate_counts(events, np.zeros(1, dtype=bool))


def test_rope_scales_with_the_control():
    assert rope_from_control(0.20, 0.10) == pytest.approx((-0.02, 0.02))
    assert rope_from_control(0.02, 0.10) == pytest.approx((-0.002, 0.002))


def test_rope_rejects_a_nonpositive_coefficient():
    with pytest.raises(ValueError, match="pct_coef"):
        rope_from_control(0.20, 0.0)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lift_effect():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    return fit_prepost(_events(0.10, 0.25), _window(), rope_pct_coef=0.10, **FIT_KW)


def test_detects_a_real_lift(lift_effect):
    assert lift_effect.estimate > 0
    assert lift_effect.prob_gt_zero > 0.95
    assert lift_effect.decision == "positive"
    assert lift_effect.lcl < lift_effect.estimate < lift_effect.ucl


def test_finds_nothing_when_nothing_changed():
    pytest.importorskip("pymc")
    effect = fit_prepost(_events(0.15, 0.15, seed=7), _window(), rope_pct_coef=0.10, **FIT_KW)
    assert effect.decision != "positive"
    assert effect.decision != "negative"


def test_records_the_window_and_the_band(lift_effect):
    assert lift_effect.window.n_days_pre == 10
    assert lift_effect.rope is not None
    # Band is ±10% of the pre-period mu, not an absolute percentage point.
    assert lift_effect.rope.rope_high == pytest.approx(0.10 * lift_effect.mu_control_mean)
    assert lift_effect.rope.rope_low == -lift_effect.rope.rope_high


def test_no_band_gives_a_direction_only_verdict():
    pytest.importorskip("pymc")
    effect = fit_prepost(_events(0.10, 0.25), _window(), **FIT_KW)
    assert effect.rope is None
    assert effect.decision in {"positive", "negative", "inconclusive"}
    # "equivalent" is unreachable without a band.
    assert "rope_low" not in effect.to_row()


def test_pooled_rate_differs_from_mu(lift_effect):
    # Not an assertion about which is bigger — only that they are distinct
    # estimands and the result exposes both without conflating them.
    assert lift_effect.pooled_rate_control != lift_effect.mu_control_mean


def test_to_row_is_flat_and_serialisable(lift_effect):
    row = lift_effect.to_row()
    assert pd.DataFrame([row]).shape[0] == 1
    assert row["n_days_pre"] == row["n_days_post"] == 10
    assert row["decision"] == lift_effect.decision
    assert not any(isinstance(v, np.ndarray) for v in row.values())


# --------------------------------------------------------------------------- #
# Every band reported side by side
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def all_bands_effect():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    return fit_prepost(
        _events(0.10, 0.25),
        _window(),
        rope_stat_coef=0.1,
        rope_pct_coef=0.10,
        rope_biz=(-0.005, 0.005),
        **FIT_KW,
    )


def test_all_three_bands_get_their_own_verdict(all_bands_effect):
    row = all_bands_effect.to_row()
    for band in ROPE_BANDS:
        assert row[f"decision_{band}"] is not None
        assert row[f"rope_{band}_low"] < row[f"rope_{band}_high"]


def test_headline_decision_is_the_first_band_supplied(all_bands_effect):
    # ROPE_BANDS order decides precedence; "stat" was supplied, so it wins.
    assert all_bands_effect.decision == all_bands_effect.ropes["stat"].decision


def test_unsupplied_bands_are_present_but_empty(lift_effect):
    # Only rope_pct_coef was given, so the schema is stable but stat/biz
    # carry no verdict — a summary table over several fits still concatenates.
    row = lift_effect.to_row()
    assert row["decision_pct"] == lift_effect.ropes["pct"].decision
    for band in ("stat", "biz"):
        assert row[f"decision_{band}"] is None
        assert np.isnan(row[f"rope_{band}_low"])


def test_str_reports_one_line_per_band(all_bands_effect):
    text = str(all_bands_effect)
    for band in ROPE_BANDS:
        assert f"{band:<4} ROPE" in text


def test_str_falls_back_to_the_headline_without_a_band():
    pytest.importorskip("pymc")
    effect = fit_prepost(_events(0.10, 0.25), _window(), **FIT_KW)
    assert "no ROPE" in str(effect)
    assert effect.decision in str(effect)


def test_periods_are_fitted_on_different_seeds(lift_effect):
    # Same seed for both would correlate the chains and shrink the delta.
    assert not np.array_equal(
        lift_effect.fit_control.mu_samples, lift_effect.fit_treatment.mu_samples
    )


def test_rejects_missing_columns():
    events = pd.DataFrame({"user_id": [1], "datepart": ["2026-01-01"]})
    with pytest.raises(ValueError, match="missing column"):
        fit_prepost(events, _window(), **FIT_KW)
