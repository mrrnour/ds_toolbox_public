"""Unit tests for :mod:`dstoolbox.ml_funcs.stat_bayes`.

Fast tests (ROPE classification, table shape) run without PyMC.
BEST-fit tests are marked ``slow`` and depend on PyMC + ArviZ; they're
skipped automatically if either library is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pm = pytest.importorskip("pymc", reason="stat_bayes MCMC tests require pymc")
az = pytest.importorskip("arviz", reason="stat_bayes MCMC tests require arviz")

from dstoolbox.ml_funcs.stat_bayes import (  # noqa: E402
    AGREEING_STATES,
    CALLED_BAYES,
    CALLED_BOTH,
    CALLED_FREQ,
    CALLED_NONE,
    INCONCLUSIVE_PROB_THRESHOLD,
    VERDICTS,
    BestResult,
    BetaBinomialResult,
    BetaPrior,
    RopeDecision,
    _classify,
    _resolve_beta_prior,
    best_two_sample,
    beta_binomial_two_sample,
    beta_prior_from_baseline,
    call_agreement,
    is_call,
    is_flagged,
    prior_overlap_table,
    prior_sensitivity,
    prior_sensitivity_verdict,
    rope_comparison_table,
    rope_decision,
    rope_decision_normal,
)


# ---------------------------------------------------------------------------
# Fast tests (no MCMC)
# ---------------------------------------------------------------------------

def test_classify_positive():
    assert _classify(0.97, 0.02, 0.01) == "positive"


def test_classify_negative():
    assert _classify(0.01, 0.02, 0.97) == "negative"


def test_classify_equivalent():
    assert _classify(0.02, 0.96, 0.02) == "equivalent"


def test_classify_inconclusive():
    assert _classify(0.4, 0.3, 0.3) == "inconclusive"


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

def test_both_decision_rules_speak_the_same_vocabulary():
    """A renderer must not need to know which rule produced a label."""
    from dstoolbox.ml_funcs.stat_bayes_hier import verdict_without_rope

    banded = {
        _classify(*masses)
        for masses in [(0.97, 0.02, 0.01), (0.01, 0.02, 0.97),
                       (0.02, 0.96, 0.02), (0.4, 0.3, 0.3)]
    }
    unbanded = {verdict_without_rope(p) for p in (0.99, 0.01, 0.5)}
    assert banded <= set(VERDICTS)
    assert unbanded <= set(VERDICTS)


@pytest.mark.parametrize(
    ("decision", "expected"),
    [("positive", True), ("negative", True),
     ("equivalent", False), ("inconclusive", False)],
)
def test_is_call_names_an_effect_but_equivalence_is_not_one(decision, expected):
    assert is_call(decision) is expected


def test_is_call_refuses_a_label_outside_the_vocabulary():
    """A typo or a stale label must read as "no call", never as one.

    Downstream code uses this as its flag test, so an unrecognised string
    coming back ``True`` would inflate every call rate that consumes it.
    """
    assert not is_call("meaningful_positive")
    assert not is_call("")


@pytest.mark.parametrize("decision", ["positive", "negative"])
def test_is_flagged_needs_no_probability_for_a_directional_verdict(decision):
    """The ROPE already decided; the promotion clause never has to run."""
    assert is_flagged(decision) is True


@pytest.mark.parametrize(
    ("prob_gt_zero", "expected"),
    [(0.97, True), (0.03, True), (0.95, True), (0.05, True), (0.60, False), (0.94, False)],
)
def test_is_flagged_promotes_an_inconclusive_verdict_at_both_tails(prob_gt_zero, expected):
    """The rule is certainty about the sign, so it is symmetric about 0.5."""
    assert is_flagged("inconclusive", prob_gt_zero) is expected


def test_is_flagged_never_promotes_equivalent():
    """Equivalence is a finding, not an absence of evidence.

    A slice can be both inside the band and certain of its sign; promoting it
    would turn "too small to act on" into a call for an effect.
    """
    assert is_flagged("equivalent", 0.999) is False
    assert is_flagged("equivalent", 0.001) is False


@pytest.mark.parametrize("prob_gt_zero", [None, float("nan")])
def test_is_flagged_falls_back_to_the_verdict_without_a_probability(prob_gt_zero):
    """Results written before the direction rule existed carry no probability."""
    assert is_flagged("inconclusive", prob_gt_zero) is False
    assert is_flagged("positive", prob_gt_zero) is True


def test_is_flagged_honours_a_caller_supplied_threshold():
    assert is_flagged("inconclusive", 0.92) is False
    assert is_flagged("inconclusive", 0.92, threshold=0.90) is True


def test_default_threshold_mirrors_a_two_sided_five_percent_test():
    """Guards the constant the CSVs record the promotion bar as."""
    assert INCONCLUSIVE_PROB_THRESHOLD == 0.95


@pytest.mark.parametrize(
    ("bayes", "other", "expected"),
    [(True, True, CALLED_BOTH), (True, False, CALLED_BAYES),
     (False, True, CALLED_FREQ), (False, False, CALLED_NONE)],
)
def test_call_agreement_names_which_arms_made_the_call(bayes, other, expected):
    assert call_agreement(bayes, other) == expected


def test_agreeing_states_pool_the_two_opposite_ways_of_agreeing():
    """Both arms calling and neither calling are agreement; one-sided is not."""
    assert AGREEING_STATES == {CALLED_BOTH, CALLED_NONE}
    assert CALLED_BAYES not in AGREEING_STATES
    assert CALLED_FREQ not in AGREEING_STATES


def test_rope_decision_equivalent_bucket():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.02, scale=0.05, size=20_000)
    dec = rope_decision(samples, rope_low=-0.5, rope_high=0.5)
    assert dec.decision == "equivalent"
    assert dec.prob_in_rope >= 0.95


def test_rope_decision_positive_bucket():
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=1.0, scale=0.05, size=20_000)
    dec = rope_decision(samples, rope_low=-0.1, rope_high=0.1)
    assert dec.decision == "positive"
    assert dec.prob_gt_high >= 0.95


def test_rope_decision_inconclusive_when_narrow():
    rng = np.random.default_rng(2)
    samples = rng.normal(loc=0.02, scale=1.0, size=20_000)
    dec = rope_decision(samples, rope_low=-0.01, rope_high=0.01)
    assert dec.decision == "inconclusive"


def test_rope_decision_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="rope_low"):
        rope_decision(np.zeros(10), rope_low=0.5, rope_high=-0.5)


@pytest.mark.parametrize(
    ("mean", "sd", "half", "expected"),
    [(0.0, 0.2, 2.6, "equivalent"), (1.0, 0.05, 0.1, "positive"),
     (-1.0, 0.05, 0.1, "negative"), (0.02, 1.0, 0.01, "inconclusive")],
)
def test_rope_decision_normal_buckets(mean, sd, half, expected):
    dec = rope_decision_normal(mean, sd, rope_low=-half, rope_high=half)
    assert dec.decision == expected


def test_rope_decision_normal_agrees_with_the_sample_based_rule():
    """The point of the closed form: same rule, masses read off a normal.

    A reporting layer re-judging stored summaries against a second band must
    land where a refit would have, or the ladder it draws is not comparable
    to the rung that was actually fitted.
    """
    rng = np.random.default_rng(11)
    for mean, sd, half in [(0.0, 0.2, 0.5), (0.5, 0.1, 0.1), (-0.5, 0.1, 0.1),
                           (0.05, 0.5, 0.05)]:
        sampled = rope_decision(
            rng.normal(mean, sd, size=200_000), rope_low=-half, rope_high=half,
        )
        closed = rope_decision_normal(mean, sd, rope_low=-half, rope_high=half)
        assert closed.decision == sampled.decision
        assert closed.prob_in_rope == pytest.approx(sampled.prob_in_rope, abs=0.01)


def test_rope_decision_normal_rejects_a_degenerate_posterior():
    """An sd of zero means the HDI it was recovered from had no width."""
    with pytest.raises(ValueError, match="sd"):
        rope_decision_normal(0.0, 0.0, rope_low=-1.0, rope_high=1.0)


def test_rope_comparison_table_shape_and_columns():
    rng = np.random.default_rng(3)
    samples = rng.normal(0.02, 0.05, size=5_000)
    table = rope_comparison_table(
        samples,
        ropes={
            "stat": (-0.5, 0.5),
            "pct": (-0.01, 0.01),
            "biz": (None, None),
        },
    )
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["stat", "pct", "biz"]
    expected_cols = {
        "rope_low", "rope_high",
        "prob_gt_high", "prob_in_rope", "prob_lt_low", "decision",
    }
    assert expected_cols.issubset(table.columns)
    assert table.loc["biz", "decision"] == "undefined"
    assert table.loc["stat", "decision"] == "equivalent"


# ---------------------------------------------------------------------------
# Slow tests (real PyMC fit)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_best_two_sample_smoke_recovers_shift():
    """Two samples with a known 1σ gap → posterior HDI excludes zero."""
    rng = np.random.default_rng(42)
    y_control = rng.normal(loc=0.0, scale=1.0, size=60)
    y_treatment = rng.normal(loc=1.0, scale=1.0, size=60)
    result = best_two_sample(
        y_control, y_treatment,
        prior="kruschke",
        draws=500, tune=500, chains=2,
        random_seed=0, progressbar=False,
    )
    assert isinstance(result, BestResult)
    assert result.n_control == 60
    assert result.n_treatment == 60
    assert abs(result.posterior_mean_delta - 1.0) < 0.3
    assert result.hdi[0] > 0.0


@pytest.mark.slow
def test_prior_sensitivity_returns_shift_table():
    rng = np.random.default_rng(43)
    y_control = rng.normal(0.0, 1.0, size=40)
    y_treatment = rng.normal(0.5, 1.0, size=40)
    results, shift = prior_sensitivity(
        y_control, y_treatment,
        priors=("kruschke", "weakly_informative"),
        draws=300, tune=300, chains=2,
        random_seed=0, progressbar=False,
    )
    assert set(results) == {"kruschke", "weakly_informative"}
    assert list(shift.columns) == [
        "prior", "mean_delta", "lcl", "ucl", "shift_from_primary",
    ]
    assert len(shift) == 2
    # Primary row (first prior) has zero shift by construction.
    assert shift.iloc[0]["shift_from_primary"] == 0.0


# ---------------------------------------------------------------------------
# Beta-Binomial primitive
# ---------------------------------------------------------------------------

def test_beta_binomial_rejects_bad_trials():
    with pytest.raises(ValueError, match="trials_control"):
        beta_binomial_two_sample(50, 0, 60, 1000)


def test_beta_binomial_rejects_successes_above_trials():
    with pytest.raises(ValueError, match=r"successes_treatment"):
        beta_binomial_two_sample(50, 1000, 1500, 1000)


def test_beta_binomial_rejects_unknown_prior():
    with pytest.raises(ValueError, match="Unknown prior spec"):
        beta_binomial_two_sample(
            50, 1000, 60, 1000, prior="haldane", draws=100, tune=100, chains=1,
        )


@pytest.mark.slow
def test_beta_binomial_recovers_known_lift():
    """4pp positive lift with large N → posterior HDI comfortably above 0."""
    result = beta_binomial_two_sample(
        successes_control=500, trials_control=10_000,     # 5.00%
        successes_treatment=610, trials_treatment=10_000,  # 6.10%
        prior="uniform",
        draws=1000, tune=500, chains=2,
        random_seed=0, progressbar=False,
    )
    assert isinstance(result, BetaBinomialResult)
    assert result.trials_control == 10_000
    assert result.trials_treatment == 10_000
    assert result.rate_control == pytest.approx(0.05, abs=1e-6)
    assert result.rate_treatment == pytest.approx(0.061, abs=1e-6)
    # Expected posterior mean ≈ 0.061 - 0.050 = 0.011.
    assert result.posterior_mean_delta == pytest.approx(0.011, abs=0.002)
    # 95% HDI should exclude zero with these sample sizes.
    assert result.hdi[0] > 0.0
    assert result.hdi[1] < 0.03


@pytest.mark.slow
def test_beta_binomial_null_effect_hdi_straddles_zero():
    """Equal rates → posterior of delta should be centered near 0."""
    result = beta_binomial_two_sample(
        successes_control=500, trials_control=10_000,
        successes_treatment=505, trials_treatment=10_000,
        prior="uniform",
        draws=1000, tune=500, chains=2,
        random_seed=1, progressbar=False,
    )
    assert abs(result.posterior_mean_delta) < 0.005
    # HDI must contain zero when the true effect is negligible.
    assert result.hdi[0] < 0.0 < result.hdi[1]


@pytest.mark.slow
def test_beta_binomial_uniform_vs_jeffreys_close():
    """With N=1000+, uniform and Jeffreys priors should give near-identical δ."""
    kw = dict(
        successes_control=50, trials_control=1000,
        successes_treatment=70, trials_treatment=1000,
        draws=500, tune=500, chains=2,
        random_seed=2, progressbar=False,
    )
    uni = beta_binomial_two_sample(prior="uniform", **kw)
    jef = beta_binomial_two_sample(prior="jeffreys", **kw)
    assert abs(uni.posterior_mean_delta - jef.posterior_mean_delta) < 0.005


# ---------------------------------------------------------------------------
# Custom Beta priors (baseline-anchored)
# ---------------------------------------------------------------------------

def test_beta_prior_from_baseline_splits_weight():
    """Beta(5, 95) — a 5% baseline carrying 100 pseudo-observations."""
    prior = beta_prior_from_baseline(0.05, 100, name="weakly_informative")
    assert prior.name == "weakly_informative"
    assert prior.alpha == pytest.approx(5.0)
    assert prior.beta == pytest.approx(95.0)
    assert prior.weight == pytest.approx(100.0)
    assert prior.mean == pytest.approx(0.05)


def test_beta_prior_from_baseline_heavy_weight_keeps_mean():
    """Beta(25, 475) — same 5% mean, 5× the stubbornness."""
    prior = beta_prior_from_baseline(0.05, 500, name="informative")
    assert prior.alpha == pytest.approx(25.0)
    assert prior.beta == pytest.approx(475.0)
    assert prior.mean == pytest.approx(0.05)


def test_beta_prior_from_baseline_rejects_nonpositive_weight():
    with pytest.raises(ValueError, match="weight"):
        beta_prior_from_baseline(0.05, 0, name="informative")


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_beta_prior_from_baseline_rejects_rate_outside_open_unit(rate):
    with pytest.raises(ValueError, match="baseline_rate"):
        beta_prior_from_baseline(rate, 100, name="informative")


def test_beta_prior_rejects_nonpositive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        BetaPrior(name="broken", alpha=0.0, beta=1.0)


def test_resolve_beta_prior_passes_through_named_specs():
    assert _resolve_beta_prior("jeffreys") == BetaPrior("jeffreys", 0.5, 0.5)
    assert _resolve_beta_prior("uniform") == BetaPrior("uniform", 1.0, 1.0)


def test_resolve_beta_prior_returns_custom_unchanged():
    custom = beta_prior_from_baseline(0.02, 250, name="informative")
    assert _resolve_beta_prior(custom) is custom


def test_resolve_beta_prior_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown prior spec"):
        _resolve_beta_prior("haldane")


@pytest.mark.slow
def test_beta_binomial_accepts_custom_prior_and_keeps_label():
    prior = beta_prior_from_baseline(0.05, 100, name="weakly_informative")
    result = beta_binomial_two_sample(
        successes_control=50, trials_control=1000,
        successes_treatment=70, trials_treatment=1000,
        prior=prior,
        draws=500, tune=500, chains=2,
        random_seed=4, progressbar=False,
    )
    assert result.prior_spec == "weakly_informative"


@pytest.mark.slow
def test_heavy_prior_shrinks_delta_toward_zero():
    """A stubborn baseline prior on both arms pulls δ back toward no-change."""
    kw = dict(
        successes_control=5, trials_control=100,      # 5%
        successes_treatment=15, trials_treatment=100,  # 15%
        draws=500, tune=500, chains=2,
        random_seed=5, progressbar=False,
    )
    reference = beta_binomial_two_sample(prior="jeffreys", **kw)
    skeptical = beta_binomial_two_sample(
        prior=beta_prior_from_baseline(0.05, 500, name="informative"), **kw,
    )
    assert abs(skeptical.posterior_mean_delta) < abs(reference.posterior_mean_delta)


# ---------------------------------------------------------------------------
# Prior-sensitivity verdict (HDI overlap)
# ---------------------------------------------------------------------------

def _shift_table(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"prior": n, "lcl": lo, "ucl": hi} for n, lo, hi in rows]
    )


def test_verdict_prior_robust_when_all_hdis_overlap():
    table = _shift_table([
        ("noninformative", -0.010, 0.030),
        ("weakly_informative", -0.005, 0.025),
        ("informative", -0.002, 0.012),
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_ROBUST"


def test_verdict_prior_driven_when_an_hdi_separates():
    table = _shift_table([
        ("noninformative", 0.020, 0.050),
        ("informative", -0.030, -0.005),
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_DRIVEN"


def test_verdict_touching_bounds_are_sensitive_not_driven():
    """Intervals that meet at a point still overlap, but they agree on nothing."""
    table = _shift_table([
        ("noninformative", 0.000, 0.020),
        ("informative", -0.020, 0.000),
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_SENSITIVE"


def test_verdict_sensitive_when_the_mean_travels_far():
    """Overlapping intervals whose centres drift are not robust."""
    table = pd.DataFrame([
        {"prior": "noninformative", "lcl": -0.010, "ucl": 0.030,
         "mean_delta": 0.010},
        {"prior": "informative", "lcl": -0.005, "ucl": 0.035,
         "mean_delta": 0.026},
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_SENSITIVE"


def test_verdict_sensitive_when_the_conclusion_flips():
    """Same data, different prior, different call: report the range, not a number."""
    table = pd.DataFrame([
        {"prior": "noninformative", "lcl": 0.001, "ucl": 0.030,
         "mean_delta": 0.015, "prob_delta_gt_0": 0.99},
        {"prior": "informative", "lcl": -0.002, "ucl": 0.028,
         "mean_delta": 0.013, "prob_delta_gt_0": 0.90},
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_SENSITIVE"


def test_overlap_table_scores_a_contained_interval_as_full_overlap():
    """A prior that only sharpens the estimate has not changed the answer."""
    table = _shift_table([
        ("noninformative", -0.010, 0.030),
        ("informative", -0.002, 0.012),
    ])
    graded = prior_overlap_table(table)
    assert list(graded["hdi_overlap"]) == [1.0, 1.0]
    assert list(graded["is_primary"]) == [True, False]


def test_overlap_table_reports_the_shift_in_reference_widths():
    table = pd.DataFrame([
        {"prior": "noninformative", "lcl": 0.0, "ucl": 0.040,
         "mean_delta": 0.020},
        {"prior": "informative", "lcl": 0.0, "ucl": 0.040,
         "mean_delta": 0.024},
    ])
    assert prior_overlap_table(table)["shift_hdi_frac"].iloc[1] == pytest.approx(0.1)


def test_verdict_honours_explicit_primary():
    table = _shift_table([
        ("informative", -0.030, -0.005),
        ("noninformative", 0.020, 0.050),
    ])
    assert prior_sensitivity_verdict(table, primary="noninformative") == "PRIOR_DRIVEN"


def test_verdict_robust_intervals_may_still_straddle_zero():
    """PRIOR_ROBUST reports prior influence, not whether the effect is conclusive."""
    table = _shift_table([
        ("noninformative", -0.010, 0.026),
        ("informative", -0.009, 0.021),
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_ROBUST"


def test_verdict_single_row_is_robust():
    assert prior_sensitivity_verdict(_shift_table([("jeffreys", -0.01, 0.01)])) == "PRIOR_ROBUST"


def test_verdict_rejects_unknown_primary():
    table = _shift_table([("jeffreys", -0.01, 0.01)])
    with pytest.raises(ValueError, match="not in shift_table"):
        prior_sensitivity_verdict(table, primary="informative")


def test_verdict_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        prior_sensitivity_verdict(pd.DataFrame({"prior": ["jeffreys"]}))

