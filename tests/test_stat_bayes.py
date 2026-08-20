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
    BestResult,
    BetaBinomialResult,
    BetaPrior,
    RopeDecision,
    _classify,
    _resolve_beta_prior,
    best_two_sample,
    beta_binomial_prior_sensitivity,
    beta_binomial_two_sample,
    beta_prior_from_baseline,
    prior_overlap_table,
    prior_sensitivity,
    prior_sensitivity_verdict,
    rope_comparison_table,
    rope_decision,
)


# ---------------------------------------------------------------------------
# Fast tests (no MCMC)
# ---------------------------------------------------------------------------

def test_classify_meaningful_positive():
    assert _classify(0.97, 0.02, 0.01) == "meaningful_positive"


def test_classify_meaningful_negative():
    assert _classify(0.01, 0.02, 0.97) == "meaningful_negative"


def test_classify_equivalent():
    assert _classify(0.02, 0.96, 0.02) == "equivalent"


def test_classify_inconclusive():
    assert _classify(0.4, 0.3, 0.3) == "inconclusive"


def test_rope_decision_equivalent_bucket():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.02, scale=0.05, size=20_000)
    dec = rope_decision(samples, rope_low=-0.5, rope_high=0.5)
    assert dec.decision == "equivalent"
    assert dec.prob_in_rope >= 0.95


def test_rope_decision_meaningful_positive_bucket():
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=1.0, scale=0.05, size=20_000)
    dec = rope_decision(samples, rope_low=-0.1, rope_high=0.1)
    assert dec.decision == "meaningful_positive"
    assert dec.prob_gt_high >= 0.95


def test_rope_decision_inconclusive_when_narrow():
    rng = np.random.default_rng(2)
    samples = rng.normal(loc=0.02, scale=1.0, size=20_000)
    dec = rope_decision(samples, rope_low=-0.01, rope_high=0.01)
    assert dec.decision == "inconclusive"


def test_rope_decision_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="rope_low"):
        rope_decision(np.zeros(10), rope_low=0.5, rope_high=-0.5)


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
    y_pre = rng.normal(loc=0.0, scale=1.0, size=60)
    y_post = rng.normal(loc=1.0, scale=1.0, size=60)
    result = best_two_sample(
        y_pre, y_post,
        prior="kruschke",
        draws=500, tune=500, chains=2,
        random_seed=0, progressbar=False,
    )
    assert isinstance(result, BestResult)
    assert result.n_pre == 60
    assert result.n_post == 60
    assert abs(result.posterior_mean_delta - 1.0) < 0.3
    assert result.hdi[0] > 0.0


@pytest.mark.slow
def test_prior_sensitivity_returns_shift_table():
    rng = np.random.default_rng(43)
    y_pre = rng.normal(0.0, 1.0, size=40)
    y_post = rng.normal(0.5, 1.0, size=40)
    results, shift = prior_sensitivity(
        y_pre, y_post,
        priors=("kruschke", "weakly_informative"),
        draws=300, tune=300, chains=2,
        random_seed=0, progressbar=False,
    )
    assert set(results) == {"kruschke", "weakly_informative"}
    assert list(shift.columns) == [
        "prior", "mean_delta", "hdi_low", "hdi_high", "shift_from_primary",
    ]
    assert len(shift) == 2
    # Primary row (first prior) has zero shift by construction.
    assert shift.iloc[0]["shift_from_primary"] == 0.0


# ---------------------------------------------------------------------------
# Beta-Binomial primitive
# ---------------------------------------------------------------------------

def test_beta_binomial_rejects_bad_trials():
    with pytest.raises(ValueError, match="trials_pre"):
        beta_binomial_two_sample(50, 0, 60, 1000)


def test_beta_binomial_rejects_successes_above_trials():
    with pytest.raises(ValueError, match=r"successes_post"):
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
        successes_pre=500,   trials_pre=10_000,   # 5.00%
        successes_post=610,  trials_post=10_000,  # 6.10%
        prior="uniform",
        draws=1000, tune=500, chains=2,
        random_seed=0, progressbar=False,
    )
    assert isinstance(result, BetaBinomialResult)
    assert result.trials_pre == 10_000
    assert result.trials_post == 10_000
    assert result.rate_pre == pytest.approx(0.05, abs=1e-6)
    assert result.rate_post == pytest.approx(0.061, abs=1e-6)
    # Expected posterior mean ≈ 0.061 - 0.050 = 0.011.
    assert result.posterior_mean_delta == pytest.approx(0.011, abs=0.002)
    # 95% HDI should exclude zero with these sample sizes.
    assert result.hdi[0] > 0.0
    assert result.hdi[1] < 0.03


@pytest.mark.slow
def test_beta_binomial_null_effect_hdi_straddles_zero():
    """Equal rates → posterior of delta should be centered near 0."""
    result = beta_binomial_two_sample(
        successes_pre=500,   trials_pre=10_000,
        successes_post=505,  trials_post=10_000,
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
        successes_pre=50, trials_pre=1000,
        successes_post=70, trials_post=1000,
        draws=500, tune=500, chains=2,
        random_seed=2, progressbar=False,
    )
    uni = beta_binomial_two_sample(prior="uniform", **kw)
    jef = beta_binomial_two_sample(prior="jeffreys", **kw)
    assert abs(uni.posterior_mean_delta - jef.posterior_mean_delta) < 0.005


@pytest.mark.slow
def test_beta_binomial_prior_sensitivity_shift_table():
    """beta_binomial_prior_sensitivity returns matching results + well-formed shift table."""
    results, shift = beta_binomial_prior_sensitivity(
        successes_pre=50,   trials_pre=1000,
        successes_post=70,  trials_post=1000,
        priors=("uniform", "jeffreys"),
        draws=500, tune=500, chains=2,
        random_seed=3, progressbar=False,
    )
    assert set(results) == {"uniform", "jeffreys"}
    assert all(isinstance(r, BetaBinomialResult) for r in results.values())
    assert list(shift.columns) == [
        "prior", "mean_delta", "hdi_low", "hdi_high", "shift_from_primary",
    ]
    assert len(shift) == 2
    assert shift.iloc[0]["shift_from_primary"] == 0.0


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
        successes_pre=50,   trials_pre=1000,
        successes_post=70,  trials_post=1000,
        prior=prior,
        draws=500, tune=500, chains=2,
        random_seed=4, progressbar=False,
    )
    assert result.prior_spec == "weakly_informative"


@pytest.mark.slow
def test_heavy_prior_shrinks_delta_toward_zero():
    """A stubborn baseline prior on both windows pulls δ back toward no-change."""
    kw = dict(
        successes_pre=5,   trials_pre=100,     # 5%
        successes_post=15, trials_post=100,    # 15%
        draws=500, tune=500, chains=2,
        random_seed=5, progressbar=False,
    )
    reference = beta_binomial_two_sample(prior="jeffreys", **kw)
    skeptical = beta_binomial_two_sample(
        prior=beta_prior_from_baseline(0.05, 500, name="informative"), **kw,
    )
    assert abs(skeptical.posterior_mean_delta) < abs(reference.posterior_mean_delta)


@pytest.mark.slow
def test_prior_sensitivity_mixes_named_and_custom_priors():
    results, shift = beta_binomial_prior_sensitivity(
        successes_pre=50,   trials_pre=1000,
        successes_post=70,  trials_post=1000,
        priors=(
            "jeffreys",
            beta_prior_from_baseline(0.05, 100, name="weakly_informative"),
            beta_prior_from_baseline(0.05, 500, name="informative"),
        ),
        draws=400, tune=400, chains=2,
        random_seed=6, progressbar=False,
    )
    assert list(results) == ["jeffreys", "weakly_informative", "informative"]
    assert list(shift["prior"]) == ["jeffreys", "weakly_informative", "informative"]
    assert shift.iloc[0]["shift_from_primary"] == 0.0


def test_prior_sensitivity_rejects_duplicate_prior_names():
    with pytest.raises(ValueError, match="unique"):
        beta_binomial_prior_sensitivity(
            50, 1000, 70, 1000,
            priors=("jeffreys", BetaPrior("jeffreys", 2.0, 2.0)),
        )


# ---------------------------------------------------------------------------
# Prior-sensitivity verdict (HDI overlap)
# ---------------------------------------------------------------------------

def _shift_table(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"prior": n, "hdi_low": lo, "hdi_high": hi} for n, lo, hi in rows]
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
        {"prior": "noninformative", "hdi_low": -0.010, "hdi_high": 0.030,
         "mean_delta": 0.010},
        {"prior": "informative", "hdi_low": -0.005, "hdi_high": 0.035,
         "mean_delta": 0.026},
    ])
    assert prior_sensitivity_verdict(table) == "PRIOR_SENSITIVE"


def test_verdict_sensitive_when_the_conclusion_flips():
    """Same data, different prior, different call: report the range, not a number."""
    table = pd.DataFrame([
        {"prior": "noninformative", "hdi_low": 0.001, "hdi_high": 0.030,
         "mean_delta": 0.015, "prob_delta_gt_0": 0.99},
        {"prior": "informative", "hdi_low": -0.002, "hdi_high": 0.028,
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
        {"prior": "noninformative", "hdi_low": 0.0, "hdi_high": 0.040,
         "mean_delta": 0.020},
        {"prior": "informative", "hdi_low": 0.0, "hdi_high": 0.040,
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

