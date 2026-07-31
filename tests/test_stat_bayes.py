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
    RopeDecision,
    _classify,
    best_two_sample,
    beta_binomial_two_sample,
    prior_sensitivity,
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
