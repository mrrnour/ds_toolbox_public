"""Tests for the hierarchical Beta-Binomial module.

Fits are kept deliberately small (few draws, 2 chains, fixed seed) — these
check wiring and invariants, not convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from dstoolbox.ml_funcs.stat_bayes import BetaPrior
from dstoolbox.ml_funcs.stat_bayes_hier import (
    hier_beta_binomial_fit,
    verdict_without_rope,
)

pytest.importorskip("pymc")
pytest.importorskip("arviz")

FIT_KW = {"draws": 300, "tune": 300, "chains": 2, "random_seed": 0}


def _counts(rate: float, n_units: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    trials = rng.integers(1, 25, n_units)
    return trials, rng.binomial(trials, rate)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("trials", "successes", "match"),
    [
        ([10, 5], [1], "same length"),
        ([], [], "empty"),
        ([10, 0], [1, 0], "must be positive"),
        ([10, 5], [11, 1], r"\[0, trials\]"),
        ([10, 5], [-1, 1], r"\[0, trials\]"),
    ],
)
def test_rejects_malformed_counts(trials, successes, match):
    with pytest.raises(ValueError, match=match):
        hier_beta_binomial_fit(trials, successes, **FIT_KW)


def test_rejects_nonpositive_kappa_prior():
    with pytest.raises(ValueError, match="kappa_prior"):
        hier_beta_binomial_fit([5, 5], [1, 2], kappa_prior=(0.0, 0.1), **FIT_KW)


# --------------------------------------------------------------------------- #
# Single-period fit
# --------------------------------------------------------------------------- #

def test_fit_recovers_the_generating_rate():
    trials, successes = _counts(0.20)
    fit = hier_beta_binomial_fit(trials, successes, **FIT_KW)

    assert fit.mu_mean == pytest.approx(0.20, abs=0.04)
    assert fit.kappa_mean > 0
    assert fit.n_units == len(trials)
    assert fit.trials == int(trials.sum())
    assert fit.successes == int(successes.sum())
    assert fit.rate == pytest.approx(fit.successes / fit.trials)
    assert fit.prior_spec == "uniform"
    assert fit.rhat_max < 1.05
    assert fit.ess_min > 50


def test_fit_records_the_named_prior():
    trials, successes = _counts(0.20)
    prior = BetaPrior(name="anchored", alpha=8.0, beta=32.0)
    assert hier_beta_binomial_fit(trials, successes, prior=prior, **FIT_KW).prior_spec == "anchored"


def test_heavier_prior_pulls_mu_toward_its_mean():
    # 40 users at a 50% rate, against a prior worth ~400 pseudo-trials at 10%.
    trials, successes = _counts(0.50, n_units=40)
    flat = hier_beta_binomial_fit(trials, successes, **FIT_KW)
    pulled = hier_beta_binomial_fit(
        trials, successes,
        prior=BetaPrior(name="strong", alpha=40.0, beta=360.0), **FIT_KW,
    )
    assert pulled.mu_mean < flat.mu_mean


# --------------------------------------------------------------------------- #
# Pre/post contrast, composed from two independent fits (how callers use it)
# --------------------------------------------------------------------------- #

def test_contrast_detects_a_real_lift():
    trials_pre, successes_pre = _counts(0.10, seed=1)
    trials_post, successes_post = _counts(0.16, seed=2)
    pre = hier_beta_binomial_fit(trials_pre, successes_pre, **FIT_KW)
    post = hier_beta_binomial_fit(trials_post, successes_post, **FIT_KW)

    delta = post.mu_samples - pre.mu_samples
    assert delta.mean() > 0
    assert verdict_without_rope(float((delta > 0).mean())) == "positive"
    assert post.mu_mean > pre.mu_mean
    assert max(pre.rhat_max, post.rhat_max) < 1.05


def test_contrast_finds_nothing_when_nothing_changed():
    # A genuine A/A: one pool drawn at a single rate, split in half. Using two
    # *different* seeds would not be null — independent draws of 250 users
    # differ by ~1pp by chance alone, which the model would rightly detect.
    trials, successes = _counts(0.12, n_units=500, seed=3)
    half = len(trials) // 2
    pre = hier_beta_binomial_fit(trials[:half], successes[:half], **FIT_KW)
    post = hier_beta_binomial_fit(trials[half:], successes[half:], **FIT_KW)

    delta = post.mu_samples - pre.mu_samples
    assert verdict_without_rope(float((delta > 0).mean())) == "inconclusive"


def test_fit_accepts_unequal_period_sizes():
    big = hier_beta_binomial_fit(*_counts(0.12, n_units=400, seed=5), **FIT_KW)
    small = hier_beta_binomial_fit(*_counts(0.12, n_units=60, seed=6), **FIT_KW)
    assert big.n_units == 400
    assert small.n_units == 60


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("prob", "expected"),
    [(0.99, "positive"), (0.95, "positive"), (0.01, "negative"),
     (0.05, "negative"), (0.50, "inconclusive"), (0.94, "inconclusive")],
)
def test_verdict_thresholds(prob, expected):
    assert verdict_without_rope(prob, 0.95) == expected


@pytest.mark.parametrize(
    ("prob", "threshold"), [(1.5, 0.95), (-0.1, 0.95), (0.5, 0.0), (0.5, 1.0)],
)
def test_verdict_rejects_out_of_range_inputs(prob, threshold):
    with pytest.raises(ValueError):
        verdict_without_rope(prob, threshold)


# --------------------------------------------------------------------------- #
# mu_weighted — the trial-weighted companion to mu
# --------------------------------------------------------------------------- #

def _shrunk(n, k, mu, kappa):
    """The per-unit posterior mean the weighted rate averages over."""
    return (mu * kappa + k) / (kappa + n)


def test_weighted_rate_centres_on_the_shrinkage_weighted_mean():
    """Bootstrap weights are mean-1, so averaging draws recovers the plug-in.

    Needs a realistic unit count: the bootstrap of a *ratio* carries an
    ``O(1/U)`` bias, which is plainly visible with a handful of units.
    """
    from dstoolbox.ml_funcs.stat_bayes_hier import _weighted_rate_samples

    rng = np.random.default_rng(3)
    n = rng.integers(1, 60, size=2_000)
    k = rng.binomial(n, 0.35)
    mu = np.full(20_000, 0.35)
    kappa = np.full(20_000, 20.0)

    got = _weighted_rate_samples(n, k, mu, kappa, random_seed=0)
    theta = _shrunk(n, k, 0.35, 20.0)
    want = float(np.sum(n * theta) / n.sum())
    assert got.mean() == pytest.approx(want, abs=5e-4)


def test_weighted_rate_spread_survives_fixed_mu_and_kappa():
    """The unit bootstrap must contribute width even with the model frozen."""
    from dstoolbox.ml_funcs.stat_bayes_hier import _weighted_rate_samples

    rng = np.random.default_rng(0)
    n = rng.integers(1, 60, size=400)
    k = rng.binomial(n, 0.3)
    mu = np.full(4_000, 0.3)
    kappa = np.full(4_000, 25.0)

    got = _weighted_rate_samples(n, k, mu, kappa, random_seed=1)
    assert got.std() > 0.0
    # Wide enough to matter, tight enough to be a rate.
    assert 1e-4 < got.std() < 0.1


def test_weighted_rate_leans_toward_the_heavy_units():
    """One dominant unit should drag the weighted rate toward its own rate."""
    from dstoolbox.ml_funcs.stat_bayes_hier import _weighted_rate_samples

    # Nine light units never convert; one huge unit converts half the time.
    n = np.array([2] * 9 + [10_000])
    k = np.array([0] * 9 + [5_000])
    mu = np.full(2_000, 0.1)
    kappa = np.full(2_000, 10.0)

    weighted = _weighted_rate_samples(n, k, mu, kappa, random_seed=0).mean()
    per_unit_mean = float(np.mean(_shrunk(n, k, 0.1, 10.0)))
    assert weighted > per_unit_mean
    assert weighted == pytest.approx(0.5, abs=0.02)


def test_weighted_rate_is_reproducible_for_a_given_seed():
    from dstoolbox.ml_funcs.stat_bayes_hier import _weighted_rate_samples

    n = np.array([5, 50, 500])
    k = np.array([1, 10, 100])
    mu = np.full(500, 0.2)
    kappa = np.full(500, 15.0)

    a = _weighted_rate_samples(n, k, mu, kappa, random_seed=7)
    b = _weighted_rate_samples(n, k, mu, kappa, random_seed=7)
    assert a == pytest.approx(b)


def test_fit_exposes_both_rates():
    n = [10, 3, 25, 40]
    k = [1, 0, 4, 30]
    fit = hier_beta_binomial_fit(n, k, draws=300, tune=300, chains=2, random_seed=0)
    assert fit.mu_weighted_samples.shape == fit.mu_samples.shape
    assert 0.0 < fit.mu_weighted_mean < 1.0
    # The heavy converter pulls the weighted rate above the per-unit one.
    assert fit.mu_weighted_mean > fit.mu_mean
