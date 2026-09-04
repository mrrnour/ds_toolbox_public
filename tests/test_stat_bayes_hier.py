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


def _counts(rate: float, n_users: int = 250, seed: int = 0):
    rng = np.random.default_rng(seed)
    trials = rng.integers(1, 25, n_users)
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
    assert fit.n_users == len(trials)
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
    trials, successes = _counts(0.50, n_users=40)
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
    trials, successes = _counts(0.12, n_users=500, seed=3)
    half = len(trials) // 2
    pre = hier_beta_binomial_fit(trials[:half], successes[:half], **FIT_KW)
    post = hier_beta_binomial_fit(trials[half:], successes[half:], **FIT_KW)

    delta = post.mu_samples - pre.mu_samples
    assert verdict_without_rope(float((delta > 0).mean())) == "inconclusive"


def test_fit_accepts_unequal_period_sizes():
    big = hier_beta_binomial_fit(*_counts(0.12, n_users=400, seed=5), **FIT_KW)
    small = hier_beta_binomial_fit(*_counts(0.12, n_users=60, seed=6), **FIT_KW)
    assert big.n_users == 400
    assert small.n_users == 60


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
