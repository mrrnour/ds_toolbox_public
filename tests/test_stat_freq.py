"""Unit tests for :mod:`dstoolbox.ml_funcs.stat_freq`.

All tests here are fast and dependency-light — ``scipy`` is a base
dependency, so nothing needs ``importorskip``. The 10k-permutation case
is marked ``slow`` because it is the only one that costs real time.

The reference points are deliberately external: the naive path is checked
against :func:`scipy.stats.ttest_ind`, and the clustered path against a
hand-evaluation of formula (6) in Deng et al. (2018) — the same formula
``gettyab.stats.var_delta`` implements.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ttest_ind

from dstoolbox.ml_funcs.stat_freq import (
    WelchTestResult,
    _collapse_clusters,
    _delta_var,
    delta_method_two_sample,
    permutation_welch_two_sample,
    student_t_two_sample,
    welch_t_two_sample,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def daily_rates() -> tuple[np.ndarray, np.ndarray]:
    """Two short unclustered series, standing in for daily conversion rates."""
    rng = np.random.default_rng(7)
    pre = rng.normal(loc=3.0, scale=0.4, size=30)
    post = rng.normal(loc=3.5, scale=0.7, size=14)
    return pre, post


def _clustered_events(
    n_clusters: int,
    reps: int,
    rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 0/1 events with strong within-cluster correlation.

    Every cluster contributes ``reps`` rows that share a single Bernoulli
    draw, so the within-cluster correlation is 1 and the naive iid
    variance understates the truth by roughly ``reps``.

    Args:
        n_clusters: Number of independent clusters.
        reps: Rows contributed by each cluster.
        rate: Success probability of the shared draw.
        seed: RNG seed.

    Returns:
        ``(y, cluster)`` — event outcomes and their cluster labels.
    """
    rng = np.random.default_rng(seed)
    draws = rng.binomial(1, rate, size=n_clusters).astype(float)
    y = np.repeat(draws, reps)
    cluster = np.repeat(np.arange(n_clusters), reps)
    return y, cluster


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #


def test_collapse_clusters_returns_counts_and_sums():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    cluster = np.array(["a", "a", "b", "b"])
    counts, sums = _collapse_clusters(y, cluster)
    assert sorted(counts.tolist()) == [2, 2]
    assert sorted(sums.tolist()) == [1.0, 2.0]


def test_delta_var_matches_deng_formula_by_hand():
    """``_delta_var`` must evaluate formula (6) of Deng et al. (2018)."""
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    sums = np.array([0.0, 1.0, 2.0, 4.0])
    k = counts.size
    n_mean, s_mean = counts.mean(), sums.mean()
    expected = (1.0 / (k * n_mean**2)) * (
        sums.var(ddof=1)
        - 2.0 * np.cov(sums, counts, ddof=1)[0, 1] * s_mean / n_mean
        + counts.var(ddof=1) * (s_mean / n_mean) ** 2
    )
    assert _delta_var(counts, sums) == pytest.approx(expected, rel=1e-12)


def test_delta_var_of_singletons_is_the_naive_variance_of_the_mean():
    """One row per cluster must collapse to ``var(y, ddof=1) / n`` exactly."""
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    counts = np.ones_like(y)
    assert _delta_var(counts, y) == pytest.approx(y.var(ddof=1) / y.size, rel=1e-12)


# --------------------------------------------------------------------------- #
# welch_t_two_sample
# --------------------------------------------------------------------------- #


def test_welch_naive_matches_scipy(daily_rates):
    pre, post = daily_rates
    res = welch_t_two_sample(pre, post, variance="naive")
    expected = ttest_ind(post, pre, equal_var=False)
    assert isinstance(res, WelchTestResult)
    assert res.t_stat == pytest.approx(expected.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(expected.pvalue, rel=1e-12)


def test_welch_without_clusters_reports_identical_variances(daily_rates):
    """No cluster ids means every row is its own cluster — the two SEs coincide."""
    pre, post = daily_rates
    res = welch_t_two_sample(pre, post)
    assert res.se_cluster == pytest.approx(res.se_naive, rel=1e-12)
    assert res.design_effect == pytest.approx(1.0, rel=1e-12)
    assert res.n_clusters_pre == pre.size
    assert res.n_clusters_post == post.size


def test_welch_singleton_cluster_ids_match_the_no_cluster_path(daily_rates):
    pre, post = daily_rates
    explicit = welch_t_two_sample(
        pre,
        post,
        cluster_pre=np.arange(pre.size),
        cluster_post=np.arange(post.size),
    )
    implicit = welch_t_two_sample(pre, post)
    assert explicit.se_cluster == pytest.approx(implicit.se_cluster, rel=1e-12)
    assert explicit.p_value == pytest.approx(implicit.p_value, rel=1e-12)


def test_clustering_inflates_the_standard_error():
    """Repeat visitors must widen the SE — this is the whole point of the correction."""
    y_pre, c_pre = _clustered_events(n_clusters=400, reps=10, rate=0.20, seed=1)
    y_post, c_post = _clustered_events(n_clusters=400, reps=10, rate=0.24, seed=2)
    res = welch_t_two_sample(y_pre, y_post, cluster_pre=c_pre, cluster_post=c_post)
    assert res.se_cluster > res.se_naive
    # 10 identical rows per cluster ⇒ variance inflated by ~10x.
    assert res.design_effect == pytest.approx(10.0, rel=0.15)
    assert res.n_clusters_pre == 400
    assert res.n_pre == 4000


def test_point_estimate_is_event_weighted_and_ignores_clustering():
    y_pre, c_pre = _clustered_events(n_clusters=50, reps=4, rate=0.30, seed=3)
    y_post, c_post = _clustered_events(n_clusters=50, reps=7, rate=0.40, seed=4)
    res = welch_t_two_sample(y_pre, y_post, cluster_pre=c_pre, cluster_post=c_post)
    assert res.mean_pre == pytest.approx(float(np.mean(y_pre)), rel=1e-12)
    assert res.mean_post == pytest.approx(float(np.mean(y_post)), rel=1e-12)


def test_variance_choice_changes_inference_but_not_the_effect():
    y_pre, c_pre = _clustered_events(n_clusters=300, reps=8, rate=0.20, seed=5)
    y_post, c_post = _clustered_events(n_clusters=300, reps=8, rate=0.26, seed=6)
    naive = welch_t_two_sample(
        y_pre, y_post, cluster_pre=c_pre, cluster_post=c_post, variance="naive"
    )
    clustered = welch_t_two_sample(
        y_pre, y_post, cluster_pre=c_pre, cluster_post=c_post, variance="cluster"
    )
    assert naive.delta_mean == pytest.approx(clustered.delta_mean, rel=1e-12)
    assert abs(clustered.t_stat) < abs(naive.t_stat)
    assert clustered.p_value > naive.p_value


def test_summary_mentions_both_standard_errors(daily_rates):
    pre, post = daily_rates
    text = welch_t_two_sample(pre, post).summary()
    assert "se_naive" in text
    assert "se_cluster" in text


# --------------------------------------------------------------------------- #
# student_t_two_sample
# --------------------------------------------------------------------------- #


def test_student_t_matches_scipy_pooled(daily_rates):
    pre, post = daily_rates
    res = student_t_two_sample(pre, post)
    expected = ttest_ind(post, pre, equal_var=True)
    assert res.t_stat == pytest.approx(expected.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(expected.pvalue, rel=1e-12)
    assert res.dof == pytest.approx(pre.size + post.size - 2)


# --------------------------------------------------------------------------- #
# permutation_welch_two_sample
# --------------------------------------------------------------------------- #


def test_permutation_is_reproducible_under_a_fixed_seed(daily_rates):
    pre, post = daily_rates
    kwargs = {"n_perm": 200, "n_boot": 200, "seed": 11}
    first = permutation_welch_two_sample(pre, post, **kwargs)
    second = permutation_welch_two_sample(pre, post, **kwargs)
    assert first.p_value_perm == second.p_value_perm
    assert first.ci == second.ci


def test_permutation_p_value_respects_the_plus_one_floor(daily_rates):
    pre, post = daily_rates
    res = permutation_welch_two_sample(pre, post, n_perm=200, n_boot=0, seed=0)
    assert res.p_value_perm >= 1.0 / 201.0
    assert res.p_value_perm <= 1.0
    assert res.n_perm == 200


def test_permutation_detects_a_large_shift():
    rng = np.random.default_rng(21)
    pre = rng.normal(0.0, 1.0, size=40)
    post = rng.normal(3.0, 1.0, size=40)
    res = permutation_welch_two_sample(pre, post, n_perm=500, n_boot=0, seed=0)
    assert res.p_value_perm < 0.01
    assert res.reject_h0


def test_permutation_stays_silent_on_an_aa_split():
    rng = np.random.default_rng(22)
    pre = rng.normal(0.0, 1.0, size=60)
    post = rng.normal(0.0, 1.0, size=60)
    res = permutation_welch_two_sample(pre, post, n_perm=500, n_boot=0, seed=0)
    assert res.p_value_perm > 0.05
    assert not res.reject_h0


def test_permutation_permutes_whole_clusters():
    """Cluster labels must travel with their rows, so the null keeps the inflation."""
    y_pre, c_pre = _clustered_events(n_clusters=120, reps=6, rate=0.20, seed=8)
    y_post, c_post = _clustered_events(n_clusters=120, reps=6, rate=0.22, seed=9)
    res = permutation_welch_two_sample(
        y_pre,
        y_post,
        cluster_pre=c_pre,
        cluster_post=c_post,
        n_perm=400,
        n_boot=0,
        seed=0,
    )
    assert res.n_clusters_pre == 120
    assert res.n_clusters_post == 120
    assert res.se_cluster > res.se_naive
    assert 0.0 < res.p_value_perm <= 1.0


def test_bootstrap_ci_brackets_the_observed_delta(daily_rates):
    pre, post = daily_rates
    res = permutation_welch_two_sample(pre, post, n_perm=200, n_boot=1000, seed=3)
    assert res.ci_method == "bootstrap-t"
    assert res.ci[0] < res.delta_mean < res.ci[1]


def test_zero_bootstrap_draws_fall_back_to_the_analytic_interval(daily_rates):
    pre, post = daily_rates
    res = permutation_welch_two_sample(pre, post, n_perm=100, n_boot=0, seed=3)
    assert res.ci_method == "student-t"


@pytest.mark.slow
def test_permutation_at_production_settings(daily_rates):
    pre, post = daily_rates
    res = permutation_welch_two_sample(pre, post, seed=0)
    assert res.n_perm == 10_000
    assert 0.0 < res.p_value_perm <= 1.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_rejects_samples_shorter_than_two():
    with pytest.raises(ValueError, match="at least 2 observations"):
        welch_t_two_sample([1.0], [1.0, 2.0, 3.0])


def test_rejects_cluster_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        welch_t_two_sample(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], cluster_pre=[1, 2], cluster_post=[1, 2, 3]
        )


def test_rejects_a_single_cluster_per_arm():
    with pytest.raises(ValueError, match="at least 2 clusters"):
        welch_t_two_sample(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            cluster_pre=[1, 1, 1],
            cluster_post=[1, 2, 3],
        )


def test_rejects_unknown_variance_choice(daily_rates):
    pre, post = daily_rates
    with pytest.raises(ValueError, match="Unknown variance"):
        welch_t_two_sample(pre, post, variance="robust")


def test_rejects_non_finite_input():
    with pytest.raises(ValueError, match="finite"):
        welch_t_two_sample([1.0, 2.0, np.nan], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# Delta-method z test — the gettyab ``run_delta`` algorithm
# --------------------------------------------------------------------------- #


def _gettyab_var_delta(y: np.ndarray, ids: np.ndarray) -> float:
    """Reference implementation copied from ``gettyab.stats.var_delta``.

    Reproduced here (mixed ddof and all) so the parity test does not need
    gettyab installed. See the ``Notes`` in the module under test for why
    the two differ by a factor of order ``(k-1)/k``.
    """
    order = np.argsort(ids, kind="stable")
    _, starts = np.unique(ids[order], return_index=True)
    counts = np.diff(np.append(starts, ids.size)).astype(float)
    sums = np.add.reduceat(y[order], starts)

    k = counts.size
    n_mean, s_mean = counts.mean(), sums.mean()
    var_n, var_s = np.var(counts), np.var(sums)
    cov_sn = np.cov(sums, counts)[0, 1]
    return float(
        (1.0 / (k * n_mean**2))
        * (var_s - 2.0 * cov_sn * s_mean / n_mean + var_n * (s_mean / n_mean) ** 2)
    )


def test_delta_method_reproduces_run_delta_steps():
    """est_diff, var_diff, z and the 1.96 interval, exactly as gettyab does them."""
    rng = np.random.default_rng(7)
    ids_pre = np.repeat(np.arange(300), 4)
    ids_post = np.repeat(np.arange(300, 650), 4)
    y_pre = np.repeat(rng.binomial(1, 0.20, 300), 4).astype(float)
    y_post = np.repeat(rng.binomial(1, 0.26, 350), 4).astype(float)

    res = delta_method_two_sample(y_pre, y_post, cluster_pre=ids_pre, cluster_post=ids_post)

    est_diff = y_post.mean() - y_pre.mean()
    var_diff = _delta_var(*_collapse_clusters(y_pre, ids_pre)[:2]) + _delta_var(
        *_collapse_clusters(y_post, ids_post)[:2]
    )
    se = np.sqrt(var_diff)

    assert res.delta_mean == pytest.approx(est_diff, rel=1e-12)
    assert res.se == pytest.approx(se, rel=1e-12)
    assert res.t_stat == pytest.approx(est_diff / se, rel=1e-12)
    assert res.ci[0] == pytest.approx(est_diff - 1.959963984540054 * se, rel=1e-9)
    assert res.ci[1] == pytest.approx(est_diff + 1.959963984540054 * se, rel=1e-9)


def test_delta_method_uses_the_normal_not_the_t():
    """``run_delta`` computes ``2 * (1 - Phi(|z|))``; dof must read as infinite."""
    from scipy.stats import norm

    rng = np.random.default_rng(8)
    ids_pre = np.repeat(np.arange(200), 5)
    ids_post = np.repeat(np.arange(200, 400), 5)
    y_pre = np.repeat(rng.normal(10.0, 2.0, 200), 5)
    y_post = np.repeat(rng.normal(10.4, 2.0, 200), 5)

    res = delta_method_two_sample(y_pre, y_post, cluster_pre=ids_pre, cluster_post=ids_post)
    assert res.dof == float("inf")
    assert res.p_value == pytest.approx(2.0 * norm.sf(abs(res.t_stat)), rel=1e-12)
    assert res.ci_method == "normal"


def test_delta_method_se_tracks_the_clustered_se_not_the_naive_one():
    """The whole point: it must inherit the cluster-robust variance."""
    rng = np.random.default_rng(9)
    ids_pre = np.repeat(np.arange(150), 6)
    ids_post = np.repeat(np.arange(150, 300), 6)
    y_pre = np.repeat(rng.binomial(1, 0.3, 150), 6).astype(float)
    y_post = np.repeat(rng.binomial(1, 0.3, 150), 6).astype(float)

    res = delta_method_two_sample(y_pre, y_post, cluster_pre=ids_pre, cluster_post=ids_post)
    assert res.se == pytest.approx(res.se_cluster, rel=1e-12)
    assert res.se > res.se_naive
    assert res.variance == "cluster"


def test_delta_method_matches_gettyab_within_the_ddof_gap():
    """Parity with gettyab is exact up to its mixed-ddof quirk (~1/k)."""
    rng = np.random.default_rng(10)
    ids_pre = np.repeat(np.arange(500), 3)
    ids_post = np.repeat(np.arange(500, 1000), 3)
    y_pre = np.repeat(rng.binomial(1, 0.25, 500), 3).astype(float)
    y_post = np.repeat(rng.binomial(1, 0.28, 500), 3).astype(float)

    res = delta_method_two_sample(y_pre, y_post, cluster_pre=ids_pre, cluster_post=ids_post)
    se_gettyab = np.sqrt(_gettyab_var_delta(y_pre, ids_pre) + _gettyab_var_delta(y_post, ids_post))
    assert res.se == pytest.approx(se_gettyab, rel=5e-3)


def test_delta_method_without_clusters_is_the_naive_z_test():
    y_pre = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    y_post = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    res = delta_method_two_sample(y_pre, y_post)
    assert res.se == pytest.approx(res.se_naive, rel=1e-12)
    assert res.se == pytest.approx(res.se_cluster, rel=1e-12)
