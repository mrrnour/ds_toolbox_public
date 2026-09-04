"""Frequentist two-sample tests with an optional cluster-robust variance.

Three tests share one result container and one variance engine:

- :func:`welch_t_two_sample` — unequal-variance t-test (Welch 1947).
- :func:`student_t_two_sample` — pooled-variance t-test, kept as a
  reference companion rather than a decision rule.
- :func:`permutation_welch_two_sample` — studentised permutation test
  with a studentised bootstrap-t interval.

The distinguishing feature is the ``cluster_control`` / ``cluster_treatment``
arguments. Passing them switches the variance from the naive iid form to
the delta method of Deng et al. (2018), formula (6), which is what an
online-experimentation ratio metric needs when one visitor contributes
many rows. Every result reports ``se_naive`` and ``se_cluster`` side by
side plus their ratio as ``design_effect``, so the size of the correction
is always visible — it is routinely an order of magnitude.

Two properties are worth stating plainly:

- The **point estimate is event-weighted** — ``sum(S_i) / sum(N_i)`` over
  clusters, i.e. the plain mean of the rows. Clustering widens the
  interval; it never moves the estimate. This matches ``gettyab``'s
  ``run_delta``.
- The **permutation shuffles whole clusters**, not rows. Row-level
  shuffling would destroy the within-cluster correlation and rebuild an
  anti-conservative null. Collapsing to per-cluster sufficient statistics
  ``(N_i, S_i)`` is also what makes the test affordable on event-level
  data: cost scales with the cluster count, not the row count.

Omitting the cluster arguments reduces every function to its textbook
form, so the module doubles as the plain daily-rate test.

Notes
-----
``gettyab.stats.var_delta`` implements the same formula but mixes
degrees of freedom — ``np.var`` (ddof=0) for the variance terms against
``np.cov`` (ddof=1) for the covariance term. This module uses ddof=1
throughout, so agreement with ``run_delta`` is to within a factor of
order ``(k-1)/k`` rather than exact.

References
----------
- Deng, A., Lu, J., & Litz, J. (2017). Trustworthy analysis of online
  A/B tests: pitfalls, challenges and solutions. *WSDM*. See also
  Deng et al. (2018), *KDD*, formula (6) — the delta-method variance of
  a ratio metric under clustering.
- Welch, B. L. (1947). The generalization of Student's problem when
  several different population variances are involved. *Biometrika*, 34.
- Satterthwaite, F. E. (1946). An approximate distribution of estimates
  of variance components. *Biometrics Bulletin*, 2(6), 110-114.
- Chung, E., & Romano, J. P. (2013). Exact and asymptotically robust
  permutation tests. *Annals of Statistics*, 41(2), 484-507 — why the
  permutation statistic must be studentised.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1), 1-26.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

_VARIANCE_CHOICES = ("naive", "cluster")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WelchTestResult:
    """Outcome of a two-sample test, carrying both variance estimates.

    Attributes
    ----------
    mean_control, mean_treatment
        Event-weighted arm means, ``sum(S_i) / sum(N_i)``.
    delta_mean
        ``mean_treatment - mean_control``.
    delta_rel_pct
        ``delta_mean`` as a percentage of ``mean_control``; ``nan`` when the
        control mean is zero.
    n_control, n_treatment
        Row counts.
    n_clusters_control, n_clusters_treatment
        Distinct cluster counts. Equal to the row counts when no cluster
        ids were supplied.
    se_naive
        Standard error of ``delta_mean`` treating rows as independent.
    se_cluster
        Standard error of ``delta_mean`` under the delta method. Equals
        ``se_naive`` when every row is its own cluster.
    se
        Whichever of the two drove ``t_stat``, ``p_value`` and ``ci``.
    variance
        ``"naive"`` or ``"cluster"`` — which estimate ``se`` refers to.
    t_stat, dof, p_value
        Analytic test statistic, Satterthwaite degrees of freedom, and
        two-sided p-value.
    ci
        ``(low, high)`` interval on ``delta_mean`` at ``1 - alpha``.
    ci_method
        ``"student-t"`` or ``"bootstrap-t"``.
    alpha
        Two-sided significance level.
    method
        Human-readable name of the test that produced this result.
    p_value_perm, n_perm, seed
        Populated only by :func:`permutation_welch_two_sample`.
    """

    mean_control: float
    mean_treatment: float
    delta_mean: float
    delta_rel_pct: float
    n_control: int
    n_treatment: int
    n_clusters_control: int
    n_clusters_treatment: int
    se_naive: float
    se_cluster: float
    se: float
    variance: str
    t_stat: float
    dof: float
    p_value: float
    ci: tuple[float, float]
    ci_method: str
    alpha: float
    method: str
    p_value_perm: float | None = None
    n_perm: int | None = None
    seed: int | None = None

    @property
    def design_effect(self) -> float:
        """Variance inflation from clustering, ``(se_cluster / se_naive) ** 2``."""
        if self.se_naive <= 0.0:
            return float("nan")
        return (self.se_cluster / self.se_naive) ** 2

    @property
    def p_value_used(self) -> float:
        """The permutation p-value when present, otherwise the analytic one."""
        return self.p_value if self.p_value_perm is None else self.p_value_perm

    @property
    def reject_h0(self) -> bool:
        """Whether :attr:`p_value_used` falls below ``alpha``."""
        return bool(self.p_value_used < self.alpha)

    def summary(self) -> str:
        """Render a one-block human-readable digest of the result."""
        verdict = "reject H0" if self.reject_h0 else "fail to reject H0"
        return (
            f"{self.method}\n"
            f"  mean_control  = {self.mean_control:.6g}  (n={self.n_control}, "
            f"clusters={self.n_clusters_control})\n"
            f"  mean_treatment= {self.mean_treatment:.6g}  (n={self.n_treatment}, "
            f"clusters={self.n_clusters_treatment})\n"
            f"  delta_mean    = {self.delta_mean:.6g}  ({self.delta_rel_pct:.3g}%)\n"
            f"  se_naive      = {self.se_naive:.6g}\n"
            f"  se_cluster    = {self.se_cluster:.6g}  "
            f"(design effect {self.design_effect:.3g}x)\n"
            f"  using         = {self.variance}\n"
            f"  t={self.t_stat:.4f}  dof={self.dof:.1f}  p={self.p_value_used:.4g}\n"
            f"  {100 * (1 - self.alpha):.0f}% CI  = "
            f"[{self.ci[0]:.6g}, {self.ci[1]:.6g}]  ({self.ci_method})\n"
            f"  verdict       = {verdict} at alpha={self.alpha}"
        )


# --------------------------------------------------------------------------- #
# Internal containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Units:
    """Per-cluster sufficient statistics: row counts, sums, and sums of squares."""

    counts: np.ndarray
    sums: np.ndarray
    sqsums: np.ndarray


@dataclass(frozen=True)
class _Arm:
    """One side of the comparison, prepared and validated."""

    units: _Units
    n: int
    n_clusters: int
    mean: float
    raw_var: float
    var_naive: float
    var_cluster: float


@dataclass(frozen=True)
class _TestSpec:
    """Knobs shared by the analytic, permutation and bootstrap paths."""

    variance: str = "cluster"
    alpha: float = 0.05
    n_perm: int = 10_000
    n_boot: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.variance not in _VARIANCE_CHOICES:
            raise ValueError(
                f"Unknown variance {self.variance!r}; expected one of {_VARIANCE_CHOICES}."
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha}.")
        if self.n_perm < 1:
            raise ValueError(f"n_perm must be >= 1; got {self.n_perm}.")
        if self.n_boot < 0:
            raise ValueError(f"n_boot must be >= 0; got {self.n_boot}.")


# --------------------------------------------------------------------------- #
# Variance engine
# --------------------------------------------------------------------------- #


def _collapse_clusters(
    values: np.ndarray,
    cluster: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce row-level values to per-cluster counts and sums.

    Args:
        values: Row-level numeric values.
        cluster: Cluster label for each row, same length as ``values``.

    Returns:
        ``(counts, sums)`` — one entry per distinct cluster, in sorted
        label order.
    """
    _, inverse = np.unique(cluster, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    counts = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=values)
    return counts, sums


def _delta_var(counts: np.ndarray, sums: np.ndarray) -> float:
    """Delta-method variance of a clustered ratio mean.

    Implements formula (6) of Deng et al. (2018) for ``sum(S) / sum(N)``,
    using ddof=1 consistently across the variance and covariance terms.

    Args:
        counts: Rows contributed by each cluster, ``N_i``.
        sums: Metric total within each cluster, ``S_i``.

    Returns:
        Variance of the ratio mean.
    """
    k = counts.size
    n_mean = float(counts.mean())
    ratio = float(sums.mean()) / n_mean
    cov_sn = float(np.cov(sums, counts, ddof=1)[0, 1])
    numerator = (
        float(sums.var(ddof=1)) - 2.0 * cov_sn * ratio + float(counts.var(ddof=1)) * ratio**2
    )
    return numerator / (k * n_mean**2)


def _satterthwaite_dof(var_a: float, dof_a: int, var_b: float, dof_b: int) -> float:
    """Welch-Satterthwaite degrees of freedom for a difference of two means.

    Args:
        var_a: Variance of the first mean.
        dof_a: Degrees of freedom behind ``var_a``.
        var_b: Variance of the second mean.
        dof_b: Degrees of freedom behind ``var_b``.

    Returns:
        Effective degrees of freedom; falls back to ``dof_a + dof_b`` if
        both variances vanish.
    """
    denominator = var_a**2 / dof_a + var_b**2 / dof_b
    if denominator <= 0.0:
        return float(dof_a + dof_b)
    return (var_a + var_b) ** 2 / denominator


# --------------------------------------------------------------------------- #
# Arm preparation
# --------------------------------------------------------------------------- #


def _as_float_vector(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Coerce an input sample to a validated 1-D float array.

    Args:
        values: The sample.
        name: Argument name, used in error messages.

    Returns:
        A 1-D float array.

    Raises:
        ValueError: If the sample has fewer than two entries or contains
            non-finite values.
    """
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 2:
        raise ValueError(f"{name} needs at least 2 observations; got {array.size}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; found nan or inf.")
    return array


def _prepare_arm(
    values: np.ndarray | Sequence[float],
    cluster: np.ndarray | Sequence[object] | None,
    label: str,
) -> _Arm:
    """Validate one arm and precompute both of its variance estimates.

    Args:
        values: Row-level metric values.
        cluster: Cluster id per row, or ``None`` to treat every row as
            its own cluster.
        label: ``"control"`` or ``"treatment"``, used in error messages.

    Returns:
        A populated :class:`_Arm`.

    Raises:
        ValueError: If the cluster ids do not align with ``values`` or
            fewer than two distinct clusters are present.
    """
    y = _as_float_vector(values, f"y_{label}")
    ids = np.arange(y.size) if cluster is None else np.asarray(cluster).ravel()
    if ids.size != y.size:
        raise ValueError(
            f"cluster_{label} and y_{label} must have the same length; "
            f"got {ids.size} and {y.size}."
        )
    counts, sums = _collapse_clusters(y, ids)
    if counts.size < 2:
        raise ValueError(f"cluster_{label} needs at least 2 clusters; got {counts.size}.")
    _, sqsums = _collapse_clusters(y**2, ids)
    raw_var = float(y.var(ddof=1))
    return _Arm(
        units=_Units(counts=counts, sums=sums, sqsums=sqsums),
        n=int(y.size),
        n_clusters=int(counts.size),
        mean=float(sums.sum() / counts.sum()),
        raw_var=raw_var,
        var_naive=raw_var / y.size,
        var_cluster=_delta_var(counts, sums),
    )


def _pick(arm: _Arm, variance: str) -> tuple[float, int]:
    """Select an arm's variance-of-the-mean and its degrees of freedom.

    Args:
        arm: The prepared arm.
        variance: ``"naive"`` or ``"cluster"``.

    Returns:
        ``(variance_of_the_mean, degrees_of_freedom)``.
    """
    if variance == "cluster":
        return arm.var_cluster, arm.n_clusters - 1
    return arm.var_naive, arm.n - 1


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def _subset_stats(units: _Units, index: np.ndarray, variance: str) -> tuple[float, float]:
    """Mean and variance-of-the-mean for an arbitrary subset of clusters.

    Recomputes the naive variance from pooled sums of squares so that
    resampled subsets stay cheap; the observed statistic is computed from
    the raw sample instead, where numerical conditioning matters.

    Args:
        units: Per-cluster sufficient statistics.
        index: Positions of the clusters forming the subset.
        variance: ``"naive"`` or ``"cluster"``.

    Returns:
        ``(mean, variance_of_the_mean)``.
    """
    counts, sums = units.counts[index], units.sums[index]
    total_n = float(counts.sum())
    total_s = float(sums.sum())
    mean = total_s / total_n
    if variance == "cluster":
        return mean, _delta_var(counts, sums)
    raw_var = (float(units.sqsums[index].sum()) - total_s**2 / total_n) / (total_n - 1.0)
    return mean, raw_var / total_n


def _studentised_delta(units: _Units, left: np.ndarray, right: np.ndarray, variance: str) -> float:
    """Studentised difference of means between two disjoint cluster subsets.

    Args:
        units: Per-cluster sufficient statistics.
        left: Cluster positions assigned to the control arm.
        right: Cluster positions assigned to the treatment arm.
        variance: ``"naive"`` or ``"cluster"``.

    Returns:
        ``(mean_right - mean_left) / se``, or ``0.0`` if the standard
        error degenerates.
    """
    mean_left, var_left = _subset_stats(units, left, variance)
    mean_right, var_right = _subset_stats(units, right, variance)
    se = float(np.sqrt(var_left + var_right))
    if not se > 0.0:
        return 0.0
    return (mean_right - mean_left) / se


def _permutation_p(
    control: _Arm,
    treatment: _Arm,
    spec: _TestSpec,
    rng: np.random.Generator,
) -> float:
    """Two-sided p-value from permuting whole clusters between arms.

    Args:
        control: Prepared control arm.
        treatment: Prepared treatment arm.
        spec: Test configuration.
        rng: Random generator, consumed in place.

    Returns:
        ``(hits + 1) / (n_perm + 1)`` — the ``+1`` keeps the p-value
        strictly positive and the test valid at finite ``n_perm``.
    """
    pooled = _Units(
        counts=np.concatenate([control.units.counts, treatment.units.counts]),
        sums=np.concatenate([control.units.sums, treatment.units.sums]),
        sqsums=np.concatenate([control.units.sqsums, treatment.units.sqsums]),
    )
    total = pooled.counts.size
    split = control.n_clusters
    observed = abs(
        _studentised_delta(pooled, np.arange(split), np.arange(split, total), spec.variance)
    )
    hits = 0
    for _ in range(spec.n_perm):
        order = rng.permutation(total)
        stat = _studentised_delta(pooled, order[:split], order[split:], spec.variance)
        if abs(stat) >= observed:
            hits += 1
    return (hits + 1.0) / (spec.n_perm + 1.0)


def _bootstrap_t_ci(
    control: _Arm,
    treatment: _Arm,
    spec: _TestSpec,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Studentised bootstrap-t interval on the difference of means.

    Clusters are resampled with replacement within each arm, so the
    interval inherits the same clustering assumption as ``se_cluster``.

    Args:
        control: Prepared control arm.
        treatment: Prepared treatment arm.
        spec: Test configuration.
        rng: Random generator, consumed in place.

    Returns:
        ``(low, high)`` at level ``1 - alpha``.
    """
    delta = treatment.mean - control.mean
    se = float(np.sqrt(_pick(control, spec.variance)[0] + _pick(treatment, spec.variance)[0]))
    pivots = np.empty(spec.n_boot, dtype=float)
    for draw in range(spec.n_boot):
        mean_c, var_c = _subset_stats(
            control.units,
            rng.integers(0, control.n_clusters, control.n_clusters),
            spec.variance,
        )
        mean_t, var_t = _subset_stats(
            treatment.units,
            rng.integers(0, treatment.n_clusters, treatment.n_clusters),
            spec.variance,
        )
        se_draw = float(np.sqrt(var_c + var_t))
        pivots[draw] = ((mean_t - mean_c) - delta) / se_draw if se_draw > 0.0 else 0.0
    low, high = np.nanpercentile(
        pivots, [100.0 * spec.alpha / 2.0, 100.0 * (1.0 - spec.alpha / 2.0)]
    )
    return (delta - float(high) * se, delta - float(low) * se)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _assemble(
    control: _Arm,
    treatment: _Arm,
    spec: _TestSpec,
    *,
    se: float,
    dof: float,
    method: str,
) -> WelchTestResult:
    """Build a result from an arm pair and an already-chosen standard error.

    Args:
        control: Prepared control arm.
        treatment: Prepared treatment arm.
        spec: Test configuration.
        se: Standard error of the difference driving the inference.
        dof: Degrees of freedom for the reference t distribution.
        method: Name to record on the result.

    Returns:
        A populated :class:`WelchTestResult` with an analytic interval.
    """
    delta = treatment.mean - control.mean
    t_stat = delta / se if se > 0.0 else float("nan")
    half_width = float(stats.t.ppf(1.0 - spec.alpha / 2.0, dof)) * se
    return WelchTestResult(
        mean_control=control.mean,
        mean_treatment=treatment.mean,
        delta_mean=delta,
        delta_rel_pct=(100.0 * delta / control.mean if control.mean != 0.0 else float("nan")),
        n_control=control.n,
        n_treatment=treatment.n,
        n_clusters_control=control.n_clusters,
        n_clusters_treatment=treatment.n_clusters,
        se_naive=float(np.sqrt(control.var_naive + treatment.var_naive)),
        se_cluster=float(np.sqrt(control.var_cluster + treatment.var_cluster)),
        se=se,
        variance=spec.variance,
        t_stat=float(t_stat),
        dof=float(dof),
        p_value=float(2.0 * stats.t.sf(abs(t_stat), dof)),
        ci=(delta - half_width, delta + half_width),
        ci_method="student-t",
        alpha=spec.alpha,
        method=method,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def welch_t_two_sample(
    y_control: np.ndarray | Sequence[float],
    y_treatment: np.ndarray | Sequence[float],
    *,
    cluster_control: np.ndarray | Sequence[object] | None = None,
    cluster_treatment: np.ndarray | Sequence[object] | None = None,
    variance: str = "cluster",
    alpha: float = 0.05,
) -> WelchTestResult:
    """Unequal-variance t-test, optionally with a cluster-robust standard error.

    With no cluster ids this is the textbook Welch test and reproduces
    ``scipy.stats.ttest_ind(y_treatment, y_control, equal_var=False)``. With
    cluster ids the standard error switches to the delta method, which is
    the correct choice for a row-level ratio metric where one unit — a
    visitor, a session — contributes many rows.

    Args:
        y_control: Control-arm values, one entry per row.
        y_treatment: Treatment-arm values, one entry per row.
        cluster_control: Cluster id per control row. ``None`` treats each
            row as independent.
        cluster_treatment: Cluster id per treatment row.
        variance: Which standard error drives the inference — ``"naive"``
            or ``"cluster"``. Both are always reported.
        alpha: Two-sided significance level.

    Returns:
        A :class:`WelchTestResult` carrying both standard errors.

    Raises:
        ValueError: On unusable input — see :func:`_prepare_arm`.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> ctl, trt = rng.normal(0, 1, 200), rng.normal(0.4, 1, 200)
        >>> res = welch_t_two_sample(ctl, trt, variance="naive")
        >>> res.p_value < 0.05
        True
    """
    spec = _TestSpec(variance=variance, alpha=alpha)
    control = _prepare_arm(y_control, cluster_control, "control")
    treatment = _prepare_arm(y_treatment, cluster_treatment, "treatment")
    var_c, dof_c = _pick(control, spec.variance)
    var_t, dof_t = _pick(treatment, spec.variance)
    return _assemble(
        control,
        treatment,
        spec,
        se=float(np.sqrt(var_c + var_t)),
        dof=_satterthwaite_dof(var_c, dof_c, var_t, dof_t),
        method=f"Welch t-test ({spec.variance} variance)",
    )


def delta_method_two_sample(
    y_control: np.ndarray | Sequence[float],
    y_treatment: np.ndarray | Sequence[float],
    *,
    cluster_control: np.ndarray | Sequence[object] | None = None,
    cluster_treatment: np.ndarray | Sequence[object] | None = None,
    alpha: float = 0.05,
) -> WelchTestResult:
    """Clustered z-test, transcribing ``gettyab.modelling.run_delta``.

    Five steps, in the order that function performs them:

    1. Collapse each arm to per-cluster ``(count, sum)`` pairs.
    2. Take each arm's delta-method variance, Deng et al. (2018) eq. (6).
    3. ``delta = mean_treatment - mean_control``.
    4. ``var_diff = var_control + var_treatment``; ``z = delta / sqrt(var_diff)``.
    5. ``p = 2 * (1 - Phi(|z|))`` and ``delta +/- z_crit * se``.

    The reference is the **standard normal**, not a t distribution, so
    :attr:`WelchTestResult.dof` reads ``inf`` and :attr:`ci_method` reads
    ``"normal"``. That is deliberate: it keeps the numbers comparable with
    the production A/B estimator instead of quietly improving on it.

    Prefer this when the goal is to match an existing ``run_delta`` read-out.
    Prefer :func:`permutation_welch_two_sample` when the goal is a defensible
    p-value on its own terms, since it shares this standard error but does
    not lean on the normal approximation.

    Args:
        y_control: Control-arm values, one entry per row.
        y_treatment: Treatment-arm values, one entry per row.
        cluster_control: Cluster id per control row. ``None`` treats every
            row as its own cluster, which reduces to the naive z-test.
        cluster_treatment: Cluster id per treatment row.
        alpha: Two-sided significance level.

    Returns:
        A :class:`WelchTestResult` whose ``se`` equals ``se_cluster``.

    Raises:
        ValueError: On unusable input — see :func:`_prepare_arm`.

    References:
        Deng, A., Knoblich, U., & Lu, J. (2018). Applying the delta method
        in metric analytics. *KDD*, 233-242.
    """
    spec = _TestSpec(variance="cluster", alpha=alpha)
    control = _prepare_arm(y_control, cluster_control, "control")
    treatment = _prepare_arm(y_treatment, cluster_treatment, "treatment")

    delta = treatment.mean - control.mean
    se = float(np.sqrt(control.var_cluster + treatment.var_cluster))
    z_stat = delta / se if se > 0.0 else float("nan")
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    half_width = z_crit * se

    return WelchTestResult(
        mean_control=control.mean,
        mean_treatment=treatment.mean,
        delta_mean=delta,
        delta_rel_pct=(100.0 * delta / control.mean if control.mean != 0.0 else float("nan")),
        n_control=control.n,
        n_treatment=treatment.n,
        n_clusters_control=control.n_clusters,
        n_clusters_treatment=treatment.n_clusters,
        se_naive=float(np.sqrt(control.var_naive + treatment.var_naive)),
        se_cluster=se,
        se=se,
        variance=spec.variance,
        t_stat=float(z_stat),
        dof=float("inf"),
        p_value=float(2.0 * stats.norm.sf(abs(z_stat))),
        ci=(delta - half_width, delta + half_width),
        ci_method="normal",
        alpha=alpha,
        method="Delta-method z-test (cluster variance)",
    )


def student_t_two_sample(
    y_control: np.ndarray | Sequence[float],
    y_treatment: np.ndarray | Sequence[float],
    *,
    cluster_control: np.ndarray | Sequence[object] | None = None,
    cluster_treatment: np.ndarray | Sequence[object] | None = None,
    alpha: float = 0.05,
) -> WelchTestResult:
    """Pooled-variance t-test, reproducing ``ttest_ind(..., equal_var=True)``.

    Kept as a reference companion. Pooling assumes equal variances across
    arms, which two-group comparisons routinely violate, so prefer
    :func:`welch_t_two_sample` for decisions. Cluster ids do not change
    the pooled statistic but are honoured when populating
    :attr:`WelchTestResult.se_cluster`, which makes the gap between the
    pooled and cluster-robust standard errors easy to read off.

    Args:
        y_control: Control-arm values, one entry per row.
        y_treatment: Treatment-arm values, one entry per row.
        cluster_control: Cluster id per control row, for reporting only.
        cluster_treatment: Cluster id per treatment row, for reporting only.
        alpha: Two-sided significance level.

    Returns:
        A :class:`WelchTestResult` whose ``variance`` field reads
        ``"naive"``.

    Raises:
        ValueError: On unusable input — see :func:`_prepare_arm`.
    """
    spec = _TestSpec(variance="naive", alpha=alpha)
    control = _prepare_arm(y_control, cluster_control, "control")
    treatment = _prepare_arm(y_treatment, cluster_treatment, "treatment")
    dof = control.n + treatment.n - 2
    pooled_var = ((control.n - 1) * control.raw_var + (treatment.n - 1) * treatment.raw_var) / dof
    se = float(np.sqrt(pooled_var * (1.0 / control.n + 1.0 / treatment.n)))
    return _assemble(
        control,
        treatment,
        spec,
        se=se,
        dof=float(dof),
        method="Student t-test (pooled)",
    )


def permutation_welch_two_sample(
    y_control: np.ndarray | Sequence[float],
    y_treatment: np.ndarray | Sequence[float],
    *,
    cluster_control: np.ndarray | Sequence[object] | None = None,
    cluster_treatment: np.ndarray | Sequence[object] | None = None,
    variance: str = "cluster",
    alpha: float = 0.05,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> WelchTestResult:
    """Studentised permutation Welch test with a bootstrap-t interval.

    The permutation reassigns whole clusters between arms and compares
    studentised statistics, which keeps the test valid under unequal
    variances (Chung & Romano 2013) and preserves within-cluster
    correlation under the null. Cost scales with the number of clusters
    times ``n_perm``, not with the number of rows — collapsing to
    per-cluster sufficient statistics is what makes event-level data
    tractable.

    The analytic ``p_value`` is still reported alongside
    ``p_value_perm``; :attr:`WelchTestResult.reject_h0` uses the
    permutation value.

    Args:
        y_control: Control-arm values, one entry per row.
        y_treatment: Treatment-arm values, one entry per row.
        cluster_control: Cluster id per control row.
        cluster_treatment: Cluster id per treatment row.
        variance: Which standard error studentises the statistic —
            ``"naive"`` or ``"cluster"``.
        alpha: Two-sided significance level.
        n_perm: Number of permutations.
        n_boot: Number of bootstrap draws for the interval. Pass ``0`` to
            skip the bootstrap and keep the analytic t interval.
        seed: Seed for the shared generator; permutations are drawn
            before bootstrap replicates.

    Returns:
        A :class:`WelchTestResult` with ``p_value_perm``, ``n_perm`` and
        ``seed`` populated.

    Raises:
        ValueError: On unusable input or an out-of-range setting.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> ctl, trt = rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)
        >>> res = permutation_welch_two_sample(ctl, trt, n_perm=200, n_boot=0)
        >>> res.p_value_perm < 0.05
        True
    """
    spec = _TestSpec(variance=variance, alpha=alpha, n_perm=n_perm, n_boot=n_boot, seed=seed)
    control = _prepare_arm(y_control, cluster_control, "control")
    treatment = _prepare_arm(y_treatment, cluster_treatment, "treatment")
    var_c, dof_c = _pick(control, spec.variance)
    var_t, dof_t = _pick(treatment, spec.variance)
    base = _assemble(
        control,
        treatment,
        spec,
        se=float(np.sqrt(var_c + var_t)),
        dof=_satterthwaite_dof(var_c, dof_c, var_t, dof_t),
        method=f"permutation Welch t-test ({spec.variance} variance)",
    )
    rng = np.random.default_rng(spec.seed)
    p_perm = _permutation_p(control, treatment, spec, rng)
    if spec.n_boot > 0:
        return replace(
            base,
            ci=_bootstrap_t_ci(control, treatment, spec, rng),
            ci_method="bootstrap-t",
            p_value_perm=p_perm,
            n_perm=spec.n_perm,
            seed=spec.seed,
        )
    return replace(base, p_value_perm=p_perm, n_perm=spec.n_perm, seed=spec.seed)
