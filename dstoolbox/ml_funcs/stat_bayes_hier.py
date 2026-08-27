"""Hierarchical (partially pooled) Beta-Binomial for clustered rate data.

The hierarchical counterpart to the flat model in
:mod:`dstoolbox.ml_funcs.stat_bayes`. Where the flat model pools every trial
into one ``(successes, trials)`` pair, this one keeps the *unit* structure —
each unit (user, visitor, session) contributes its own ``(n_i, k_i)`` and a
population distribution ties the per-unit rates together::

    mu      ~ Beta(alpha, beta)                  population mean rate
    kappa   ~ Gamma(a, b)                        concentration
    theta_i ~ Beta(mu*kappa, (1-mu)*kappa)       per-unit rate
    k_i     ~ Binomial(n_i, theta_i)

``theta_i`` is marginalised analytically, so the likelihood is
``BetaBinomial(n_i, mu*kappa, (1-mu)*kappa)`` and only ``(mu, kappa)`` are
sampled — the fit cost does not grow with unit count.

Use it when trials cluster within units, which breaks the flat model's
independence assumption.

The fit reports **two** population rates, and they answer different
questions. ``mu`` is *unit-averaged*: one unit, one vote, whatever its trial
count. ``mu_weighted`` is *trial-weighted*, ``sum(n_i theta_i)/sum(n_i)``,
reconstructed from the conjugate posterior of the shrunk per-unit rates.

They can move in opposite directions whenever heavy and light units convert
at different rates and the mix shifts between periods, so pick the one that
matches the decision: ``mu`` for "did the average user get a better
experience", ``mu_weighted`` for "did total conversions per trial go up".
``mu`` tracks the raw per-unit mean closely and is *not* the pooled rate
``sum(k_i)/sum(n_i)``; ``mu_weighted`` is the shrunk counterpart of that
pooled rate. Because ``mu`` averages over whoever is in the sample, it is
also not invariant to the observation window. See
:mod:`dstoolbox.ml_funcs.stat_bayes_group` for the two-group workflow built on
this model, which enforces equal-length windows for that reason.

``pymc`` / ``arviz`` are optional — ``pip install 'dstoolbox[bayes]'``.

References
----------
- Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed., §5.1, §5.3.
- Kruschke (2015). *Doing Bayesian Data Analysis*, 2nd ed., ch. 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._base_import_warning import optional_import
from .stat_bayes import BetaPrior, _resolve_beta_prior

_pm = optional_import("pymc", "stat_bayes_hier")
_az = optional_import("arviz", "stat_bayes_hier")

#: ``Gamma(alpha, beta)`` prior on ``kappa``. Mean 20, SD ~14 — weak, but
#: rules out the degenerate ``kappa -> 0`` corner where every unit is either
#: always or never a success.
DEFAULT_KAPPA_PRIOR: tuple[float, float] = (2.0, 0.1)


@dataclass(frozen=True)
class HierBetaBinomialFit:
    """Posterior for one period.

    ``diagnostics`` holds the ArviZ ``r_hat`` / ``ess_bulk`` / ``ess_tail`` /
    ``mcse_mean`` rows for ``mu`` and ``kappa``. ``trace`` is kept so callers
    can render trace and rank plots.
    """

    trace: object  # arviz.InferenceData
    mu_samples: np.ndarray
    kappa_samples: np.ndarray
    mu_weighted_samples: np.ndarray
    diagnostics: pd.DataFrame
    divergences: int
    n_units: int
    successes: int
    trials: int
    prior_spec: str

    @property
    def mu_mean(self) -> float:
        """Posterior mean of the unit-level population rate — one unit, one vote."""
        return float(self.mu_samples.mean())

    @property
    def mu_weighted_mean(self) -> float:
        """Posterior mean of the trial-weighted rate — one *trial*, one vote.

        Sits between :attr:`mu_mean` and :attr:`rate`: it answers the same
        question as the pooled rate but through shrunk per-unit estimates,
        so units with few trials pull toward the population mean instead of
        contributing their noisy raw ratio.
        """
        return float(self.mu_weighted_samples.mean())

    @property
    def kappa_mean(self) -> float:
        """Posterior mean concentration."""
        return float(self.kappa_samples.mean())

    @property
    def rate(self) -> float:
        """Empirical pooled rate, ``successes / trials``."""
        return self.successes / self.trials

    @property
    def rhat_max(self) -> float:
        """Worst Gelman-Rubin statistic across ``mu`` and ``kappa``."""
        return float(self.diagnostics["r_hat"].max())

    @property
    def ess_min(self) -> float:
        """Smallest bulk effective sample size across ``mu`` and ``kappa``."""
        return float(self.diagnostics["ess_bulk"].min())


def _as_counts(trials, successes) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a ``(trials, successes)`` pair to validated integer arrays."""
    n = np.asarray(trials).ravel()
    k = np.asarray(successes).ravel()
    if n.shape != k.shape:
        raise ValueError(
            f"trials and successes must have the same length; "
            f"got {n.shape[0]} and {k.shape[0]}."
        )
    if n.size == 0:
        raise ValueError("trials is empty — nothing to fit.")
    n, k = n.astype(int), k.astype(int)
    if (n <= 0).any():
        raise ValueError("every entry of trials must be positive.")
    if (k < 0).any() or (k > n).any():
        raise ValueError("every entry of successes must lie in [0, trials].")
    return n, k


def _weighted_rate_samples(
    n: np.ndarray,
    k: np.ndarray,
    mu_samples: np.ndarray,
    kappa_samples: np.ndarray,
    *,
    random_seed: int | None = None,
    max_elements: int = 20_000_000,
) -> np.ndarray:
    """Posterior draws of the trial-weighted rate ``sum(n_i th_i) / sum(n_i)``.

    ``theta_i`` is marginalised out of the sampler, but its conditional
    posterior is conjugate and closed-form::

        theta_i | mu, kappa, data ~ Beta(mu*kappa + k_i,
                                         (1-mu)*kappa + n_i - k_i)

    so its mean is ``(mu*kappa + k_i) / (kappa + n_i)`` — the usual shrinkage
    estimator. Plugging that mean in rather than drawing ``theta_i`` is a
    deliberate approximation: the weighted average runs over every unit, so
    the spread contributed by the ``theta_i`` draws is ``O(1/U)`` and
    negligible beside the two terms kept below.

    Each draw is then reweighted by a **Bayesian bootstrap** over units:
    ``g ~ Dirichlet(1, ..., 1)`` replaces the fixed ``1/U`` unit weights, so
    the interval covers *which units you would see next*, not just this
    sample. Without it the estimand collapses to a finite-population
    quantity that the observed counts almost determine, and the credible
    interval comes out an order of magnitude too narrow — units are the
    resampling stratum here because trials cluster inside them.

    Works in draw-sized chunks because the intermediate is
    ``n_draws x n_units`` — 8k draws over 40k units is 3.2e8 floats, which
    will not fit in memory as one array.
    """
    n_f = n.astype(float)
    k_f = k.astype(float)
    rng = np.random.default_rng(random_seed)

    chunk = max(1, int(max_elements // max(1, n_f.size)))
    out = np.empty(mu_samples.size, dtype=float)
    for start in range(0, mu_samples.size, chunk):
        mu_c = mu_samples[start:start + chunk, None]
        kap_c = kappa_samples[start:start + chunk, None]
        theta = (mu_c * kap_c + k_f) / (kap_c + n_f)
        # Dirichlet(1,...,1) via normalised exponentials; the normalising
        # constant cancels in the ratio, so it is never formed.
        g = rng.standard_exponential(size=theta.shape)
        w = g * n_f
        out[start:start + chunk] = (w * theta).sum(axis=1) / w.sum(axis=1)
    return out


def hier_beta_binomial_fit(
    trials,
    successes,
    *,
    prior: str | BetaPrior = "uniform",
    kappa_prior: tuple[float, float] = DEFAULT_KAPPA_PRIOR,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = None,
    progressbar: bool = False,
) -> HierBetaBinomialFit:
    """Fit the hierarchical Beta-Binomial to one period of unit-level counts.

    Parameters
    ----------
    trials, successes
        Equal-length array-likes, one entry per unit. ``successes[i]`` must
        lie in ``[0, trials[i]]``; ``trials[i]`` must be positive.
    prior
        Prior on ``mu``: ``"uniform"``, ``"jeffreys"``, or a
        :class:`~dstoolbox.ml_funcs.stat_bayes.BetaPrior`.
    kappa_prior
        ``(alpha, beta)`` of the Gamma prior on the concentration.
    draws, tune, chains, target_accept, random_seed, progressbar
        Forwarded to :func:`pymc.sample`.

    Returns
    -------
    HierBetaBinomialFit

    Example
    -------
    >>> fit = hier_beta_binomial_fit([10, 3, 25], [1, 0, 4], random_seed=0)
    >>> 0.0 < fit.mu_mean < 1.0
    True
    """
    n, k = _as_counts(trials, successes)
    spec = _resolve_beta_prior(prior)
    kappa_alpha, kappa_beta = kappa_prior
    if kappa_alpha <= 0 or kappa_beta <= 0:
        raise ValueError(f"kappa_prior parameters must be > 0; got {kappa_prior}.")

    with _pm.Model():  # type: ignore[union-attr]
        mu = _pm.Beta("mu", alpha=spec.alpha, beta=spec.beta)  # type: ignore[union-attr]
        kappa = _pm.Gamma("kappa", alpha=kappa_alpha, beta=kappa_beta)  # type: ignore[union-attr]
        # BetaBinomial == Binomial(n, theta) with theta ~ Beta(mu*kappa,
        # (1-mu)*kappa) marginalised out, so per-unit rates never enter the
        # sampler and fit cost is independent of unit count.
        _pm.BetaBinomial(  # type: ignore[union-attr]
            "obs", n=n, alpha=mu * kappa, beta=(1.0 - mu) * kappa, observed=k,
        )
        trace = _pm.sample(  # type: ignore[union-attr]
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    diagnostics = _az.summary(  # type: ignore[union-attr]
        trace, var_names=["mu", "kappa"], round_to=6
    )[["r_hat", "ess_bulk", "ess_tail", "mcse_mean"]]

    mu_samples = np.asarray(trace.posterior["mu"]).ravel()        # type: ignore[attr-defined]
    kappa_samples = np.asarray(trace.posterior["kappa"]).ravel()  # type: ignore[attr-defined]

    return HierBetaBinomialFit(
        trace=trace,
        mu_samples=mu_samples,
        kappa_samples=kappa_samples,
        mu_weighted_samples=_weighted_rate_samples(
            n, k, mu_samples, kappa_samples, random_seed=random_seed
        ),
        diagnostics=diagnostics,
        divergences=int(np.asarray(trace.sample_stats["diverging"]).sum()),  # type: ignore[attr-defined]
        n_units=int(n.size),
        successes=int(k.sum()),
        trials=int(n.sum()),
        prior_spec=spec.name,
    )


def verdict_without_rope(
    prob_gt_zero: float,
    credibility_threshold: float = 0.95,
) -> str:
    """Direction-only verdict for analyses with no agreed equivalence band.

    The counterpart to :func:`~dstoolbox.ml_funcs.stat_bayes.rope_decision`.
    Returns ``"positive"`` / ``"negative"`` rather than
    ``"meaningful_positive"`` / ``"meaningful_negative"``: nothing here
    establishes the effect is large enough to matter, only that its sign is
    credible.

    Parameters
    ----------
    prob_gt_zero
        Posterior probability the effect exceeds zero.
    credibility_threshold
        Posterior mass required before a direction is declared.

    Returns
    -------
    str
        ``"positive"``, ``"negative"`` or ``"inconclusive"``.

    Example
    -------
    >>> verdict_without_rope(0.97)
    'positive'
    >>> verdict_without_rope(0.60)
    'inconclusive'
    """
    if not 0.0 <= prob_gt_zero <= 1.0:
        raise ValueError(f"prob_gt_zero must be in [0, 1]; got {prob_gt_zero}.")
    if not 0.0 < credibility_threshold < 1.0:
        raise ValueError(
            f"credibility_threshold must be in (0, 1); got {credibility_threshold}."
        )
    if prob_gt_zero >= credibility_threshold:
        return "positive"
    if (1.0 - prob_gt_zero) >= credibility_threshold:
        return "negative"
    return "inconclusive"
