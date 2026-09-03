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

The fit reports one population rate, ``mu``, and it is *unit*-averaged: one
unit, one vote, whatever its trial count. It is deliberately **not** the
pooled rate ``sum(k_i)/sum(n_i)``, which weights by trial and so is
dominated by heavy units. The two answer different questions and can move
in opposite directions when the heavy/light mix shifts; say which one you
are quoting. Because ``mu`` averages over whoever is in the sample, it is
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
    diagnostics: pd.DataFrame
    divergences: int
    n_users: int
    successes: int
    trials: int
    prior_spec: str

    @property
    def mu_mean(self) -> float:
        """Posterior mean of the unit-level population rate — one unit, one vote."""
        return float(self.mu_samples.mean())

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
        diagnostics=diagnostics,
        divergences=int(np.asarray(trace.sample_stats["diverging"]).sum()),  # type: ignore[attr-defined]
        n_users=int(n.size),
        successes=int(k.sum()),
        trials=int(n.sum()),
        prior_spec=spec.name,
    )


def verdict_without_rope(
    prob_gt_zero: float,
    credibility_threshold: float = 0.95,
) -> str:
    """Direction-only verdict for analyses with no agreed equivalence band.

    The counterpart to :func:`~dstoolbox.ml_funcs.stat_bayes.rope_decision`,
    and returns the same four labels so a caller can render a verdict without
    knowing which rule produced it. The label therefore does not record that
    no band was consulted — a ``"positive"`` here only clears zero, whereas one
    from ``rope_decision`` cleared a band edge. Anything that must tell those
    apart has to carry the band alongside the label.

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
