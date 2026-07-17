"""Capability inspection for ML/forecasting models.

Lets callers (e.g. ``ml_comparison``) discover which optional surfaces a model
implements — point predictions, prediction intervals, sample paths, component
decomposition, attention weights — without try/except probes scattered through
the training code.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline


#: All capability flags this module knows about. Order matters for printing.
CAPABILITIES: tuple[str, ...] = (
    "point",
    "interval",
    "samples",
    "components",
    "attention",
)


def _unwrap(model: object) -> object:
    """Return the final step of a sklearn ``Pipeline``; otherwise ``model``."""
    return model[-1] if isinstance(model, Pipeline) else model


def model_capabilities(model: object) -> set[str]:
    """Return the set of capability flags ``model`` implements.

    A model "implements" a capability when the corresponding method exists on
    the bare estimator (after unwrapping a ``Pipeline``). No methods are
    called, so this is cheap and side-effect-free.

    Parameters
    ----------
    model
        A fitted or unfitted estimator. Plain sklearn regressors expose only
        ``point``; forecaster adapters (Silverkite, Darts, statsforecast) may
        add ``interval``, ``samples``, ``components``, ``attention``.

    Returns
    -------
    set[str]
        Subset of :data:`CAPABILITIES`.
    """
    base = _unwrap(model)
    caps: set[str] = set()

    if hasattr(base, "predict") or hasattr(base, "predict_proba"):
        caps.add("point")
    if hasattr(base, "predict_interval"):
        caps.add("interval")
    if hasattr(base, "predict_samples"):
        caps.add("samples")
    if hasattr(base, "components"):
        caps.add("components")
    if hasattr(base, "attention_weights"):
        caps.add("attention")

    return caps


def _resolve_name(model: object, fallback: str | None = None) -> str:
    """Best-effort display name for a model (pipeline-aware)."""
    if fallback is not None:
        return fallback
    if isinstance(model, Pipeline):
        return " -> ".join(step.__class__.__name__ for _, step in model.steps)
    return model.__class__.__name__


def capability_matrix(
    models: Iterable[object],
    names: Sequence[str] | None = None,
    capabilities: Sequence[str] = CAPABILITIES,
) -> pd.DataFrame:
    """Build a model × capability matrix of check marks.

    Parameters
    ----------
    models
        Iterable of estimators / forecaster adapters.
    names
        Optional display names, aligned with ``models``. If omitted, names are
        derived from the estimators (or their ``Pipeline`` steps).
    capabilities
        Which capabilities to include as columns. Defaults to
        :data:`CAPABILITIES`.

    Returns
    -------
    pandas.DataFrame
        Rows are models, columns are capability flags, values are ``"✓"`` or
        ``""``. ``model`` is the index name.
    """
    models = list(models)
    if names is None:
        names = [_resolve_name(m) for m in models]
    if len(names) != len(models):
        raise ValueError(f"len(names)={len(names)} != len(models)={len(models)}")

    rows = []
    for model in models:
        caps = model_capabilities(model)
        rows.append({c: ("✓" if c in caps else "") for c in capabilities})

    df = pd.DataFrame(rows, index=pd.Index(names, name="model"))
    return df


def print_capability_matrix(
    models: Iterable[object],
    names: Sequence[str] | None = None,
    capabilities: Sequence[str] = CAPABILITIES,
) -> pd.DataFrame:
    """Convenience: build *and* pretty-print the capability matrix.

    Returns the same DataFrame as :func:`capability_matrix` so callers can
    keep it for downstream use (e.g. branching on ``"interval"`` support).
    """
    df = capability_matrix(models, names=names, capabilities=capabilities)
    print("Model capabilities:")
    print(df.to_string())
    return df


def is_forecaster(model: object) -> bool:
    """Heuristic: ``True`` if ``model`` exposes a forecaster-style method.

    A forecaster has at least one of: ``predict_interval``, ``predict_samples``,
    ``components``. Plain sklearn regressors/classifiers return ``False``.
    """
    caps = model_capabilities(model)
    return bool(caps & {"interval", "samples", "components"})


# ===== imports preserved from public (needed by extras below) =====
from sklearn.base import is_classifier


# ===== public-only extensions (preserved on vendor merge) =====

def task_kind(model: object) -> str:
    """Return ``"classification"``, ``"forecasting"``, or ``"regression"``."""
    if is_classifier(_unwrap(model)):
        return "classification"
    if is_forecaster(model):
        return "forecasting"
    return "regression"
