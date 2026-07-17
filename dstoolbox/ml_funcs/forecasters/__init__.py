"""Forecaster adapters (sklearn-compatible) for time-series backends.

All adapters expose the standard sklearn surface ``fit(X, y) / predict(X)`` so
they plug into :func:`dstoolbox.ml_funcs.ml_comparison` alongside plain
regressors. Each adapter additionally requires ``X`` to contain a date column
(``date_col=``).

Light adapters (no heavy deps): :class:`MeanBaseline`, :class:`SeasonalNaive`.

Heavy adapters (optional deps loaded lazily — import them only if the
underlying backend is installed):

- :class:`AutoArimaSklearn` — needs ``statsforecast``
- :class:`SilverkiteSklearn` — needs ``greykite``
- :class:`DartsThetaSklearn`, :class:`DartsNBEATSSklearn` — need ``darts``

Missing backends do not break the package import; the adapter class itself
remains importable and raises a descriptive :class:`ImportError` only when
its underlying module is touched.
"""

from __future__ import annotations

# Light adapters — always importable.
from .naive import MeanBaseline, SeasonalNaive
from .windowed import WindowedForecaster

# Heavy adapters — wrap in try/except so a missing backend doesn't break the
# package. The adapter classes themselves still import fine because each
# uses :func:`optional_import` for its backend.
__all__ = [
    "MeanBaseline",
    "SeasonalNaive",
    "WindowedForecaster",
    "AutoArimaSklearn",
    "SilverkiteSklearn",
    "DartsThetaSklearn",
    "DartsNBEATSSklearn",
    "available_backends",
]

try:
    from .arima import AutoArimaSklearn  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    AutoArimaSklearn = None  # type: ignore[assignment]
    _arima_err = _e
else:
    _arima_err = None

try:
    from .greykite import SilverkiteSklearn  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    SilverkiteSklearn = None  # type: ignore[assignment]
    _greykite_err = _e
else:
    _greykite_err = None

try:
    from .darts import DartsThetaSklearn, DartsNBEATSSklearn  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    DartsThetaSklearn = None  # type: ignore[assignment]
    DartsNBEATSSklearn = None  # type: ignore[assignment]
    _darts_err = _e
else:
    _darts_err = None


def available_backends() -> dict[str, bool]:
    """Return ``{adapter_name: backend_installed}`` for all heavy adapters.

    Probes whether the *underlying* backend package is importable — not just
    whether our adapter class loads (it always does, because backends are
    imported lazily). Useful for notebooks to pick a model set based on
    what's actually installed::

        from dstoolbox.ml_funcs.forecasters import available_backends, SeasonalNaive
        models = [SeasonalNaive()]
        if available_backends()["AutoArimaSklearn"]:
            from dstoolbox.ml_funcs.forecasters import AutoArimaSklearn
            models.append(AutoArimaSklearn(season_length=7))
    """
    import importlib.util as _util

    def _has(mod: str) -> bool:
        return _util.find_spec(mod) is not None

    return {
        "AutoArimaSklearn": _has("statsforecast"),
        "SilverkiteSklearn": _has("greykite"),
        "DartsThetaSklearn": _has("darts"),
        "DartsNBEATSSklearn": _has("darts"),
    }


# ===== public-only extensions (preserved on vendor merge) =====

from ._base import DropColumns, align_forecast  # noqa: E402,F401
from .sklearn_lag import LagRegressor  # noqa: E402,F401
from .factory import build_forecaster  # noqa: E402,F401

__all__ += ["DropColumns", "align_forecast", "LagRegressor", "build_forecaster"]
