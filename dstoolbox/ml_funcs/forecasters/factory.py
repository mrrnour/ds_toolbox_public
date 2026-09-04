"""Config-driven factory: registry name → forecaster instance.

The dispatch table maps short names (``"auto_arima"``, ``"greykite"``,
``"windowed"``, ...) to the sklearn-compatible adapters exported from
:mod:`dstoolbox.ml_funcs.forecasters`. Notebooks and CLI/YAML pipelines get
a one-line way to instantiate the right adapter without duplicating the
wiring code (``date_col``, ``freq``, holiday kwargs, recursive ``base``
unpacking for windowed).

The ``"windowed"`` entry recurses: its ``params.base`` is itself a
``{"name": ..., "params": {...}}`` dict, so callers can compose e.g. a
windowed seasonal-naive with one call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import (
    AutoArimaSklearn,
    DartsNBEATSSklearn,
    DartsThetaSklearn,
    MeanBaseline,
    SeasonalNaive,
    SilverkiteSklearn,
    WindowedForecaster,
)


def _holiday_kwargs(holidays: Any) -> dict[str, Any]:
    """Normalize a holidays spec (dict, dataclass, pydantic model, or ``None``) to Silverkite kwargs.

    Recognised fields — all optional — with sensible fallbacks:

    - ``enabled``       (bool, default True if any other field is set)
    - ``countries``     (list[str], default ``["US"]``)
    - ``pre_days``      (int, default 0)
    - ``post_days``     (int, default 0)

    Returns an empty dict when holidays is ``None`` or ``enabled`` is false.
    """
    if holidays is None:
        return {}
    get = (
        holidays.get
        if isinstance(holidays, Mapping)
        else (lambda k, default=None: getattr(holidays, k, default))
    )
    enabled = get("enabled", True)
    if not enabled:
        return {}
    return {
        "holiday_countries": get("countries", ["US"]),
        "holiday_pre_num_days": get("pre_days", 0),
        "holiday_post_num_days": get("post_days", 0),
    }


def build_forecaster(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    freq: str,
    date_col: str = "ts",
    holidays: Any = None,
):
    """Instantiate one dsToolbox forecaster from a name + params dict.

    Parameters
    ----------
    name
        Registry key. One of: ``mean_baseline``, ``seasonal_naive``,
        ``auto_arima``, ``greykite``, ``darts_theta``, ``darts_nbeats``,
        ``windowed``.
    params
        Model-specific kwargs (matches YAML ``models[i].params``). For
        ``windowed`` this must contain a nested ``base = {"name": ...,
        "params": {...}}`` sub-spec.
    freq
        Pandas offset string (``"D"``, ``"W"``, ...). Required by every
        backend adapter.
    date_col
        Name of the date column in ``X``. Defaults to ``"ts"`` — the
        convention used by :func:`to_Xy`-style preprocessing.
    holidays
        Optional holiday spec forwarded to Silverkite only. Accepts a
        ``Mapping`` (``{"enabled": True, "countries": [...], "pre_days": 0,
        "post_days": 0}``) or any object exposing those attributes.

    Raises
    ------
    ValueError
        If ``name`` is not in the registry, or if ``name == "windowed"`` and
        ``params["base"]`` is missing / malformed.
    """
    p = dict(params or {})

    if name == "mean_baseline":
        return MeanBaseline(date_col=date_col)
    if name == "seasonal_naive":
        return SeasonalNaive(
            date_col=date_col,
            season_length=p.get("season_length", 7),
            freq=freq,
        )
    if name == "auto_arima":
        return AutoArimaSklearn(
            date_col=date_col,
            season_length=p.get("season_length", 1),
            freq=freq,
            **{k: v for k, v in p.items() if k != "season_length"},
        )
    if name == "greykite":
        # ``model_template`` is fixed to AUTO inside the adapter; drop if present.
        return SilverkiteSklearn(
            date_col=date_col,
            freq=freq,
            **_holiday_kwargs(holidays),
            **{k: v for k, v in p.items() if k != "model_template"},
        )
    if name == "darts_theta":
        return DartsThetaSklearn(
            date_col=date_col,
            freq=freq,
            season_length=p.get("season_length"),
            **{k: v for k, v in p.items() if k != "season_length"},
        )
    if name == "darts_nbeats":
        return DartsNBEATSSklearn(date_col=date_col, freq=freq, **p)
    if name == "windowed":
        base_spec = p.get("base")
        if not isinstance(base_spec, Mapping) or "name" not in base_spec:
            raise ValueError(
                'windowed forecaster requires params.base = {"name": ..., "params": {...}}'
            )
        base = build_forecaster(
            base_spec["name"],
            base_spec.get("params"),
            freq=freq,
            date_col=date_col,
            holidays=holidays,
        )
        return WindowedForecaster(
            base_model=base,
            window=p.get("window", "14D"),
            weight_col=p.get("weight_col"),
            date_col=date_col,
        )
    raise ValueError(f"unknown forecaster name: {name!r}")
