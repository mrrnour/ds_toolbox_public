"""Time-series intervention / synthetic-control analysis primitives.

Given a fitted (or fittable) time-series forecaster, this module turns the
pre/post arrays of an intervention study into:

* a per-day frame with observed, synthetic-control prediction, prediction
  interval, point-effect, and an ``outside_band`` flag, and
* a one-row scalar summary (daily / cumulative / relative effect, count of
  days outside the PI).

The module is cfg-free — it does not import any project ``ExperimentConfig``
or read YAML. Callers translate their own configs into the primitive
``(model, X_train, y_train, X_post, y_post, model_name)`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .training import _collect_predictions, _fit_one_fold


@dataclass
class InterventionResult:
    """Per-model pre/post intervention analysis output.

    ``frame`` columns: ``ts, y_true, y_pred, y_lo, y_hi, effect, outside_band, model``
    — the same prediction-frame schema used by :class:`BacktestReport` and
    :mod:`forecasters.windowed`. ``y_pred`` is the model's forecast on the
    post window (i.e. what it expects if no intervention had occurred);
    ``y_lo`` / ``y_hi`` are the prediction-interval bounds.
    """

    model: str
    frame: pd.DataFrame
    daily_effect: float
    cumulative_effect: float
    relative_lift: float


def _fit_predict_one(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_post: pd.DataFrame,
    y_post: pd.Series,
    model_name: str,
    interval_level: float,
) -> InterventionResult:
    # Reuse the CV loop's fit + predict helpers so intervention forecasts go
    # through the same code path (Pipeline unwrapping, predict_interval retry,
    # y_lower/y_upper naming) as ml_comparison. `_fit_one_fold` here degrades
    # to a plain .fit() — no early stopping, no eval_set.
    _fit_one_fold(model, X_train, y_train, X_post, y_post, None, None)
    preds = _collect_predictions(model, X_post, y_post, classifier=False, interval_level=interval_level)

    n = len(X_post)
    y_lo = preds["y_lower"].to_numpy(dtype=float) if "y_lower" in preds else np.full(n, np.nan)
    y_hi = preds["y_upper"].to_numpy(dtype=float) if "y_upper" in preds else np.full(n, np.nan)

    frame = pd.DataFrame({
        "ts": pd.to_datetime(X_post["ts"].to_numpy()),
        "y_true": preds["y_true"].to_numpy(dtype=float),
        "y_pred": preds["y_pred"].to_numpy(dtype=float),
        "y_lo": y_lo,
        "y_hi": y_hi,
        "model": model_name,
    })
    frame["effect"] = frame["y_true"] - frame["y_pred"]
    frame["outside_band"] = (frame["y_true"] < frame["y_lo"]) | (frame["y_true"] > frame["y_hi"])

    daily = float(np.nanmean(frame["effect"]))
    cumulative = float(np.nansum(frame["effect"]))
    denom = float(np.nansum(frame["y_pred"]))
    relative = cumulative / denom if denom else float("nan")
    return InterventionResult(
        model=model_name,
        frame=frame,
        daily_effect=daily,
        cumulative_effect=cumulative,
        relative_lift=relative,
    )


def estimate_intervention_effect(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_post: pd.DataFrame,
    y_post: pd.Series,
    model_name: str | None = None,
    interval_level: float = 0.95,
) -> InterventionResult | dict[str, InterventionResult]:
    """Fit one or many forecasters and return per-model intervention results.

    Pure compute step — caller is responsible for slicing pre/post, masking
    anomalies, and column renaming. ``X_post`` must contain a ``"ts"``
    timestamp column.

    Two calling conventions:

    * **Single model.** ``model`` is an estimator and ``model_name`` is the
      string label → returns an :class:`InterventionResult`.
    * **Multiple models.** ``model`` is a ``{name: estimator}`` dict (and
      ``model_name`` is left ``None``) → returns a ``{name: InterventionResult}``
      dict, with each entry computed by the single-model path above.
    """
    if isinstance(model, dict):
        if model_name is not None:
            raise TypeError("model_name must be None when passing a dict of models")
        return {
            name: _fit_predict_one(m, X_train, y_train, X_post, y_post, name, interval_level)
            for name, m in model.items()
        }
    if model_name is None:
        raise TypeError("model_name is required when passing a single model")
    return _fit_predict_one(model, X_train, y_train, X_post, y_post, model_name, interval_level)


def effect_summary(results: dict[str, InterventionResult]) -> pd.DataFrame:
    """Side-by-side daily / cumulative / relative effects per model."""
    return pd.DataFrame([
        {
            "model": name,
            "daily_effect": r.daily_effect,
            "cumulative_effect": r.cumulative_effect,
            "relative_lift": r.relative_lift,
            "n_outside_band": int(r.frame["outside_band"].sum()),
        }
        for name, r in results.items()
    ])


def effect_from_preds(
    preds: pd.DataFrame,
    *,
    split_col: str = "split",
    split_value: str = "val",
    model_col: str = "model",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-day effect + per-model summary from an ``ml_comparison`` preds frame.

    Preds-frame counterpart to :func:`estimate_intervention_effect` +
    :func:`effect_summary`, for the ``ml_comparison`` + 1-fold ``HoldoutSplit``
    workflow (train = pre-event, val = post-event). Avoids re-fitting each
    model just to compute the effect summary — the forecasts in ``preds``
    are reused directly.

    ``preds`` must carry columns ``y_true, y_pred, y_lo, y_hi, <split_col>,
    <model_col>``. Rows where ``<split_col> == <split_value>`` are treated
    as the post-intervention window.

    Returns ``(post_preds, summary)``:

    * ``post_preds`` — post-window rows + ``effect`` and ``outside_band`` columns.
    * ``summary`` — one row per model with ``daily_effect``,
      ``cumulative_effect``, ``relative_lift``, ``n_outside_band``.
    """
    post = preds[preds[split_col] == split_value].copy()
    post["effect"] = post["y_true"] - post["y_pred"]
    post["outside_band"] = (
        (post["y_true"] < post["y_lo"]) | (post["y_true"] > post["y_hi"])
    )

    grp = post.groupby(model_col)
    summary = pd.DataFrame({
        "daily_effect":      grp["effect"].mean(),
        "cumulative_effect": grp["effect"].sum(),
        "relative_lift":     grp["effect"].sum() / grp["y_pred"].sum(),
        "n_outside_band":    grp["outside_band"].sum().astype(int),
    }).reset_index()
    return post, summary


def sc_results_to_backtest_preds(
    results: dict[str, InterventionResult],
    df_pre: pd.DataFrame | None = None,
    *,
    date_col: str = "ts",
    value_col: str = "y",
) -> pd.DataFrame:
    """Convert per-model :class:`InterventionResult` frames to the long shape
    consumed by :class:`~dsToolbox.ml_funcs.backtest_plots.BacktestReport`.

    Treats the intervention as a single-fold backtest per model — the same
    ``(ts, y_true, y_pred, y_lo, y_hi)`` prediction schema as a CV fold, just
    with ``fold=0`` and ``split=="val"`` on the post window. Pre-event actuals
    from ``df_pre`` are stitched on as ``split=="train"`` rows so downstream
    plotters can draw training context without a refit.

    Returned columns: ``model, ts, fold, split, y_true, y_pred, y_lo, y_hi``.
    """
    rows: list[pd.DataFrame] = []
    for name, sc in results.items():
        f = sc.frame
        rows.append(pd.DataFrame({
            "model": name,
            "ts": pd.to_datetime(f["ts"]),
            "fold": 0,
            "split": "val",
            "y_true": f["y_true"].to_numpy(dtype=float),
            "y_pred": f["y_pred"].to_numpy(dtype=float),
            "y_lo": f["y_lo"].to_numpy(dtype=float),
            "y_hi": f["y_hi"].to_numpy(dtype=float),
        }))
        if df_pre is not None and len(df_pre):
            rows.append(pd.DataFrame({
                "model": name,
                "ts": pd.to_datetime(df_pre[date_col]),
                "fold": 0,
                "split": "train",
                "y_true": pd.to_numeric(df_pre[value_col], errors="coerce").to_numpy(dtype=float),
                "y_pred": np.nan,
                "y_lo": np.nan,
                "y_hi": np.nan,
            }))
    return pd.concat(rows, ignore_index=True)
