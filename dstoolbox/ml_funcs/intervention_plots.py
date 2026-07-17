"""Plotly figures for intervention / synthetic-control analysis.

Generic counterparts to the project-specific plots that previously lived in
``synthetic_control/src/plotting.py``. Takes the ``post_preds`` frame produced
by :func:`dstoolbox.ml_funcs.ts_intervention.effect_from_preds` — no project
``ExperimentConfig`` dependency.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_cumulative_effect_from_preds(
    post_preds: pd.DataFrame,
    *,
    date_col: str = "ts",
    model_col: str = "model",
    effect_col: str = "effect",
) -> go.Figure:
    """Cumulative observed-minus-forecast lift, one line per model.

    ``post_preds`` is the frame returned by
    :func:`dstoolbox.ml_funcs.ts_intervention.effect_from_preds` — the
    post-window rows with an ``effect`` column already computed.
    """
    fig = go.Figure()
    for name, g in post_preds.sort_values(date_col).groupby(model_col):
        fig.add_trace(go.Scatter(
            x=g[date_col], y=g[effect_col].cumsum(), mode="lines", name=name,
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title="Cumulative effect (observed − forecast)",
        xaxis_title=date_col,
        yaxis_title="cumulative lift",
    )
    return fig


# ===== imports preserved from public (needed by extras below) =====
from .ts_intervention import InterventionResult
from .ts_plots import plot_acf
from plotly.subplots import make_subplots


# ===== public-only extensions (preserved on vendor merge) =====

_PI_FILL = "rgba(111,67,214,0.20)"


_PI_LINE = "rgba(111,67,214,0.0)"


def plot_intervention(
    result: InterventionResult,
    df_ts: pd.DataFrame | None = None,
    *,
    event_date: pd.Timestamp | str | None = None,
    date_col: str = "ts",
    value_col: str = "y",
    lookback_days: int = 60,
) -> go.Figure:
    """Observed vs forecast + PI ribbon, with optional ``lookback_days`` of pre context.

    ``result.frame`` must carry ``ts, y_true, y_pred, y_lo, y_hi``.
    ``df_ts`` (optional) is the full pre-event series used to draw lookback
    context; when omitted, only the post window is shown.
    """
    frame = result.frame
    fig = go.Figure()

    if df_ts is not None and event_date is not None:
        event = pd.Timestamp(event_date)
        context_start = event - pd.Timedelta(days=lookback_days)
        pre_ctx = df_ts[(df_ts[date_col] >= context_start) & (df_ts[date_col] < event)]
        fig.add_trace(
            go.Scatter(
                x=pre_ctx[date_col], y=pre_ctx[value_col],
                name="pre observed", mode="markers",
            )
        )

    fig.add_trace(go.Scatter(x=frame["ts"], y=frame["y_true"], name="post observed", mode="markers"))
    fig.add_trace(go.Scatter(x=frame["ts"], y=frame["y_pred"], name="forecast", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=list(frame["ts"]) + list(frame["ts"][::-1]),
            y=list(frame["y_hi"]) + list(frame["y_lo"][::-1]),
            fill="toself", fillcolor=_PI_FILL,
            line={"width": 0}, name="forecast PI",
        )
    )
    if event_date is not None:
        fig.add_vline(x=pd.Timestamp(event_date), line_dash="dash", line_color="red")
    fig.update_layout(
        title=f"{result.model} — analysis",
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    return fig


def plot_forecast_faceted(
    results: dict[str, InterventionResult],
    df_ts: pd.DataFrame | None = None,
    *,
    event_date: pd.Timestamp | str | None = None,
    date_col: str = "ts",
    value_col: str = "y",
    lookback_days: int = 60,
    cols: int = 1,
) -> go.Figure:
    """One subplot per model: pre-context + observed + forecast + PI ribbon."""
    event = pd.Timestamp(event_date) if event_date is not None else None
    pre_ctx = None
    if df_ts is not None and event is not None:
        context_start = event - pd.Timedelta(days=lookback_days)
        pre_ctx = df_ts[(df_ts[date_col] >= context_start) & (df_ts[date_col] < event)]

    names = list(results.keys())
    n = len(names)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, shared_xaxes=True)
    for i, name in enumerate(names):
        r, c = i // cols + 1, i % cols + 1
        frame = results[name].frame
        fig.add_trace(
            go.Scatter(
                x=list(frame["ts"]) + list(frame["ts"][::-1]),
                y=list(frame["y_hi"]) + list(frame["y_lo"][::-1]),
                fill="toself", fillcolor=_PI_FILL,
                line={"width": 0, "color": _PI_LINE},
                name="PI", showlegend=False, hoverinfo="skip",
            ),
            row=r, col=c,
        )
        if pre_ctx is not None:
            fig.add_trace(
                go.Scatter(
                    x=pre_ctx[date_col], y=pre_ctx[value_col],
                    mode="markers", name="pre observed",
                    legendgroup="pre", showlegend=(i == 0),
                    marker={"color": "#15C089", "size": 5},
                ),
                row=r, col=c,
            )
        fig.add_trace(
            go.Scatter(
                x=frame["ts"], y=frame["y_true"],
                mode="markers", name="post observed",
                legendgroup="post", showlegend=(i == 0),
                marker={"color": "#2A2A2A", "size": 5},
            ),
            row=r, col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=frame["ts"], y=frame["y_pred"],
                mode="lines", name="forecast",
                legendgroup="sc", showlegend=(i == 0),
                line={"color": "#6F43D6"},
            ),
            row=r, col=c,
        )
        if event is not None:
            fig.add_vline(x=event, line_dash="dash", line_color="red", row=r, col=c)
    fig.update_layout(title="Analysis — by model", height=320 * rows)
    return fig


def plot_cumulative_effect(results: dict[str, InterventionResult]) -> go.Figure:
    """Cumulative observed-minus-forecast lift over the post window."""
    fig = go.Figure()
    for name, res in results.items():
        f = res.frame.sort_values("ts").copy()
        f["cum_effect"] = f["effect"].cumsum()
        fig.add_trace(go.Scatter(x=f["ts"], y=f["cum_effect"], mode="lines", name=name))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title="Cumulative effect (observed − forecast)",
        xaxis_title="ts",
        yaxis_title="cumulative lift",
    )
    return fig


def plot_residual_acf(result: InterventionResult, nlags: int = 40) -> go.Figure:
    """ACF of post-window residuals (``y_true - y_pred``).

    Sanity check only — the post window is where the intervention may be
    present, so non-trivial residual autocorrelation does not necessarily
    indicate a misspecified model. For backtest residuals, compute
    ``preds["y_true"] - preds["y_pred"]`` and call :func:`plot_acf` directly.
    """
    resid = (result.frame["y_true"] - result.frame["y_pred"]).to_numpy()
    return plot_acf(resid, nlags=nlags, title=f"{result.model} — post residual ACF")
