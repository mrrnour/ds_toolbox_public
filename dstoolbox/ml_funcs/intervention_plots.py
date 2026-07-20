"""Plotly figures for intervention / synthetic-control analysis.

Generic counterparts to the project-specific plots that previously lived in
``synthetic_control/src/plotting.py``. Takes the ``post_preds`` frame produced
by :func:`dstoolbox.ml_funcs.ts_intervention.effect_from_preds` — no project
``ExperimentConfig`` dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_cumulative_effect_from_preds(
    post_preds: pd.DataFrame,
    *,
    date_col: str = "ts",
    model_col: str = "model",
    effect_col: str = "effect",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    focus_model: str | None = None,
    intervention_date: "pd.Timestamp | str | None" = None,
    effect_summary: pd.DataFrame | None = None,
    experiment: str | None = None,
    show_decomposition: bool = True,
) -> go.Figure:
    """Cumulative observed-minus-forecast lift, one line per model.

    ``post_preds`` is the frame returned by
    :func:`dstoolbox.ml_funcs.ts_intervention.effect_from_preds` — the
    post-window rows with an ``effect`` column already computed.

    Enrichment parameters (all optional; backward compatible when omitted):

    * ``focus_model`` — restrict the figure to a single model. Renames its
      cumulative-effect line to ``Σ(y − ŷ)  effect`` so it pairs cleanly
      with the decomposition traces.
    * ``show_decomposition`` — when a ``focus_model`` is set (and the
      ``y_true`` / ``y_pred`` columns are present), overlay cumulative
      observed (Σy) and cumulative forecast (Σŷ) traces. The vendor's
      line equals Σy − Σŷ, so this makes the decomposition explicit.
    * ``intervention_date`` — draw a red dashed vertical marker + label.
      Added via ``add_shape`` + ``add_annotation`` (not ``add_vline``) to
      sidestep a Plotly bug where ``add_vline`` computes
      ``_mean([x0, x1])`` via ``float(sum([x, x])) / 2`` — that path
      fails for both ``pd.Timestamp`` ("Addition/subtraction of integers
      … with Timestamp is no longer supported") and ISO strings
      ("unsupported operand types int + str").
    * ``effect_summary`` — the per-model summary returned by
      :func:`effect_from_preds`. When provided together with
      ``focus_model``, the title becomes
      ``[<experiment>] <model> — cumulative Σy vs Σŷ vs Σ(y − ŷ)`` with a
      ``<sup>relative lift | n_post | cumulative lift</sup>`` subtitle,
      plus an endpoint annotation on the effect line and a footnote
      spelling out the avg-daily-lift formula. ``n_outside_band`` is
      deliberately omitted — only meaningful once PI calibration is
      verified.
    """
    frame = post_preds.sort_values(date_col)
    if focus_model is not None:
        frame = frame[frame[model_col] == focus_model]

    fig = go.Figure()
    for name, g in frame.groupby(model_col):
        trace_name = "Σ(y − ŷ)  effect" if focus_model is not None else name
        line_kwargs: dict = {"width": 2}
        marker_kwargs: dict | None = None
        mode = "lines"
        if focus_model is not None:
            mode = "lines+markers"
            line_kwargs["color"] = "#6f43d6"
            marker_kwargs = {"color": "#6f43d6", "size": 7, "symbol": "circle"}
        scatter_kwargs: dict = {
            "x": g[date_col], "y": g[effect_col].cumsum(),
            "mode": mode, "name": trace_name, "line": line_kwargs,
        }
        if marker_kwargs is not None:
            scatter_kwargs["marker"] = marker_kwargs
        fig.add_trace(go.Scatter(**scatter_kwargs))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")

    focus = None
    if focus_model is not None and not frame.empty:
        focus = frame  # already filtered
        if show_decomposition and {y_true_col, y_pred_col}.issubset(focus.columns):
            fig.add_trace(go.Scatter(
                x=focus[date_col], y=focus[y_true_col].cumsum(),
                mode="lines+markers", name="Σy  observed",
                line={"color": "#15C089", "dash": "solid"},
                marker={"color": "#15C089", "size": 7, "symbol": "circle"},
            ))
            fig.add_trace(go.Scatter(
                x=focus[date_col], y=focus[y_pred_col].cumsum(),
                mode="lines+markers", name="Σŷ  forecast",
                line={"color": "#F1A340", "dash": "dash"},
                marker={"color": "#F1A340", "size": 7, "symbol": "diamond"},
            ))

    if intervention_date is not None:
        iv = pd.Timestamp(intervention_date)
        # See docstring: add_vline auto-annotation bug on datetime x — use
        # add_shape + add_annotation to avoid the _mean([x, x]) code path.
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=iv, x1=iv, y0=0, y1=1,
            line={"color": "red", "dash": "dash"},
        )
        fig.add_annotation(
            x=iv, y=1, xref="x", yref="paper",
            text="intervention", showarrow=False,
            xanchor="left", yanchor="top",
        )

    layout_updates: dict = {
        "title": "Cumulative effect (observed − forecast)",
        "xaxis_title": date_col,
        "yaxis_title": "cumulative lift",
    }

    if focus is not None and effect_summary is not None:
        row = effect_summary.loc[effect_summary[model_col] == focus_model]
        if not row.empty:
            r = row.iloc[0]
            n_days = len(focus)
            final_cum = float(focus[effect_col].sum()) if n_days else float("nan")

            if n_days:
                fig.add_annotation(
                    x=focus[date_col].iloc[-1], y=final_cum,
                    text=f"Σ(y − ŷ)={final_cum:.4g}",
                    showarrow=True, arrowhead=2, ax=-40, ay=-30,
                )

            fig.add_annotation(
                x=0, y=-0.22, xref="paper", yref="paper",
                text=(
                    f"cumulative lift = Σ(y − ŷ) over post window "
                    f"= {r['cumulative_effect']:.4g}<br>"
                    f"relative lift = Σ(y − ŷ) / Σŷ "
                    f"= {r['relative_lift']:.2%}"
                ),
                showarrow=False,
                xanchor="left", yanchor="top", align="left",
                font={"size": 10, "color": "gray"},
            )

            prefix = f"[{experiment}] " if experiment else ""
            layout_updates["title"] = (
                f"{prefix}{focus_model} — cumulative Σy vs Σŷ vs Σ(y − ŷ)"
                f"<br><sup>"
                f"relative lift={r['relative_lift']:.2%} | "
                f"n_post={n_days} days | "
                f"cumulative lift={r['cumulative_effect']:.4g}"
                f"</sup>"
            )
            layout_updates["yaxis_title"] = "cumulative value"
            layout_updates["margin"] = {"b": 110}

    fig.update_layout(**layout_updates)
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
    intervention_date: pd.Timestamp | str | None = None,
    date_col: str = "ts",
    value_col: str = "y",
    lookback_days: int = 60,
) -> go.Figure:
    """Observed vs forecast + PI ribbon, with optional ``lookback_days`` of pre context.

    ``result.frame`` must carry ``ts, y_true, y_pred, y_lo, y_hi``.
    ``df_ts`` (optional) is the full pre-intervention series used to draw lookback
    context; when omitted, only the post window is shown.
    """
    frame = result.frame
    fig = go.Figure()

    if df_ts is not None and intervention_date is not None:
        intervention = pd.Timestamp(intervention_date)
        context_start = intervention - pd.Timedelta(days=lookback_days)
        pre_ctx = df_ts[(df_ts[date_col] >= context_start) & (df_ts[date_col] < intervention)]
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
    if intervention_date is not None:
        fig.add_vline(x=pd.Timestamp(intervention_date), line_dash="dash", line_color="red")
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
    intervention_date: pd.Timestamp | str | None = None,
    date_col: str = "ts",
    value_col: str = "y",
    lookback_days: int = 60,
    cols: int = 1,
) -> go.Figure:
    """One subplot per model: pre-context + observed + forecast + PI ribbon."""
    intervention = pd.Timestamp(intervention_date) if intervention_date is not None else None
    pre_ctx = None
    if df_ts is not None and intervention is not None:
        context_start = intervention - pd.Timedelta(days=lookback_days)
        pre_ctx = df_ts[(df_ts[date_col] >= context_start) & (df_ts[date_col] < intervention)]

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
        if intervention is not None:
            fig.add_vline(x=intervention, line_dash="dash", line_color="red", row=r, col=c)
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


# ===== audience-friendly cumulative-effect view =====================
# Same numbers as `plot_cumulative_effect_from_preds` (Σy, Σŷ, Σ(y−ŷ)),
# re-framed for a non-technical audience: green = "what actually
# happened", orange dashed = "what we expected without the launch",
# shaded band = daily gap (green = gain, red = loss), purple dotted =
# running total of the gap, black long-dashed = average-rate reference
# line. Single y-axis (cumulative percentage points, added up daily).

_PLAIN_GREEN = "#15C089"
_PLAIN_ORANGE = "#F1A340"
_PLAIN_PURPLE = "#6f43d6"
_PLAIN_NEG = "#c83c3c"
_PLAIN_BAND_GAIN = "rgba(21,192,137,0.18)"
_PLAIN_BAND_LOSS = "rgba(220,60,60,0.18)"
_INVISIBLE = "rgba(0,0,0,0)"


def _plain_add_signed_band(
    fig: go.Figure,
    ts: pd.Series,
    cum_obs: np.ndarray,
    cum_cf: np.ndarray,
) -> None:
    """Add two invisible baselines + two tonexty envelopes so the gap
    between observed and forecast is shaded green above the forecast and
    red below it. Base traces are hidden from legend; only the two fill
    traces show.
    """
    upper_env = np.maximum(cum_obs, cum_cf)
    lower_env = np.minimum(cum_obs, cum_cf)
    base = {"mode": "lines", "line": {"width": 0, "color": _INVISIBLE},
            "showlegend": False, "hoverinfo": "skip"}
    fig.add_trace(go.Scatter(x=ts, y=cum_cf, **base))
    fig.add_trace(go.Scatter(
        x=ts, y=upper_env, mode="lines",
        line={"width": 0, "color": _INVISIBLE},
        fill="tonexty", fillcolor=_PLAIN_BAND_GAIN,
        name="Extra gain from the launch", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=ts, y=cum_cf, **base))
    fig.add_trace(go.Scatter(
        x=ts, y=lower_env, mode="lines",
        line={"width": 0, "color": _INVISIBLE},
        fill="tonexty", fillcolor=_PLAIN_BAND_LOSS,
        name="Loss from the launch", hoverinfo="skip",
    ))


def _plain_add_level_traces(
    fig: go.Figure,
    ts: pd.Series,
    cum_obs: pd.Series,
    cum_cf: pd.Series,
) -> None:
    """Add the two solid-story lines: observed (green) and forecast (orange dashed)."""
    fig.add_trace(go.Scatter(
        x=ts, y=cum_obs, mode="lines+markers",
        name="What actually happened",
        line={"color": _PLAIN_GREEN, "width": 3},
        marker={"color": _PLAIN_GREEN, "size": 7, "symbol": "circle"},
        hovertemplate="%{x|%b %d}<br>cumulative observed = %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=cum_cf, mode="lines+markers",
        name="What we expected without the launch",
        line={"color": _PLAIN_ORANGE, "width": 3, "dash": "dash"},
        marker={"color": _PLAIN_ORANGE, "size": 7, "symbol": "diamond"},
        hovertemplate="%{x|%b %d}<br>cumulative expected = %{y:.3f}<extra></extra>",
    ))


def _plain_add_effect_traces(
    fig: go.Figure,
    focus_ts: pd.Series,
    cum_effect: pd.Series,
    intervention_date: pd.Timestamp,
    final_gain: float,
    daily_lift: float,
    n_days: int,
    rate_color: str,
) -> None:
    """Add the running-total purple line + straight average-rate reference line
    + a mid-line annotation showing the per-day slope value.
    """
    fig.add_trace(go.Scatter(
        x=focus_ts, y=cum_effect, mode="lines+markers",
        name="Running total of the extra gain",
        line={"color": _PLAIN_PURPLE, "width": 2, "dash": "dot"},
        marker={"color": _PLAIN_PURPLE, "size": 6, "symbol": "circle-open"},
        hovertemplate="%{x|%b %d}<br>running gain = %{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[intervention_date, focus_ts.iloc[-1]],
        y=[0.0, final_gain], mode="lines",
        name=f"Average rate ({daily_lift:+.3f} pct pts/day)",
        line={"color": rate_color, "width": 2.5, "dash": "longdash"},
        hovertemplate=(
            f"average rate = {daily_lift:+.3f} pct pts/day"
            f"<br>= total {final_gain:+.3f} pct pts ÷ {n_days} days"
            "<extra></extra>"
        ),
    ))
    mid_idx = n_days // 2
    fig.add_annotation(
        x=focus_ts.iloc[mid_idx], y=daily_lift * (mid_idx + 1),
        text=f"<b>{daily_lift:+.3f} pct pts/day</b>",
        showarrow=False, xanchor="center", yanchor="bottom",
        font={"size": 10, "color": rate_color},
        bgcolor="rgba(255,255,255,0.85)",
    )


def _plain_add_intervention_marker(
    fig: go.Figure,
    intervention_date: pd.Timestamp,
) -> None:
    """Add the red dashed vertical marker + label at the intervention date.
    Uses add_shape + add_annotation to sidestep the same add_vline datetime
    bug documented in ``plot_cumulative_effect_from_preds``.
    """
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=intervention_date, x1=intervention_date, y0=0, y1=1,
        line={"color": "red", "dash": "dash"},
    )
    fig.add_annotation(
        x=intervention_date, y=1, xref="x", yref="paper",
        text=f"Launch — {intervention_date.date()}",
        showarrow=False, xanchor="left", yanchor="top",
        font={"color": "red"},
    )


def _plain_add_endpoint_annotation(
    fig: go.Figure,
    x_last: pd.Timestamp,
    final_gain: float,
    n_days: int,
    rel_lift: float,
) -> None:
    """Add the endpoint call-out with total gain + relative lift. Border
    tints red when the endpoint is negative so a loss doesn't sit inside a
    purple (gain-colored) box.
    """
    border = _PLAIN_PURPLE if final_gain >= 0 else _PLAIN_NEG
    fig.add_annotation(
        x=x_last, y=final_gain,
        text=(
            f"total extra gain over {n_days} days = {final_gain:+.3f} pct pts"
            f"<br><b>relative lift = {rel_lift:+.1%}</b>"
        ),
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=border, borderwidth=1,
    )


def _plain_footer_text(
    rel_lift: float,
    final_gain: float,
    daily_lift: float,
    n_days: int,
    cum_expected: float,
    cum_observed: float,
) -> str:
    """Return the plain-English footer text: how to read the plot + formula
    breakdown for each headline number (relative lift, total, avg daily).
    Kept as a helper so the main function stays under the 40-line cap.
    """
    return (
        "<b>How to read this</b> — everything is on one y-axis "
        "(cumulative percentage points of conversion rate, added up daily).<br>"
        "Green line = what really happened, added up day by day.<br>"
        "Orange dashed line = what the pre-launch trend said we should have seen.<br>"
        "Shaded band = daily difference between the two — "
        f"<span style='color:{_PLAIN_GREEN}'><b>green</b></span> when the launch helped "
        "(observed above expected), "
        f"<span style='color:{_PLAIN_NEG}'><b>red</b></span> when it hurt "
        "(observed below expected).<br>"
        "Purple dotted line = running total of that signed gain, day by day "
        "(goes negative when the launch is under-performing). It looks small "
        "next to the green/orange lines because the launch moves a big "
        "baseline by a small-but-real amount.<br>"
        "Black long-dashed line = <b>average rate</b>: straight line from launch "
        "day (gain = 0) to the endpoint, so its slope is the average daily lift "
        "in percentage points per day. Purple above the black line means gains "
        "are accelerating; purple below means they are decelerating.<br>"
        "<br>"
        "<b>What the headline numbers mean</b> "
        "<i>(y = observed, ŷ = pre-launch expected, n = number of post-launch days)</i><br>"
        f"<b>{rel_lift:+.1%}</b> = <i>relative lift</i>: total extra gain ÷ what "
        "we expected without the launch (i.e. the green band as a fraction of the "
        f"orange dashed total). &nbsp;<i>Formula:</i> Σ(y − ŷ) / Σŷ "
        f"= ({final_gain:+.3f}) / ({cum_expected:.3f}).<br>"
        f"<b>{final_gain:+.3f} pct pts</b> = <i>total extra gain</i>: sum of the "
        f"daily gaps over {n_days} days, in percentage points of conversion rate. "
        f"&nbsp;<i>Formula:</i> Σ(y − ŷ) = Σy − Σŷ "
        f"= ({cum_observed:.3f}) − ({cum_expected:.3f}).<br>"
        f"<b>{daily_lift:+.3f} pct pts/day</b> = <i>average daily lift</i>: the "
        f"same total ÷ {n_days} days. &nbsp;<i>Formula:</i> Σ(y − ŷ) / n "
        f"= ({final_gain:+.3f}) / {n_days}."
    )


def _plain_layout_kwargs(
    experiment: str | None,
    intervention_date: pd.Timestamp,
    n_days: int,
    rel_lift: float,
    daily_lift: float,
) -> dict:
    """Build the update_layout kwargs (title, axes, legend, margin, height)."""
    prefix = f"[{experiment}] " if experiment else ""
    direction = "above" if rel_lift >= 0 else "below"
    return {
        "title": {
            "text": (
                f"{prefix}Did the launch on {intervention_date.date()} actually help?"
                f"<br><sup>Over {n_days} days since launch, conversion is "
                f"{rel_lift:+.1%} {direction} the pre-launch baseline "
                f"(≈ {daily_lift:+.3f} percentage points per day on average).</sup>"
            ),
            "y": 0.97, "yanchor": "top",
        },
        "xaxis_title": "date",
        "yaxis_title": (
            "cumulative conversion rate<br>"
            "<sub>(percentage points, added up daily)</sub>"
        ),
        "legend": {
            "orientation": "h",
            "xanchor": "center", "yanchor": "bottom",
            "x": 0.5, "y": 1.02,
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "lightgrey", "borderwidth": 1,
        },
        # Bottom margin sized to hold the two-block footer (~180px).
        "margin": {"t": 140, "b": 220, "l": 110, "r": 60},
        "height": 780,
        "hovermode": "x unified",
    }


def plot_cumulative_effect_plain(
    post_preds: pd.DataFrame,
    *,
    focus_model: str,
    intervention_date: "pd.Timestamp | str",
    effect_summary: pd.DataFrame,
    experiment: str | None = None,
    date_col: str = "ts",
    model_col: str = "model",
    effect_col: str = "effect",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> go.Figure:
    """Audience-friendly cumulative-effect plot for a single focus model.

    Same numbers as :func:`plot_cumulative_effect_from_preds` (Σy, Σŷ,
    Σ(y−ŷ)) but framed for a non-technical audience: "what actually
    happened" vs "what we expected without the launch", with the daily
    gap shaded (green = gain, red = loss). All series share one y-axis
    (cumulative percentage points of conversion rate). The purple
    running-total line will look small next to the green/orange level
    lines — that IS the point (small effect on a big baseline).

    Args:
        post_preds: Post-window frame from
            :func:`dstoolbox.ml_funcs.ts_intervention.effect_from_preds`
            (needs ``ts``, ``model``, ``effect``, ``y_true``, ``y_pred``).
        focus_model: The single model to plot.
        intervention_date: Launch date; used for the red dashed marker,
            the average-rate line origin, and the title.
        effect_summary: Per-model summary from :func:`effect_from_preds`;
            the row for ``focus_model`` supplies ``daily_effect`` and
            ``relative_lift`` for the annotations.
        experiment: Optional experiment tag prepended to the title.
        date_col, model_col, effect_col, y_true_col, y_pred_col: Column
            overrides (defaults match ``effect_from_preds``).

    Returns:
        Plotly figure with a single y-axis and the enriched layout.

    Example:
        >>> post_preds, effect_summary = effect_from_preds(forecast_preds)
        >>> fig = plot_cumulative_effect_plain(
        ...     post_preds,
        ...     focus_model="auto_arima",
        ...     intervention_date="2026-07-02",
        ...     effect_summary=effect_summary,
        ...     experiment="video_and_oldest_stills",
        ... )
    """
    iv = pd.Timestamp(intervention_date)
    focus = (
        post_preds[post_preds[model_col] == focus_model]
        .sort_values(date_col)
        .copy()
    )
    if focus.empty:
        raise ValueError(
            f"plot_cumulative_effect_plain: no rows for focus_model={focus_model!r} "
            f"in post_preds."
        )
    row = effect_summary.loc[effect_summary[model_col] == focus_model]
    if row.empty:
        raise ValueError(
            f"plot_cumulative_effect_plain: no row for focus_model={focus_model!r} "
            f"in effect_summary."
        )

    focus["cum_obs"] = focus[y_true_col].cumsum()
    focus["cum_cf"] = focus[y_pred_col].cumsum()
    focus["cum_effect"] = focus[effect_col].cumsum()

    r = row.iloc[0]
    n_days = len(focus)
    final_gain = float(focus["cum_effect"].iloc[-1])
    rel_lift = float(r["relative_lift"])
    daily_lift = float(r["daily_effect"])
    cum_expected = float(focus["cum_cf"].iloc[-1])
    cum_observed = float(focus["cum_obs"].iloc[-1])
    rate_color = "#111111" if final_gain >= 0 else _PLAIN_NEG

    fig = go.Figure()
    _plain_add_signed_band(
        fig, focus[date_col],
        focus["cum_obs"].to_numpy(), focus["cum_cf"].to_numpy(),
    )
    _plain_add_level_traces(fig, focus[date_col], focus["cum_obs"], focus["cum_cf"])
    _plain_add_effect_traces(
        fig, focus[date_col], focus["cum_effect"],
        iv, final_gain, daily_lift, n_days, rate_color,
    )
    _plain_add_intervention_marker(fig, iv)
    _plain_add_endpoint_annotation(
        fig, focus[date_col].iloc[-1], final_gain, n_days, rel_lift,
    )
    fig.add_annotation(
        x=0, y=-0.18, xref="paper", yref="paper",
        text=_plain_footer_text(
            rel_lift, final_gain, daily_lift, n_days,
            cum_expected, cum_observed,
        ),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font={"size": 10, "color": "gray"},
    )
    fig.update_layout(**_plain_layout_kwargs(
        experiment, iv, n_days, rel_lift, daily_lift,
    ))
    return fig

