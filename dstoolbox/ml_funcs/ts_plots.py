"""Time-series Plotly figures: series, seasonality, autocorrelation, cross-correlation.

Data layer lives in :mod:`dstoolbox.ml_funcs.ts_eda` (``acf``, ``acf_confint``,
``seasonal_table``). These are pure renderers — no fitting, no I/O.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from . import ts_eda


def _fit_control_limits(
    s: pd.Series,
    method: str,
    coef: float = 3.0,
) -> tuple[float, float, float]:
    """Return ``(lower, center, upper)`` for ``sigma`` or ``imr`` on a clean series.

    - ``sigma``: ``mean ± coef·std`` (assumes stationarity).
    - ``imr``: Shewhart Individuals chart ``mean ± 2.66·MR̄`` where
      ``MR̄ = mean(|diff|)``. Robust to mild drift via the moving-range.
    """
    x = s.dropna().astype(float)
    if len(x) < 2:
        nan = float("nan")
        return nan, nan, nan
    mu = float(x.mean())
    if method == "sigma":
        sd = float(x.std(ddof=0))
        return mu - coef * sd, mu, mu + coef * sd
    if method == "imr":
        mr = float(x.diff().abs().mean())
        return mu - 2.66 * mr, mu, mu + 2.66 * mr
    raise ValueError(f"unknown control-limit method: {method!r}")


def _as_event_list(event_date) -> list[pd.Timestamp]:
    """Normalize ``event_date`` (scalar / list / None) to a list of Timestamps."""
    if event_date is None:
        return []
    if isinstance(event_date, (list, tuple)):
        return [pd.Timestamp(e) for e in event_date]
    return [pd.Timestamp(event_date)]


def _as_anomaly_groups(anomalies) -> list[list]:
    """Normalize ``anomalies`` to a list of groups (each group is a list of windows).

    Accepts a flat list of windows (single group) or a nested list of lists
    (one group per sublist, e.g. for plotting several anomaly classes in
    different colors).
    """
    if not anomalies:
        return []
    first = next(iter(anomalies))
    if isinstance(first, (list, tuple)):
        return [list(g) for g in anomalies]
    return [list(anomalies)]


def _baseline_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    baseline,
    event_date: pd.Timestamp | str | None,
) -> pd.Series:
    """Slice the value column to the requested baseline window for limit-fitting."""
    if baseline is None:
        return df[value_col]
    if baseline == "pre":
        events = _as_event_list(event_date)
        if not events:
            raise ValueError('baseline="pre" requires event_date')
        cut = min(events)
        return df.loc[df[date_col] < cut, value_col]
    start, end = baseline
    mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    return df.loc[mask, value_col]


def plot_series(
    df_ts: pd.DataFrame,
    date_col: str,
    value_col: str,
    event_date: pd.Timestamp | str | list | None = None,
    anomalies: list = (),
    title: str = "",
    moving_average: int | list[int] | None = None,
    control_limits: dict | None = None,
) -> go.Figure:
    """Raw time series with optional event-date markers, anomaly windows, moving-average overlay,
    and SPC-style control limits.

    ``event_date`` is a single timestamp or a list of timestamps; each is drawn
    as a dashed black vertical line. ``moving_average`` is an int window (rows,
    not days) or a list of windows to overlay.

    ``anomalies`` may be either a flat list of windows (single group) or a
    nested list of lists (multiple groups rendered in distinct colors). Each
    window must expose ``.start`` / ``.end`` timestamp attributes.

    ``control_limits`` overlays SPC limits. Supported keys:

    - ``method``: ``"sigma"`` (mean ± coef·σ), ``"imr"`` (Shewhart I-chart,
      mean ± 2.66·MR̄ — default), or ``"rolling_sigma"`` (rolling mean ±
      coef·rolling-σ, drawn as a shaded band that tracks the level).
    - ``coef``: float, default 3.0. Used by ``sigma`` and ``rolling_sigma``.
    - ``window``: int, default 28. Used by ``rolling_sigma``.
    - ``baseline``: ``"pre"`` (fit on rows before the earliest ``event_date``)
      or ``(start, end)`` timestamps to fit limits on a specific window only.
      For ``rolling_sigma`` this is ignored (limits are local by construction).

    For synthetic-control work, ``baseline="pre"`` is recommended — it keeps
    the event itself out of the limit fit so post-event excursions stand out.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_ts[date_col], y=df_ts[value_col], mode="lines+markers", name=value_col)
    )
    if moving_average is not None:
        windows = [moving_average] if isinstance(moving_average, int) else list(moving_average)
        s = df_ts.sort_values(date_col)
        for w in windows:
            fig.add_trace(
                go.Scatter(
                    x=s[date_col],
                    y=s[value_col].rolling(window=w, min_periods=max(1, w // 2)).mean(),
                    mode="lines",
                    name=f"MA({w})",
                    line={"width": 2},
                )
            )
    if control_limits:
        method = control_limits.get("method", "imr")
        coef = float(control_limits.get("coef", 3.0))
        baseline = control_limits.get("baseline")
        if method == "rolling_sigma":
            window = int(control_limits.get("window", 28))
            s = df_ts.sort_values(date_col)
            mu = s[value_col].rolling(window, min_periods=max(2, window // 2)).mean()
            sd = s[value_col].rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
            fig.add_trace(
                go.Scatter(
                    x=s[date_col], y=mu + coef * sd, mode="lines",
                    line={"color": "black", "width": 1, "dash": "dot"},
                    name=f"UCL (rolling, {coef}σ, w={window})",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=s[date_col], y=mu - coef * sd, mode="lines",
                    line={"color": "black", "width": 1, "dash": "dot"},
                    fill="tonexty", fillcolor="rgba(0,0,0,0.06)",
                    name=f"LCL (rolling, {coef}σ, w={window})",
                )
            )
        else:
            base = _baseline_series(df_ts, date_col, value_col, baseline, event_date)
            lcl, mu, ucl = _fit_control_limits(base, method=method, coef=coef)
            label = "I-MR" if method == "imr" else f"{coef}σ"
            base_tag = " (baseline=pre)" if baseline == "pre" else (
                " (baseline=window)" if baseline else ""
            )
            for y, name, color in [
                (ucl, f"UCL {label}{base_tag}", "black"),
                (mu, f"mean {label}{base_tag}", "grey"),
                (lcl, f"LCL {label}{base_tag}", "black"),
            ]:
                fig.add_hline(
                    y=y, line_dash="dash", line_color=color,
                    annotation_text=name, annotation_position="top right",
                )
    if event_date is not None:
        event_colors = ("blue", "red", "black", "green", "purple", "darkorange")
        for i, ev in enumerate(_as_event_list(event_date)):
            fig.add_vline(
                x=ev, line_dash="dash",
                line_color=event_colors[i % len(event_colors)],
            )
    anomaly_colors = ("orange", "purple", "teal", "brown", "magenta", "olive")
    for group_idx, group in enumerate(_as_anomaly_groups(anomalies)):
        color = anomaly_colors[group_idx % len(anomaly_colors)]
        for window in group:
            fig.add_vrect(
                x0=pd.Timestamp(window.start),
                x1=pd.Timestamp(window.end),
                fillcolor=color,
                opacity=0.15,
                line_width=0,
            )
    fig.update_layout(
        title=title or value_col,
        xaxis_title=date_col,
        yaxis_title=value_col,
        xaxis={"autorange": True},
        yaxis={"autorange": True},
    )
    return fig


def plot_seasonality_box(
    df: pd.DataFrame, date_col: str, value_col: str, period: str = "dayofweek"
) -> go.Figure:
    """Boxplot of values grouped by a seasonal bucket (``dayofweek`` / ``month`` / ``weekofyear``)."""
    table = ts_eda.seasonal_table(df, date_col, value_col, period=period)
    fig = px.box(
        table.sort_values("bucket"),
        x="bucket",
        y=value_col,
        points="outliers",
        title=f"{value_col} by {period}",
    )
    return fig


def plot_acf(values, nlags: int = 40, title: str = "Autocorrelation") -> go.Figure:
    """ACF bar chart with Bartlett 95% CI bands."""
    rho = ts_eda.acf(values, nlags=nlags)
    n_valid = int(np.sum(~np.isnan(np.asarray(values, dtype=float))))
    ci = ts_eda.acf_confint(n_valid) if n_valid > 1 else float("nan")
    lags = np.arange(len(rho))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=rho, name="ACF"))
    fig.add_hline(y=ci, line_dash="dot", line_color="grey")
    fig.add_hline(y=-ci, line_dash="dot", line_color="grey")
    fig.update_layout(title=title, xaxis_title="lag", yaxis_title="ACF")
    return fig


def plot_paired_acf(
    values_a,
    values_b,
    nlags: int = 20,
    labels: tuple[str, str] = ("A", "B"),
    colors: tuple[str, str] = ("#1f77b4", "#d62728"),
    title: str = "Paired autocorrelation",
    skip_lag_zero: bool = True,
) -> go.Figure:
    """Grouped-bar ACF for two aligned series with a shared Bartlett CI band.

    Uses :func:`ts_eda.acf` for both series so the bars are directly
    comparable. The Bartlett band is sized from the shorter of the two
    series' valid-sample count.

    Args:
        values_a / values_b: Numeric series (missing values dropped).
        nlags: Maximum lag to plot.
        labels: Legend / hover labels ``(label_a, label_b)``.
        colors: Bar colors ``(color_a, color_b)``.
        title: Chart title.
        skip_lag_zero: Drop the trivial ``lag=0`` bar (``ρ = 1``).

    Returns:
        Plotly ``go.Figure`` ready for ``.show()`` or ``.write_html``.
    """
    rho_a = ts_eda.acf(values_a, nlags=nlags)
    rho_b = ts_eda.acf(values_b, nlags=nlags)
    n_a = int(np.sum(~np.isnan(np.asarray(values_a, dtype=float))))
    n_b = int(np.sum(~np.isnan(np.asarray(values_b, dtype=float))))
    n_ref = min(n_a, n_b)
    ci = ts_eda.acf_confint(n_ref) if n_ref > 1 else float("nan")

    start = 1 if skip_lag_zero else 0
    lags = list(range(start, len(rho_a)))
    y_a = rho_a[start:]
    y_b = rho_b[start:]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=y_a, name=labels[0], marker_color=colors[0], opacity=0.7))
    fig.add_trace(go.Bar(x=lags, y=y_b, name=labels[1], marker_color=colors[1], opacity=0.7))
    fig.add_hline(y=ci, line_dash="dot", line_color="grey",
                  annotation_text=f"+{ci:.2f}", annotation_position="right")
    fig.add_hline(y=-ci, line_dash="dot", line_color="grey",
                  annotation_text=f"-{ci:.2f}", annotation_position="right")
    fig.add_hline(y=0, line_color="black", line_width=0.5)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="lag",
        yaxis_title="autocorrelation",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=80, t=80, b=60),
    )
    return fig


def plot_prewhitening_diagnostic(
    original,
    whitened,
    label: str = "series",
    title: str = "Original vs whitened residual",
    color: str = "#1f77b4",
) -> go.Figure:
    """Overlay the original series and its whitened counterpart.

    Assumes ``whitened`` is one sample shorter than ``original`` (as
    produced by :func:`ts_trend.tfpw_y` and similar). If lengths match,
    both share the same x axis.

    Args:
        original: Raw residual (or level) series.
        whitened: Series returned by a prewhitening step.
        label: Legend prefix (e.g. ``"pre"``).
        title: Chart title.
        color: Base color; the whitened trace uses a lighter dashed variant.

    Returns:
        Plotly ``go.Figure``.
    """
    orig = np.asarray(original, dtype=float)
    whit = np.asarray(whitened, dtype=float)
    x_orig = np.arange(orig.size)
    x_whit = np.arange(orig.size - whit.size, orig.size)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_orig, y=orig.tolist(), mode="lines",
        name=f"{label} (original)",
        line=dict(color=color, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_whit, y=whit.tolist(), mode="lines",
        name=f"{label} (whitened)",
        line=dict(color=color, width=1.5, dash="dot"),
        opacity=0.7,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="index",
        yaxis_title="value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=80, b=60),
    )
    return fig


def plot_per_day_delta_bar(
    per_day_df: pd.DataFrame,
    x_col: str = "label",
    y_col: str = "delta",
    title: str = "Per-day-of-week Δ",
    y_label: str = "Δ slope (per step)",
    x_label: str = "day of week",
) -> go.Figure:
    """Bar chart of per-day Δ slopes with a zero reference line.

    Designed for the DataFrame returned by
    :func:`ts_trend.per_day_delta_slopes`.
    """
    fig = px.bar(
        per_day_df, x=x_col, y=y_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
    )
    fig.update_traces(hovertemplate="%{x}<br>Δ=%{y:.4g}<extra></extra>")
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    return fig


def plot_vbh_per_season(
    decompositions,
    *,
    alpha: float = 0.05,
    experiment: str | None = None,
    positive_color: str = "#1f77b4",
    negative_color: str = "#c62828",
    row_height: int = 340,
) -> go.Figure:
    """Bar chart of per-season Kendall S with VBH χ² decomposition per arm.

    Renders one row per arm (dict key), each panel showing per-season S
    with ``S=±N`` / ``Z=±N.NN`` labels. Panel titles carry ΣS,
    χ²_trend, χ²_het and the homogeneous / inhomogeneous verdict against
    ``alpha``. Mirrors the two-worlds illustration in Van Belle & Hughes
    (1984).

    Parameters
    ----------
    decompositions : mapping[str, VbhDecomposition]
        Ordered mapping of arm name → :class:`ts_trend.VbhDecomposition`.
    alpha : float, default 0.05
        Threshold for the χ²_het homogeneity verdict.
    experiment : str, optional
        Prepended to the figure title.
    positive_color, negative_color : str
        Bar colors for ``S >= 0`` and ``S < 0``.
    row_height : int, default 340
        Pixel height per subplot row.
    """
    from plotly.subplots import make_subplots

    arms = list(decompositions.items())
    if not arms:
        raise ValueError("decompositions must contain at least one arm")

    subplot_titles = []
    for name, d in arms:
        verdict = "inhomogeneous" if d.is_inhomogeneous(alpha) else "homogeneous"
        subplot_titles.append(
            f"<b>{name}</b>  ΣS={d.S_total:+.0f} → "
            f"χ²_trend={d.chi2_trend:.2f} (p={d.p_trend:.3f}, df={d.df_trend}) · "
            f"χ²_het={d.chi2_het:.2f} (p={d.p_het:.3f}, df={d.df_het}) ← {verdict}"
        )

    n_arms = len(arms)
    fig = make_subplots(
        rows=n_arms, cols=1,
        shared_xaxes=True, vertical_spacing=0.22,
        subplot_titles=subplot_titles,
    )
    for row, (_, d) in enumerate(arms, start=1):
        df = d.per_season
        colors = [positive_color if s >= 0 else negative_color for s in df["S"]]
        text = [f"S={s:+.0f}<br>Z={z:+.2f}" for s, z in zip(df["S"], df["Z"])]
        fig.add_trace(
            go.Bar(
                x=df["label"], y=df["S"],
                text=text, textposition="outside",
                marker_color=colors, showlegend=False,
            ),
            row=row, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey", row=row, col=1)
        fig.update_yaxes(title_text="per-season MK statistic S_g", row=row, col=1)

    title = "Van Belle-Hughes χ² decomposition (per season)"
    if experiment:
        title = f"<b>{experiment}</b> — {title}"
    fig.update_layout(
        template="plotly_white",
        height=row_height * n_arms,
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=50, r=20, t=90, b=40),
    )
    return fig


def plot_pacf(
    values,
    nlags: int = 40,
    method: str = "ywm",
    title: str = "Partial autocorrelation",
) -> go.Figure:
    """PACF bar chart with Bartlett 95% CI bands.

    Mirrors :func:`plot_acf`. ``method`` is forwarded to :func:`ts_eda.pacf`.
    A spike at lag ``p`` with cutoff afterwards suggests an AR(``p``) component;
    likewise PACF cuts off at the seasonal lag ``s`` for seasonal AR terms.
    """
    rho = ts_eda.pacf(values, nlags=nlags, method=method)
    n_valid = int(np.sum(~np.isnan(np.asarray(values, dtype=float))))
    ci = ts_eda.acf_confint(n_valid) if n_valid > 1 else float("nan")
    lags = np.arange(len(rho))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=rho, name="PACF"))
    fig.add_hline(y=ci, line_dash="dot", line_color="grey")
    fig.add_hline(y=-ci, line_dash="dot", line_color="grey")
    fig.update_layout(title=title, xaxis_title="lag", yaxis_title="PACF")
    return fig


def plot_stl_decomposition(
    values,
    period: int,
    *,
    robust: bool = True,
    title: str | None = None,
) -> go.Figure:
    """STL decomposition plot: observed / trend / seasonal / residual panels.

    Wraps :func:`ts_seasonality.stl_decompose`. Pass a pandas Series with a
    DatetimeIndex to get calendar-labelled x-axes. Residual is shown as bars
    around zero (should look like white noise for a good period choice).
    """
    from plotly.subplots import make_subplots

    from . import ts_seasonality

    df = ts_seasonality.stl_decompose(values, period=period, robust=robust)
    x = df.index

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("observed", "trend", "seasonal", "residual"),
    )
    fig.add_trace(go.Scatter(x=x, y=df["observed"], mode="lines", name="observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["trend"], mode="lines", name="trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["seasonal"], mode="lines", name="seasonal"), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=df["resid"], name="residual"), row=4, col=1)
    fig.add_hline(y=0, row=4, col=1, line_color="grey", line_width=1)
    fig.update_layout(
        height=700, showlegend=False,
        title=title or f"STL decomposition (period={period})",
        template="plotly_white",
    )
    return fig


def plot_residual_diagnostics(
    residuals,
    *,
    lags: int | None = None,
    title: str = "Residual diagnostics",
) -> go.Figure:
    """Four-panel residual diagnostics: series, histogram, Q-Q, ACF.

    Plotly counterpart to ``statsmodels`` ``ARIMAResults.plot_diagnostics``:

    - row 1: standardized residuals over index | histogram + KDE + N(0, 1),
    - row 2: normal Q-Q | ACF with Bartlett 95% band.

    Ljung-Box is intentionally text-only — see ``ts_eda.ljung_box``.
    """
    from plotly.subplots import make_subplots
    from scipy import stats

    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    if n < 4:
        raise ValueError(f"need at least 4 residuals, got {n}")
    std = r.std(ddof=1) or 1.0
    r_std = (r - r.mean()) / std
    if lags is None:
        lags = min(40, max(5, n // 3))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Standardized residuals",
            "Histogram + KDE + N(0,1)",
            "Normal Q-Q",
            f"ACF (Bartlett 95%, n={n})",
        ),
    )
    fig.add_trace(
        go.Scatter(
            y=r_std, mode="lines+markers", name="resid_std",
            marker={"size": 4, "color": "#4C78A8"},
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, row=1, col=1, line_color="grey", line_width=1)

    fig.add_trace(
        go.Histogram(x=r_std, histnorm="probability density",
                     name="hist", opacity=0.6),
        row=1, col=2,
    )
    try:
        kde = stats.gaussian_kde(r_std)
        grid = np.linspace(float(r_std.min()), float(r_std.max()), 200)
        fig.add_trace(
            go.Scatter(x=grid, y=kde(grid), mode="lines", name="KDE",
                       line={"color": "#F58518"}),
            row=1, col=2,
        )
    except Exception:
        pass
    xg = np.linspace(-4, 4, 200)
    fig.add_trace(
        go.Scatter(x=xg, y=stats.norm.pdf(xg), mode="lines", name="N(0,1)",
                   line={"color": "grey", "dash": "dot"}),
        row=1, col=2,
    )

    osm, osr = stats.probplot(r_std, dist="norm", fit=False)
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode="markers", name="qq",
                   marker={"size": 5, "color": "#4C78A8"}),
        row=2, col=1,
    )
    lo, hi = float(osm.min()), float(osm.max())
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ref",
                   line={"color": "grey", "dash": "dot"}),
        row=2, col=1,
    )

    # Skip lag 0 (always == 1 by definition, carries no information and
    # would visually dominate the plot).
    rho = ts_eda.acf(r, nlags=lags)[1:]
    ci = ts_eda.acf_confint(n)
    ks = np.arange(1, len(rho) + 1)
    fig.add_trace(go.Bar(x=ks, y=rho, name="acf"), row=2, col=2)
    fig.add_hline(y=ci, row=2, col=2, line_dash="dot", line_color="grey")
    fig.add_hline(y=-ci, row=2, col=2, line_dash="dot", line_color="grey")

    fig.update_layout(height=700, showlegend=False, title=title,
                      template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Public-only helpers kept from the pre-vendor-sync dstoolbox surface.
# Not part of the trend_analysis vendor slice; retained so downstream users
# that import `plot_eda_overview` / `plot_ccf` / `lag_plot` don't break.
# ---------------------------------------------------------------------------

def plot_eda_overview(
    df: pd.DataFrame, date_col: str, value_col: str, rolling: int = 14
) -> go.Figure:
    """Raw series with a rolling-mean overlay; quick visual sanity check."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode="lines", name=value_col))
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[value_col].rolling(rolling, min_periods=1).mean(),
            mode="lines",
            name=f"rolling mean ({rolling})",
            line={"width": 2, "color": "orange"},
        )
    )
    fig.update_layout(title="Series + rolling mean", xaxis_title=date_col, yaxis_title=value_col)
    return fig


def plot_ccf(x, y, lags: int = 40, title: str = "Cross-correlation") -> go.Figure:
    """Stem plot of the cross-correlation function with a Bartlett 95% band.

    Lag k is ``corr(x_t, y_{t+k})``; positive k means ``y`` leads ``x``.
    Useful for spotting which exogenous regressors (or which lag of them)
    explain ``value_col``.
    """
    from statsmodels.tsa.stattools import ccf

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rho = ccf(x_arr, y_arr)[: lags + 1]
    ci = 2.0 / np.sqrt(len(y_arr))
    ks = np.arange(rho.size)

    fig = go.Figure()
    for k, r in zip(ks, rho):
        fig.add_trace(
            go.Scatter(x=[k, k], y=[0, r], mode="lines",
                       line={"color": "#6F43D6"}, showlegend=False)
        )
    fig.add_trace(
        go.Scatter(x=ks, y=rho, mode="markers",
                   marker={"color": "#6F43D6", "size": 6}, name="CCF")
    )
    fig.add_hline(y=ci, line_dash="dot", line_color="grey")
    fig.add_hline(y=-ci, line_dash="dot", line_color="grey")
    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(title=title, xaxis_title="lag", yaxis_title="CCF",
                      template="plotly_white")
    return fig


def lag_plot(x, y=None, nlags=24):
    """Plot autocorrelation (or cross-correlation) scatters at multiple lags.

    Renders a 4-column grid of matplotlib subplots; each subplot scatters
    ``x`` vs. itself at lag ``i`` (or ``y`` vs. ``x.shift(i)`` when ``y``
    is given). Matplotlib counterpart to :func:`plot_acf` / :func:`plot_ccf`.

    Parameters
    ----------
    x : pandas.Series
        Primary time series.
    y : pandas.Series or None, optional
        If provided, render cross-correlation panels instead of
        autocorrelation. Default ``None``.
    nlags : int, optional
        Number of lag panels to draw. Default 24.
    """
    with sns.plotting_context("paper"):
        fig, ax = plt.subplots(nrows=math.ceil((nlags) / 4), ncols=4, figsize=[15, 10])

        if y is None:
            fig.suptitle(f"Auto correlation plot {x.name}", fontsize=30)
            for i, ax_ in enumerate(ax.flatten()):
                pd.plotting.lag_plot(x, lag=i + 1, ax=ax_)
                ax_.ticklabel_format(style="sci", scilimits=(0, 0))
                ax_.set_ylabel(f"{x.name}$_t$")
                ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")
        else:
            fig.suptitle(f"Cross correlation plot {x.name} vs {y.name}", fontsize=30)
            for i, ax_ in enumerate(ax.flatten()):
                ax_.scatter(y=y, x=x.shift(periods=i), s=10)
                ax_.set_ylabel(f"{y.name}$_{{t}}$")
                ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")
