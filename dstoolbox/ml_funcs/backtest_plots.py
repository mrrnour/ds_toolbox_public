"""Plotly figures for backtest / time-series cross-validation diagnostics.

``BacktestReport`` is the main entry point: instantiate it with the long
``preds`` frame produced by ``ml_comparison`` (and optionally the per-fold
``metrics`` leaderboard), then call its render methods.

``plot_backtest_splits`` is a separate free function — it visualizes split
geometry from a splitter + a frame with a datetime column, before any model
has been fit, so it doesn't share state with ``BacktestReport``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PI_FILL = "rgba(111,67,214,0.20)"
_PI_LINE = "rgba(111,67,214,0.0)"

_REQUIRED_COLS: frozenset[str] = frozenset({
    "model", "ts", "fold", "split", "y_true", "y_pred",
})


def _ribbon_trace(x, lo, hi, name: str) -> go.Scatter:
    return go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself",
        fillcolor=_PI_FILL,
        line={"width": 0, "color": _PI_LINE},
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _model_grid(models: list[str], cols: int = 2) -> tuple[int, int]:
    n = len(models)
    rows = (n + cols - 1) // cols
    return rows, cols


@dataclass
class BacktestReport:
    """Plotly facade over a long-format backtest predictions frame.

    Parameters
    ----------
    preds
        Long DataFrame with columns ``model, ts, fold, split, y_true, y_pred``
        and optional ``y_lo`` / ``y_hi``. ``split`` has values ``"train"`` /
        ``"val"``; ``fold`` is a per-model integer fold index.
    metrics
        Optional per-origin leaderboard: one row per model × fold with one
        column per metric (``model`` column required). Needed only by
        :meth:`plot_metric_box` and :meth:`plot_metric_by_fold`.
    model_params
        Optional ``{model_name: {hyperparam: value}}``; appended to subplot
        titles when set.
    series
        Optional ``ts`` + value-column DataFrame; drawn as a thin grey
        backdrop behind faceted backtest plots so the full training span
        is visible.
    cols
        Default subplot grid width.
    """

    preds: pd.DataFrame
    metrics: pd.DataFrame | None = None
    model_params: dict[str, dict] | None = None
    series: pd.DataFrame | None = None
    cols: int = 2
    _palette: tuple[str, ...] = field(
        default_factory=lambda: tuple(px.colors.qualitative.Plotly), repr=False,
    )

    def __post_init__(self) -> None:
        missing = _REQUIRED_COLS - set(self.preds.columns)
        if missing:
            raise ValueError(f"preds missing columns: {sorted(missing)}")
        # normalize once
        self.preds = self.preds.copy()
        self.preds["ts"] = pd.to_datetime(self.preds["ts"])
        if "y_lo" not in self.preds.columns:
            self.preds["y_lo"] = np.nan
        if "y_hi" not in self.preds.columns:
            self.preds["y_hi"] = np.nan

    @cached_property
    def models(self) -> list[str]:
        return sorted(self.preds["model"].unique().tolist())

    @cached_property
    def color_map(self) -> dict[str, str]:
        return {m: self._palette[i % len(self._palette)] for i, m in enumerate(self.models)}

    @cached_property
    def _fold_info(self) -> dict[tuple[str, int], dict]:
        """Per-(model, fold) train/val sample counts and date ranges."""
        gb = self.preds.groupby(["model", "fold", "split"])
        sizes = gb.size().unstack("split", fill_value=0)
        ts_min = gb["ts"].min().unstack("split")
        ts_max = gb["ts"].max().unstack("split")
        info: dict[tuple[str, int], dict] = {}
        for (m, fold), row in sizes.iterrows():
            d = {"train_n": int(row.get("train", 0)), "val_n": int(row.get("val", 0))}
            tr_s, tr_e = ts_min.loc[(m, fold)].get("train"), ts_max.loc[(m, fold)].get("train")
            va_s, va_e = ts_min.loc[(m, fold)].get("val"), ts_max.loc[(m, fold)].get("val")
            d["train_range"] = (
                f"{pd.Timestamp(tr_s).date()} → {pd.Timestamp(tr_e).date()}"
                if pd.notna(tr_s) else "—"
            )
            d["val_range"] = (
                f"{pd.Timestamp(va_s).date()} → {pd.Timestamp(va_e).date()}"
                if pd.notna(va_s) else "—"
            )
            info[(m, int(fold))] = d
        return info

    def _title(self, name: str) -> str:
        if not self.model_params or not self.model_params.get(name):
            return name
        params = ", ".join(f"{k}={v}" for k, v in self.model_params[name].items())
        return f"{name}<br><sub>{params}</sub>"

    def _series_xy(self) -> tuple[pd.Series, pd.Series] | None:
        if self.series is None or not len(self.series):
            return None
        s = self.series.copy()
        if "ts" not in s.columns:
            raise ValueError("`series` must have a 'ts' column")
        val_col = next((c for c in s.columns if c != "ts"), None)
        if val_col is None:
            raise ValueError("`series` must have a value column besides 'ts'")
        s = s.sort_values("ts")
        return s["ts"], s[val_col]

    # ------------------------------------------------------------------ plots

    def plot_origin(self, origin: pd.Timestamp, model: str) -> go.Figure:
        """Forecast vs actual + PI ribbon for a single backtest origin."""
        sub = self.preds[
            (self.preds.get("origin") == origin) & (self.preds["model"] == model)
        ].sort_values("ts") if "origin" in self.preds.columns else self.preds[
            (self.preds["fold"] == origin) & (self.preds["model"] == model)
        ].sort_values("ts")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["ts"], y=sub["y_true"], name="actual", mode="markers"))
        fig.add_trace(go.Scatter(x=sub["ts"], y=sub["y_pred"], name="forecast", mode="lines"))
        fig.add_trace(
            go.Scatter(
                x=list(sub["ts"]) + list(sub["ts"][::-1]),
                y=list(sub["y_hi"]) + list(sub["y_lo"][::-1]),
                fill="toself",
                fillcolor=_PI_FILL,
                line={"width": 0},
                name="PI",
                showlegend=True,
            )
        )
        fig.update_layout(title=f"{model} — backtest origin {pd.Timestamp(origin).date() if hasattr(origin, 'date') or isinstance(origin, pd.Timestamp) else origin}")
        return fig

    def plot_faceted(
        self,
        fold: int | None = None,
        show: str = "all",
    ) -> go.Figure:
        """Faceted forecast-vs-actual; one subplot per model.

        Default: overlay every backtest fold per model — one forecast line
        per fold (colored, labeled by its first val timestamp), actuals
        drawn once as black markers, and a PI ribbon for the latest fold
        only (drawing 10 ribbons is unreadable).

        Parameters
        ----------
        fold
            Drill into a single fold (forecast line + PI ribbon for just
            that fold). Default: overlay all folds.
        show
            What to render. One of:

            - ``"val"`` — only val-window forecasts, actuals, and PI ribbon.
            - ``"training"`` — only the grey full-series backdrop passed via
              ``self.series`` (no val-window content).
            - ``"all"`` *(default)* — both: val-window content plus the
              grey backdrop.
        """
        valid_show = {"val", "training", "all"}
        if show not in valid_show:
            raise ValueError(f"show must be one of {sorted(valid_show)}, got {show!r}")

        draw_val = show in {"val", "all"}
        draw_history = show in {"training", "all"}

        preds = self.preds[self.preds["split"] == "val"]
        if fold is not None:
            preds = preds[preds["fold"] == fold]
        if draw_val and preds.empty:
            raise ValueError("no val-split rows in preds for the requested fold")

        rows, cols = _model_grid(self.models, cols=self.cols)
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[self._title(m) for m in self.models],
            shared_xaxes=True,
        )
        series_xy = self._series_xy() if draw_history else None

        for i, name in enumerate(self.models):
            r, c = i // cols + 1, i % cols + 1
            m_preds = (
                preds[preds["model"] == name].sort_values("ts")
                if draw_val
                else preds.iloc[0:0]
            )
            folds = sorted(m_preds["fold"].unique().tolist())
            latest = folds[-1] if folds else None

            if series_xy is not None:
                fig.add_trace(
                    go.Scatter(
                        x=series_xy[0], y=series_xy[1],
                        mode="lines", name="training",
                        legendgroup="training", showlegend=False,
                        line={"color": "#9aa0a6", "width": 1},
                        hoverinfo="skip",
                    ),
                    row=r, col=c,
                )

            for fk in folds:
                f = m_preds[m_preds["fold"] == fk].drop_duplicates(subset=["ts"]).sort_values("ts")
                fig.add_trace(
                    go.Scatter(
                        x=f["ts"], y=f["y_true"],
                        mode="lines+markers", name="actual",
                        legendgroup="actual", showlegend=False,
                        marker={"color": "#2A2A2A", "size": 4},
                        line={"color": "#2A2A2A", "width": 1},
                    ),
                    row=r, col=c,
                )

            for j, fk in enumerate(folds):
                sub = m_preds[m_preds["fold"] == fk].sort_values("ts")
                color = self._palette[j % len(self._palette)]
                label = str(pd.Timestamp(sub["ts"].iloc[0]).date())
                if fk == latest:
                    fig.add_trace(
                        _ribbon_trace(sub["ts"], sub["y_lo"], sub["y_hi"], f"{name} PI"),
                        row=r, col=c,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=sub["ts"], y=sub["y_pred"],
                        mode="lines", name=label,
                        legendgroup=label, showlegend=False,
                        line={"color": color, "width": 1.5},
                        opacity=1.0 if fk == latest else 0.55,
                    ),
                    row=r, col=c,
                )

        title = (
            f"Backtest fold {fold} — by model"
            if fold is not None
            else "Backtest — all folds, by model (latest fold has PI)"
        )
        fig.update_layout(title=title, height=300 * rows, showlegend=False)
        return fig

    def plot_folds_per_model(self, train_lookback: int | None = None, cols: int | None = None) -> dict[str, go.Figure]:
        """One figure per model; each is a grid of n_folds subplots.

        For each fold panel:
          - **blue line+markers** = training actuals (``split == 'train'``)
          - **dashed red line** = in-sample forecast (train rows' ``y_pred``)
          - **black line+markers** = val-window actuals (``split == 'val'``)
          - **red line + PI** = val-window forecast
          - **dashed vertical** = origin (train→val boundary)
        """
        n_folds = self.preds["fold"].nunique()
        if n_folds == 0:
            raise ValueError("preds has no folds")
        cols = cols if cols is not None else self.cols
        rows = (n_folds + cols - 1) // cols

        train_color = "#1f77b4"
        actual_color = "#2A2A2A"
        forecast_color = "#d62728"

        figures: dict[str, go.Figure] = {}
        for name in self.models:
            m_preds = self.preds[self.preds["model"] == name]
            m_train = m_preds[m_preds["split"] == "train"].sort_values(["fold", "ts"])
            m_val = m_preds[m_preds["split"] == "val"].sort_values(["fold", "ts"])
            folds_for_model = sorted(m_val["fold"].unique().tolist())

            title = name
            if self.model_params and self.model_params.get(name):
                params = ", ".join(f"{k}={v}" for k, v in self.model_params[name].items())
                title = f"{name} — {params}"

            subplot_titles: list[str] = []
            xaxis_titles: list[str] = []
            for i, f in enumerate(folds_for_model):
                tr = m_train[m_train["fold"] == f]
                va = m_val[m_val["fold"] == f]
                n_train = len(tr)
                full_start = tr["ts"].min().date() if n_train else "—"
                full_end = tr["ts"].max().date() if n_train else "—"
                pr_start = va["ts"].min().date()
                pr_end = va["ts"].max().date()
                shown = (
                    f" (showing last {min(train_lookback, n_train)} of {n_train})"
                    if train_lookback and n_train and train_lookback < n_train
                    else ""
                )
                subplot_titles.append(f"Fold {i + 1}")
                xaxis_titles.append(
                    f"train {full_start} → {full_end}{shown}<br>forecast {pr_start} → {pr_end}"
                )

            total_height = 420 * rows + 120
            vspace = min(0.18, 80 / total_height) if rows > 1 else 0.0
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                shared_yaxes=False,
                vertical_spacing=vspace,
                horizontal_spacing=0.08,
            )

            for i, f in enumerate(folds_for_model):
                r, c = i // cols + 1, i % cols + 1
                show_legend = (i == 0)

                tr = m_train[m_train["fold"] == f]
                if train_lookback and len(tr) > train_lookback:
                    tr = tr.tail(train_lookback)
                va = m_val[m_val["fold"] == f]
                origin = va["ts"].min()

                # Bridge across the fold boundary: extend the training-side
                # traces forward by one point (first val row) so the blue
                # training-actuals line and the dashed in-sample forecast
                # line visually reach the val start without duplicating
                # markers on the val side.
                va_first = va.head(1)
                tr_actual = pd.concat([tr[["ts", "y_true"]], va_first[["ts", "y_true"]]]) if not va_first.empty else tr[["ts", "y_true"]]

                if not tr.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=tr_actual["ts"], y=tr_actual["y_true"],
                            mode="lines+markers", name="training actuals",
                            legendgroup="training",
                            marker={"color": train_color, "size": 5},
                            line={"color": train_color, "width": 1.5},
                            showlegend=show_legend,
                        ),
                        row=r, col=c,
                    )
                    tr_pred = tr.dropna(subset=["y_pred"])
                    if not tr_pred.empty:
                        bridge_pred = pd.concat([tr_pred[["ts", "y_pred"]], va_first[["ts", "y_pred"]]])
                        fig.add_trace(
                            go.Scatter(
                                x=bridge_pred["ts"], y=bridge_pred["y_pred"],
                                mode="lines+markers", name="in-sample forecast",
                                legendgroup="insample",
                                marker={"color": forecast_color, "size": 4, "symbol": "circle-open"},
                                line={"color": forecast_color, "width": 1.2, "dash": "dot"},
                                opacity=0.7,
                                showlegend=show_legend,
                            ),
                            row=r, col=c,
                        )

                fig.add_trace(
                    go.Scatter(
                        x=va["ts"], y=va["y_true"],
                        mode="lines+markers", name="forecast-period actuals",
                        legendgroup="actual",
                        marker={"color": actual_color, "size": 6},
                        line={"color": actual_color, "width": 1.5},
                        showlegend=show_legend,
                    ),
                    row=r, col=c,
                )
                fig.add_trace(
                    _ribbon_trace(va["ts"], va["y_lo"], va["y_hi"], f"{name} PI"),
                    row=r, col=c,
                )
                fig.add_trace(
                    go.Scatter(
                        x=va["ts"], y=va["y_pred"],
                        mode="lines+markers", name="forecast",
                        legendgroup="forecast",
                        marker={"color": forecast_color, "size": 5},
                        line={"color": forecast_color, "width": 2},
                        showlegend=show_legend,
                    ),
                    row=r, col=c,
                )
                fig.add_vline(
                    x=pd.Timestamp(origin), line_dash="dash", line_color="#888",
                    row=r, col=c,
                )
                fig.update_xaxes(
                    title_text=xaxis_titles[i],
                    title_font={"size": 9, "color": "#555"},
                    row=r, col=c,
                )

            fig.update_layout(
                title={"text": title, "y": 0.995, "yanchor": "top"},
                height=total_height,
                margin={"t": 140, "b": 80, "l": 60, "r": 40},
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.0 + 60 / total_height,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
            fig.update_annotations(font_size=11)
            figures[name] = fig

        return figures

    def plot_residuals_by_fold(
        self,
        model: str,
        split: str = "val",
        show_ljung_box: bool = True,
        height_per_fold: int = 220,
    ) -> go.Figure:
        """Per-fold residual line plot for one model.

        Stacked subplot (one row per fold) of ``y_true - y_pred`` vs ``ts``
        with a zero reference line. When ``show_ljung_box=True`` and
        ``dsToolbox.ml_funcs.ts_eda`` is importable, each subplot title
        includes the per-fold Ljung-Box Q and p-value so window-specific
        autocorrelation is visible alongside the aggregate diagnostic.

        Parameters
        ----------
        model
            Model name to plot; must exist in ``preds["model"]``.
        split
            Which split to draw residuals for. Defaults to ``"val"``
            (held-out generalization residuals).
        show_ljung_box
            Whether to compute and annotate per-fold Ljung-Box in titles.
        height_per_fold
            Row height in pixels; total figure height scales with fold count.
        """
        if model not in set(self.preds["model"]):
            raise ValueError(f"model {model!r} not in preds; have {self.models}")

        sub = self.preds[
            (self.preds["model"] == model) & (self.preds["split"] == split)
        ].copy()
        if sub.empty:
            raise ValueError(f"no rows for model={model!r} split={split!r}")
        sub["resid"] = sub["y_true"] - sub["y_pred"]

        lb = None
        if show_ljung_box:
            try:
                from dsToolbox.ml_funcs.ts_eda import ljung_box as lb
            except Exception:
                lb = None

        folds = sorted(sub["fold"].dropna().unique())
        titles: list[str] = []
        for k in folds:
            r = sub.loc[sub["fold"] == k, "resid"].dropna().to_numpy()
            if lb is not None and r.size >= 4:
                q_k, p_k = lb(r, lags=min(10, max(2, r.size // 5)))
                titles.append(f"fold {int(k)} — n={r.size}, LB Q={q_k:.2f}, p={p_k:.3f}")
            else:
                titles.append(f"fold {int(k)} — n={r.size}")

        fig = make_subplots(
            rows=len(folds), cols=1, shared_xaxes=False, subplot_titles=titles,
        )
        color = self.color_map.get(model, "#1f77b4")
        for i, k in enumerate(folds, start=1):
            s = sub.loc[sub["fold"] == k].sort_values("ts")
            fig.add_trace(
                go.Scatter(
                    x=s["ts"], y=s["resid"],
                    mode="lines+markers",
                    line={"color": color},
                    marker={"color": color, "size": 5},
                    name=f"fold {int(k)}",
                    showlegend=False,
                ),
                row=i, col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=i, col=1)

        fig.update_layout(
            title=f"{model} — per-fold {split} residuals",
            height=height_per_fold * max(1, len(folds)),
        )
        return fig

    # ------------------------------ rolling calendar-window residuals

    def rolling_window_residuals(
        self,
        window: str = "14D",
        weights: pd.Series | None = None,
        model: str | list[str] | None = None,
        split: str = "val",
    ) -> pd.DataFrame:
        """Aggregate predictions into non-overlapping calendar windows.

        Pools the ``split`` predictions of one or more models across all
        folds (deduping on ``ts`` per model) and slices the resulting
        timeline into fixed calendar periods — e.g. ``"7D"``, ``"14D"``,
        ``"1M"``. Window count is set by the pre-intervention history length
        divided by ``window``.

        Returns one row per (model, window) with columns
        ``model, window_start, window_end, n_days, obs, fcst, residual,
        rel_residual``.

        Parameters
        ----------
        window
            pandas offset string; boundaries are calendar-anchored.
        weights
            Optional ``pd.Series`` indexed by ``ts``. When provided, the
            per-window ``obs`` and ``fcst`` are ``sum(y * w) / sum(w)`` —
            i.e. the A/B-style ratio; otherwise a uniform average.
        model
            One model name, a list of model names, or ``None`` (all models
            in :attr:`models`).
        split
            Which split's predictions to aggregate. Defaults to ``"val"``.
        """
        sub = self.preds[self.preds["split"] == split]
        if sub.empty:
            raise ValueError(f"no rows for split={split!r}")

        if model is None:
            targets = list(self.models)
        elif isinstance(model, str):
            targets = [model]
        else:
            targets = list(model)
        available = set(sub["model"])
        missing = [m for m in targets if m not in available]
        if missing:
            raise ValueError(f"model(s) not in preds for split={split!r}: {missing}")

        if weights is not None:
            w_idx = pd.to_datetime(weights.index)
            w_series = pd.Series(weights.to_numpy(), index=w_idx)
        else:
            w_series = None

        rows: list[dict] = []
        for m in targets:
            m_sub = (
                sub[sub["model"] == m]
                .sort_values("ts")
                .drop_duplicates(subset="ts", keep="last")
                .copy()
            )
            m_sub["ts"] = pd.to_datetime(m_sub["ts"])
            for start, g in m_sub.groupby(pd.Grouper(key="ts", freq=window), sort=True):
                if g.empty:
                    continue
                if w_series is not None:
                    w = w_series.reindex(g["ts"]).to_numpy()
                    w = np.where(np.isnan(w), 0.0, w)
                    if w.sum() == 0:
                        w = np.ones(len(g))
                else:
                    w = np.ones(len(g))
                obs = float(np.average(g["y_true"], weights=w))
                fcst = float(np.average(g["y_pred"], weights=w))
                rows.append({
                    "model": m,
                    "window_start": pd.Timestamp(start),
                    "window_end": g["ts"].max(),
                    "n_days": len(g),
                    "obs": obs,
                    "fcst": fcst,
                    "residual": obs - fcst,
                    "rel_residual": (obs - fcst) / obs if obs != 0 else np.nan,
                })
        if not rows:
            raise ValueError(f"grouper freq={window!r} produced no windows")
        return pd.DataFrame(rows)

    def plot_rolling_window_residuals(
        self,
        window: str = "14D",
        weights: pd.Series | None = None,
        model: str | None = None,
        split: str = "val",
        weight_label: str | None = None,
        rolling: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Per-window residual (``obs − fcst``) bars for one model."""
        if model is None:
            model = self.models[0]
        df = (
            rolling[rolling["model"] == model].copy()
            if rolling is not None
            else self.rolling_window_residuals(window=window, weights=weights, model=model, split=split)
        ).sort_values("window_start")
        if df.empty:
            raise ValueError(f"no rolling windows for model={model!r}")
        df["label"] = df["window_start"].dt.strftime("%Y-%m-%d")

        _label = weight_label or ("weighted" if weights is not None else "uniform")
        color = self.color_map.get(model, "#1f77b4")

        hover = [
            f"{ws} → {we}<br>n_days={n}<br>obs={ob:.4g}<br>fcst={fc:.4g}<br>"
            f"residual={r:.4g}<br>rel={rr:.3%}"
            for ws, we, n, ob, fc, r, rr in zip(
                df["label"], df["window_end"].dt.strftime("%Y-%m-%d"),
                df["n_days"], df["obs"], df["fcst"], df["residual"], df["rel_residual"],
            )
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df["label"], y=df["residual"], name="residual",
                marker={"color": color}, hovertext=hover, hoverinfo="text",
                showlegend=False,
            ),
        )
        fig.add_hline(y=0, line_dash="dot")
        fig.update_yaxes(title_text="obs − fcst")
        fig.update_xaxes(title_text="window start")
        fig.update_layout(
            title=f"{model} — residual (obs − fcst) per {window} window ({_label})",
            height=380, bargap=0.2,
        )
        return fig

    def plot_rolling_window_residuals_box(
        self,
        window: str = "14D",
        weights: pd.Series | None = None,
        split: str = "val",
        weight_label: str | None = None,
        rolling: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Box plot of per-window absolute residuals (``obs − fcst``) across all models."""
        wr = (
            rolling
            if rolling is not None
            else self.rolling_window_residuals(window=window, weights=weights, model=None, split=split)
        )
        if "model" not in wr.columns:
            raise ValueError("rolling frame must contain a 'model' column (pass model=None)")
        _label = weight_label or ("weighted" if weights is not None else "uniform")

        fig = px.box(
            wr, x="model", y="residual", points="all",
            hover_data=["window_start", "window_end", "n_days", "obs", "fcst", "rel_residual"],
            title=f"Absolute residual (obs − fcst) per {window} window — {_label}",
        )
        fig.add_hline(y=0, line_dash="dot")
        fig.update_yaxes(title_text="obs − fcst")
        fig.update_xaxes(title_text="model")
        fig.update_layout(height=480, boxmode="group", showlegend=False)
        return fig

    def plot_metric_box(self, metric: str) -> go.Figure:
        """Boxplot of a per-origin metric across models.

        Requires ``metrics`` to be set at construction. Shows the mean as a
        dashed line inside each box and annotates its value above the box.
        """
        if self.metrics is None:
            raise ValueError("plot_metric_box requires metrics= at construction")
        fig = px.box(
            self.metrics, x="model", y=metric, points="all",
            title=f"Per-origin {metric} — all models",
        )
        fig.update_traces(boxmean=True)

        means = (
            self.metrics.groupby("model", sort=False)[metric]
            .mean()
            .dropna()
        )
        y_max = self.metrics[metric].max()
        y_min = self.metrics[metric].min()
        offset = (y_max - y_min) * 0.05 if y_max != y_min else abs(y_max) * 0.05 or 1.0
        for model_name, mean_val in means.items():
            fig.add_annotation(
                x=model_name, y=mean_val + offset,
                text=f"μ={mean_val:.3g}",
                showarrow=False,
                font={"size": 11, "color": "#444"},
                yanchor="bottom",
            )
        return fig

    def plot_metric_by_fold(self, metrics: str | list[str]) -> go.Figure:
        """Per-fold metric(s) line chart across models (x = fold index).

        First subplot is a bar chart of val sample count per fold; the
        rest are per-metric lines across models. Requires ``metrics=`` at
        construction.
        """
        if self.metrics is None:
            raise ValueError("plot_metric_by_fold requires metrics= at construction")
        metric_list = [metrics] if isinstance(metrics, str) else list(metrics)
        df = self.metrics.copy()
        df["fold"] = df.groupby("model").cumcount()

        # one (fold → train_n) mapping; take the max across models so a model
        # that failed predict(train) for a fold doesn't leave the bar empty
        n_by_fold: dict[int, int] = {}
        for fold in sorted({f for _, f in self._fold_info}):
            vals = [
                self._fold_info[(m, fold)]["train_n"]
                for m in self.models
                if (m, fold) in self._fold_info
            ]
            if vals:
                n_by_fold[fold] = max(vals)

        n_rows = len(metric_list) + 1
        titles = ["n training samples per fold", *metric_list]
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=titles,
        )

        if n_by_fold:
            folds_sorted = sorted(n_by_fold)
            y_vals = [n_by_fold[f] for f in folds_sorted]
            fig.add_trace(
                go.Bar(
                    x=folds_sorted,
                    y=y_vals,
                    marker_color="#1f77b4",
                    name="train n",
                    showlegend=False,
                    text=y_vals,
                    textposition="outside",
                ),
                row=1, col=1,
            )
            fig.update_yaxes(
                title_text="n train",
                range=[0, max(y_vals) * 1.18],
                row=1, col=1,
            )
            fig.update_xaxes(
                tickmode="linear", tick0=0, dtick=1,
                range=[folds_sorted[0] - 0.5, folds_sorted[-1] + 0.5],
                row=1, col=1,
            )

        for r, metric in enumerate(metric_list, start=2):
            for m in self.models:
                sub = df[df["model"] == m][["fold", metric]].dropna().copy()
                if sub.empty:
                    continue
                sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
                hover = []
                for fold, val in zip(sub["fold"], sub[metric]):
                    info = self._fold_info.get((m, int(fold)), {})
                    lines = [
                        f"<b>{m}</b> — fold {fold}",
                        f"{metric}: {val:.4g}",
                    ]
                    if info:
                        lines.append(f"train: n={info['train_n']}  {info.get('train_range', '')}")
                        lines.append(f"val:   n={info['val_n']}  {info.get('val_range', '')}")
                    hover.append("<br>".join(lines))
                fig.add_trace(
                    go.Scatter(
                        x=sub["fold"], y=sub[metric],
                        mode="lines+markers", name=m,
                        legendgroup=m, showlegend=(r == 2),
                        line={"color": self.color_map[m]},
                        hovertext=hover, hoverinfo="text",
                    ),
                    row=r, col=1,
                )
            fig.update_yaxes(title_text=metric, row=r, col=1)

        fig.update_xaxes(title_text="fold", row=n_rows, col=1)

        fig.update_layout(
            title="Per-fold metrics — does more training data help?",
            height=220 * n_rows + 120,
            legend_title_text="model",
            hovermode="x unified",
        )
        return fig


def plot_backtest_splits(
    splitter,
    X: pd.DataFrame,
    *,
    ts_col: str = "ts",
    intervention_date: pd.Timestamp | str | None = None,
    title: str = "Backtest folds — train (blue) → forecast (orange)",
    show_arrow: bool = False,
    annotate_fold: int | None = -1,
) -> go.Figure:
    """Timeline of every backtest fold: train window + forecast window per origin.

    One horizontal row per fold (top = fold 1). Train window in blue, gap
    (if any) as a dotted grey segment, forecast horizon in orange. ``X`` is
    the frame passed to ``splitter.split`` — must carry a datetime column
    named ``ts_col`` so positional fold indices can be mapped back to dates.

    ``show_arrow`` keeps the arrowhead at the end of each forecast segment
    (off by default — implied a directional flow that isn't there).
    ``annotate_fold`` picks one fold (negative index allowed) on which to
    overlay labelled brackets for **gap**, **horizon**, and **step**
    (step is measured against the previous fold's forecast start). Set to
    ``None`` to disable the overlay.
    """
    if ts_col not in X.columns:
        raise ValueError(f"X is missing required '{ts_col}' column")
    ts = pd.to_datetime(X[ts_col]).reset_index(drop=True)
    folds = list(splitter.split(X))
    if not folds:
        raise ValueError("splitter produced 0 folds for this X")

    fig = go.Figure()
    showed_legend = {"train": False, "gap": False, "forecast": False}
    n = len(folds)
    n_train_per_fold: list[int] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        y = n - i  # fold 1 on top
        t_start, t_end = ts.iloc[train_idx[0]], ts.iloc[train_idx[-1]]
        v_start, v_end = ts.iloc[val_idx[0]], ts.iloc[val_idx[-1]]
        n_train = len(train_idx)
        n_val = len(val_idx)
        n_train_per_fold.append(n_train)
        fig.add_trace(
            go.Scatter(
                x=[t_start, t_end], y=[y, y],
                mode="lines",
                line={"color": "#1f77b4", "width": 8},
                name="train",
                legendgroup="train",
                showlegend=not showed_legend["train"],
                hovertemplate=(
                    f"fold {i + 1} train<br>{t_start.date()} → {t_end.date()}"
                    f"<br>n_train = {n_train}<extra></extra>"
                ),
            )
        )
        # Mid-segment label so the train size is visible without hovering.
        fig.add_annotation(
            x=t_start + (t_end - t_start) / 2,
            y=y,
            text=f"n={n_train}",
            showarrow=False,
            font={"color": "white", "size": 10},
            yshift=0,
        )
        showed_legend["train"] = True
        if t_end < v_start - pd.Timedelta(days=1):
            fig.add_trace(
                go.Scatter(
                    x=[t_end, v_start], y=[y, y],
                    mode="lines",
                    line={"color": "grey", "width": 2, "dash": "dot"},
                    name="gap",
                    legendgroup="gap",
                    showlegend=not showed_legend["gap"],
                    hovertemplate=f"fold {i + 1} gap<br>{t_end.date()} → {v_start.date()}<extra></extra>",
                )
            )
            showed_legend["gap"] = True
        forecast_trace = {
            "x": [v_start, v_end],
            "y": [y, y],
            "mode": "lines+markers" if show_arrow else "lines",
            "line": {"color": "#ff7f0e", "width": 8},
            "name": "forecast",
            "legendgroup": "forecast",
            "showlegend": not showed_legend["forecast"],
            "hovertemplate": f"fold {i + 1} forecast<br>{v_start.date()} → {v_end.date()}<extra></extra>",
        }
        if show_arrow:
            forecast_trace["marker"] = {"symbol": "arrow-right", "size": 12, "color": "#ff7f0e"}
        fig.add_trace(go.Scatter(**forecast_trace))
        showed_legend["forecast"] = True

    if intervention_date is not None:
        fig.add_vline(x=pd.Timestamp(intervention_date), line_dash="dash", line_color="red")

    if annotate_fold is not None and n > 0:
        k = annotate_fold if annotate_fold >= 0 else n + annotate_fold
        if 0 <= k < n:
            _annotate_split_geometry(fig, folds, ts, k, n)

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n, 0, -1)),
        ticktext=[f"origin {i + 1}" for i in range(n)],
        title="",
    )
    fig.update_layout(
        title=title,
        xaxis_title=ts_col,
        height=max(220, 40 * n + 160),
        showlegend=True,
    )
    return fig


def _annotate_split_geometry(fig, folds, ts, k: int, n: int) -> None:
    """Overlay labelled brackets for gap, horizon, and step on fold ``k``."""
    train_idx, val_idx = folds[k]
    y = n - k
    t_end = ts.iloc[train_idx[-1]]
    v_start = ts.iloc[val_idx[0]]
    v_end = ts.iloc[val_idx[-1]]
    horizon = len(val_idx)
    if k > 0:
        prev_v_start = ts.iloc[folds[k - 1][1][0]]
        step = folds[k][1][0] - folds[k - 1][1][0]
    else:
        prev_v_start = None
        step = None
    gap = val_idx[0] - train_idx[-1] - 1

    def bracket(x0, x1, y_off: float, label: str, color: str) -> None:
        y_line = y + y_off
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y_line, y1=y_line,
            line={"color": color, "width": 2},
        )
        cap = 0.08
        for x in (x0, x1):
            fig.add_shape(
                type="line", x0=x, x1=x, y0=y_line - cap, y1=y_line + cap,
                line={"color": color, "width": 2},
            )
        fig.add_annotation(
            x=x0 + (x1 - x0) / 2, y=y_line,
            text=label, showarrow=False,
            font={"color": color, "size": 11},
            yshift=14 if y_off > 0 else -14,
            bgcolor="rgba(255,255,255,0.85)",
        )

    bracket(v_start, v_end, +0.35, f"horizon = {horizon}", "#ff7f0e")
    if gap > 0:
        bracket(t_end, v_start, +0.35, f"gap = {gap}", "grey")
    if step is not None:
        bracket(prev_v_start, v_start, -0.45, f"step = {step}", "#555")
