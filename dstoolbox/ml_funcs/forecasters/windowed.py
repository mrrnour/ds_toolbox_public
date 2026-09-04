"""Window-aggregating meta-forecaster.

Wraps any base forecaster (ARIMA, Silverkite, Darts, naive, ...) and turns
its daily point predictions into non-overlapping calendar-window
weighted-average predictions — matching the estimand an A/B readout reports
(``sum(y * w) / sum(w)`` over a 1-2 week window).

The wrapper is sklearn-compatible: :meth:`predict` returns one value per row
of ``X`` (each daily row gets its own window's weighted-average forecast, so
predictions are piecewise-constant across each window). This lets the same
class flow through :func:`ml_comparison`, :class:`BacktestReport` plotting,
and :func:`estimate_intervention_effect` alongside the daily models — no
custom plotting code needed at the notebook level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..mixins import IntervalMixin
from ._base import as_datetime_index


class WindowedForecaster(BaseEstimator, RegressorMixin, IntervalMixin):
    """Aggregate a base forecaster's daily predictions into fixed calendar windows.

    Parameters
    ----------
    base_model
        Any sklearn-compatible forecaster with ``fit(X, y) / predict(X)``.
        Instantiated by the caller (e.g. ``AutoArimaSklearn(...)``).
    window
        Pandas frequency string for the non-overlapping calendar buckets
        (``"14D"``, ``"7D"``, ``"1W"``, ...). Buckets are anchored on the
        pandas epoch grid so train / val calls share the same grid.
    weight_col
        Column in ``X`` holding per-row weights (e.g. ``"n_searches"``).
        If missing, falls back to ``weight_series`` then to uniform weights.
    weight_series
        Optional pre-built ``pd.Series`` indexed by timestamp — used when
        ``weight_col`` is ``None`` or not in ``X``. Convenient when the
        weights live in the raw frame but weren't passed as exog to
        :func:`to_Xy`.
    date_col
        Column in ``X`` holding the per-row timestamp.

    Notes
    -----
    Metrics computed by :func:`ml_comparison` will compare **daily** ``y_true``
    against a **piecewise-constant** window prediction, so within-window
    variance shows up as an irreducible noise floor. That's the intended
    behaviour: it puts the daily and windowed models on a comparable scale
    and mirrors what an A/B readout actually measures.
    """

    def __init__(
        self,
        base_model: Any,
        window: str = "14D",
        weight_col: str | None = None,
        weight_series: pd.Series | None = None,
        date_col: str = "ts",
    ) -> None:
        self.base_model = base_model
        self.window = window
        self.weight_col = weight_col
        self.weight_series = weight_series
        self.date_col = date_col

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> WindowedForecaster:
        self.base_model.fit(X, y)
        self._fitted_ = True
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["_fitted_"])
        daily = np.asarray(self.base_model.predict(X), dtype=float)
        return self._aggregate(X, daily)

    def predict_interval(
        self,
        X: pd.DataFrame,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["_fitted_"])
        if not hasattr(self.base_model, "predict_interval"):
            n = len(X)
            nan = np.full(n, np.nan, dtype=float)
            return nan, nan.copy()
        lo_daily, hi_daily = self.base_model.predict_interval(X, level=level)
        lo = self._aggregate(X, np.asarray(lo_daily, dtype=float))
        hi = self._aggregate(X, np.asarray(hi_daily, dtype=float))
        return lo, hi

    # ---------------------------------------------------------- aggregation
    def _weights_for(self, X: pd.DataFrame, ts: pd.DatetimeIndex) -> np.ndarray:
        if self.weight_col and self.weight_col in X.columns:
            w = pd.to_numeric(X[self.weight_col], errors="coerce").to_numpy()
        elif self.weight_series is not None:
            w = pd.Series(self.weight_series).reindex(ts).to_numpy()
        else:
            w = np.ones(len(ts), dtype=float)
        w = np.where(np.isfinite(w), w, 0.0)
        return w.astype(float)

    def _aggregate(self, X: pd.DataFrame, daily: np.ndarray) -> np.ndarray:
        ts = as_datetime_index(X, self.date_col)
        w = self._weights_for(X, ts)
        buckets = _bucket_starts(ts, self.window)

        df = pd.DataFrame({"bucket": buckets, "y": daily, "w": w})
        num = df.groupby("bucket", sort=False)["y"].transform(
            lambda s: np.average(s, weights=df.loc[s.index, "w"])
            if df.loc[s.index, "w"].sum() > 0
            else s.mean()
        )
        return num.to_numpy(dtype=float)

    # -------------------------------------------------- classmethod helpers
    @classmethod
    def aggregate_preds(
        cls,
        preds: pd.DataFrame,
        window: str = "14D",
        weight_series: pd.Series | None = None,
        weight_col: str | None = None,
        prefer_split: str = "val",
        model_col: str = "model",
        ts_col: str = "ts",
    ) -> pd.DataFrame:
        """Bucket a daily backtest preds frame into per-window weighted averages.

        Same aggregation as :meth:`_aggregate` but operates on a preds frame
        (e.g. ``BacktestReport.preds``) so callers get a ``rolling`` frame
        without re-fitting anything. When both train and val rows exist for
        the same ``(model, ts)``, ``prefer_split`` wins (default: val).

        Returns a frame with columns ``[model, window_start, window_end,
        n_days, obs, fcst, residual, rel_residual]`` — the exact schema
        :meth:`BacktestReport.plot_rolling_window_residuals_box` /
        :meth:`plot_rolling_window_residuals` expects via their ``rolling=``
        kwarg.
        """
        p = preds.copy()
        p[ts_col] = pd.to_datetime(p[ts_col])
        if "split" in p.columns:
            rank = {"train": 0, "val": 1}
            other = 1 if prefer_split == "val" else 0
            p["_rank"] = p["split"].map(rank).fillna(other)
            p = (
                p.sort_values([model_col, ts_col, "_rank"])
                .drop_duplicates(subset=[model_col, ts_col], keep="last")
                .drop(columns="_rank")
            )
        p = p.dropna(subset=["y_pred"]).sort_values([model_col, ts_col]).reset_index(drop=True)

        w_source = pd.Series(weight_series).astype(float) if weight_series is not None else None
        if w_source is not None:
            w_source.index = pd.to_datetime(w_source.index)

        has_fold = "fold" in p.columns
        has_split = "split" in p.columns

        def _agg(g: pd.DataFrame) -> pd.Series:
            ts = pd.DatetimeIndex(g[ts_col])
            if weight_col and weight_col in g.columns:
                w = pd.to_numeric(g[weight_col], errors="coerce").to_numpy()
            elif w_source is not None:
                w = w_source.reindex(ts).to_numpy()
            else:
                w = np.ones(len(g), dtype=float)
            w = np.where(np.isfinite(w), w, 0.0)
            if w.sum() == 0:
                w = np.ones(len(g), dtype=float)
            obs = float(np.average(g["y_true"], weights=w))
            fcst = float(np.average(g["y_pred"], weights=w))
            out = {
                "window_start": ts.min(),
                "window_end": ts.max(),
                "n_days": len(g),
                "obs": obs,
                "fcst": fcst,
                "residual": obs - fcst,
                "rel_residual": (obs - fcst) / obs if obs != 0 else np.nan,
            }
            if has_fold:
                # Prefer the fold of val rows in this window when any exist,
                # otherwise the mode across all rows. Windows are calendar-
                # aligned to the pre-intervention grid so a single fold usually wins.
                if has_split:
                    val_folds = g.loc[g["split"] == prefer_split, "fold"]
                    src = val_folds if not val_folds.empty else g["fold"]
                else:
                    src = g["fold"]
                out["fold"] = int(src.mode().iloc[0])
            return pd.Series(out)

        rolling = (
            p.groupby([model_col, pd.Grouper(key=ts_col, freq=window)], sort=True)
            .apply(_agg)
            .reset_index()
            .drop(columns=[ts_col])
            .sort_values([model_col, "window_start"])
            .reset_index(drop=True)
        )
        return rolling

    @classmethod
    def sliding_weighted_avg_preds(
        cls,
        preds: pd.DataFrame,
        window: int,
        *,
        weight_series: pd.Series | None = None,
        weight_col: str | None = None,
        split: str | None = "val",
        ts_col: str = "ts",
        model_col: str = "model",
        fold_col: str = "fold",
        label: str = "start",
    ) -> pd.DataFrame:
        """Overlapping sliding-window weighted-average preds.

        For each ``(model, fold)`` group of ``preds`` (filtered to ``split``
        when that column is present), sort by ``ts_col`` and slide a
        length-``window`` window one row at a time. Each window emits one
        row whose ``y_true`` / ``y_pred`` are the weighted averages of the
        daily values inside the window; consecutive windows overlap by
        ``window - 1`` rows.

        Complements :meth:`aggregate_preds` (non-overlapping calendar
        buckets) for use cases where an A/B-style rolling smoother of the
        daily backtest forecast is wanted instead of hard bucket edges.

        Parameters
        ----------
        preds
            Daily preds frame (e.g. ``BacktestReport.preds``) with at
            minimum ``model_col``, ``ts_col``, ``y_true``, ``y_pred``.
        window
            Number of consecutive daily rows in each sliding window.
        weight_series
            Optional ``pd.Series`` indexed by timestamp with per-day weights
            (e.g. ``n_searches``). Falls back to ``weight_col`` on ``preds``,
            then to uniform weights.
        weight_col
            Column name on ``preds`` carrying per-row weights.
        split
            When ``preds`` has a ``"split"`` column, keep only rows where
            ``split`` equals this value. Pass ``None`` to disable filtering.
            The emitted rows always carry ``split == split`` (or ``"val"``
            when ``split is None``).
        label
            Where each window is timestamped: ``"start"`` (default — first
            window sits at day 0 of the val segment, no gap after train
            end), ``"end"``, or ``"center"``.

        Returns
        -------
        pd.DataFrame
            Preds-shaped frame with columns ``model, fold, split, ts,
            window_start, window_end, y_true, y_pred, y_lo, y_hi``. Feed
            it back to a fresh :class:`BacktestReport` to reuse the
            standard fold / residual visuals at window resolution.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if label not in {"start", "end", "center"}:
            raise ValueError(f"label must be one of 'start', 'end', 'center'; got {label!r}")

        p = preds.copy()
        p[ts_col] = pd.to_datetime(p[ts_col])
        if split is not None and "split" in p.columns:
            p = p[p["split"] == split]
        p = p.dropna(subset=["y_pred"]).reset_index(drop=True)

        out_split = split if split is not None else "val"
        cols = [
            model_col,
            "fold",
            "split",
            ts_col,
            "window_start",
            "window_end",
            "y_true",
            "y_pred",
            "y_lo",
            "y_hi",
        ]

        if p.empty:
            return pd.DataFrame(columns=cols)

        if weight_col and weight_col in p.columns:
            p["_w"] = pd.to_numeric(p[weight_col], errors="coerce").fillna(0.0)
        elif weight_series is not None:
            ws = pd.Series(weight_series).astype(float)
            ws.index = pd.to_datetime(ws.index)
            p["_w"] = p[ts_col].map(ws).fillna(0.0)
        else:
            p["_w"] = 1.0

        has_fold = fold_col in p.columns
        group_cols = [model_col, fold_col] if has_fold else [model_col]

        def _one(g: pd.DataFrame) -> list[dict]:
            g = g.sort_values(ts_col).reset_index(drop=True)
            n = len(g)
            if n < window:
                return []
            y_true = g["y_true"].to_numpy(dtype=float)
            y_pred = g["y_pred"].to_numpy(dtype=float)
            w = g["_w"].to_numpy(dtype=float)
            ts = g[ts_col].to_numpy()
            rows: list[dict] = []
            for i in range(n - window + 1):
                ww = w[i : i + window]
                if not np.isfinite(ww).all() or ww.sum() == 0:
                    ww = np.ones(window, dtype=float)
                if label == "start":
                    label_ts = ts[i]
                elif label == "end":
                    label_ts = ts[i + window - 1]
                else:  # center
                    label_ts = ts[i + window // 2]
                rows.append(
                    {
                        model_col: g[model_col].iloc[0],
                        "fold": int(g[fold_col].iloc[0]) if has_fold else 0,
                        "split": out_split,
                        ts_col: label_ts,
                        "window_start": ts[i],
                        "window_end": ts[i + window - 1],
                        "y_true": float(np.average(y_true[i : i + window], weights=ww)),
                        "y_pred": float(np.average(y_pred[i : i + window], weights=ww)),
                        "y_lo": np.nan,
                        "y_hi": np.nan,
                    }
                )
            return rows

        parts: list[dict] = []
        for _, g in p.groupby(group_cols, sort=True):
            parts.extend(_one(g))
        if not parts:
            return pd.DataFrame(columns=cols)
        return (
            pd.DataFrame(parts)
            .reindex(columns=cols)
            .sort_values([model_col, "fold", ts_col] if has_fold else [model_col, ts_col])
            .reset_index(drop=True)
        )

    @staticmethod
    def as_backtest_preds(rolling: pd.DataFrame) -> pd.DataFrame:
        """Convert a windowed ``rolling`` frame into a preds-shaped frame.

        Lets you feed windowed results back into a fresh
        :class:`BacktestReport` for the generic plots
        (``plot_faceted``, ``plot_folds_per_model``, ...) — one row per
        window instead of one row per day. When ``rolling`` carries a
        ``fold`` column (added by :meth:`aggregate_preds` when the source
        preds had one), it flows through so per-fold panels stay meaningful.
        """
        fold = rolling["fold"].astype(int).to_numpy() if "fold" in rolling.columns else 0
        out = pd.DataFrame(
            {
                "model": rolling["model"],
                "ts": pd.to_datetime(rolling["window_start"]),
                "fold": fold,
                "split": "val",
                "y_true": rolling["obs"].astype(float),
                "y_pred": rolling["fcst"].astype(float),
                "y_lo": np.nan,
                "y_hi": np.nan,
            }
        )
        return out.reset_index(drop=True)


def _bucket_starts(ts: pd.DatetimeIndex, window: str) -> np.ndarray:
    """Non-overlapping calendar-bucket start dates for each timestamp.

    Uses :func:`pd.Timestamp.floor` with the pandas offset so day-multiple
    aliases like ``"14D"`` (which :meth:`pd.Series.dt.to_period` rejects)
    still work.
    """
    offset = pd.tseries.frequencies.to_offset(window)
    return pd.DatetimeIndex(ts).floor(offset).to_numpy()
