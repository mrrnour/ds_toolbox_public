"""Smoke tests for :func:`dstoolbox.ml_funcs.plot_metrics_and_residuals`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from dstoolbox.ml_funcs import plot_metrics_and_residuals


def _make_preds(n_folds: int = 3, n_val: int = 20, n_train: int = 60) -> pd.DataFrame:
    """Two-model long preds frame with per-fold train + val rows."""
    rng = np.random.default_rng(0)
    rows: list[pd.DataFrame] = []
    for fold in range(n_folds):
        for split, n in [("train", n_train), ("val", n_val)]:
            for model, noise_sd in [("m_a", 0.3), ("m_b", 0.9)]:
                y = rng.normal(size=n) * 2 + 5
                yhat = y + rng.normal(size=n) * noise_sd
                rows.append(
                    pd.DataFrame(
                        {
                            "model": model,
                            "CV_Iteration": fold,
                            "split": split,
                            "y_true": y,
                            "y_pred": yhat,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _make_metrics(n_folds: int = 3) -> pd.DataFrame:
    """Per-fold metrics with mase / R2 / mae, wired so ``m_a`` beats ``m_b``."""
    rng = np.random.default_rng(1)
    rows: list[dict[str, float | int | str]] = []
    for fold in range(n_folds):
        rows.append({
            "model": "m_a",
            "CV": f"CV_{fold + 1}",
            "mase": 0.4 + rng.normal(scale=0.05),
            "R2":   0.85 + rng.normal(scale=0.03),
            "mae":  0.24 + rng.normal(scale=0.03),
        })
        rows.append({
            "model": "m_b",
            "CV": f"CV_{fold + 1}",
            "mase": 1.1 + rng.normal(scale=0.05),
            "R2":   0.35 + rng.normal(scale=0.05),
            "mae":  0.72 + rng.normal(scale=0.05),
        })
    return pd.DataFrame(rows)


def test_basic_layout_and_ordering():
    preds = _make_preds()
    metrics = _make_metrics()

    fig, diag = plot_metrics_and_residuals(
        preds, metrics,
        metric_cols=["mase", "R2", "mae"],
        higher_is_better=["R2"],
        log_summary=False,
    )

    assert fig is not None
    # 1 residual row + 3 metric rows.
    assert len(fig.layout.annotations) == 4
    assert diag["metric_cols"] == ["mase", "R2", "mae"]
    assert diag["last_fold"] == 2
    assert diag["n_folds"] == 3
    # m_a's mase (~0.4) is lower than m_b's (~1.1) → m_a first (mase is lower-is-better).
    assert diag["model_order"][0] == "m_a"


def test_higher_is_better_flips_order():
    preds = _make_preds()
    metrics = _make_metrics()

    _, diag = plot_metrics_and_residuals(
        preds, metrics,
        metric_cols=["R2", "mase"],
        higher_is_better=["R2"],
        log_summary=False,
    )
    # First metric now R² (higher = better) → m_a still first (R² ~0.85 vs 0.35).
    assert diag["model_order"][0] == "m_a"

    _, diag_flip = plot_metrics_and_residuals(
        preds, metrics,
        metric_cols=["R2", "mase"],
        higher_is_better=[],  # treat R² as lower-is-better
        log_summary=False,
    )
    assert diag_flip["model_order"][0] == "m_b"


def test_diagnostics_returns_train_overlap():
    preds = _make_preds()
    metrics = _make_metrics()

    _, diag = plot_metrics_and_residuals(
        preds, metrics,
        metric_cols=["mase"],
        diagnostics=True,
        log_summary=False,
    )
    assert "train_overlap" in diag
    assert [r["fold"] for r in diag["train_overlap"]["rows"]] == [0, 1, 2]


def test_empty_val_returns_none():
    preds = _make_preds()
    preds = preds[preds["split"] == "train"]
    metrics = _make_metrics()

    fig, diag = plot_metrics_and_residuals(
        preds, metrics, metric_cols=["mase"], log_summary=False,
    )
    assert fig is None
    assert diag == {}


def test_empty_metrics_returns_none():
    preds = _make_preds()
    metrics = _make_metrics().iloc[0:0]

    fig, diag = plot_metrics_and_residuals(
        preds, metrics, metric_cols=["mase"], log_summary=False,
    )
    assert fig is None
    assert diag == {}


def test_missing_metric_col_raises():
    preds = _make_preds()
    metrics = _make_metrics()

    with pytest.raises(KeyError):
        plot_metrics_and_residuals(
            preds, metrics, metric_cols=["not_a_metric"], log_summary=False,
        )


def test_empty_metric_cols_raises():
    preds = _make_preds()
    metrics = _make_metrics()

    with pytest.raises(ValueError):
        plot_metrics_and_residuals(preds, metrics, metric_cols=[], log_summary=False)
