"""YAML-friendly metric label → ``metric_dict`` key translation.

Config files (YAML, CLI, notebook top cells) typically use short, human-facing
metric names like ``rmse``, ``mae``, ``mape``. The scoring pipeline in
:mod:`dstoolbox.ml_funcs.scores` keys off the full sklearn-style names
registered in ``metric_dict``. :data:`METRIC_ALIASES` maps between the two;
:func:`map_metric_names` applies that mapping to a user list while preserving
order and removing duplicates.

RMSE and the interval-coverage variants are handled by the caller — RMSE is
computed as ``sqrt(mean_squared_error)`` in the leaderboard step, and
``coverage_80`` / ``coverage_95`` both resolve to a single
``interval_coverage`` entry whose quantile is set elsewhere.
"""

from __future__ import annotations


METRIC_ALIASES: dict[str, str] = {
    "rmse": "mean_squared_error",  # caller takes sqrt in the leaderboard
    "mae": "mean_absolute_error",
    "mape": "mean_absolute_percentage_error",
    "r2": "R2",
    "smape": "smape",
    "mase": "mase",
    "coverage_80": "interval_coverage",
    "coverage_95": "interval_coverage",
}


def map_metric_names(metrics: list[str]) -> list[str]:
    """Translate config metric labels to ``metric_dict`` keys.

    Order-preserving and deduplicated: the first occurrence of each resolved
    key wins, later duplicates are dropped. Unknown labels pass through
    unchanged so callers can extend the registry ad hoc without editing this
    table.
    """
    seen: list[str] = []
    for m in metrics:
        key = METRIC_ALIASES.get(m, m)
        if key not in seen:
            seen.append(key)
    return seen
