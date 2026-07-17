"""Cross-validation splitters for time-series and panel data.

For a single time series, prefer :class:`sklearn.model_selection.TimeSeriesSplit`
directly — this module exists for two things sklearn doesn't cover:

1. :class:`PanelTimeSeriesSplit` — time-aware splits for panel data (multiple
   series stacked in one ``DataFrame``). Splits by *date*, not by row index,
   so every series contributes the same train/val date windows in each fold.
2. :func:`time_series_split_from_config` — translate a "backtest" config block
   (``initial_train``, ``step``, ``horizon``, ``gap``, optional rolling window)
   into a stock :class:`TimeSeriesSplit`. Keeps notebooks free of fold-count
   arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class BacktestConfig:
    """Plain config object for translating into a :class:`TimeSeriesSplit`.

    Attributes mirror the YAML ``backtest:`` block used in time-series
    notebooks. ``rolling_window`` set to ``None`` (default) yields an
    expanding-window split; an int yields rolling.
    """

    initial_train: int
    step: int
    horizon: int
    gap: int = 0
    rolling_window: int | None = None


def _n_splits_for(n_samples: int, cfg: BacktestConfig) -> int:
    """How many folds fit in ``n_samples`` given an initial train + step + horizon."""
    usable = n_samples - cfg.initial_train - cfg.gap - cfg.horizon
    if usable < 0:
        raise ValueError(
            f"n_samples={n_samples} too small for initial_train={cfg.initial_train} "
            f"+ gap={cfg.gap} + horizon={cfg.horizon}"
        )
    return usable // cfg.step + 1


class ExpandingBacktestSplit:
    """Expanding (or rolling) window backtest with explicit ``initial_train`` + ``step``.

    sklearn's :class:`TimeSeriesSplit` end-aligns folds and forces
    ``step == test_size`` — it can't model the common "train at least N
    samples, advance by S each fold, forecast H ahead" backtest. This
    splitter does, while exposing the same ``.split() / .get_n_splits()``
    interface so it's a drop-in.

    Parameters
    ----------
    initial_train, step, horizon, gap, max_train_size
        Same semantics as :class:`BacktestConfig`. ``max_train_size`` set to
        an int turns the splitter into a rolling-window backtest; ``None``
        keeps it expanding.
    n_folds, n_samples
        Optional — pin the number of folds instead of (or in addition to)
        ``initial_train``/``step``. Exactly two of
        ``{initial_train, step, n_folds}`` must be supplied; the third is
        derived. ``n_samples`` (the length of the series being split) is
        required whenever ``n_folds`` is used to derive one of the others.
    """

    def __init__(
        self,
        initial_train: int | None = None,
        step: int | None = None,
        horizon: int | None = None,
        gap: int = 0,
        max_train_size: int | None = None,
        *,
        n_folds: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        if horizon is None:
            raise ValueError("horizon is required")
        initial_train, step = self._resolve_train_step(
            initial_train=initial_train,
            step=step,
            horizon=horizon,
            gap=gap,
            n_folds=n_folds,
            n_samples=n_samples,
        )
        self.initial_train = initial_train
        self.step = step
        self.horizon = horizon
        self.gap = gap
        self.max_train_size = max_train_size

    @staticmethod
    def _resolve_train_step(
        *,
        initial_train: int | None,
        step: int | None,
        horizon: int,
        gap: int,
        n_folds: int | None,
        n_samples: int | None,
    ) -> tuple[int, int]:
        provided = sum(x is not None for x in (initial_train, step, n_folds))
        if provided < 2:
            raise ValueError(
                "provide at least two of {initial_train, step, n_folds}; "
                f"got initial_train={initial_train}, step={step}, n_folds={n_folds}"
            )
        if n_folds is not None and (initial_train is None or step is None):
            if n_samples is None:
                raise ValueError("n_samples is required when deriving initial_train or step from n_folds")
            if n_folds < 1:
                raise ValueError(f"n_folds must be >= 1, got {n_folds}")
            if initial_train is None:
                # step + n_folds + n_samples known
                initial_train = n_samples - gap - horizon - (n_folds - 1) * step
            else:
                # initial_train + n_folds + n_samples known
                usable = n_samples - initial_train - gap - horizon
                if n_folds == 1:
                    step = max(1, usable)
                else:
                    if usable < n_folds - 1:
                        raise ValueError(
                            f"n_samples={n_samples} too small for {n_folds} folds "
                            f"with initial_train={initial_train}, gap={gap}, horizon={horizon}"
                        )
                    step = usable // (n_folds - 1)
        if initial_train is None or step is None:
            raise ValueError("could not resolve both initial_train and step")
        if initial_train <= 0:
            raise ValueError(f"derived initial_train must be > 0, got {initial_train}")
        if step <= 0:
            raise ValueError(f"derived step must be > 0, got {step}")
        return initial_train, step

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        train_end = self.initial_train
        while train_end + self.gap + self.horizon <= n:
            val_start = train_end + self.gap
            val_end = val_start + self.horizon
            train_start = (
                max(0, train_end - self.max_train_size)
                if self.max_train_size is not None
                else 0
            )
            yield np.arange(train_start, train_end), np.arange(val_start, val_end)
            train_end += self.step

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return 0
        n = len(X)
        usable = n - self.initial_train - self.gap - self.horizon
        return max(0, usable // self.step + 1)


def time_series_split_from_config(
    cfg: BacktestConfig | Mapping[str, int],
    n_samples: int | None = None,  # noqa: ARG001 — kept for backwards-compat
) -> ExpandingBacktestSplit:
    """Build an :class:`ExpandingBacktestSplit` from a backtest config.

    The translation: ``initial_train`` / ``step`` / ``horizon`` / ``gap``
    pass through directly; ``max_train_size = rolling_window`` (``None`` for
    expanding). ``n_samples`` is unused (kept in the signature only so older
    notebooks that pass it keep working).
    """
    if not isinstance(cfg, BacktestConfig):
        cfg = BacktestConfig(**dict(cfg))
    return ExpandingBacktestSplit(
        initial_train=cfg.initial_train,
        step=cfg.step,
        horizon=cfg.horizon,
        gap=cfg.gap,
        max_train_size=cfg.rolling_window,
    )


class HoldoutSplit:
    """Single train/val split — sklearn splitter that yields exactly one fold.

    Time-series analogue of the notebook's "fit on ``train``, predict
    ``val``" pattern. Use it with :func:`ml_comparison` when you want one
    number per model (quick baseline sanity check) instead of a full CV
    loop. Indices are positional and chronological — no shuffling, no
    stratification.

    Parameters
    ----------
    train_size
        Either an ``int`` number of training rows or a ``float`` in
        ``(0, 1)`` interpreted as a proportion of ``len(X)``.
    horizon
        Validation length. ``None`` (default) means "everything after the
        train block (minus ``gap``)".
    gap
        Rows dropped between train and validation.

    Examples
    --------
    >>> HoldoutSplit(train_size=0.75)              # last 25% as val
    >>> HoldoutSplit(train_size=200, horizon=24)   # 200 train rows, 24 val rows
    """

    def __init__(
        self,
        train_size: int | float,
        horizon: int | None = None,
        gap: int = 0,
    ) -> None:
        if isinstance(train_size, float) and not 0 < train_size < 1:
            raise ValueError(f"train_size as float must be in (0, 1), got {train_size}")
        if isinstance(train_size, int) and train_size <= 0:
            raise ValueError(f"train_size as int must be > 0, got {train_size}")
        if horizon is not None and horizon <= 0:
            raise ValueError(f"horizon must be > 0 when set, got {horizon}")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        self.train_size = train_size
        self.horizon = horizon
        self.gap = gap

    def _train_end(self, n: int) -> int:
        return int(self.train_size * n) if isinstance(self.train_size, float) else int(self.train_size)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        train_end = self._train_end(n)
        val_start = train_end + self.gap
        val_end = n if self.horizon is None else min(n, val_start + self.horizon)
        if train_end <= 0 or val_start >= n or val_end <= val_start:
            raise ValueError(
                f"HoldoutSplit: empty fold for n={n}, train_end={train_end}, "
                f"val_start={val_start}, val_end={val_end}"
            )
        yield np.arange(0, train_end), np.arange(val_start, val_end)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1


class PanelTimeSeriesSplit:
    """Time-aware CV split for panel (multi-series) data.

    Splits the *unique sorted dates* in ``X[date_col]`` using a stock
    :class:`TimeSeriesSplit`, then expands each train/val date set back to row
    positions so every series in the panel contributes the same date window
    to each fold.

    Drop-in replacement for :class:`TimeSeriesSplit` when your ``X`` contains
    a date column shared across multiple series (typical panel/longitudinal
    layout). For a single series, just use :class:`TimeSeriesSplit` directly.

    Parameters
    ----------
    date_col
        Column in ``X`` holding the per-row timestamp. Must be parseable by
        ``pd.to_datetime``.
    n_splits
        Number of CV folds. Default 5 (matches sklearn).
    test_size
        Number of *unique dates* in each validation fold. ``None`` lets
        sklearn pick (``n_dates // (n_splits + 1)``).
    gap
        Number of *unique dates* to drop between train and validation in
        each fold.
    max_train_size
        Cap on number of unique training dates per fold (rolling window).
        ``None`` for expanding window.

    Notes
    -----
    - All series in the panel are assumed to share the same date grid (or at
      least be drawn from the same superset). Missing dates per series are
      fine — that series simply contributes fewer rows in that fold.
    - The yielded ``train_idx`` / ``val_idx`` are **positional** indices into
      ``X`` (i.e. compatible with ``X.iloc[...]``).
    """

    def __init__(
        self,
        date_col: str,
        n_splits: int = 5,
        test_size: int | None = None,
        gap: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        self.date_col = date_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, val_idx)`` positional indices for each fold."""
        if self.date_col not in X.columns:
            raise KeyError(
                f"PanelTimeSeriesSplit: date_col={self.date_col!r} not in X"
            )

        dates = pd.to_datetime(X[self.date_col]).to_numpy()
        unique_dates = np.sort(np.unique(dates))

        inner = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap,
            max_train_size=self.max_train_size,
        )

        positions = np.arange(len(X))
        for train_date_idx, val_date_idx in inner.split(unique_dates):
            train_dates = unique_dates[train_date_idx]
            val_dates = unique_dates[val_date_idx]
            train_mask = np.isin(dates, train_dates)
            val_mask = np.isin(dates, val_dates)
            yield positions[train_mask], positions[val_mask]

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        groups: object | None = None,
    ) -> int:
        """Number of folds. Matches sklearn splitter protocol."""
        return self.n_splits
