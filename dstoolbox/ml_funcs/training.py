"""Training/CV orchestration: cross-validated prediction, model comparison, nested CV, classifier batch eval."""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .inspection import print_capability_matrix
from .performance_plots import ml_comparison_plot
from .scores import ml_scores

logger = logging.getLogger(__name__)


def ml_prediction_sub_epochs(model):
    """Extract per-epoch train/validation metrics from a fitted XGBoost model.

    Parameters
    ----------
    model : xgboost estimator
        A model that has been fit with an ``eval_set`` of two entries
        (training and validation).

    Returns
    -------
    pandas.DataFrame
        One row per epoch with ``Train_<metric>`` and
        ``Validation_<metric>`` columns plus an ``epochs`` column and a
        ``best_ntree`` column carrying the best iteration index.
    """
    results = model.evals_result()
    df_epochs = pd.DataFrame()
    for metric_key in results["validation_0"]:
        val0 = results["validation_0"][metric_key]
        val1 = results["validation_1"][metric_key]
        tmp = pd.DataFrame(
            [val0, val1], index=[f"Train_{metric_key}", f"Validation_{metric_key}"]
        ).T
        df_epochs = pd.concat([df_epochs, tmp], axis=1)

    df_epochs.index.name = "epochs"
    df_epochs = df_epochs.reset_index()
    df_epochs["best_ntree"] = model.best_iteration

    return df_epochs


def _unwrap_pipeline(model: Any) -> Any:
    """Return the final estimator of a ``Pipeline``; otherwise ``model``."""
    return model[-1] if isinstance(model, Pipeline) else model


def _resolve_cv(
    X: pd.DataFrame,
    y: pd.Series,
    sk_fold: Any,
) -> Iterable[tuple[Iterable[int], Iterable[int]]]:
    """Translate ``sk_fold`` (None | (X_val, y_val) tuple | splitter) into folds.

    Mutates neither ``X`` nor ``y``. Returns an iterable of ``(train_idx, val_idx)``
    suitable for ``X.iloc[...]``. For the (X_val, y_val) tuple form, the caller
    is responsible for concatenating before iterating — kept here only for
    backwards compatibility with the legacy ``ml_prediction`` shape.
    """
    if sk_fold is None:
        warnings.warn(
            "sk_fold=None: training and validation sets are identical.",
            UserWarning,
            stacklevel=3,
        )
        return zip([range(X.shape[0])], [range(X.shape[0])], strict=False)
    if isinstance(sk_fold, list | tuple) and len(sk_fold) == 2:
        # Legacy: caller passed (X_val, y_val). Caller must concat before calling _resolve_cv.
        raise TypeError("_resolve_cv does not support (X_val, y_val) tuples; handle in caller.")
    return sk_fold.split(X, y)


def _early_stopping_rounds(model: Any) -> int | None:
    """Return ``early_stopping_rounds`` if model is XGBoost-like, else ``None``."""
    base = _unwrap_pipeline(model)
    if "xgb" not in base.__class__.__name__.lower():
        return None
    return getattr(base, "early_stopping_rounds", None)


def _build_y_train_by_fold(X: pd.DataFrame, y: pd.Series, sk_fold: Any) -> dict[int, pd.Series]:
    """Materialize per-fold training targets keyed by CV iteration index.

    Used by ``ml_comparison`` to feed MASE its fold-specific history without
    refitting models. The splitter is consumed once; callers needing the splits
    afterwards should re-call ``.split(...)``.
    """
    if sk_fold is None or (isinstance(sk_fold, list | tuple) and len(sk_fold) == 2):
        return {}
    return {i: y.iloc[train_idx] for i, (train_idx, _) in enumerate(sk_fold.split(X, y))}


def _fit_one_fold(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int | None,
    callbacks: list | None,
) -> pd.DataFrame | None:
    """Fit ``model`` on one fold; return per-epoch metrics if early-stopping is on."""
    if early_stopping_rounds is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            callbacks=callbacks,
            verbose=10,
        )
        return ml_prediction_sub_epochs(model)
    model.fit(X_train, y_train)
    return None


def _collect_predictions(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    classifier: bool,
    interval_level: float | None = None,
) -> pd.DataFrame:
    """Build a ``y_model`` slice for one fold: ``y_pred``, ``y_true``, (probs, intervals).

    Parameters
    ----------
    model, X_val, y_val, classifier
        Fitted estimator (possibly wrapped in a ``Pipeline``), the fold's
        validation features/target, and whether ``model`` is a classifier.
    interval_level : float or None, default None
        Two-sided prediction-interval level in ``(0, 1)``, e.g. ``0.95`` for a
        95 % PI. When set **and** the unwrapped model exposes
        ``predict_interval(X, level=...)``, this call yields ``y_lower`` /
        ``y_upper`` columns alongside ``y_pred`` / ``y_true``. If the model
        can't produce intervals or the call raises, the columns are simply
        omitted (a UserWarning is emitted) — downstream concat/metrics stay
        NaN-safe. ``None`` (default) skips the ``predict_interval`` call
        entirely and returns the two-column point-forecast shape.
    """
    if classifier:
        probs = pd.DataFrame(model.predict_proba(X_val), index=y_val.index)
        return pd.concat(
            [probs, probs.idxmax(axis=1).rename("y_pred"), y_val.rename("y_true")],
            axis=1,
        )
    preds = pd.DataFrame(model.predict(X_val), index=y_val.index)
    out = pd.concat([preds, y_val], axis=1)
    out.columns = ["y_pred", "y_true"]

    if interval_level is not None and hasattr(_unwrap_pipeline(model), "predict_interval"):
        try:
            X_for_interval = model[:-1].transform(X_val) if isinstance(model, Pipeline) else X_val
            lower, upper = _unwrap_pipeline(model).predict_interval(
                X_for_interval, level=interval_level
            )
            out["y_lower"] = pd.Series(lower, index=y_val.index)
            out["y_upper"] = pd.Series(upper, index=y_val.index)
        except Exception as e:
            warnings.warn(
                f"predict_interval failed for {type(model).__name__}: {e}",
                UserWarning,
                stacklevel=2,
            )
    return out


def ml_prediction(
    ml_model,
    X,
    y,
    sk_fold,
    X_test=None,
    y_test=None,
    callbacks=None,
    verbose=False,
    interval_level: float | None = None,
    include_train_predictions: bool = False,
):
    """
    Perform machine learning prediction with cross-validation and optional early stopping.

    Parameters
    ----------
    ml_model
        An sklearn-style estimator or ``Pipeline`` ending in one.
    X, y
        Feature matrix and target aligned by index.
    sk_fold
        One of: ``None`` (no CV, train == val — warns), a splitter with a
        ``.split(X, y)`` method (``KFold`` / ``TimeSeriesSplit`` / …), or a
        legacy ``[X_val, y_val]`` list for a single hold-out.
    X_test, y_test
        Optional external eval set. When passed with a CV splitter and no
        early stopping, each fold overwrites the previous fold's predictions
        (a UserWarning is emitted).
    callbacks, verbose
        Forwarded to XGBoost's ``fit`` when early stopping is on.
    interval_level : float or None, default None
        Two-sided prediction-interval level (e.g. ``0.95``). When set,
        forecasters that expose ``predict_interval`` populate ``y_lower`` /
        ``y_upper`` in the returned ``y_model`` for every fold. ``None``
        preserves the historical two-column ``(y_pred, y_true)`` output.
    include_train_predictions : bool, default False
        Also call ``.predict()`` on each fold's training slice (right after
        the fit — no refit cost). The returned ``y_model`` gains a ``split``
        column with values ``"train"`` / ``"val"``. Callers that compute
        metrics from this frame **must** filter to ``split == "val"`` first;
        the train rows are for in-sample-forecast plots only.

    Returns
    -------
    (y_model, ml_models, df_epochs)
        ``y_model`` is the long predictions frame described above;
        ``ml_models`` is the list of fitted estimators (one entry per fold);
        ``df_epochs`` is the XGBoost per-epoch metric table, or ``None`` when
        early stopping is off.
    """
    y_model = pd.DataFrame([])
    df_epochs = pd.DataFrame([])
    ml_models = []

    base = _unwrap_pipeline(ml_model)
    classifier = is_classifier(base)
    early_stopping_rounds = _early_stopping_rounds(ml_model)

    if (
        (X_test is not None)
        and isinstance(sk_fold, StratifiedKFold | TimeSeriesSplit)
        and early_stopping_rounds is None
    ):
        warnings.warn(
            "X_test passed together with a CV splitter and no early stopping; "
            "CV folds will overwrite X_test predictions per iteration.",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(sk_fold, list) and len(sk_fold) == 2 and isinstance(sk_fold[0], pd.DataFrame):
        logger.info("no cross validation ")
        X_val_extra, y_val_extra = sk_fold
        train_no = X.shape[0]
        X = pd.concat([X, X_val_extra], axis=0)
        y = pd.concat([y, y_val_extra], axis=0)
        cv = zip([range(train_no)], [range(train_no, X.shape[0])], strict=False)
    else:
        cv = _resolve_cv(X, y, sk_fold)

    for cv_itr, (train_index, val_index) in enumerate(cv):
        if verbose:
            logger.info("CV Itreation %d", cv_itr + 1)
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        df_epochs_tmp = _fit_one_fold(
            ml_model,
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds,
            callbacks,
        )
        if df_epochs_tmp is not None:
            df_epochs_tmp["CV_Iteration"] = cv_itr
            df_epochs = pd.concat([df_epochs, df_epochs_tmp], axis=0)
            logger.debug("best_ntree=%s, best_score=%s", base.best_iteration, base.best_score)

        ml_models.append(ml_model)

        X_score = X_test if X_test is not None else X_val
        y_score = y_test if y_test is not None else y_val
        y_model0 = _collect_predictions(
            ml_model, X_score, y_score, classifier, interval_level=interval_level
        )
        y_model0["CV_Iteration"] = cv_itr
        if include_train_predictions:
            y_model0["split"] = "val"
        y_model = pd.concat([y_model, y_model0], axis=0)

        if include_train_predictions:
            try:
                y_train_pred = _collect_predictions(
                    ml_model, X_train, y_train, classifier, interval_level=None
                )
                y_train_pred["CV_Iteration"] = cv_itr
                y_train_pred["split"] = "train"
                y_model = pd.concat([y_model, y_train_pred], axis=0)
            except Exception as exc:  # noqa: BLE001 — surface model-side failure, keep going
                warnings.warn(
                    f"in-sample predict failed at fold {cv_itr} for {type(ml_model).__name__}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

    if early_stopping_rounds is not None:
        df_epochs["best_ntree"] = df_epochs["best_ntree"] == df_epochs["epochs"]
    else:
        df_epochs = None

    return y_model, ml_models, df_epochs


def ml_comparison(
    ml_models,
    X,
    y,
    scores_names,
    sk_fold,
    map_names=None,
    multi_class="ovo",
    average="macro",
    plot=True,
    verbose=True,
    show_capabilities=True,
    interval_level: float | None = None,
    season_length: int = 1,
    quantile: float = 0.5,
    mapNames=None,
    return_predictions: bool = False,
    include_train_predictions: bool = False,
):
    """
    Compare multiple machine learning models using cross-validation and return their performance metrics.

    Parameters
    ----------
    ml_models
        Iterable of sklearn-style estimators / forecasters / ``Pipeline`` objects.
    X, y
        Feature matrix and target aligned by index.
    scores_names
        List of metric keys resolved via ``ml_scores`` / ``metric_dict``
        (e.g. ``['rmse', 'mae', 'mase', 'smape', 'interval_coverage']``).
    sk_fold
        CV splitter with ``.split(X, y)``. ``None`` disables CV and warns.
    map_names, mapNames
        ``{index: display_name}`` dict for pretty model names in outputs.
        ``mapNames`` is the deprecated camelCase alias — pass one, not both.
    multi_class, average
        Forwarded to sklearn multi-class metrics.
    plot, verbose, show_capabilities
        UI toggles: fold-comparison plot, per-model progress printing, and
        the one-line capability matrix (point / interval / samples /
        components / attention) printed before fitting.

    Time-series knobs
    -----------------
    interval_level : float or None, default None
        Two-sided prediction-interval level (e.g. ``0.95`` for a 95 % PI).
        When set, every interval-capable forecaster is asked for
        ``predict_interval`` on each fold's validation window and the
        returned predictions frame gains ``y_lower`` / ``y_upper`` columns.
        Those columns are what ``interval_coverage`` and ``interval_width``
        score on, and what downstream ribbon plots draw. ``None`` (default)
        skips the ``predict_interval`` call entirely and yields point
        forecasts only. Models without ``predict_interval`` are unaffected
        either way.
    season_length : int, default 1
        Seasonal-naive baseline period used **only** by ``mase`` — it's
        forwarded straight to ``ml_scores(..., season_length=...)`` and lands
        in MASE's denominator ``mean(|y_t - y_{t-m}|)``. Set to the true
        seasonal period of the target series (``7`` for daily data with a
        weekly cycle, ``12`` for monthly-with-yearly, ``24`` for hourly-with-
        daily, etc.); the default ``1`` reduces MASE to a *naive-1* baseline,
        which is the wrong benchmark for seasonal series. **Not** a model
        knob — each forecaster's own seasonal period is set on the
        forecaster itself (e.g. ``AutoArimaSklearn(season_length=7)``,
        ``StatsForecastAutoARIMA(season_length=7)``); this parameter never
        reaches the models. Silently ignored when ``'mase'`` is not in
        ``scores_names``.
    quantile : float, default 0.5
        Forwarded to ``pinball_loss``; ignored when ``'pinball_loss'`` is not
        in ``scores_names``. ``0.5`` gives the median (½ * MAE); use e.g.
        ``0.9`` to score an upper-quantile forecast.

    When ``mase`` is in ``scores_names``, per-fold training history is
    reconstructed from ``X`` / ``y`` and the CV splitter once and reused for
    every model (via ``ml_scores(y_train_by_fold=...)``); models are not
    refit for the baseline.

    Returns
    -------
    metrics_all : DataFrame
        Long frame of per-fold and aggregate scores, one row per
        ``(model, CV_Iteration)`` plus mean / std summary rows.
    predictions : DataFrame, optional
        Returned when ``return_predictions=True``: concatenation of each
        model's per-fold ``y_model`` (columns ``y_pred``, ``y_true``,
        ``CV_Iteration``, optional ``y_lower`` / ``y_upper``, optional
        ``split``) with an added ``model`` column. Lets callers reuse the
        same fits for downstream plots instead of re-running ``ml_prediction``.

    When ``include_train_predictions=True``, the returned predictions also
    carry per-fold in-sample rows (``split="train"``) alongside the
    validation rows (``split="val"``). Metrics are computed on the val rows
    only — the train rows are persisted for downstream plotting (e.g.
    in-sample-forecast overlays) without paying for a refit.
    """
    if mapNames is not None:
        if map_names is not None:
            raise TypeError("pass map_names OR mapNames, not both")
        warnings.warn(
            "mapNames is deprecated; use map_names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        map_names = mapNames
    if map_names is None:
        map_names = {}

    if show_capabilities:
        derived_names = [
            map_names.get(i)
            or getattr(m, "display_name", None)
            or (
                "-->".join(s.__class__.__name__ for s in m)
                if isinstance(m, Pipeline)
                else m.__class__.__name__
            )
            for i, m in enumerate(ml_models)
        ]
        print_capability_matrix(ml_models, names=derived_names)
        print()

    y_train_by_fold = _build_y_train_by_fold(X, y, sk_fold) if "mase" in scores_names else None

    with warnings.catch_warnings():
        if verbose:
            warnings.simplefilter("default")
        else:
            warnings.simplefilter("ignore")
        metrics_all = pd.DataFrame()
        preds_all: list[pd.DataFrame] = []
        for con, model in enumerate(ml_models):
            if con in map_names:
                model_name = map_names.get(con)
            elif getattr(model, "display_name", None):
                model_name = model.display_name
            elif isinstance(model, Pipeline):
                model_name = "-->".join([i.__class__.__name__ for i in model])
            else:
                model_name = model.__class__.__name__

            start_time = dt.datetime.now()
            logger.info("%s...", model_name)

            y_model, _, _ = ml_prediction(
                model,
                X,
                y,
                sk_fold,
                interval_level=interval_level,
                include_train_predictions=include_train_predictions,
            )

            if return_predictions:
                preds_all.append(y_model.assign(model=model_name))

            # Score on val rows only; train rows (when present) are for plotting.
            y_model_for_scores = (
                y_model[y_model["split"] == "val"] if "split" in y_model.columns else y_model
            )
            cv_results = ml_scores(
                y_model_for_scores,
                scores_names,
                multi_class=multi_class,
                average=average,
                y_train_by_fold=y_train_by_fold,
                season_length=season_length,
                quantile=quantile,
            )

            tmp = cv_results
            tmp.insert(0, "model", model_name)
            end_time = dt.datetime.now()
            run_time = end_time - start_time
            tmp["elapsed_time"] = run_time

            metrics_all = pd.concat([metrics_all, tmp], axis=0)

            if verbose:
                txt = metrics_all.loc[
                    metrics_all["CV"].isin(
                        [
                            "CV_scores_Mean",
                            "CV_scores_STD",
                        ]
                    ),
                    :,
                ]
                logger.info("models summary:\n%s", txt)
                logger.info("-------------------------------------------")
        if plot:
            ml_comparison_plot(metrics_all, outputFile=None)
        if return_predictions:
            predictions = pd.concat(preds_all, axis=0) if preds_all else pd.DataFrame()
            return metrics_all, predictions
        return metrics_all


# ===== imports preserved from public (needed by extras below) =====
from .classifier_report import ProbabilisticClassifierReport
from .performance_plots import (
    plot_confusion_matrix_multi,
)

# ===== public-only extensions (preserved on vendor merge) =====


def classifier_performance_batch(
    y_model,
    map_lbls=None,
    scores_names=None,
    multi_class="raise",
    average="balanced",
):
    """Evaluate classifier performance with confusion matrices, PR/ROC curves, and score metrics.

    Parameters
    ----------
    y_model : pandas.DataFrame
        Predictions frame with at least ``y_true`` and ``prob`` columns.
    map_lbls : dict, optional
        Class-index -> display-label map. Defaults to
        ``{0: 'Low Loss', 1: 'High Loss'}``.
    scores_names : list of str, optional
        Metric names to compute. Defaults to
        ``['accuracy', 'recall', 'precision']``.
    multi_class : {'raise', 'ovr', 'ovo'}, default ``'raise'``
        Passed to metric functions.
    average : str, default ``'balanced'``
        Averaging mode passed to :func:`ml_scores`.

    Returns
    -------
    (pandas.DataFrame, dict)
        ``(scores, confusion_matrices)``.
    """
    if map_lbls is None:
        map_lbls = {0: "Low Loss", 1: "High Loss"}
    if scores_names is None:
        scores_names = [
            "accuracy",
            # 'balanced_accuracy',
            "recall",
            "precision",
            # 'roc_auc',
            # 'aucpr',
        ]
    confMats = plot_confusion_matrix_multi(y_model, map_lbls, outputFile=None)

    model_prob = y_model["prob"]  # y_model[map_lbls.get(1)]
    pos_label = 1
    pcr = ProbabilisticClassifierReport(y=y_model["y_true"], prob=model_prob, pos_label=pos_label)
    df_rp, thresholds = pcr.plot_precision_recall(outputFile=None)
    df_auc, thresholds = pcr.plot_roc(outputFile=None)

    scores = ml_scores(y_model, scores_names, multi_class=multi_class, average=average)
    return scores, confMats


def ml_prediction_nested_cv(ml_model, X, y, outer_fold, inner_fold):
    """Nested cross-validation: refit the best-iteration model on each outer fold.

    For each outer split, inner splits drive early-stopping (via
    :func:`ml_prediction_sub_epochs`); the best iteration is then used to
    refit on the outer train+val and score on the outer test set. XGBoost
    specific for now.

    Parameters
    ----------
    ml_model : xgboost.XGBClassifier or compatible
        Model with early-stopping support (``best_iteration``, ``best_score``).
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    outer_fold : sklearn cross-validator
        Provides the outer test folds.
    inner_fold : sklearn cross-validator
        Provides the inner validation folds.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(y_model, df_epochs)`` — out-of-fold predictions and per-epoch metrics.
    """
    from xgboost import XGBClassifier

    y_model = pd.DataFrame([])
    df_epochs = pd.DataFrame([])

    for cv_outer, (train_val_index, test_index) in enumerate(outer_fold.split(X, y)):
        X_train_val, X_test = X.iloc[train_val_index, :], X.iloc[test_index, :]
        y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]

        for cv_inner, (train_index, val_index) in enumerate(
            inner_fold.split(X_train_val, y_train_val)
        ):
            # print("Itreation ",cv_inner)

            X_train, X_val = X_train_val.iloc[train_index, :], X_train_val.iloc[val_index, :]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            ##TODO: it is only for xgboost, cover other ml_models
            eval_set = [(X_train, y_train), (X_val, y_val)]

            ml_model.fit(X_train, y_train, eval_set=eval_set, verbose=200)
            df_epochs_tmp = ml_prediction_sub_epochs(ml_model)
            df_epochs_tmp["CV_Iteration"] = f"{cv_outer}_{cv_inner}"
            df_epochs = pd.concat([df_epochs, df_epochs_tmp], axis=0)

            print("best_ntree=", ml_model.best_iteration)
            print("best_score=", ml_model.best_score)

            ml_model2 = XGBClassifier(n_estimators=ml_model.best_iteration)

            ml_model2.fit(
                X_train_val,
                y_train_val,
            )

            y_model0 = pd.DataFrame(ml_model2.predict_proba(X_test), index=X_test.index)
            #   y_model0.rename(columns=map_lbls,inplace=True)
            y_model0 = pd.concat([y_model0, y_model0.idxmax(axis=1).rename("y_pred")], axis=1)

            y_model0["CV_Iteration"] = f"{cv_outer}_{cv_inner}"
            y_model0["y_true"] = y_test

            y_model = pd.concat([y_model, y_model0], axis=0)

    # print("--------------------------------------------------------")

    df_epochs["best_ntree"] = df_epochs["best_ntree"] == df_epochs["epochs"]

    return y_model, df_epochs
