"""ML metric registry and scoring helpers."""

import inspect
import logging

import numpy as np
import pandas as pd
from sklearn import metrics

logger = logging.getLogger(__name__)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent: ``100 * mean(2 * |y - yhat| / (|y| + |yhat|))``.

    Bounded in ``[0, 200]``; safer than MAPE when ``y`` crosses zero or is small.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp)
    diff = np.abs(yt - yp)
    mask = denom > 0
    safe = np.where(mask, denom, 1.0)
    return float(100.0 * np.mean(np.where(mask, 2.0 * diff / safe, 0.0)))


def mase(y_true, y_pred, y_train=None, season_length: int = 1) -> float:
    """Mean Absolute Scaled Error vs seasonal-naive on ``y_train``.

    ``mase < 1`` means the model beats a seasonal-naive baseline trained on the
    same history. Returns NaN if ``y_train`` is missing or too short for the
    requested ``season_length``.
    """
    if y_train is None:
        return float("nan")
    yt_train = np.asarray(y_train, dtype=float)
    if yt_train.size <= season_length:
        return float("nan")
    naive_err = float(np.mean(np.abs(yt_train[season_length:] - yt_train[:-season_length])))
    if naive_err == 0:
        return float("nan")
    err = float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
    return err / naive_err


def pinball_loss(y_true, y_pred, quantile: float = 0.5) -> float:
    """Quantile (pinball) loss at the given quantile (default median).

    Lower is better. For median forecasts, equals 0.5 * MAE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def interval_coverage(y_true, y_pred=None, y_lower=None, y_upper=None) -> float:
    """Fraction of ``y_true`` falling inside ``[y_lower, y_upper]``.

    ``y_pred`` is accepted only to satisfy the ``(y_true, y_pred, ...)`` calling
    convention; it's not used. Returns NaN if interval columns are missing or
    all-NaN (model lacks ``predict_interval``).
    """
    if y_lower is None or y_upper is None:
        return float("nan")
    yt = np.asarray(y_true, dtype=float)
    yl = np.asarray(y_lower, dtype=float)
    yu = np.asarray(y_upper, dtype=float)
    valid = ~(np.isnan(yl) | np.isnan(yu))
    if not valid.any():
        return float("nan")
    inside = (yt[valid] >= yl[valid]) & (yt[valid] <= yu[valid])
    return float(np.mean(inside))


def interval_width(y_true=None, y_pred=None, y_lower=None, y_upper=None) -> float:
    """Mean width of the prediction interval; NaN if intervals are missing."""
    if y_lower is None or y_upper is None:
        return float("nan")
    yl = np.asarray(y_lower, dtype=float)
    yu = np.asarray(y_upper, dtype=float)
    valid = ~(np.isnan(yl) | np.isnan(yu))
    if not valid.any():
        return float("nan")
    return float(np.mean(yu[valid] - yl[valid]))


metric_dict={
            'accuracy'                          : metrics.accuracy_score,
            'balanced_accuracy'                 : metrics.balanced_accuracy_score,
            'top_k_accuracy'                    : metrics.top_k_accuracy_score,
            'average_precision'                 : metrics.average_precision_score,
            'aucpr'                             : metrics.average_precision_score,
            'brier_score'                       : metrics.brier_score_loss,
            'f1'                                : metrics.f1_score,
            'f1_samples'                        : metrics.f1_score,
            'log_loss'                          : metrics.log_loss,
            'precision'                         : metrics.precision_score,
            'recall'                            : metrics.recall_score,
            'jaccard'                           : metrics.jaccard_score,
            'auc'                               : metrics.roc_auc_score,
            'roc_auc'                           : metrics.roc_auc_score,
            'adjusted_mutual_info_score'        : metrics.adjusted_mutual_info_score,
            'adjusted_rand_score'               : metrics.adjusted_rand_score,
            'completeness_score'                : metrics.completeness_score,
            'fowlkes_mallows_score'             : metrics.fowlkes_mallows_score,
            'homogeneity_score'                 : metrics.homogeneity_score,
            'mutual_info_score'                 : metrics.mutual_info_score,
            'normalized_mutual_info_score'      : metrics.normalized_mutual_info_score,
            'rand_score'                        : metrics.rand_score,
            'v_measure_score'                   : metrics.v_measure_score,

            'explained_variance'                : metrics.explained_variance_score,
            'max_error'                         : metrics.max_error,
            'mean_absolute_error'               : metrics.mean_absolute_error,
            'mean_squared_error'                : metrics.mean_squared_error,
            'mean_squared_log_error'            : metrics.mean_squared_log_error,
            'median_absolute_error'             : metrics.median_absolute_error,
            'R2'                                : metrics.r2_score,
            'mean_poisson_deviance'             : metrics.mean_poisson_deviance,
            'mean_gamma_deviance'               : metrics.mean_gamma_deviance,
            'mean_absolute_percentage_error'    : metrics.mean_absolute_percentage_error,

            'mcc'                               : metrics.matthews_corrcoef,
            'kappa'                             : metrics.cohen_kappa_score,

            # Time-series metrics (see functions above).
            'mase'                              : mase,
            'smape'                             : smape,
            'pinball_loss'                      : pinball_loss,
            'interval_coverage'                 : interval_coverage,
            'interval_width'                    : interval_width,
            }


_AUTO_COLS = ("y_lower", "y_upper")


def _call_metric(umetric, df: pd.DataFrame, kwargs_dict: dict) -> float:
    """Invoke ``umetric(y_true, y_pred, ...)`` injecting only the kwargs/cols it accepts.

    - Columns ``y_lower`` / ``y_upper`` are pulled from ``df`` when present and
      the metric's signature names them (used by interval metrics).
    - Anything in ``kwargs_dict`` that the metric accepts is forwarded as-is.
    """
    params = inspect.signature(umetric).parameters
    call_kwargs = {}
    for col in _AUTO_COLS:
        if col in params and col in df.columns:
            call_kwargs[col] = df[col]
    for k, v in kwargs_dict.items():
        if k in params:
            call_kwargs[k] = v
    return umetric(df['y_true'], df['y_pred'], **call_kwargs)


def ml_scores(y_model, scores_names,
              multi_class='ovo',
              average='macro',  # {'micro', 'macro', 'samples', 'weighted'}
              y_train=None,
              y_train_by_fold: dict | None = None,
              season_length: int = 1,
              quantile: float = 0.5,
              ):
    """Compute the requested metrics on ``y_model``.

    ``y_model`` must have columns ``y_true``, ``y_pred``, ``CV_Iteration``;
    optional ``y_lower`` / ``y_upper`` columns are picked up automatically by
    interval metrics (coverage, width).

    Time-series metric plumbing
    ---------------------------
    - ``mase`` needs the training history; pass it via ``y_train`` (used for the
      whole-data row) and/or ``y_train_by_fold={cv_iter: y_train_series}`` (used
      per fold). If both are given, ``y_train_by_fold`` wins per-fold.
    - ``mase`` also reads ``season_length`` (default 1, i.e. naive-1).
    - ``pinball_loss`` reads ``quantile`` (default 0.5).
    """

    if 'CV_Iteration' not in y_model.columns:
        y_model['CV_Iteration'] = 'All_data'

    scores_all = pd.Series(index=scores_names, dtype='float64', name='scores_all')
    scores = pd.DataFrame(index=y_model['CV_Iteration'].unique(), columns=scores_names)

    if (y_model.shape[1] > 5) & (average == 'binary'):
        logger.warning("It is a multiclass problem: average argument changed to 'macro'")
        average = 'macro'

    base_kwargs = {
        'average': average,
        'multi_class': multi_class,
        'season_length': season_length,
        'quantile': quantile,
    }
    if y_train is not None:
        base_kwargs['y_train'] = y_train

    for score_name in scores_names:
        umetric = metric_dict.get(score_name)
        if umetric is None:
            logger.info("%s not in metric_dict; skipped", score_name)
            continue

        try:
            scores_all[score_name] = _call_metric(umetric, y_model, base_kwargs)
        except Exception as e:
            logger.warning("%s wasn't added to scores_all:\n  %s", score_name, e)
            continue

        for cv_iter, x in y_model.groupby('CV_Iteration'):
            fold_kwargs = dict(base_kwargs)
            if y_train_by_fold is not None and cv_iter in y_train_by_fold:
                fold_kwargs['y_train'] = y_train_by_fold[cv_iter]
            try:
                scores.loc[cv_iter, score_name] = _call_metric(umetric, x, fold_kwargs)
            except Exception as e:
                logger.warning("%s fold %s failed:\n  %s", score_name, cv_iter, e)

    scores = pd.concat([
        scores,
        pd.DataFrame(scores.mean(axis=0)).T.rename({0: 'CV_scores_Mean'}, axis=0),
        pd.DataFrame(scores.std(axis=0)).T.rename({0: 'CV_scores_STD'}, axis=0),
        pd.DataFrame(scores_all).T,
    ], axis=0)
    scores = scores.reset_index().rename({'index': 'CV'}, axis=1)

    return scores


# ===== public-only extensions (preserved on vendor merge) =====

def ml_scores_crossvalidate(**kwargs):
    """
    Perform cross-validation on a given estimator and return the results as a DataFrame.
    This function uses scikit-learn's `cross_validate` to perform cross-validation on the provided estimator
    and returns the results in a pandas DataFrame. The DataFrame includes the mean and standard deviation
    of the cross-validation scores.
    Parameters:
    **kwargs:
      Keyword arguments to be passed to `sklearn.model_selection.cross_validate`. These typically include:
      - estimator: The object to use to fit the data.
      - X: The data to fit.
      - y: The target variable to try to predict.
      - scoring: A single string or a callable to evaluate the predictions on the test set.
      - cv: Determines the cross-validation splitting strategy.
      - return_train_score: Whether to include train scores.
    Returns:
    pandas.DataFrame:
      A DataFrame containing the cross-validation results. The DataFrame includes the mean and standard
      deviation of the cross-validation scores, with the keys 'CV_scores_Mean' and 'CV_scores_STD' respectively.
      The 'fit_time' and 'score_time' columns are removed from the results.
    """
    from sklearn.model_selection import cross_validate
    ##NOTE: you can't use cross_validate for early stopping
    ####scoring for cross_validate
    # scoring=[
    #         'accuracy',
    #         'roc_auc',
    #         'recall' ,
    #         'f1',
    #         'kappa',
    #         'mcc',
    #         'average_precision',
    #         'balanced_accuracy',
    #         'precision',
    #         ]
    # scoring_dict=dict(zip(scoring, scoring))
    # ## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    # scoring_dict['mcc']=make_scorer(matthews_corrcoef)
    # scoring_dict['kappa']=make_scorer(cohen_kappa_score)

    cv_results = cross_validate(**kwargs)
    cv_results = pd.DataFrame(cv_results)
    cv_results = cv_results.drop(
        columns=[c for c in ('fit_time', 'score_time') if c in cv_results.columns]
    )
    cv_results = pd.concat([
        cv_results,
        cv_results.mean(axis=0).to_frame().T.rename(index={0: 'CV_scores_Mean'}),
        cv_results.std(axis=0).to_frame().T.rename(index={0: 'CV_scores_STD'}),
    ])
    return cv_results
