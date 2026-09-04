"""Multi-label classification helpers: tag binarization, stratified split, evaluation."""

import numpy as np
import pandas as pd

from ..utils.dataframes import unify_cols
from .scores import ml_scores


def binarize_multilabel_tags(tags):
    """Convert per-sample tag lists into a one-hot binary matrix.

    Parameters
    ----------
    tags : list of list of str
        One list of tags per sample.

    Returns
    -------
    pandas.DataFrame
        Rows are samples, columns are the unique tags, values are 0/1.

    Examples
    --------
    >>> tags = [['tag1', 'tag2'], ['tag2', 'tag3'], ['tag1']]
    >>> binarize_multilabel_tags(tags)
       tag1  tag2  tag3
    0     1     1     0
    1     0     1     1
    2     1     0     0
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    tags_seri = pd.Series(tags)
    mlb = MultiLabelBinarizer()
    out = pd.DataFrame(mlb.fit_transform(tags_seri), columns=mlb.classes_, index=tags_seri.index)
    return out


def split_multilabel_data_indices(X, y, test_size, random_state=None):
    """Iteratively stratified train/test row-index split for multi-label targets.

    Uses ``iterstrat.MultilabelStratifiedShuffleSplit`` to keep label
    marginals balanced across the two subsets.

    Parameters
    ----------
    X : array-like
        Feature matrix (only its shape is used).
    y : array-like of shape (n_samples, n_labels)
        Binary multi-label indicator matrix.
    test_size : float in (0, 1)
        Fraction of rows to place in the test split.
    random_state : int, numpy.random.RandomState, or None, optional
        Seed forwarded to the stratifier.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(train_indexes, test_indexes)`` row positions.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    stratifier = MultilabelStratifiedShuffleSplit(
        n_splits=2, test_size=test_size, random_state=random_state
    )
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes, test_indexes


def split_multilabel_data(df_samples, binarized_tags, random_state=None):
    """Stratified 70/21/9 train/eval/test split of multi-label samples.

    First splits 70/30 (train / eval+test), then 70/30 within eval+test
    (eval / test). Adds a ``Set`` column to both frames.

    Parameters
    ----------
    df_samples : pandas.DataFrame
        Sample features; a ``Set`` column will be added in place.
    binarized_tags : pandas.DataFrame
        Multi-label indicator frame aligned to ``df_samples``.
    random_state : int or None, optional
        Seed forwarded to the second (eval/test) split.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(df_samples, binarized_tags)`` with a ``Set`` column added
        to each (values ``'train'``, ``'eval'``, ``'test'``).
    """
    binarized_tags_lst = binarized_tags.apply(lambda x: x.tolist(), axis=1)

    train_rows, evalNtest_rows = split_multilabel_data_indices(
        df_samples.to_numpy(),
        np.array(binarized_tags_lst.tolist()),
        test_size=0.30,
        random_state=None,
    )
    df_samples__eval_test = df_samples.iloc[evalNtest_rows]
    eval_rows, test_rows = split_multilabel_data_indices(
        df_samples__eval_test.to_numpy(),
        np.array(binarized_tags_lst.iloc[evalNtest_rows].tolist()),
        test_size=0.30,
        random_state=random_state,
    )

    train_idx, eval_idx, test_idx = (
        df_samples.iloc[train_rows].index,
        df_samples__eval_test.iloc[eval_rows].index,
        df_samples__eval_test.iloc[test_rows].index,
    )

    df_samples.loc[df_samples.index.isin(train_idx), "Set"] = "train"
    df_samples.loc[df_samples.index.isin(eval_idx), "Set"] = "eval"
    df_samples.loc[df_samples.index.isin(test_idx), "Set"] = "test"
    binarized_tags["Set"] = df_samples["Set"]

    return df_samples, binarized_tags


def evaluate_multilabel(y_pred, y_true, average_op="binary", scores_names=None):
    """Compute per-tag and aggregate (macro/micro/weighted) multi-label scores.

    Parameters
    ----------
    y_pred : pandas.DataFrame
        Predicted binary indicators, one column per tag.
    y_true : pandas.DataFrame
        True binary indicators aligned to ``y_pred``.
    average_op : str, default ``'binary'``
        Averaging mode passed to per-tag metric functions.
    scores_names : list of str, optional
        Metric names to compute. Defaults to a fixed classification set
        (recall, precision, accuracy, auc_weighted, f1, kappa, mcc).

    Returns
    -------
    (dict, pandas.DataFrame)
        ``(model_performance, y_model)`` where ``model_performance`` has
        keys ``'yScore'`` (per-tag + aggregate scores frame) and
        ``'accuracy_overall'`` (exact-match subset accuracy over the
        full multi-label output), and ``y_model`` is the long-form
        prediction frame renamed to use ``Class`` for the tag column.
    """
    if scores_names is None:
        scores_names = [
            "recall",
            "precision",
            "accuracy",
            "auc_weighted",
            # 'balanced_accuracy',
            # 'roc_auc',
            # 'aucpr',
            "f1",
            "kappa",
            "mcc",
        ]
    y_pred, y_true = unify_cols(y_pred, y_true, "y_pred", "y_true")

    y_model = (
        pd.concat(
            [
                y_true.melt(value_name="y_true").set_index("variable"),
                y_pred.melt(value_name="y_pred").set_index("variable"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"variable": "CV_Iteration"})
    )

    tmp = y_model.groupby("CV_Iteration")[["y_pred", "y_true"]].sum().sum(axis=1)
    # print('Number of tags in each CV_Iteration:', tmp)
    y_model = y_model[y_model["CV_Iteration"].isin(tmp[tmp > 0].index)]
    # plot_confusion_matrix_multi(y_model, map_lbls={0:'N',1:'Y'}, ncol=5)

    yScore = ml_scores(y_model, scores_names, multi_class="ovo", average=average_op).set_index("CV")
    yScore.index.name = "Tag"

    map_dict = {
        "CV_scores_Mean": "macro_avg",
        "CV_scores_STD": "macro_avg_STD",
        "scores_all": "micro_avg",
    }
    yScore = yScore.rename(index=map_dict)
    yScore["Support_number"] = y_model.groupby("CV_Iteration")["y_true"].sum()

    idx = yScore.index.str.contains("_avg")
    tmp = pd.DataFrame(
        yScore[~idx].apply(
            lambda x: np.average(x, weights=yScore.loc[~idx, "Support_number"]), axis=0
        ),
        columns=["weighted_avg"],
    ).T
    yScore = pd.concat([yScore, tmp], axis=0)

    idx = yScore.index.str.contains("_avg")
    yScore.loc[idx, "Support_number"] = yScore.loc[~idx, "Support_number"].sum()
    yScore["Support_number"] = yScore["Support_number"].astype(int)

    yScore_labels = yScore[~idx].sort_values(by=["mcc"], ascending=False)
    yScore_overall = yScore[idx]
    yScore = pd.concat([yScore_labels, yScore_overall], axis=0)

    from sklearn.metrics import accuracy_score

    accuracy_overall = accuracy_score(y_true, y_pred)
    print("Accuracy Score of selecting entire sets of tags: ", round(accuracy_overall, 4))
    model_performance = {
        "yScore": yScore,
        "accuracy_overall": accuracy_overall,
    }

    return model_performance, y_model.rename({"CV_Iteration": "Class"}, axis=1)
