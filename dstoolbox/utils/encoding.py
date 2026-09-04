"""Sparse / multi-label encoding helpers (faster alternatives to pandas.get_dummies)."""

import numpy as np
import pandas as pd


def _get_dummies2(x, allLabels):
    """Internal helper for :func:`fast_get_dummies`."""
    ind = np.in1d(allLabels, x) * 1
    return ind


def _sparseLabel(x, splitter="; "):
    """Internal helper for :func:`sparse_label_encoding`."""
    # TODO: how we can apply in two mat in parallel
    try:
        tmp = x[1].split(splitter)
        x2 = x[2:]
        x2[np.where(x2)] = np.fromstring(x[0], dtype=int, sep=splitter)
    except Exception as err:
        raise RuntimeError(f"_sparseLabel failed on row: {x!r}") from err

    # if len(set(tmp))!=len(tmp):
    #     tmp3=pd.Series(data=np.fromstring(x[0], dtype=int, sep=splitter)  , index=tmp)
    #     tmp3=tmp3.groupby(tmp3.index).sum()
    #     x2[np.where(x2)]=tmp3
    # else:
    #     x2[np.where(x2)]=np.fromstring(x[0], dtype=int, sep=splitter)
    return x2


def fast_get_dummies(df, splitter="; "):
    """Fast alternative to ``pandas.get_dummies`` for a delimited-string column.

    Explodes ``df`` (a Series of ``splitter``-joined tokens) into a dense
    one-hot indicator matrix. Designed to sidestep the memory blow-ups
    ``pandas.get_dummies`` triggers on large frames with many unique
    tokens per row.

    Parameters
    ----------
    df : pandas.Series
        String Series where each cell is a ``splitter``-delimited list
        of labels.
    splitter : str, optional
        Delimiter between labels within each cell. Default ``'; '``.

    Returns
    -------
    pandas.DataFrame
        One-hot indicator frame indexed like ``df``; columns are the
        sorted unique labels.
    """

    df = df.str.split(splitter, expand=True)
    tmp = df.values.flatten()
    tmp = tmp[~(pd.isnull(tmp))]
    allLabels = np.sort(np.unique(tmp))
    tmp1 = np.apply_along_axis(_get_dummies2, 1, df.values, allLabels=allLabels)
    tmp1 = pd.DataFrame(tmp1, columns=allLabels, index=df.index)
    return tmp1


def sparse_label_encoding(df, prodCol, priceCol, splitter="; "):
    """One-hot expand ``prodCol`` and fill each active indicator with the matching value from ``priceCol``.

    Both ``df[prodCol]`` and ``df[priceCol]`` must be strings of the same
    ``splitter``-delimited length so their tokens line up positionally.
    The resulting frame has one column per unique product token; each
    cell is that row's price for the product (or 0 if the product is
    absent from the row).

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    prodCol : str
        Column of ``splitter``-joined product tokens.
    priceCol : str
        Column of ``splitter``-joined numeric strings; must be positionally
        aligned with ``prodCol``.
    splitter : str, optional
        Delimiter used inside both columns. Default ``'; '``.

    Returns
    -------
    pandas.DataFrame
        Wide numeric frame (``dtype=int16``) indexed like ``df``; columns
        are uppercased unique product tokens.

    Examples
    --------
    ``prodCol='INT10; SPN; TMN; INT20'`` with ``priceCol='10; 20; 30; 5'``
    produces columns ``[INT10, SPN, TMN, INT20]`` with values
    ``[10, 20, 30, 5]`` in that row.
    """
    tmp1 = fast_get_dummies(df[prodCol], splitter)
    print("dummies generated")

    tmp2 = pd.concat([df[priceCol], df[prodCol], tmp1], axis=1)
    out0 = np.apply_along_axis(_sparseLabel, 1, tmp2, splitter=splitter)

    out = pd.DataFrame(out0, dtype=np.int16, columns=tmp1.columns.str.upper(), index=df.index)
    return out
