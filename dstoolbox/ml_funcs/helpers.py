"""Misc ML helpers (class weighting)."""

import numpy as np
import pandas as pd


def compute_class_weights(class_weight, y):
    """Build a per-sample weight series from a class-weight spec.

    Parameters
    ----------
    class_weight : dict, ``'balanced'``, or None
        Weights associated with classes in the form
        ``{class_label: weight}``. ``'balanced'`` computes weights
        inversely proportional to class frequency. ``None`` returns a
        vector of ones.
    y : pandas.Series
        Target labels of shape ``[n_samples]``.

    Returns
    -------
    pandas.Series
        Per-sample weights aligned to ``y``.

    Notes
    -----
    Passes ``classes=`` and ``y=`` as keyword arguments to
    ``sklearn.utils.class_weight.compute_class_weight``, which is
    required by scikit-learn >= 1.0.
    """
    if class_weight == 'balanced':
        from sklearn.utils import class_weight as _sk_cw
        tmp = np.round(_sk_cw.compute_class_weight(
            'balanced',
            classes=np.unique(y.sort_values()),
            y=y,
        ), 2)
        class_weight_map = dict(zip(y.sort_values().unique().tolist(), tmp))
        return y.map(class_weight_map)
    if class_weight is None:
        return pd.Series(np.tile(1, y.size))
    return class_weight
