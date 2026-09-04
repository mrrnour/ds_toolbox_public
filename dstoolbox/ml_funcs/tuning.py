"""Hyperparameter tuning: Optuna-style ml_tuner, hyperopt-based search, XGBoost step-wise tuner."""

import ast

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
except ImportError:  # optional dep — fail lazily so importing the package doesn't require hyperopt
    from ._base_import_warning import _MissingDependency

    _hp_missing = _MissingDependency(
        "hyperopt",
        used_by="dstoolbox.ml_funcs.tuning",
        install_hint="pip install 'dstoolbox[ml_tuning]'",
    )
    fmin = tpe = hp = STATUS_OK = Trials = _hp_missing

from typing import Any

from .scores import ml_scores
from .training import ml_prediction


def _apply_trial_suggest(expr: str, trial):
    """Parse and dispatch a ``trial.suggest_*(...)`` string safely.

    Replaces the previous ``eval(par, var_in_model_params, local_vars)``
    call site. Only accepts a single call of shape
    ``trial.suggest_<name>(<literal-args>, <literal-kwargs>)`` and rejects
    anything else with :class:`ValueError`.

    Parameters
    ----------
    expr : str
        The suggest expression, e.g. ``"trial.suggest_int('max_depth', 3, 10)"``.
    trial : optuna.trial.Trial
        Active trial to draw from.

    Returns
    -------
    Any
        The value returned by the matching ``trial.suggest_*`` method.
    """
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"Invalid suggest expression: {expr!r}") from e

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "trial"
        and node.func.attr.startswith("suggest_")
    ):
        raise ValueError(f"Expected 'trial.suggest_*(...)' expression, got: {expr!r}")

    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return getattr(trial, node.func.attr)(*args, **kwargs)


def ml_tuner(
    trial,
    sk_model,
    model_params,
    X,
    y,
    sk_fold,
    var_in_model_params,
    Umetric="auc",
    use_early_Stopping=False,
    early_stopping_rounds=300,
    use_callbacks=False,
):
    """Optuna trial objective: train ``sk_model`` and return cross-validated score.

    Resolves any ``trial.suggest_*`` strings in ``model_params`` against
    ``trial`` (and any names in ``var_in_model_params``), instantiates the
    model, runs cross-validation via :func:`ml_prediction`, and returns
    the mean ``Umetric`` score for Optuna to maximize.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial.
    sk_model : sklearn estimator class or instance
        Class to instantiate with ``model_params``, or an already-built
        estimator (``model_params`` then ignored).
    model_params : dict or None
        Hyperparameters to pass to ``sk_model``. Values that are strings
        starting with ``'trial.suggest_'`` are evaluated to draw a sample.
    X, y : pandas.DataFrame, pandas.Series
        Training features and target.
    sk_fold : sklearn cross-validation splitter
        Forwarded to :func:`ml_prediction`.
    var_in_model_params : dict
        Extra namespace used when ``eval``-ing the suggest expressions.
    Umetric : str, optional
        Score key from ``metric_dict`` to optimize. Default ``'auc'``.
    use_early_Stopping : bool, optional
        Enable early stopping for XGBoost-like models. Default False.
    early_stopping_rounds : int, optional
        Patience for early stopping. Default 300.
    use_callbacks : bool, optional
        Wire up Optuna's pruning callback (XGBoost only). Default False.

    Returns
    -------
    float
        Mean cross-validation score on ``Umetric``.
    """
    import optuna

    ##TODO: it is not working:
    # if 'Pipeline' in  str(type(sk_model)):
    #   model_sub = sk_model.steps[-1][1]
    #   sk_model.steps[-1][1]=model_sub(**model_params)
    #   model = sk_model
    # else:
    #   model = sk_model(**model_params)

    ##TODO: Revise logic:
    if model_params is not None:
        model = sk_model(**model_params)
        ##Umetric is used when use_early_Stopping=True, otherwise it was infered from model_params('eval_metric')
        model_params = {
            key: (
                _apply_trial_suggest(par, trial)
                if isinstance(par, str) and par.startswith("trial.suggest_")
                else par
            )
            for (key, par) in model_params.items()
        }
        print(model_params)
    else:
        model = sk_model

    if use_early_Stopping:
        ##TODO: add all metric to it:
        eval_metric_dict = {
            "auc": "auc",
            "aucpr": "aucpr",
        }

        Umetric = eval_metric_dict.get(model_params.get("eval_metric"))

    if use_callbacks:
        # when len(eval_set)=2:
        observation_key = "validation_1-" + model_params["eval_metric"]
        # when len(eval_set)=1:
        # observation_key="validation_0-"+model_params['eval_metric']

        # Add a callback for pruning.
        pruning_callback = [optuna.integration.XGBoostPruningCallback(trial, observation_key)]

    else:
        pruning_callback = None

    y_model, _ = ml_prediction(
        model,
        X,
        y,
        sk_fold,
        use_early_Stopping=use_early_Stopping,
        early_stopping_rounds=early_stopping_rounds,
        pruning_callback=pruning_callback,
    )

    scores = ml_scores(y_model, [Umetric], multi_class="ovo", average="macro")
    print(scores)
    scores = scores.loc[scores["CV"] == "CV_scores_Mean", Umetric]

    return scores


def hyperparameter_tuning(
    space: dict[str, float | int],
    X: pd.DataFrame,
    y: pd.Series,
    sk_fold,  # [X_test,y_test]
    early_stopping_rounds: int = 50,
    Umetric: callable = accuracy_score,
) -> dict[str, Any]:
    """Hyperopt objective for an ``XGBClassifier``: returns ``-score`` for minimization.

    Parameters
    ----------
    space : dict
        Hyperparameter sample drawn by hyperopt. Keys ``max_depth`` and
        ``reg_alpha`` are coerced to ``int``.
    X, y : pandas.DataFrame, pandas.Series
        Training data.
    sk_fold : sklearn cross-validation splitter
        Passed to :func:`ml_prediction`.
    early_stopping_rounds : int, optional
        Added to the model parameter dict. Default 50.
    Umetric : callable, optional
        Scorer key (string or callable) passed to :func:`ml_scores`.
        Default ``accuracy_score``.

    Returns
    -------
    dict
        ``{'loss': -score, 'status': STATUS_OK, 'model': <XGBClassifier>}``
        in the form hyperopt's ``fmin`` expects.
    """
    from xgboost import XGBClassifier

    int_vals = ["max_depth", "reg_alpha"]
    space = {k: (int(val) if k in int_vals else val) for k, val in space.items()}
    space["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**space)
    y_model, _, df_epochs = ml_prediction(
        model,
        X,
        y,
        sk_fold,
    )

    scores = ml_scores(y_model, [Umetric], multi_class="ovo", average="macro")
    scores_sub = scores.loc[scores["CV"] == "CV_scores_Mean", Umetric]

    return {"loss": -scores_sub, "status": STATUS_OK, "model": model}


def xgb_tuner(X_train, y_train, X_test, y_test, random_state, metric=roc_auc_score, stepWise=True):
    """Step-wise hyperopt search for XGBoost hyperparameters.

    When ``stepWise=True``, runs hyperopt 5 times, each time exploring one
    small group of related parameters (tree depth, stochastic sampling,
    regularization, gamma, learning rate) and feeding the best back into
    the next round (200 evals per round). When ``stepWise=False``, does a
    single 1500-eval run over the full space.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame, pandas.Series
        Training data.
    X_test, y_test : pandas.DataFrame, pandas.Series
        Validation data passed to :func:`hyperparameter_tuning`.
    random_state : int
        Seed pinned in every sampled model.
    metric : callable, optional
        Scorer to maximize. Default :func:`sklearn.metrics.roc_auc_score`.
    stepWise : bool, optional
        Use the staged search (default) or a single broad search.

    Returns
    -------
    tuple
        ``(best_params, hyperopt_trials)``.
    """

    from hyperopt import Trials, fmin, hp, tpe

    params = {"random_state": random_state}
    rounds = [
        {
            "max_depth": hp.quniform("max_depth", 1, 8, 1),  # tree
            "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
        },
        {
            "subsample": hp.uniform("subsample", 0.5, 1),  # stochastic
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        },
        {
            "reg_alpha": hp.uniform("reg_alpha", 0, 10),
            "reg_lambda": hp.uniform("reg_lambda", 1, 10),
        },
        {"gamma": hp.loguniform("gamma", -10, 10)},  # regularization
        {"learning_rate": hp.loguniform("learning_rate", -7, 0)},  # boosting
    ]
    if not stepWise:
        rounds = [
            {
                "max_depth": hp.quniform("max_depth", 1, 8, 1),  # tree
                "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
                "subsample": hp.uniform("subsample", 0.5, 1),  # stochastic
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                "reg_alpha": hp.uniform("reg_alpha", 0, 10),
                "reg_lambda": hp.uniform("reg_lambda", 1, 10),
                "gamma": hp.loguniform("gamma", -10, 10),  # regularization
                "learning_rate": hp.loguniform("learning_rate", -7, 0),
            }  # boosting
        ]
    for round in rounds:
        params = {**params, **round}
        trials = Trials()
        best = fmin(
            fn=lambda space: hyperparameter_tuning(
                space, X_train, y_train, X_test, y_test, metric=metric
            ),
            space=params,
            algo=tpe.suggest,
            max_evals=200 if stepWise else 1500,
            trials=trials,
        )
    params = {**params, **best}

    params["max_depth"] = int(params["max_depth"])

    return params, trials
