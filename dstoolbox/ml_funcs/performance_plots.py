"""Plotting helpers for ML evaluation: comparison boxplots, learning curves, gain/lift, PR/ROC, confusion matrix."""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

#: Default seaborn style applied per-plot (not at module import). Override by
#: passing ``style=None`` to plotting functions, or by calling ``sns.set_style``
#: yourself before invoking them.
_DEFAULT_SNS_STYLE = "darkgrid"
_DEFAULT_FIGSIZE = (30, 20)


def _apply_style(style: str | None = _DEFAULT_SNS_STYLE) -> None:
    """Apply seaborn style only when the caller wants it. No-op if ``style is None``."""
    if style is not None:
        sns.set_style(style)


def ml_comparison_plot(metrics_all, outputFile=None):
    """Generate a comparison box plot for machine learning model metrics.

    Parameters
    ----------
    metrics_all : pandas.DataFrame
        DataFrame containing the metrics for different models. Expected
        columns include ``'CV'``, ``'model'``, ``'elapsed_time'``,
        ``'Feature_nos'``, and one column per metric.
    outputFile : str, optional
        Path to save the output plot. If ``None``, the plot is not saved.
        Default is ``None``.

    Returns
    -------
    None
        Displays the plot and optionally saves it to ``outputFile``.

    Notes
    -----
    - Rows where ``'CV'`` is ``"CV_scores_Mean"``, ``"CV_scores_STD"``, or
      ``"scores_all"`` are filtered out.
    - If a ``'model'`` column is present it is used as the hue; otherwise
      ``'CV'`` is used.
    - Uses a seaborn boxplot with rotated x-axis labels and no outliers.
    """
    _apply_style()
    df_tmp = metrics_all.loc[
        ~metrics_all["CV"].isin(["CV_scores_Mean", "CV_scores_STD", "scores_all"]), :
    ]

    if "model" in df_tmp.columns.tolist():
        df_tmp = df_tmp.drop("CV", axis=1)
        hue = id_vars = "model"
    else:
        id_vars = "CV"
        hue = "scores"
    ucols = [col for col in df_tmp if col not in ["elapsed_time", "Feature_nos"]]
    df_long = pd.melt(df_tmp[ucols], id_vars=[id_vars], var_name=["scores"])
    # sns.set_style("darkgrid")
    plt.figure(figsize=(25, 15))

    uplot = sns.boxplot(
        x="scores",
        y="value",
        hue=hue,
        data=df_long,
        # orient='h',  ##it takes forever
        showfliers=False,
    )

    uplot.set_xticklabels(uplot.get_xticklabels(), rotation=90)
    uplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    uplot.grid()

    if outputFile is not None:
        # graphfile=os.path.join(outputFile,'compare_models.png')
        logger.info("plot save in %s", outputFile)
        plt.savefig(outputFile)
        plt.show()
        plt.close()


def plot_confusion_matrix_multi(y_model, map_lbls, outputFile=None, ncol=3, all_data_flag=True):
    """
    Plots confusion matrices for model predictions, optionally saving the plot to a file.
    """
    ##y_model=pd.concat([y_true, y_pred],axis=1)

    from sklearn.metrics import confusion_matrix

    y_model1 = y_model.copy()

    if all_data_flag:
        ###TODO: use it for x-validation and not for multilabel
        if y_model1["CV_Iteration"].nunique() != 1:
            y_model1["CV_Iteration"] = "cv_" + y_model1["CV_Iteration"].astype(str)
            y_model_all = y_model.copy()

            y_model_all["CV_Iteration"] = "All_data"
            y_model1 = pd.concat([y_model1, y_model_all], axis=0)

        fig_size = (25, 17)
    else:
        y_model1["CV_Iteration"] = "All_data"
        ncol = 1
        fig_size = (10, 5)

    # print(y_model1)
    confMats = pd.Series([], dtype=object)
    # confMats=pd.Series([],index=y_model['CV_Iteration'].unique())

    fig, axs = plt.subplots(
        math.ceil(y_model1["CV_Iteration"].nunique() / ncol), ncol, figsize=fig_size
    )

    axs = np.array([axs]) if ncol == 1 else axs

    for cont, (cv, y_model_sub) in enumerate(y_model1.groupby(["CV_Iteration"])):
        cv = cv[0] if isinstance(cv, tuple) else cv

        y_true = y_model_sub[["y_true"]]
        y_pred = y_model_sub[["y_pred"]]

        confMat = pd.DataFrame(confusion_matrix(y_true, y_pred))
        confMat = confMat.rename(columns=map_lbls).rename(map_lbls, axis=1).rename(map_lbls, axis=0)
        confMat.index.name = "True label"
        confMat.columns.name = "Predicted label"

        confMats[cv] = confMat
        uPlot = sns.heatmap(
            ax=axs.flatten()[cont], data=confMat, annot=True, cmap="YlGnBu", fmt="g", cbar=False
        )

        axs.flatten()[cont].set_title(f"{cv}")
        fig.tight_layout()

    if outputFile is not None:
        figure = uPlot.get_figure()
        figure.savefig(outputFile, bbox_inches="tight")
        plt.close("all")

    return confMats


# ===== public-only extensions (preserved on vendor merge) =====


def learning_curve_early_stopping(df_epochs, outputFile=None):
    """
    Plots the learning curve with early stopping for XGBoost models.
    This function visualizes the training and validation performance over epochs
    for cross-validation iterations, highlighting the point of early stopping.
    Parameters:
    df_epochs (pd.DataFrame): DataFrame containing the epochs data. It should include columns for epochs,
                  CV_Iteration, and best_ntree, along with performance metrics for training
                  and validation.
    outputFile (str, optional): Path to save the output plot. If None, the plot is not saved. Default is None.
    Returns:
    None
    """
    ###https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/

    ##TODO: it is only for xgb now (best_ntree)
    cols = df_epochs.columns[~df_epochs.columns.str.contains("Validation_|Train_")].tolist()
    df_epochs_melted = df_epochs.melt(id_vars=cols)
    uPlot = sns.relplot(
        data=df_epochs_melted,
        y="value",
        x="epochs",
        col="CV_Iteration",
        hue="variable",
        style="variable",
        kind="line",
        #             markers=True,
        palette=["green", "black"],
        col_wrap=3,
    )

    axes = uPlot.axes.flatten()

    sns.set(rc={"figure.figsize": (60, 30)})
    for con, ax in enumerate(axes):
        data_tmp = df_epochs_melted[df_epochs_melted["CV_Iteration"] == con]
        xc = data_tmp.loc[data_tmp["best_ntree"], "epochs"]
        ax.axvline(xc.iloc[0], ls="-", linewidth=3, color="red", alpha=0.75)

    if outputFile is not None:
        figure = uPlot.get_figure()
        # ,"learning_curve.png")
        figure.savefig(outputFile, bbox_inches="tight")
        plt.close("all")
