"""Binary classifier probability diagnostic plots, bundled as a single report.

A :class:`ProbabilisticClassifierReport` carries one ``(y, prob, pos_label)``
triple and exposes the four canonical diagnostic plots as methods:

* :meth:`plot_gain_lift` — gain + lift charts (resource-allocation tool)
* :meth:`plot_precision_recall` — PR curve with threshold ticks
* :meth:`plot_roc` — ROC curve with threshold ticks
* :meth:`plot_reliability` — calibration curve (raw + normalized)

This replaces the standalone ``gainNlift`` / ``precision_recall_curve2`` /
``roc_curve2`` / ``reliability_diagram`` free functions that used to live in
:mod:`dstoolbox.ml_funcs.performance_plots`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ProbabilisticClassifierReport:
    """Probability-diagnostic plots for one binary classifier.

    Parameters
    ----------
    y : pd.Series
        Ground-truth labels.
    prob : pd.Series
        Predicted probabilities for the positive class.
    pos_label : int | str, default 1
        Value in ``y`` treated as the positive class.
    """

    y: pd.Series
    prob: pd.Series
    pos_label: int | str = 1

    # ------------------------------------------------------------------
    # Gain & lift
    # ------------------------------------------------------------------
    def plot_gain_lift(
        self,
        group_no: int = 25,
        outputFile: tuple[str, str] | list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gain + lift charts. Returns ``(summary_df, gain_chart_df, lift_chart_df)``.

        ``outputFile`` is a 2-element sequence of paths (gain, lift). Pass
        ``None`` to display only.
        """
        df = pd.concat([self.y, self.prob], axis=1)
        df.sort_values(by=df.columns[1], ascending=False, inplace=True)

        def gain_stp1(subset):
            pos_event = sum(subset[self.y.name] == self.pos_label)
            return len(subset), pos_event

        tmp = list(map(gain_stp1, np.array_split(df, group_no)))
        out = pd.DataFrame(tmp, columns=['case', 'event'])
        out['event%'] = out['event'] / out['event'].sum() * 100
        out['cum_case'] = out['case'].cumsum()
        out['cum_case%'] = out['cum_case'] / out['case'].sum() * 100
        out['gain'] = out['event%'].cumsum()
        out['cum_lift'] = out['gain'] / out['cum_case%']

        row_no = int(out.shape[0])

        df_gain_chart = pd.DataFrame(
            out['gain'].tolist() + out['cum_case%'].tolist(),
            columns=['values'],
        )
        df_gain_chart['x'] = pd.Series(out['cum_case%'].tolist() * 2)
        df_gain_chart['selection method'] = pd.Series(['model'] * row_no + ['random'] * row_no)
        df_gain_chart = pd.concat(
            [df_gain_chart,
             pd.DataFrame.from_dict({'values': [0, 0], 'x': [0, 0],
                                     'selection method': ['model', 'random']})],
            ignore_index=True,
        )

        df_lift_chart = pd.DataFrame(
            out['cum_lift'].tolist() + [1] * row_no, columns=['values'],
        )
        df_lift_chart['x'] = pd.Series(out['cum_case%'].tolist() * row_no)
        df_lift_chart['selection method'] = pd.Series(['model'] * row_no + ['random'] * row_no)

        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        uPlot1 = sns.lineplot(
            data=df_gain_chart, ax=ax[0], x='x', y='values',
            hue='selection method', style='selection method', markers=True,
        )
        uPlot1.set(xlabel='', ylabel='% of events')
        ax[0].set_title('Gain Chart')

        uPlot2 = sns.lineplot(
            data=df_lift_chart, ax=ax[1], x='x', y='values',
            hue='selection method', style='selection method', markers=True,
        )
        uPlot2.set(xlabel='% 0f data sets', ylabel='Lift')
        ax[1].set_title('Lift Chart')

        plt.ylim(0, int(df_lift_chart['values'].max() + 1))
        plt.xlim(0, 100)

        if outputFile is not None:
            uPlot1.get_figure().savefig(outputFile[0], bbox_inches='tight')
            uPlot2.get_figure().savefig(outputFile[1], bbox_inches='tight')
            plt.close('all')

        return out, df_gain_chart, df_lift_chart

    # ------------------------------------------------------------------
    # Precision-Recall
    # ------------------------------------------------------------------
    def plot_precision_recall(
        self, outputFile: str | None = None, **kwargs,
    ) -> tuple[pd.DataFrame, list[int]]:
        """Precision-recall curve with threshold-annotated x-ticks. Returns ``(df, tick_idx)``."""
        from sklearn.metrics import precision_recall_curve, auc

        plt.clf()
        model_precision, model_recall, thresholds = precision_recall_curve(
            y_true=self.y, probas_pred=self.prob, pos_label=self.pos_label, **kwargs,
        )
        model_auc_rp = auc(model_recall, model_precision)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(True, which='major', linestyle='--', alpha=0.7, color='gray')
        ax.grid(True, which='minor', linestyle=':', alpha=0.4, color='gray')
        ax.minorticks_on()
        ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

        df_rp = pd.DataFrame({
            'Precision': model_precision[:-1],
            'Recall': model_recall[:-1],
            'thresholds': thresholds,
        })
        ax.plot(df_rp['Recall'], df_rp['Precision'],
                marker='o', markersize=4, alpha=0.1, linestyle='-', linewidth=1)
        ax.set_title('Precision Recall Curve', pad=20, fontsize=12)
        ax.set_xlabel('Recall/(threshold)', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)

        no_skill = len(self.y[self.y == 1]) / len(self.y)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
        ax.text(0.9, no_skill, 'No skill line', color='black', fontsize=10)

        df_rp_unique = df_rp.drop_duplicates(subset=['Recall']).reset_index()
        interval_no = min(15, df_rp_unique.shape[0])
        idx = list(np.linspace(
            df_rp_unique.index.min(), df_rp_unique.index.max(),
            interval_no, endpoint=True, dtype='int',
        ))
        recall_values = df_rp_unique.iloc[idx]['Recall']
        threshold_values = df_rp_unique.iloc[idx]['thresholds'].round(3).astype(str)
        ax.set_xticks(recall_values)
        ax.set_xticklabels(
            [f"{round(x, 2)}(t={y})" for x, y in zip(recall_values, threshold_values)],
            rotation=90,
        )
        ax.annotate(
            f'ROC of Precision Recall curve={model_auc_rp:.3f}',
            xy=(0.4, 0), xycoords='axes fraction',
            xytext=(-20, 25), textcoords='offset pixels',
            horizontalalignment='right', verticalalignment='bottom', fontsize=10,
        )
        plt.tight_layout()

        if outputFile is not None:
            plt.savefig(outputFile, bbox_inches='tight', dpi=300)
            plt.show()

        return df_rp, idx

    # ------------------------------------------------------------------
    # ROC
    # ------------------------------------------------------------------
    def plot_roc(
        self, outputFile: str | None = None, **kwargs,
    ) -> tuple[pd.DataFrame, float]:
        """ROC curve with threshold-annotated x-ticks. Returns ``(df, auc)``."""
        from sklearn.metrics import roc_auc_score, roc_curve

        model_auc = roc_auc_score(y_true=self.y, y_score=self.prob, **kwargs)
        model_fpr, model_tpr, thresholds = roc_curve(
            y_true=self.y, y_score=self.prob, pos_label=self.pos_label,
        )
        thresholds[0] = 1
        df_roc = pd.DataFrame(
            [model_fpr, model_tpr, thresholds],
            index=['False_Positive_Rate', 'True_Positive_Rate', 'thresholds'],
        ).T

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        plt.plot(model_fpr, model_tpr, marker='.', label='Model')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        ax.legend(loc='upper right', frameon=True)
        ax.annotate(
            'ROC AUC=%.3f (random selection=.5)' % (model_auc),
            xy=(1, 0), xycoords='axes fraction',
            xytext=(-20, 20), textcoords='offset pixels',
            horizontalalignment='right', verticalalignment='bottom',
        )

        df_roc_tmp = df_roc.drop_duplicates(subset=['False_Positive_Rate']).reset_index()
        interval_no = min(15, df_roc_tmp.shape[0])
        idx = list(np.linspace(
            df_roc_tmp.index.min(), df_roc_tmp.index.max(),
            interval_no, endpoint=True, dtype='int',
        ))
        plt.xticks(df_roc_tmp.iloc[idx]['False_Positive_Rate'])
        ticks_loc = ax.get_xticks().tolist()
        threshs = df_roc_tmp.iloc[idx]['thresholds'].round(3).astype(str)
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(
            [str(round(x, 2)) + "(t=" + y + ")" for x, y in zip(ticks_loc, threshs)],
        )
        plt.xticks(rotation=90)

        if outputFile is not None:
            plt.savefig(outputFile, bbox_inches='tight')
            plt.close('all')

        return df_roc, model_auc

    # ------------------------------------------------------------------
    # Reliability / calibration
    # ------------------------------------------------------------------
    def plot_reliability(
        self, outputFile: str | None = None, **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reliability diagram (raw + normalized). Returns the 4 calibration arrays."""
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(
            y_true=self.y, y_prob=self.prob, n_bins=50, normalize=False, **kwargs,
        )
        prob_true_norm, prob_pred_norm = calibration_curve(
            y_true=self.y, y_prob=self.prob, n_bins=50, normalize=True, **kwargs,
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot([0, 1], [0, 1])
        plt.plot(prob_pred_norm, prob_true_norm, label='Normlized')
        plt.plot(prob_pred, prob_true, label='Original')
        plt.grid()
        plt.xlabel("Average probability")
        plt.ylabel("Fraction of positive")
        plt.title("Reliability diagram")
        ax.legend(loc='upper right', frameon=True)

        if outputFile is not None:
            plt.savefig(outputFile, bbox_inches='tight')
            plt.close('all')

        return prob_true, prob_pred, prob_true_norm, prob_pred_norm
