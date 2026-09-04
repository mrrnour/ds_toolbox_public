"""ml_funcs package: ML pipelines, training, scoring, plots, tuning, PCA/CCA, multilabel, regression assumptions, time-series outlier detection/imputation."""

from .assumptions import (
    LinearRegressionAssumptionsChecker,
    ridge_lasso_notes,
)
from .backtest_plots import (
    BacktestReport,
    plot_backtest_splits,
    plot_metrics_and_residuals,
)
from .classifier_report import ProbabilisticClassifierReport
from .feature_importance import (
    feature_importance_batch,
    pdp_plot_batch,
    shap_plots_batch,
)
from .forecasters import (
    AutoArimaSklearn,
    DartsNBEATSSklearn,
    DartsThetaSklearn,
    DropColumns,
    LagRegressor,
    MeanBaseline,
    SeasonalNaive,
    SilverkiteSklearn,
    WindowedForecaster,
    available_backends,
    build_forecaster,
)
from .helpers import (
    compute_class_weights,
)
from .inspection import (
    CAPABILITIES,
    capability_matrix,
    is_forecaster,
    model_capabilities,
    print_capability_matrix,
    task_kind,
)
from .intervention_plots import (
    plot_cumulative_effect,
    plot_cumulative_effect_from_preds,
    plot_cumulative_effect_plain,
    plot_forecast_faceted,
    plot_intervention,
    plot_residual_acf,
)
from .metric_aliases import (
    METRIC_ALIASES,
    map_metric_names,
)
from .mixins import (
    ComponentsMixin,
    IntervalMixin,
    ProbabilisticMixin,
)
from .multilabel import (
    binarize_multilabel_tags,
    evaluate_multilabel,
    split_multilabel_data,
    split_multilabel_data_indices,
)
from .pca import (
    canonical_correlation_analysis,
    cca_batch,
    pc_draw_vectors,
    pca_explained_var,
    pca_important_features,
    pca_ortho_rotation,
    pca_plot,
)
from .performance_plots import (
    learning_curve_early_stopping,
    ml_comparison_plot,
    plot_confusion_matrix_multi,
)
from .scores import (
    metric_dict,
    ml_scores,
    ml_scores_crossvalidate,
)
from .splits import (
    BacktestConfig,
    ExpandingBacktestSplit,
    HoldoutSplit,
    PanelTimeSeriesSplit,
    backtest_split,
    time_series_split_from_config,
)
from .stat_bayes import (
    AGREEING_STATES,
    CALLED_BAYES,
    CALLED_BOTH,
    CALLED_FREQ,
    CALLED_NONE,
    INCONCLUSIVE_PROB_THRESHOLD,
    VERDICTS,
    BestResult,
    BetaBinomialResult,
    BetaPrior,
    RopeDecision,
    best_two_sample,
    beta_binomial_two_sample,
    beta_prior_from_baseline,
    call_agreement,
    is_call,
    is_flagged,
    plot_beta_binomial_report,
    plot_kruschke_report,
    plot_prior_sensitivity,
    plot_rope_decision,
    prior_overlap_table,
    prior_sensitivity,
    prior_sensitivity_verdict,
    rope_comparison_table,
    rope_decision,
    rope_decision_normal,
)
from .stat_bayes_group import (
    ROPE_BANDS,
    GroupCounts,
    GroupEffect,
    PrePostWindow,
    aggregate_counts,
    fit_group_comparison,
    fit_prepost,
    rope_from_control,
    rope_from_control_se,
    split_by_window,
)
from .stat_bayes_group_plots import (
    plot_convergence,
    plot_effect,
    plot_forest,
    plot_prior_forest,
    plot_summary,
    verdict_style,
)
from .stat_bayes_group_tools import (
    matched_sequential_windows,
    prior_forest_rows,
    prior_sensitivity_groups,
    prior_shape_table,
    sequential_scan,
)
from .stat_bayes_hier import (
    DEFAULT_KAPPA_PRIOR,
    HierBetaBinomialFit,
    hier_beta_binomial_fit,
    verdict_without_rope,
)
from .stat_freq import (
    WelchTestResult,
    delta_method_two_sample,
    permutation_welch_two_sample,
    student_t_two_sample,
    welch_t_two_sample,
)
from .templates import (
    classifiers_template,
    regressors_template,
)
from .training import (
    classifier_performance_batch,
    ml_comparison,
    ml_prediction,
    ml_prediction_nested_cv,
    ml_prediction_sub_epochs,
)
from .ts_eda import (
    acf,
    acf_confint,
    adf_test,
    ljung_box,
    missing_summary,
    pacf,
    seasonal_table,
    stationarity_report,
)
from .ts_intervention import (
    InterventionResult,
    effect_from_preds,
    effect_report,
    effect_summary,
    estimate_intervention_effect,
    sc_results_to_backtest_preds,
)
from .ts_outliers import (
    detect_iqr,
    detect_mad,
    detect_outliers,
    detect_rolling_mad,
    detect_rolling_zscore,
    detect_stl_resid,
    detect_zscore,
    impute_ffill_bfill,
    impute_linear,
    impute_outliers,
    impute_rolling_median,
    impute_seasonal_mean,
    impute_stl_recon,
    impute_time,
    mask_anomalies,
    replace_outliers,
)
from .ts_plots import (
    lag_plot,
    plot_acf,
    plot_ccf,
    plot_eda_overview,
    plot_pacf,
    plot_paired_acf,
    plot_per_day_delta_bar,
    plot_prewhitening_diagnostic,
    plot_seasonality_box,
    plot_series,
    plot_vbh_per_season,
)
from .ts_seasonality import (
    acf_top_periods,
    detect_seasonality,
    friedman_seasonality_test,
    periodogram_top_periods,
    stl_decompose,
    stl_seasonal_strength,
)
from .ts_trend import (
    AcfRegime,
    MbbDeltaResult,
    MKAdaptiveCoreArmResult,
    MKAdaptiveCoreResult,
    RegionalHomogeneityResult,
    SeasonalTrendResult,
    TrendResult,
    VbhBranchResult,
    VbhDecomposition,
    classify_acf_regime,
    correlated_seasonal_mk,
    deseason,
    lag1_acf,
    mk_3pw,
    mk_adaptive_core,
    mk_adaptive_core_arm,
    mk_adaptive_mbb,
    mk_hamed_rao,
    mk_original,
    mk_pw,
    mk_tfpw,
    mk_vbh,
    mk_yue_wang,
    partial_mk,
    per_day_delta_slopes,
    regional_homogeneity,
    seasonal_mk,
    sen_slope_ci,
    vbh_chi2_decomposition,
)
from .tuning import (
    hyperparameter_tuning,
    ml_tuner,
    xgb_tuner,
)
