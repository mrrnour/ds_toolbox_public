"""ml_funcs package: ML pipelines, training, scoring, plots, tuning, PCA/CCA, multilabel, regression assumptions, time-series outlier detection/imputation."""

from .scores import (
    metric_dict,
    ml_scores,
    ml_scores_crossvalidate,
)

from .templates import (
    classifiers_template,
    regressors_template,
)

from .training import (
    ml_prediction_sub_epochs,
    ml_prediction,
    ml_comparison,
    classifier_performance_batch,
    ml_prediction_nested_cv,
)

from .performance_plots import (
    ml_comparison_plot,
    learning_curve_early_stopping,
    plot_confusion_matrix_multi,
)

from .classifier_report import ProbabilisticClassifierReport

from .feature_importance import (
    feature_importance_batch,
    pdp_plot_batch,
    shap_plots_batch,
)

from .tuning import (
    ml_tuner,
    hyperparameter_tuning,
    xgb_tuner,
)

from .pca import (
    pca_plot,
    pca_explained_var,
    pca_ortho_rotation,
    pca_important_features,
    pc_draw_vectors,
    cca_batch,
    canonical_correlation_analysis,
)

from .assumptions import (
    LinearRegressionAssumptionsChecker,
    ridge_lasso_notes,
)

from .helpers import (
    compute_class_weights,
)

from .multilabel import (
    binarize_multilabel_tags,
    split_multilabel_data_indices,
    split_multilabel_data,
    evaluate_multilabel,
)

from .inspection import (
    CAPABILITIES,
    capability_matrix,
    is_forecaster,
    model_capabilities,
    print_capability_matrix,
    task_kind,
)

from .splits import (
    BacktestConfig,
    ExpandingBacktestSplit,
    HoldoutSplit,
    PanelTimeSeriesSplit,
    time_series_split_from_config,
)

from .mixins import (
    ComponentsMixin,
    IntervalMixin,
    ProbabilisticMixin,
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
)

from .ts_intervention import (
    InterventionResult,
    effect_from_preds,
    effect_summary,
    estimate_intervention_effect,
    sc_results_to_backtest_preds,
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

from .ts_seasonality import (
    acf_top_periods,
    detect_seasonality,
    friedman_seasonality_test,
    periodogram_top_periods,
    stl_decompose,
    stl_seasonal_strength,
)

from .ts_outliers import (
    detect_outliers,
    detect_iqr,
    detect_mad,
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
    replace_outliers,
)

from .ts_trend import (
    PairedSlopeResult,
    RegionalHomogeneityResult,
    SeasonalTrendResult,
    TrendResult,
    mk_3pw,
    mk_hamed_rao,
    mk_original,
    mk_pw,
    mk_tfpw,
    mk_yue_wang,
    paired_slope_test,
    paired_slope_test_3pw,
    paired_slope_test_ar1,
    paired_slope_test_boot,
    regional_homogeneity,
    seasonal_mk,
    sen_slope_ci,
)

from .ts_plots import (
    plot_acf,
    plot_ccf,
    plot_eda_overview,
    plot_pacf,
    plot_seasonality_box,
    plot_series,
    lag_plot,
)

from .backtest_plots import (
    BacktestReport,
    plot_backtest_splits,
)

from .intervention_plots import (
    plot_cumulative_effect,
    plot_cumulative_effect_from_preds,
    plot_forecast_faceted,
    plot_intervention,
    plot_residual_acf,
)
