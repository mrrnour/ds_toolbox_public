"""General-purpose utilities.

Re-exports the public names of each submodule so callers can use
``from dstoolbox import utils`` and access everything as ``utils.<name>``.
"""

from .lists import (
    regex_filter_list,
    flatten_list,
    unique_list,
    remove_extra_none,
)

from .text import (
    normalize_text,
    clean_column_names,
    find_fuzzy_matches,
    create_venn_diagram,
    compare_lists,
    retrieve_name,
    rle_encode,
)

from .datetime_utils import (
    check_timestamps,
    pass_days,
    seconds_to_dhms,
    monthly_first_dates,
)

from .paths import (
    check_path,
    copy_ymls,
    setup_output_folder,
    load_config,
    load_params,
)

from .logging_utils import (
    make_logger,
    custom_print,
    redirect_stdio_to_logger,
    setup_logger,
)

from .sql import (
    parse_sql_file,
    strip_sql_comments,
)

from .encoding import (
    fast_get_dummies,
    sparse_label_encoding,
)

from .dataframes import (
    movecol,
    merge_between,
    cell_share_of_total,
    compare_dataframes_columns,
    categorical_to_codes,
    reduce_mem_usage,
    null_per_column,
    unify_cols,
    percent_agg,
    fill_with_colnames,
    join_non_zero,
    clean_product_descriptions,
    condense_cols,
    encode_categoricals,
    flexible_join,
    dates_to_months_since_min,
)

from .stats import (
    compare_univariate_features,
    hypothesis_test,
    hypothesis_test_batch_pars,
    find_low_variance,
    kruskal_wallis_by_group,
    chi2_contingency_pvalue,
    corr_pointbiserial,
    find_high_correlations,
    discretizer,
    jitter,
    extract_equation,
    analyze_categorical_data,
    interpret_results_analyze_categorical,
    analyze_cat_num,
    sigma_limit,
    sigma_limit_grpby,
    sigma_limit_cols_grpby,
    i_mr_sigma_limits,
    control_limit,
    control_limit_grpby,
    i_mr_ctrl_limits,
    sc_post_gap_test,
)

from .plots import (
    corrmap,
    sankey,
    wordcloud_graph,
    plot_3d_scatter,
    categorical_color_map,
    plotly_group_stack,
    stack_plotly_subplots,
    PlotConfig,
    DistributionReport,
    figures_to_html,
    save_plotly_fig,
    save_fig,
    hist_plot,
    plot_I_MR,
    plot_series_overlay,
)
