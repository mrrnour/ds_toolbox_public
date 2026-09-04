"""General-purpose utilities.

Re-exports the public names of each submodule so callers can use
``from dstoolbox import utils`` and access everything as ``utils.<name>``.
"""

from .dataframes import (
    categorical_to_codes,
    cell_share_of_total,
    clean_product_descriptions,
    compare_dataframes_columns,
    condense_cols,
    dates_to_months_since_min,
    encode_categoricals,
    fill_with_colnames,
    flexible_join,
    join_non_zero,
    merge_between,
    movecol,
    null_per_column,
    percent_agg,
    reduce_mem_usage,
    unify_cols,
)
from .datetime_utils import (
    check_timestamps,
    monthly_first_dates,
    pass_days,
    seconds_to_dhms,
)
from .encoding import (
    fast_get_dummies,
    sparse_label_encoding,
)
from .lists import (
    flatten_list,
    regex_filter_list,
    remove_extra_none,
    unique_list,
)
from .logging_utils import (
    custom_print,
    make_logger,
    redirect_stdio_to_logger,
    setup_logger,
)
from .paths import (
    check_path,
    copy_ymls,
    load_config,
    load_params,
    setup_output_folder,
)
from .plots import (
    DistributionReport,
    PlotConfig,
    categorical_color_map,
    corrmap,
    figures_to_html,
    hist_plot,
    plot_3d_scatter,
    plot_I_MR,
    plot_series_overlay,
    plotly_group_stack,
    sankey,
    save_fig,
    save_plotly_fig,
    stack_plotly_subplots,
    wordcloud_graph,
)
from .sql import (
    parse_sql_file,
    strip_sql_comments,
)
from .stats import (
    analyze_cat_num,
    analyze_categorical_data,
    chi2_contingency_pvalue,
    compare_univariate_features,
    control_limit,
    control_limit_grpby,
    corr_pointbiserial,
    discretizer,
    extract_equation,
    find_high_correlations,
    find_low_variance,
    hypothesis_test,
    hypothesis_test_batch_pars,
    i_mr_ctrl_limits,
    i_mr_sigma_limits,
    interpret_results_analyze_categorical,
    jitter,
    kruskal_wallis_by_group,
    sc_post_gap_test,
    sigma_limit,
    sigma_limit_cols_grpby,
    sigma_limit_grpby,
)
from .text import (
    clean_column_names,
    compare_lists,
    create_venn_diagram,
    find_fuzzy_matches,
    normalize_text,
    retrieve_name,
    rle_encode,
)
