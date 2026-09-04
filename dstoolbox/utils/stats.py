"""Statistical analysis: feature selection, hypothesis tests, correlations,
discretization, categorical/numeric analyses, plus Statistical Process
Control (sigma limits and I-MR control limits).
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def compare_univariate_features(X, y, univar_fea_lst):
    """Compare several univariate feature-selection scores side-by-side and plot them.

    Runs each method in ``univar_fea_lst`` on ``(X, y)``, melts the
    results into long form, and renders a seaborn scatter plot of
    p-values per feature with a dashed reference line at p=0.05.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    univar_fea_lst : list of str
        Names of sklearn callables (e.g. ``'mutual_info_classif'``,
        ``'chi2'``, ``'f_classif'``, ``'SelectFdr'``, ``'SelectFwe'``).
        These are resolved via ``eval``.

    Returns
    -------
    pandas.DataFrame
        Wide-format scores frame, one row per feature, one column per
        method.
    """
    ##TODO: refactor it:
    from sklearn.feature_selection import (
        SelectFdr,
        SelectFpr,
        SelectFwe,
        SelectKBest,
        SelectPercentile,
        chi2,
        f_classif,
        mutual_info_classif,
        mutual_info_regression,
    )

    _UNIVAR_REGISTRY = {
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
        "chi2": chi2,
        "SelectPercentile": SelectPercentile,
        "SelectFpr": SelectFpr,
        "SelectFdr": SelectFdr,
        "SelectFwe": SelectFwe,
        "f_classif": f_classif,
    }

    arr = np.empty((0, X.shape[1]), float)
    for univar in univar_fea_lst:
        print(univar)
        try:
            uFunc = _UNIVAR_REGISTRY[univar]
        except KeyError as e:
            raise ValueError(
                f"Unknown univariate feature selector {univar!r}; "
                f"expected one of {sorted(_UNIVAR_REGISTRY)}."
            ) from e
        score = uFunc(X, y)
        if univar in ["mutual_info_classif", "mutual_info_regression"]:
            score = score
        elif univar in ["chi2"]:
            score = score[1]
        elif univar in ["SelectFdr", "SelectFdr", "SelectFwe", "f_classif"]:
            ###TODO: correct it:
            selector = SelectKBest(uFunc, k="all")
            selector.fit(X, y)
            score = selector.pvalues_
        #     cols = selector.get_support(indices=True)

        if score is None:
            score = np.empty(X.shape[1])
            score[:] = np.nan
        arr = np.append(arr, [score], axis=0)
    scores = pd.DataFrame(data=arr, index=univar_fea_lst, columns=X.columns)
    scores_long = pd.melt(
        scores.T.dropna(how="all").reset_index().rename(columns={"index": "feature"}),
        id_vars=["feature"],
        value_name="P_value",
        var_name="Feature_selection_Method",
    )

    fig, ax = plt.subplots(figsize=(25, 15))
    uplot = sns.scatterplot(
        y="feature",
        x="P_value",
        hue="Feature_selection_Method",
        style="Feature_selection_Method",
        size="Feature_selection_Method",
        data=scores_long,
        ax=ax,
    )
    cutter_values = [0.05]
    for con, xl in enumerate(cutter_values):
        ax.axvline(x=xl, color="red", linestyle="--")
        ax.text(xl, con + 5, f"P_value={xl}", rotation=90, size=10)

    ## plt.xticks(rotation=90)
    plt.show()
    plt.close()

    return scores.T


def hypothesis_test(
    df,
    par,
    group,
    group_names,
):
    """Welch's t-test on a numeric column split by a boolean group, with a worded summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame; ``df[group]`` is coerced to bool.
    par : str
        Numeric column to compare.
    group : str
        Boolean group column.
    group_names : sequence of str of length 2
        Display names for the two groups (True, False).

    Returns
    -------
    tuple
        ``(des, res, summary)`` — descriptive stats, raw t-test results
        table (with a ``summary`` row), and a 1-cell summary frame
        labelling the comparison ``'lower'``, ``'higher'``, or
        ``'no difference'``.
    """
    import researchpy as rp

    df[group] = df[group].astype("bool")
    X1 = df[par][df[group]]
    X2 = df[par][~df[group]]

    group1_name, group2_name = group_names[0], group_names[1]
    des, res = rp.ttest(
        X1,
        X2,
        group1_name=group1_name,
        group2_name=group2_name,
        equal_variances=False,
        paired=False,
        # correction= None
    )
    res = res.set_index(res.columns[0])
    res.columns = [par]

    if res.loc["Two side test p value = "][0] != 0:
        txt = f"{par}: There is no difference between {group1_name} and {group2_name}"
        txt2 = "no difference"
    elif (res.loc["Two side test p value = "][0] == 0) & (
        res.loc["Difference < 0 p value = "][0] == 0
    ):
        txt = f"{par}: {group1_name} is lower "  # than {group2_name}"
        txt2 = "lower"
    elif (res.loc["Two side test p value = "][0] == 0) & (
        res.loc["Difference > 0 p value = "][0] == 0
    ):
        txt = f"{par}: {group1_name} is higher"  # than {group2_name}"
        txt2 = "higher"
    else:
        txt2 = txt = ""

    res.loc["summary"] = txt
    #   print(txt)

    summary = pd.DataFrame(txt2, index=[par], columns=[group1_name])
    #   print(summary)
    return des, res, summary


def hypothesis_test_batch_pars(df, pars, group, group_names):
    """Run :func:`hypothesis_test` over many numeric columns and stack the results.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    pars : list of str
        Numeric columns to compare.
    group : str
        Boolean group column.
    group_names : sequence of str of length 2
        Display names for the two groups.

    Returns
    -------
    tuple
        ``(stats, tsts, summary)`` — concatenated descriptive stats, raw
        test tables, and one-row-per-parameter worded summary.
    """
    tsts = pd.DataFrame()
    stats = pd.DataFrame()
    summary = pd.DataFrame()
    for par in pars:
        par1_stats_tmp, par1_test_tmp, summary_tmp = hypothesis_test(
            df, par=par, group=group, group_names=group_names
        )
        stats = pd.concat(
            [stats, par1_stats_tmp],
            axis=0,
            # keys=[par]
        )

        tsts = pd.concat([tsts, par1_test_tmp], axis=1)

        summary = pd.concat(
            [summary, summary_tmp],
            axis=0,
            #                     keys=[par]
        )
    return stats, tsts, summary


def find_low_variance(df, thresh=0.0):
    """Return names of columns whose variance is at most ``thresh``.

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric frame.
    thresh : float, optional
        Variance cutoff (inclusive). Default ``0.0`` (only constant
        columns).

    Returns
    -------
    list of str
        Column names meeting the cutoff.
    """

    variance = df.var(skipna=True)
    low_variance = list(variance[variance <= thresh].index)
    return low_variance


def kruskal_wallis_by_group(x, y):
    """Kruskal-Wallis H-test on ``x`` split by group labels ``y``.

    Convenience wrapper over ``scipy.stats.mstats.kruskalwallis`` that
    accepts a two-column layout (values + group labels) rather than
    pre-split arrays.

    Parameters
    ----------
    x : pandas.Series
        Observations.
    y : pandas.Series
        Group labels aligned with ``x``.

    Returns
    -------
    scipy.stats.mstats.KruskalResult
        Named tuple ``(statistic, pvalue)``.
    """
    groupednumbers = {}
    for grp in y.unique():
        groupednumbers[grp] = x.values[y == grp]
    args = groupednumbers.values()
    tmp = stats.mstats.kruskalwallis(*args)
    return tmp


def chi2_contingency_pvalue(x, y):
    """Return the p-value of a chi-square test of independence between ``x`` and ``y``.

    Builds a contingency table with ``pd.crosstab``, runs
    ``scipy.stats.chi2_contingency``, and returns only the p-value.
    Empty crosstabs return ``None``; test errors return ``0``.

    Parameters
    ----------
    x, y : pandas.Series
        Aligned categorical variables.

    Returns
    -------
    float or None
        P-value of the chi-square test; ``None`` if the contingency
        table is empty.
    """
    xtab = pd.crosstab(x, y)
    pval = None
    if xtab.size != 0:
        try:
            _, pval, _, _ = stats.chi2_contingency(xtab)
        except Exception:
            pval = 0
    return pval


def corr_pointbiserial(binary_data, continuous_data, data):
    """Point-biserial correlation between a binary column and a continuous column.

    Parameters
    ----------
    binary_data : str
        Name of a dichotomous column in ``data``.
    continuous_data : str
        Name of a continuous column in ``data``.
    data : pandas.DataFrame
        Frame containing both columns.

    Returns
    -------
    float
        Point-biserial correlation coefficient.

    Notes
    -----
    NaN handling is not applied; drop NaNs upstream if needed.
    """

    bd_unique = data[binary_data].unique()

    g0 = data[data[binary_data] == bd_unique[0]][continuous_data]
    g1 = data[data[binary_data] == bd_unique[1]][continuous_data]

    s_y = np.std(data[continuous_data])
    n = len(data[binary_data])
    n0 = len(g0)
    n1 = len(g1)
    m0 = g0.mean()
    m1 = g1.mean()
    out = (m0 - m1) * math.sqrt((n0 * n1) / n**2) / s_y
    return out


def find_high_correlations(corrMat, df_scores, thershold):
    """Identify features to drop based on a correlation threshold and a per-feature score.

    For each pair of features whose absolute correlation exceeds
    ``thershold``, the one with the lower score in ``df_scores`` is
    flagged for removal.

    Parameters
    ----------
    corrMat : pandas.DataFrame
        Symmetric correlation matrix.
    df_scores : pandas.Series or dict-like
        Per-feature score; the higher-scoring member of each correlated
        pair is kept.
    thershold : float
        Absolute correlation cutoff.

    Returns
    -------
    tuple
        ``(to_drop, mat_ind, high_corrs)``: list of features to drop,
        list of all features involved in any high-correlation pair, and
        a dict mapping the kept column to its correlated partner.
    """
    # TODO: add comment
    corr_matrix = corrMat.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than x
    drop_cols = [x for x in upper.columns if any(upper[x] > thershold)]
    drops_rows = [x for x in upper.index if any(upper.loc[x] > thershold)]

    to_drop = list()
    mat_ind = list()
    for x, y in zip(drop_cols, drops_rows, strict=False):
        tmp = y if df_scores[x] > df_scores[x] else x
        to_drop.append(tmp)
        mat_ind.append(x)
        mat_ind.append(y)

    high_corrs = dict(zip(drop_cols, drops_rows, strict=False))

    return to_drop, mat_ind, high_corrs


def discretizer(x, y, labels=None, method="cut"):
    """Bin a continuous variable into ordered categories using one of four strategies.

    Parameters
    ----------
    x : pandas.Series
        Continuous variable to bin.
    y : pandas.Series
        Target variable (only used for the tree-based methods).
    labels : list of str or None, optional
        Bin labels. Defaults to ``['Q1', 'Q2', 'Q3', 'Q4']``.
    method : {'cut', 'tree1', 'tree2', anything else}, optional
        * ``'cut'`` (default): MinMax-scale ``x`` then split at
          ``[0, 0.33, 0.66, 1]``.
        * ``'tree1'``: depth-1 decision tree on ``y`` chooses one cut.
        * ``'tree2'``: depth-2 decision tree chooses cuts.
        * fallback: rank-based ``pd.qcut`` into 3 bins.

    Returns
    -------
    tuple
        ``(out, bins)`` — the categorical Series and the bin edges
        (as floats).
    """
    # print('discretizer:'+method)
    if labels is None:
        labels = ["Q1", "Q2", "Q3", "Q4"]

    if method == "cut":
        from sklearn import preprocessing

        min_max_scaler = preprocessing.MinMaxScaler()
        out, bins = pd.cut(
            min_max_scaler.fit_transform(x.values.reshape(-1, 1)).reshape(-1),
            bins=[-0.1, 0.33, 0.66, 1.1],
            labels=labels,
            retbins=True,
        )

    elif method == "tree1":
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
        clf.fit(x.to_frame(), y)

        # if len(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]))==1:
        #     # print ("max_depth increased")
        #     clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
        #     clf.fit(x.to_frame(),y)

        bins = np.sort(
            np.append(
                np.unique(clf.tree_.threshold[clf.tree_.threshold != -2]),
                [x.max() + 1 / 1e6, x.min() - 1 / 1e6],
            )
        ).tolist()
        out = pd.cut(x, bins=bins, labels=labels[: (len(bins) - 1)])

    elif method == "tree2":
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=2)
        clf.fit(x.to_frame(), y)

        bins = np.sort(
            np.append(
                np.unique(clf.tree_.threshold[clf.tree_.threshold != -2]),
                [x.max() + 1 / 1e6, x.min() - 1 / 1e6],
            )
        ).tolist()
        out = pd.cut(x, bins=bins, labels=labels[: (len(bins) - 1)])

    else:
        ##TODO: it needs to be corrected:
        out, bins = pd.qcut(x.rank(method="first"), q=3, labels=labels, retbins=True)
        # out,bins=pd.qcut(x,q=3,labels=labels,retbins=True, duplicates='drop')
        # out,bins=pd.qcut(x+ jitter(x),q=3,labels=labels,retbins=True)

    bins = [np.float(x) for x in bins]
    print(x.name + ":\n" + str(bins))
    return (out, bins)


def jitter(a_series, noise_reduction=1000000):
    """Generate small mean-zero noise scaled to the series' standard deviation.

    Useful for breaking ties before ``pd.qcut`` on a series with many
    repeated values.

    Parameters
    ----------
    a_series : pandas.Series
        Series whose std controls the noise scale.
    noise_reduction : float, optional
        Larger values give smaller noise. Default ``1_000_000``.

    Returns
    -------
    numpy.ndarray
        Noise vector of the same length as ``a_series``.

    References
    ----------
    https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
    """
    return (np.random.random(len(a_series)) * a_series.std() / noise_reduction) - (
        a_series.std() / (2 * noise_reduction)
    )


def extract_equation(results_pars):
    """Format a regression's coefficients as a human-readable equation string.

    Recognizes ``np.power(<feat>, <n>)`` index entries and rewrites them
    as ``<feat>**<n>`` in the output.

    Parameters
    ----------
    results_pars : pandas.Series
        Indexed by parameter name (or ``'Intercept'``); values are
        coefficients.

    Returns
    -------
    str
        Equation as a string, e.g. ``"+1.23+0.45*x-0.07*x**2"``.
    """
    vars = results_pars.reset_index()
    vars[0] = np.round(vars[0], 2).astype(str)
    vars["ploys"] = vars["index"].str.extract(r"np.power\((.+?),")
    vars["power"] = np.where(vars["ploys"].isnull(), np.nan, vars["index"].str[-2:-1])
    vars["index"] = np.where(
        vars["ploys"].isnull(), vars["index"], vars["ploys"] + "**" + vars["power"]
    )
    equation = ""
    for row in vars.iterrows():
        sign = "" if (np.sign(float(row[1][0])) == -1) or (row[0] == 0) else "+"
        tmp = (
            f"{sign}{row[1][0]}"
            if row[1]["index"] == "Intercept"
            else f"{sign}{row[1][0]}*{row[1]['index']}"
        )
        equation += tmp
    return equation


def analyze_categorical_data(data, independent_var, dependent_var, alpha=0.05):
    """Perform a chi-square test of independence between two categorical variables.

    Computes the contingency table, chi-square statistic, p-value, and
    Cramer's V; renders a heatmap of the contingency table; and returns
    a worded interpretation.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing both variables.
    independent_var : str
        Name of the independent variable column.
    dependent_var : str
        Name of the dependent variable column.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        Keys: ``contingency_table``, ``chi_square_statistic``,
        ``p_value``, ``degrees_of_freedom``, ``cramers_v``,
        ``significant``, ``interpretation``.
    """
    from scipy.stats import chi2_contingency

    # Create contingency table
    contingency_table = pd.crosstab(data[independent_var], data[dependent_var])

    # Perform Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    # Create result dictionary
    results = {
        "contingency_table": contingency_table,
        "chi_square_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "cramers_v": cramer_v,
        "significant": p_value < alpha,
    }

    # Add interpretation
    results["interpretation"] = interpret_results_analyze_categorical(results, alpha)

    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap="YlOrRd", fmt="d")
    plt.title(f"Contingency Table: {independent_var} vs {dependent_var}")
    plt.tight_layout()

    return results


def interpret_results_analyze_categorical(results, alpha):
    """Turn a chi-square result dict into a human-readable interpretation string.

    Parameters
    ----------
    results : dict
        Output of :func:`analyze_categorical_data`; must contain
        ``p_value`` and ``cramers_v``.
    alpha : float
        Significance level used to phrase the significance verdict.

    Returns
    -------
    str
        Two-sentence summary: significance verdict plus strength band
        (negligible / weak / moderate / strong) for Cramer's V.
    """
    interpretation = []

    # Chi-square test interpretation
    if results["p_value"] < alpha:
        interpretation.append(
            f"There is a statistically significant relationship between the variables (p-value = {results['p_value']:.4f} < {alpha})."
        )
    else:
        interpretation.append(
            f"There is no statistically significant relationship between the variables (p-value = {results['p_value']:.4f} > {alpha})."
        )

    # Cramer's V interpretation
    cramers_v = results["cramers_v"]
    if cramers_v < 0.1:
        strength = "negligible"
    elif cramers_v < 0.3:
        strength = "weak"
    elif cramers_v < 0.5:
        strength = "moderate"
    else:
        strength = "strong"

    interpretation.append(
        f"The strength of the association is {strength} (Cramer's V = {cramers_v:.3f})."
    )

    return " ".join(interpretation)


# TODO:make it better
def analyze_cat_num(
    data: pd.DataFrame, categorical_var: str, numeric_var: str, alpha: float = 0.05
) -> dict[str, str | float | dict]:
    """Compare a numeric variable across a categorical grouping, picking the appropriate test.

    Chooses among independent t-test, Welch's t-test, Mann-Whitney U,
    one-way ANOVA, and Kruskal-Wallis based on group count, normality
    (Shapiro-Wilk), and variance homogeneity (Levene).

    Parameters
    ----------
    data : pandas.DataFrame
        Frame containing both variables.
    categorical_var : str
        Categorical independent variable column.
    numeric_var : str
        Numeric dependent variable column.
    alpha : float, optional
        Significance level used for assumption tests and the final
        verdict. Default ``0.05``.

    Returns
    -------
    dict
        Keys: ``test_type``, ``test_statistic``, ``p_value``,
        ``significant``, ``descriptive_stats``, ``assumption_tests``
        (``normality`` + ``homogeneity_of_variance``).

    Raises
    ------
    TypeError
        ``data`` is not a DataFrame.
    ValueError
        A named variable is missing or fewer than two groups exist.
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if categorical_var not in data.columns or numeric_var not in data.columns:
        raise ValueError("Specified variables must exist in the DataFrame")

    # Get unique categories and check number of groups
    categories = data[categorical_var].unique()
    n_groups = len(categories)

    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")

    # Create groups for analysis
    groups = [data[data[categorical_var] == cat][numeric_var].dropna() for cat in categories]

    # Calculate descriptive statistics
    descriptive_stats = (
        data.groupby(categorical_var)[numeric_var]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .to_dict("index")
    )

    # Test for normality in each group
    normality_tests = {}
    for cat, group in zip(categories, groups, strict=False):
        if len(group) >= 3:  # Shapiro-Wilk test requires at least 3 samples
            stat, p_val = stats.shapiro(group)
            normality_tests[cat] = {"statistic": stat, "p_value": p_val, "normal": p_val > alpha}

    # Test for homogeneity of variances
    levene_stat, levene_p = stats.levene(*groups)

    # Perform appropriate statistical test based on number of groups
    if n_groups == 2:
        # Perform t-test if normal, Mann-Whitney U test if not
        all_normal = all(test["normal"] for test in normality_tests.values())
        equal_var = levene_p > alpha

        if all_normal:
            stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
            test_type = "Independent t-test"
            if not equal_var:
                test_type += " with Welch's correction"
        else:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_type = "Mann-Whitney U test"
    else:
        # Perform one-way ANOVA if normal, Kruskal-Wallis if not
        all_normal = all(test["normal"] for test in normality_tests.values())
        equal_var = levene_p > alpha

        if all_normal and equal_var:
            stat, p_value = stats.f_oneway(*groups)
            test_type = "One-way ANOVA"
        else:
            stat, p_value = stats.kruskal(*groups)
            test_type = "Kruskal-Wallis H test"

    # Compile results
    results = {
        "test_type": test_type,
        "test_statistic": stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "descriptive_stats": descriptive_stats,
        "assumption_tests": {
            "normality": normality_tests,
            "homogeneity_of_variance": {
                "statistic": levene_stat,
                "p_value": levene_p,
                "equal_variances": levene_p > alpha,
            },
        },
    }

    return results


# ---------------------------------------------------------------------------
# Statistical Process Control (SPC): sigma limits and I-MR control limits
# ---------------------------------------------------------------------------

# Shewhart I-MR constants for subgroup size n=2 (individuals chart).
#   E2 = 3 / d2  = 3 / 1.128 ≈ 2.66  scales MR-bar into ±3σ for the I chart.
#   D4 for n=2  = 3.267                upper control limit for the MR chart.
# Reference: NIST/SEMATECH e-Handbook of Statistical Methods, §6.3.2.
_IMR_INDIVIDUAL_CONST = 2.66
_IMR_RANGE_UCL_CONST = 3.267

_DEFAULT_SIGMA_COEF = 3


def _sigma_columns(col: str) -> list[str]:
    """Column labels for a per-column sigma-limit block: LL / AVG / UL."""
    return [f"LL_{col}", f"AVG_{col}", f"UL_{col}"]


def _imr_columns(col: str) -> list[str]:
    """Column labels for a per-column I-MR limit block (six labels)."""
    return [
        f"I_LL_{col}",
        f"I_AVG_{col}",
        f"I_UL_{col}",
        f"MR_LL_{col}",
        f"MR_AVG_{col}",
        f"MR_UL_{col}",
    ]


def sigma_limit(df, coef: float = _DEFAULT_SIGMA_COEF) -> tuple[float, float, float]:
    """Compute sigma-based control limits (mean ± coef*std) for a series.

    Parameters
    ----------
    df : pandas.Series or array-like
        Numeric values.
    coef : float, optional
        Multiplier of the standard deviation; default 3 (3-sigma).

    Returns
    -------
    tuple of float
        ``(LCL, mean, UCL)`` — lower limit, mean, upper limit.
    """
    data_mean, data_std = np.mean(df), np.std(df)
    LCL = data_mean - coef * data_std
    UCL = data_mean + coef * data_std
    return (LCL, data_mean, UCL)


def sigma_limit_grpby(
    df: pd.DataFrame,
    col: str,
    grpby_col: list[str] | None = None,
    coef: float = _DEFAULT_SIGMA_COEF,
) -> pd.DataFrame:
    """Compute sigma limits for one column, optionally grouped.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    col : str
        Name of the column to compute limits on.
    grpby_col : list of str or None, optional
        Group-by columns. ``None`` (default) computes a single set of limits
        across the whole frame.
    coef : float, optional
        Sigma multiplier; default 3.

    Returns
    -------
    pandas.DataFrame
        Columns ``LL_<col>``, ``AVG_<col>``, ``UL_<col>``, indexed by group
        (or one row when ``grpby_col`` is empty).
    """
    if grpby_col:
        cls = df.groupby(grpby_col)[col].apply(sigma_limit, coef=coef)
    else:
        cls = sigma_limit(df[col], coef=coef)
    return pd.DataFrame(
        cls.tolist(),
        index=cls.index,
        columns=_sigma_columns(col),
    ).rename_axis(cls.index.name)


def sigma_limit_cols_grpby(
    df: pd.DataFrame,
    cols: list[str],
    grpby_col: list[str],
    coef: float = _DEFAULT_SIGMA_COEF,
) -> pd.DataFrame:
    """Compute sigma limits across multiple columns, grouped.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    cols : list of str
        Columns to compute limits on.
    grpby_col : list of str
        Group-by columns.
    coef : float, optional
        Sigma multiplier; default 3.

    Returns
    -------
    pandas.DataFrame
        One row per group, with three columns (LL/AVG/UL) per requested
        column in ``cols``. The group key(s) are reset to a column.
    """
    per_col = [sigma_limit_grpby(df, col=c, grpby_col=grpby_col, coef=coef) for c in cols]
    return pd.concat(per_col, axis=1).reset_index()


def i_mr_sigma_limits(
    df: pd.DataFrame,
    cols: list[str],
    grpby_col: list[str],
) -> pd.DataFrame:
    """Compute Individual (I) and Moving-Range (MR) sigma limits for I-MR control charts.

    Parameters
    ----------
    df : pandas.DataFrame
        Source time-series data; must contain all columns in ``grpby_col``
        plus every column listed in ``cols``.
    cols : list of str
        Numeric columns to compute limits on.
    grpby_col : list of str
        Group-by columns (e.g. ``['machine']`` or ``['asset_id']``).

    Returns
    -------
    pandas.DataFrame
        One row per group, with ``I_LL_<col>``, ``I_AVG_<col>``,
        ``I_UL_<col>``, ``MR_LL_<col>``, ``MR_AVG_<col>``, ``MR_UL_<col>``
        for each column in ``cols``.
    """
    I_chart_CLs = sigma_limit_cols_grpby(df, cols=cols, grpby_col=grpby_col).set_index(grpby_col)

    MR_df = df.copy()
    MR_df[cols] = MR_df.groupby(grpby_col)[cols].diff().abs()
    MR_chart_CLs = sigma_limit_cols_grpby(MR_df, cols=cols, grpby_col=grpby_col).set_index(
        grpby_col
    )

    I_chart_CLs.columns = [f"I_{c}" for c in I_chart_CLs.columns]
    MR_chart_CLs.columns = [f"MR_{c}" for c in MR_chart_CLs.columns]

    return pd.concat([I_chart_CLs, MR_chart_CLs], axis=1).reset_index()


def control_limit(df) -> tuple[float, float, float, float, float, float]:
    """Compute I-MR control-chart limits using Shewhart constants.

    Uses :data:`_IMR_INDIVIDUAL_CONST` (≈ 2.66) to scale the moving-range
    mean for the Individual chart, and :data:`_IMR_RANGE_UCL_CONST`
    (≈ 3.267) for the Moving-Range chart upper limit. The MR lower limit
    is 0.

    Parameters
    ----------
    df : pandas.Series
        Numeric time series.

    Returns
    -------
    tuple of float
        ``(x_LCL, x_avg, x_UCL, MR_LCL, MR_avg, MR_UCL)``.
    """
    x_avg = np.mean(df)
    MR_avg = np.mean(df.diff().abs())

    x_LCL = x_avg - MR_avg * _IMR_INDIVIDUAL_CONST
    x_UCL = x_avg + MR_avg * _IMR_INDIVIDUAL_CONST
    MR_LCL = 0.0
    MR_UCL = MR_avg * _IMR_RANGE_UCL_CONST

    return (x_LCL, x_avg, x_UCL, MR_LCL, MR_avg, MR_UCL)


def control_limit_grpby(
    df: pd.DataFrame,
    col: str,
    grpby_col: list[str] | None = None,
    coef: float | None = None,
) -> pd.DataFrame:
    """Compute I-MR control limits for one column, optionally grouped.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    col : str
        Name of the column to compute limits on.
    grpby_col : list of str or None, optional
        Group-by columns. ``None`` (default) computes one set of limits.
    coef : float or None, optional
        Accepted for API symmetry with :func:`sigma_limit_grpby` and
        :func:`i_mr_ctrl_limits`; ignored (Shewhart constants are fixed).

    Returns
    -------
    pandas.DataFrame
        Columns ``I_LL_<col>``, ``I_AVG_<col>``, ``I_UL_<col>``,
        ``MR_LL_<col>``, ``MR_AVG_<col>``, ``MR_UL_<col>`` indexed by group.
    """
    del coef  # documented as ignored
    if grpby_col:
        cls = df.groupby(grpby_col)[col].apply(control_limit)
    else:
        cls = control_limit(df[col])
    return pd.DataFrame(
        cls.tolist(),
        index=cls.index,
        columns=_imr_columns(col),
    ).rename_axis(cls.index.name)


def i_mr_ctrl_limits(
    df: pd.DataFrame,
    cols: list[str],
    grpby_col: list[str],
    coef: float | None = None,
) -> pd.DataFrame:
    """Compute I-MR control limits across multiple columns, grouped.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data.
    cols : list of str
        Columns to compute limits on.
    grpby_col : list of str
        Group-by columns.
    coef : float or None, optional
        Forwarded to :func:`control_limit_grpby` (ignored there; retained
        for API symmetry with the sigma variants).

    Returns
    -------
    pandas.DataFrame
        One row per group, six columns per item in ``cols``.
    """
    per_col = [control_limit_grpby(df, col=c, grpby_col=grpby_col, coef=coef) for c in cols]
    return pd.concat(per_col, axis=1).reset_index()


def sc_post_gap_test(
    post_preds,
    *,
    model=None,
    pre_residuals=None,
    n_permutations=10_000,
    random_state=42,
    alpha=0.05,
):
    """Test whether counterfactual (y_pred) and real (y_true) conversion-rate
    lines diverge significantly in the SC post-intervention window.

    Replaces a raw lift number with three complementary statistical tests:

    1. Sign-flip permutation test (primary) - SC gold-standard (Abadie 2010).
       Randomly flips signs of each daily effect; checks if observed mean is
       extreme under the null. Valid even with autocorrelation.
    2. Wilcoxon signed-rank (secondary) - non-parametric, no normality assumed;
       tests whether median effect = 0.
    3. KS test: post effects vs. pre residuals (diagnostic, optional) - tests
       whether post deviations follow the same distribution as pre-period model
       noise. Only runs when pre_residuals is supplied.

    Parameters
    ----------
    post_preds : pandas.DataFrame
        Output of effect_from_preds; needs columns y_true, y_pred, model.
    model : str, optional
        Filter to one model. If None all rows are used.
    pre_residuals : array-like, optional
        Pre-period (y_true - y_pred) array for the KS diagnostic.
    n_permutations : int
        Number of sign-flip draws (default 10 000).
    random_state : int
        RNG seed (default 42).
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict
        n_post, gap_mean, gap_std,
        perm_pvalue, perm_significant,
        wilcoxon_pvalue, wilcoxon_significant,
        ks_stat, ks_pvalue, ks_significant (None if no pre_residuals),
        significant (True when both perm + wilcoxon reject at alpha),
        direction ('positive'/'negative'/'none'), alpha.

    Examples
    --------
    >>> result = sc_post_gap_test(post_preds, model="auto_arima")
    >>> result["gap_mean"], result["perm_pvalue"]

    With KS diagnostic:
    >>> pre_resid = (pre["y_true"] - pre["y_pred"]).to_numpy()
    >>> result = sc_post_gap_test(post_preds, model="auto_arima",
    ...                           pre_residuals=pre_resid)
    """
    from scipy.stats import ks_2samp
    from scipy.stats import wilcoxon as _wilcoxon

    rows = post_preds if model is None else post_preds[post_preds["model"] == model]
    if rows.empty:
        raise ValueError(
            f"No rows for model={model!r}. " f"Available: {post_preds['model'].unique().tolist()}"
        )

    effects = (rows["y_true"] - rows["y_pred"]).to_numpy(dtype=float)
    n = len(effects)
    if n < 5:
        raise ValueError(f"Only {n} post-period observations - too few for a meaningful test.")

    gap_mean = float(np.mean(effects))
    gap_std = float(np.std(effects, ddof=1))

    # 1. Sign-flip permutation test
    rng = np.random.default_rng(random_state)
    perm_means = np.array(
        [np.mean(effects * rng.choice([-1.0, 1.0], size=n)) for _ in range(n_permutations)]
    )
    perm_pvalue = float(np.mean(np.abs(perm_means) >= abs(gap_mean)))

    # 2. Wilcoxon signed-rank
    try:
        _, wilcoxon_pvalue = _wilcoxon(effects, alternative="two-sided")
        wilcoxon_pvalue = float(wilcoxon_pvalue)
    except ValueError:
        wilcoxon_pvalue = 1.0  # all effects identical - no divergence

    # 3. KS diagnostic (post effects vs. pre residuals)
    if pre_residuals is not None:
        pre_arr = np.asarray(pre_residuals, dtype=float)
        pre_arr = pre_arr[np.isfinite(pre_arr)]
        ks_stat, ks_pvalue = ks_2samp(pre_arr, effects)
        ks_stat, ks_pvalue = float(ks_stat), float(ks_pvalue)
        ks_significant = ks_pvalue < alpha
    else:
        ks_stat = ks_pvalue = ks_significant = None

    both_sig = (perm_pvalue < alpha) and (wilcoxon_pvalue < alpha)
    direction = ("positive" if gap_mean > 0 else "negative") if both_sig else "none"

    return {
        "n_post": n,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "perm_pvalue": perm_pvalue,
        "perm_significant": perm_pvalue < alpha,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "wilcoxon_significant": wilcoxon_pvalue < alpha,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ks_significant": ks_significant,
        "significant": both_sig,
        "direction": direction,
        "alpha": alpha,
    }
