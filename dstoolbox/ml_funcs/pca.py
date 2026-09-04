"""PCA + canonical-correlation analysis helpers."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


##TODO: merge pca_plot and pca_explained_var:
def pca_plot(df):
    """Render a 2-component PCA scatter plot of customer/segment-indexed data.

    Each row is plotted as a labeled marker; segments are coloured from
    the Plotly Set3 qualitative palette (cycled if there are more
    segments than colours).

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric features. The index is treated as the segment label and
        is rendered above each marker.

    Returns
    -------
    None
        The figure is shown via ``fig.show()``.
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # Figure 4: PCA Visualization (using original data)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)

    segments = df.index.unique().tolist()

    # Create a color palette - fix for the Set3 issue
    colors = px.colors.qualitative.Set3[: len(segments)]
    if len(segments) > len(px.colors.qualitative.Set3):
        # Cycle through existing colors
        colors = [
            px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            for i in range(len(segments))
        ]

    # Create the scatter plot
    fig = go.Figure()

    for i, (segment, color) in enumerate(zip(segments, colors, strict=False)):
        fig.add_trace(
            go.Scatter(
                x=[X_pca[i, 0]],
                y=[X_pca[i, 1]],
                mode="markers+text",
                marker=dict(size=15, color=color, opacity=0.7, line=dict(width=2, color="black")),
                text=segment,
                textposition="top center",
                textfont=dict(size=12, color="black"),
                name=segment,
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Customer Segments - PCA Visualization", font=dict(size=16, color="black"), x=0.5
        ),
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
        width=900,
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
    )

    fig.show()


def pca_explained_var(pcaML, output_folder=None, plot_type="cumulative"):
    """Plot per-component and cumulative explained-variance ratios for a fitted PCA.

    Parameters
    ----------
    pcaML : sklearn.decomposition.PCA
        A fitted PCA-like object exposing ``explained_variance_``,
        ``explained_variance_ratio_``, and ``n_components_``.
    output_folder : str or None, optional
        If given, writes the plot(s) to this folder as PNG.
    plot_type : {'cumulative', 'scree', 'both'}, default 'cumulative'
        Which plot(s) to draw.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(explained_var_percent, eigen_values)``.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    eigen_values = pcaML.explained_variance_
    explained_var = np.round(pcaML.explained_variance_ratio_, decimals=3) * 100
    explained_var_cumsum = np.cumsum(explained_var)

    if plot_type in ["scree", "both"]:
        # Scree plot
        scree_fig = go.Figure()
        scree_fig.add_trace(
            go.Scatter(
                x=np.arange(1, pcaML.n_components_ + 1),
                y=explained_var,
                mode="lines+markers",
                name="Explained Variance",
            )
        )
        scree_fig.update_layout(
            title="Scree Plot",
            xaxis_title="Principal Component",
            yaxis_title="Variance Explained (%)",
        )
        if output_folder:
            pio.write_image(scree_fig, os.path.join(output_folder, "scree_plot.png"))
        scree_fig.show()

    if plot_type in ["cumulative", "both"]:
        # Cumulative explained variance plot
        cumulative_fig = go.Figure()
        cumulative_fig.add_trace(
            go.Scatter(
                x=np.arange(1, pcaML.n_components_ + 1),
                y=explained_var_cumsum,
                mode="lines+markers",
                name="Cumulative Explained Variance",
            )
        )
        cumulative_fig.update_layout(
            title="Cumulative Explained Variance Ratio by Principal Components",
            xaxis_title="Number of Principal Components",
            yaxis_title="Cumulative Explained Variance (%)",
            yaxis=dict(range=[0, 100]),
        )
        if output_folder:
            pio.write_image(cumulative_fig, os.path.join(output_folder, "_cumulative_variance.png"))
        cumulative_fig.show()

    return explained_var, eigen_values


def pca_ortho_rotation(lam, method="varimax", gamma=None, eps=1e-6, itermax=100):
    """
    ##TODO: document it
    ## A VARIMAX rotation is a change of coordinates used in principal component analysis1 (PCA) that maximizes the sum of the variances of the squared loadings
    ## https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
    Return orthogal rotation matrix
    TODO: - other types beyond
    """
    if gamma is None:
        if method == "varimax":
            gamma = 1.0
        if method == "quartimax":
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for _i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot**2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot**3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R


def pca_important_features(transformed_features, components_, columns):
    ##TODO: check it and make a function
    ###http://benalexkeen.com/principle-component-analysis-in-python/
    """
    This function will return the most "important"
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:, 0])
    yvector = components_[1] * max(transformed_features[:, 1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = {
        columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2) for i in range(num_columns)
    }
    important_features = pd.Series(important_features)
    important_features = important_features.sort_values(ascending=[False])
    return important_features


def pc_draw_vectors(transformed_features, components_, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling

    https://benalexkeen.com/principle-component-analysis-in-python/
    """
    plt.style.use("ggplot")

    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:, 0])
    yvector = components_[1] * max(transformed_features[:, 1])

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(
            0, 0, xvector[i], yvector[i], color="b", width=0.0005, head_width=0.02, alpha=0.75
        )
        plt.text(xvector[i] * 1.2, yvector[i] * 1.2, list(columns)[i], color="b", alpha=0.75)

    return ax


##TODO:rempve it after checking canonical_correlation_analysis is the same function
def cca_batch(X1_sub, X2_sub, n_comp=2):
    """Run Canonical Correlation Analysis on two feature sets and return correlations + coefs.

    Parameters
    ----------
    X1_sub, X2_sub : pandas.DataFrame
        The two feature blocks (rows must align).
    n_comp : int, optional
        Number of canonical components to extract. Default 2.

    Returns
    -------
    tuple
        ``(component_correlations, raw_coefficients,
        normalized_coefficients, fitted_cca_model)``. ``component_correlations``
        is a pandas Series indexed ``CC1..CCn``. The normalized coefficient
        matrix is rescaled to ``[-1, 1]`` for visualisation.

    Notes
    -----
    Marked as a duplicate of :func:`canonical_correlation_analysis`;
    intended for retirement once equivalence is verified.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    cca = CCA(scale=True, n_components=n_comp)
    cca.fit(X1_sub, X2_sub)
    X1_c, X2_c = cca.transform(
        X1_sub, X2_sub
    )  # transform our datasests to obtain canonical variates

    comp_corr = pd.Series(
        [np.corrcoef(X1_c[:, i], X2_c[:, i])[1][0] for i in range(n_comp)],
        index=[f"CC{i}" for i in range(1, n_comp + 1)],
    )

    #  coef_: ndarray of shape (n_targets, n_features)
    # The coefficients of the linear model such that Y is approximated as Y = X @ coef_.T + intercept_.
    coef_df = pd.DataFrame(np.round(cca.coef_, 3), index=X2_sub.columns, columns=X1_sub.columns)

    coef_df_nrm = pd.DataFrame(
        2 * scaler.fit_transform(coef_df) - 1, index=coef_df.index, columns=coef_df.columns
    )

    return comp_corr, coef_df, coef_df_nrm, cca


def canonical_correlation_analysis(first_dataset, second_dataset, num_components=2):
    """
    Perform Canonical Correlation Analysis (CCA) on two datasets.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import MinMaxScaler

    # Initialize preprocessing and CCA model
    coefficient_scaler = MinMaxScaler()
    cca_model = CCA(scale=True, n_components=num_components)

    # Fit CCA model to find canonical correlations
    cca_model.fit(first_dataset, second_dataset)

    # Transform datasets to obtain canonical variates
    first_canonical_variates, second_canonical_variates = cca_model.transform(
        first_dataset, second_dataset
    )

    # Calculate correlation between corresponding canonical components
    component_correlations = pd.Series(
        [
            np.corrcoef(first_canonical_variates[:, i], second_canonical_variates[:, i])[1][0]
            for i in range(num_components)
        ],
        index=[f"CC{i}" for i in range(1, num_components + 1)],
    )

    # Create coefficient matrix (second_dataset features × first_dataset features)
    raw_coefficients = pd.DataFrame(
        np.round(cca_model.coef_, 3), index=second_dataset.columns, columns=first_dataset.columns
    )

    # Normalize coefficients to [-1, 1] range for visualization
    normalized_coefficients = pd.DataFrame(
        2 * coefficient_scaler.fit_transform(raw_coefficients) - 1,
        index=raw_coefficients.index,
        columns=raw_coefficients.columns,
    )

    return component_correlations, raw_coefficients, normalized_coefficients, cca_model
