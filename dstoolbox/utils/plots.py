"""Plotting utilities: correlation heatmap, sankey, word clouds, 3D scatter,
plotly subplots, distribution analysis, Statistical Process Control (I-MR
and series-overlay charts), plus plotly figure I/O (HTML dashboards and
multi-format export).
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def corrmap(df0, method='kendall', diagonal_plot=True, **kwargs):
    """Plot a correlation heatmap for numeric columns of ``df0``.

    Drops rows with any NaN, dedupes, computes ``DataFrame.corr``, and
    renders a seaborn heatmap with a diverging colormap. The upper
    triangle can be masked out via ``diagonal_plot``.

    Parameters
    ----------
    df0 : pandas.DataFrame
        Source frame (``samples × features``).
    method : {'pearson', 'kendall', 'spearman'} or callable, optional
        Correlation method forwarded to ``DataFrame.corr``. Default
        ``'kendall'``.
    diagonal_plot : bool, optional
        If True (default), mask the upper triangle to show only the
        lower-triangular half.
    **kwargs :
        Forwarded to ``DataFrame.corr`` and ``seaborn.heatmap`` (routed
        by parameter name).

    Returns
    -------
    tuple
        ``(corr, figure)`` — the correlation matrix and the matplotlib
        Figure.
    """
    import inspect

    corr_args = list(inspect.signature(pd.DataFrame.corr).parameters)
    kwargs_corr = {k: kwargs.pop(k) for k in dict(kwargs) if k in corr_args}

    heatmap_args = list(inspect.signature(sns.heatmap).parameters)
    kwargs_heatmap = {k: kwargs.pop(k) for k in dict(kwargs) if k in heatmap_args}

    corr = df0.dropna(how='any',axis=0).drop_duplicates().corr(method=method,**kwargs_corr)
    # Generate a mask for the upper triangle

    if diagonal_plot:
      mask = np.zeros_like(corr)
      mask[np.triu_indices_from(mask)] = True
    else:
      mask=None

    plt.figure(figsize = (30,20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    snsPlot = sns.heatmap(
                            corr,
                            mask=mask,
                            cmap=cmap,
                            center=0,
                            square=True,
                            linewidths=.5,
                            cbar_kws={"shrink": .5},
                            fmt=".1f",
                            annot=True,
                            **kwargs_heatmap,
                            )
    figure = snsPlot.get_figure()
    # figure.savefig(os.path.join(outputFolder,"corr_map.png"), bbox_inches='tight')
    plt.show()
    plt.close()

    return corr, figure


def sankey(left, right, value, thershold, utitle, filename):
    """Render a two-column Sankey diagram of ``left → right`` flows.

    Aggregates each ``(left, right)`` pair by summing ``value``, drops
    pairs whose row count is below ``thershold``, and writes an
    interactive Plotly HTML file.

    Parameters
    ----------
    left : pandas.Series
        Source node label per transaction.
    right : pandas.Series
        Target node label per transaction.
    value : pandas.Series
        Flow magnitude per transaction.
    thershold : float
        Minimum row count per ``(left, right)`` pair to keep in the
        diagram (note: filters on **count**, not summed value).
    utitle : str
        Plot title.
    filename : str
        Output HTML path.

    Returns
    -------
    pandas.DataFrame
        The filtered, aggregated transactions frame with columns
        ``left``, ``right``, ``value``.
    """

    tranactions0 = pd.concat(
        [left.rename('left'), right.rename('right'), value.rename('value')], axis=1)
    tranactions = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('sum')
    counts = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('count')
    tranactions = tranactions.loc[counts['value'] > thershold, :]
    tranactions.sort_values(['value'], ascending=[False], inplace=True)
    left = tranactions['left']
    right = tranactions['right']
    values = tranactions['value']

    #import chart_studio.plotly as py
    import plotly

    lbLeft = list(pd.unique(left))
    lbRight = list(pd.unique(right))

    # label=lbLeft+lbRight
    source = []
    target = []
    value = []
    for i in list(range(left.shape[0])):
        tmpSource = np.where(
            np.asarray(lbLeft) == np.asarray(
                left.iloc[i]))[0].tolist()
        source = source + tmpSource

        tmpTarget = np.where(
            np.asarray(lbRight) == np.asarray(
                right.iloc[i]))[0].tolist()
        target = target + tmpTarget

        tmpValue = [values.iloc[i]]
        value = value + tmpValue

    target = [x + len(lbLeft) for x in target]

    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5
            ),
            label=list(pd.unique(left)) + list(pd.unique(right))
            # color = ["blue", "blue", "blue", "blue", "blue", "blue"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))

    layout = dict(
        title=utitle,
        font=dict(
            size=10
        )
    )
    fig = dict(data=[data], layout=layout)
    plotly.offline.plot(fig, filename=filename)
    return tranactions


def wordcloud_graph(txtSeries_df, outputFile):
    """Render a word cloud from a text Series or a frequency 2-column frame.

    Parameters
    ----------
    txtSeries_df : pandas.Series or pandas.DataFrame
        If a Series, all values are concatenated and tokenized. If a
        DataFrame, the first two columns are treated as
        ``(word, frequency)``.
    outputFile : str
        Destination path for the rendered PNG.

    Returns
    -------
    None
        Writes the figure to ``outputFile``.
    """
    from wordcloud import WordCloud, STOPWORDS

    if  type(txtSeries_df)==pd.core.series.Series:
        txt=' '.join(txtSeries_df.astype('str'))
        wordcloud_instance = WordCloud(width=800, height=400).generate(txt)
    else:
        wordcloud_instance = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(txtSeries_df.iloc[:,0] ,txtSeries_df.iloc[:,1])))

    ## Generate plot
    plt.figure(figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud_instance)
    plt.axis("off")
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close('all')


def plot_3d_scatter(udata, uY, xyzLabels, utitle, outPutFile):
    """3D scatter plot of ``udata`` coloured by class labels ``uY``.

    Parameters
    ----------
    udata : numpy.ndarray or pandas.DataFrame
        ``samples × 3`` coordinate matrix.
    uY : pandas.Series or numpy.ndarray
        Class labels used for colouring; up to 4 classes
        (colours cycle through ``['r', 'g', 'b', 'y']``).
    xyzLabels : sequence of str, length 3
        Axis labels for X, Y, Z.
    utitle : str
        Plot title.
    outPutFile : str
        PNG output path.

    Returns
    -------
    None
        Saves the figure to ``outPutFile``.
    """

    # %matplotlib notebook
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xyzLabels[0], fontsize=15)
    ax.set_ylabel(xyzLabels[1], fontsize=15)
    ax.set_zlabel(xyzLabels[2], fontsize=15)
    ax.set_title(utitle, fontsize=20)
    targets = pd.unique(uY)
    colors = ['r', 'g', 'b', 'y']
    for target, color in list(zip(targets, colors)):
        indicesToKeep = uY.squeeze() == target
        ax.scatter(udata[indicesToKeep, 0], udata[indicesToKeep, 1],
                   udata[indicesToKeep, 2], c=color, s=50, alpha=.5)
    ax.legend(pd.unique(uY).astype('str'))
    ax.grid()

    plt.tight_layout()
    plt.savefig(outPutFile, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('Plot saved in ' + outPutFile)
    # for angle in list(range(0, 360,60)):
    #     ax.view_init(30, angle)
    #     # plt.draw()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outputFolder,fileName.split('.')[0]+str(angle)+'.'+fileName.split('.')[1]) , format='png', dpi=300, bbox_inches='tight')
    #     plt.pause(.001)


def categorical_color_map(label_col, color_palette=None):
  """Map every unique value in a Series to a colour from a Plotly qualitative palette.

  Parameters
  ----------
  label_col : pandas.Series
      Categorical labels to colour-code.
  color_palette : list of str or None, optional
      Hex/rgb colour list. Defaults to
      ``px.colors.qualitative.Alphabet``.

  Returns
  -------
  tuple
      ``(c, color_map)`` — a Series of colours aligned with
      ``label_col`` and the underlying ``{label: color}`` mapping.
  """
  # color_palette=px.colors.qualitative.Antique
  if color_palette is None:
    ## https://plotly.com/python/discrete-color/
    color_palette=px.colors.qualitative.Alphabet  ##Light24  ##Plotly
  domain=label_col.unique()
  if len(domain)>len(color_palette):
    print(f"number of available colors({len(color_palette)}) is more than categorizes({len(domain)}), change the palette")
  color_map = dict(zip(domain, color_palette[:len(domain)+2]))
  c=label_col.map(color_map)
  return c, color_map


def plotly_group_stack(df_plot,
                        col2grp,
                        col2stack,
                        col2c,
                        date_col,
                        title,
                        color_palette=px.colors.qualitative.Light24,
                        patterns=None
                                                ):
  """Build a grouped + stacked plotly bar chart with date on the x-axis.

  X-axis is a 2-level (date, group) index; bars are stacked by
  ``col2stack`` (each stack value gets a different fill ``pattern``).

  Parameters
  ----------
  df_plot : pandas.DataFrame
      Long-format frame containing ``date_col``, ``col2grp``,
      ``col2stack``, and ``col2c``.
  col2grp : str
      Column whose unique values become groups within each date.
  col2stack : str
      Column whose unique values stack on top of each other.
  col2c : str
      Numeric column plotted as bar height.
  date_col : str
      Datetime column for the x-axis.
  title : str
      Plot title.
  color_palette : list of str, optional
      Plotly qualitative palette used per group. Default ``Light24``.
  patterns : list of str or None, optional
      Per-stack pattern strings (e.g. ``''``, ``'/'``). Defaults to
      ``['', '/']``.

  Returns
  -------
  plotly.graph_objects.Figure
  """
  import plotly.graph_objects as go
  if patterns is None:
    patterns = ['', '/']

  x = [list(df_plot[date_col].dt.date.values),list(df_plot[col2grp].values)]

  colors,color_map=categorical_color_map(df_plot[col2grp], color_palette=color_palette)

  fig = go.Figure()

  for shift_name, upattern in zip(df_plot[col2stack].unique(), patterns):
    df_tmp = df_plot.mask(df_plot[col2stack]!=shift_name, pd.NA)
    for Machine in df_plot[col2grp].unique():
      y = df_tmp[col2c].mask(df_plot[col2grp]!=Machine, pd.NA)
      fig.add_bar(
                  x=x,
                  y=y,
                  name=f"{Machine} - {shift_name}",
                  hovertext = df_plot[col2stack],
                  marker_color=color_map[Machine],
                  marker_pattern_shape=upattern,
                  legendgroup=shift_name,
                  legendgrouptitle_text=shift_name,
                  hovertemplate="Date: %{x[0]}<br>"+
                                "Machine: %{x[1]}<br>"+
                                "Efficiency: %{y}<br>"+
                                "DayNight: %{hovertext}<br>",
                )

  fig.update_layout(
                    barmode="relative",
                    xaxis_title="Date",
                    yaxis_title=col2c,
                    legend_title_text='',
                    title=title
                    )
  return fig


def stack_plotly_subplots(figs, sub_titles, main_title):
  """Stack a list of Plotly Express figures into a single 3-row subplot figure.

  Parameters
  ----------
  figs : list of plotly.graph_objects.Figure
      Figures to combine; legends from all but the first are hidden.
  sub_titles : list of str
      Title for each subplot row.
  main_title : str
      Overall figure title.

  Returns
  -------
  plotly.graph_objects.Figure
      Combined 3-row subplot figure (one column).
  """
  from plotly.subplots import make_subplots
  figure_traces=[]
  for con, fig_sub in enumerate(figs):
    figure_traces_sub=[]
    for trace in range(len(fig_sub["data"])):
        if con>0:
          fig_sub["data"][trace]['showlegend'] = False
        figure_traces_sub.append(fig_sub["data"][trace])
    figure_traces.append(figure_traces_sub)
  figure = make_subplots(rows = 3, cols = 1, subplot_titles =sub_titles)
  figure.update_layout(height = 500, width = 1200, title_text =main_title, title_font_size = 25)
  for con, figure_traces_sub in enumerate(figure_traces):
    for traces in figure_traces_sub:
        figure.append_trace(traces, row = con+1, col = 1)
  return figure




@dataclass
class PlotConfig:
    """Rendering options for :class:`DistributionReport`.

    Attributes
    ----------
    height, width : int
        Figure dimensions in pixels.
    colors : list of str or None
        Trace colour cycle; defaults to a five-colour palette.
    theme : {'white', 'dark'}
        Plotly template suffix.
    title_font_size, axis_font_size : int
        Font sizes for the main title and axis titles.
    bins : int
        Histogram bin count.
    boxpoints : {'all', 'outliers', False}
        Passed through to ``plotly.graph_objects.Box``.
    kde_points : int
        Number of samples used to draw the KDE curve.
    jitter : float
        Horizontal jitter for box-plot points.
    violin_points : bool
        Whether to overlay all points on the violin plot.
    first_plot : {'box', 'violin'}
        Which chart occupies the top-left subplot.
    """
    height: int = 800
    width: int = 1200
    colors: Optional[List[str]] = None
    theme: str = 'white'  # or 'dark'
    title_font_size: int = 24
    axis_font_size: int = 14
    bins: int = 20
    boxpoints: Literal['all', 'outliers', False] = 'outliers'
    kde_points: int = 100
    jitter: float = 0.3
    violin_points: bool = True
    first_plot: Literal['box', 'violin'] = 'box'

    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if self.first_plot not in ['box', 'violin']:
            raise ValueError("first_plot must be either 'box' or 'violin'")


def _validate_input(data: pd.DataFrame, dependent_var: str, group_var: str) -> None:
    """Validate that ``data`` has both columns, ``dependent_var`` is numeric, and there are ≤ 10 groups."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")

    if dependent_var not in data.columns:
        raise ValueError(f"Column '{dependent_var}' not found in DataFrame")

    if group_var not in data.columns:
        raise ValueError(f"Column '{group_var}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(data[dependent_var]):
        raise ValueError(f"Column '{dependent_var}' must be numeric")

    if len(data[group_var].unique()) > 10:
        raise ValueError("Too many groups (>10) for meaningful visualization")


@dataclass
class DistributionReport:
    """Per-group distribution diagnostics for one numeric variable.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the data.
    dependent_var : str
        Name of the numeric column to analyze.
    group_var : str
        Name of the categorical grouping column (<= 10 unique values).
    config : PlotConfig, optional
        Rendering options. Defaults to ``PlotConfig()``.
    """

    data: pd.DataFrame
    dependent_var: str
    group_var: str
    config: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self):
        _validate_input(self.data, self.dependent_var, self.group_var)

    def plot(self):
        """Render a 2x2 figure: box/violin, histogram+KDE, ECDF, Q-Q.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import scipy.stats

        config = self.config
        data = self.data
        dependent_var = self.dependent_var
        group_var = self.group_var

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{config.first_plot.capitalize()} Plot',
                'Histogram with KDE',
                'Empirical CDF',
                'Q-Q Plot'
            )
        )

        groups = sorted(data[group_var].unique())

        # 1. Box or Violin plot (top-left)
        for i, group in enumerate(groups):
            group_data = data[data[group_var] == group][dependent_var]
            if config.first_plot == 'box':
                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=group,
                        boxpoints=config.boxpoints,
                        jitter=config.jitter,
                        pointpos=-1.8,
                        marker_color=config.colors[i % len(config.colors)],
                        marker=dict(size=6),
                        legendgroup=group,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            else:  # violin plot
                fig.add_trace(
                    go.Violin(
                        y=group_data,
                        name=group,
                        box_visible=True,
                        meanline_visible=True,
                        points="all" if config.violin_points else None,
                        marker_color=config.colors[i % len(config.colors)],
                        legendgroup=group,
                        showlegend=True
                    ),
                    row=1, col=1
                )

        # 2. Histogram with KDE (top-right)
        for i, group in enumerate(groups):
            group_data = data[data[group_var] == group][dependent_var]

            kde_x = np.linspace(group_data.min(), group_data.max(), config.kde_points)
            kde = scipy.stats.gaussian_kde(group_data)
            kde_y = kde(kde_x)

            hist, bin_edges = np.histogram(group_data, bins=config.bins)
            scaling_factor = max(hist) / max(kde_y)
            kde_y = kde_y * scaling_factor

            fig.add_trace(
                go.Histogram(
                    x=group_data,
                    name=group,
                    opacity=0.7,
                    nbinsx=config.bins,
                    marker_color=config.colors[i % len(config.colors)],
                    legendgroup=group,
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    name=group,
                    line=dict(color=config.colors[i % len(config.colors)]),
                    legendgroup=group,
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. ECDF (bottom-left)
        for i, group in enumerate(groups):
            group_data = data[data[group_var] == group][dependent_var]

            sorted_data = np.sort(group_data)
            n = len(sorted_data)
            ecdf = np.arange(1, n + 1) / n

            fig.add_trace(
                go.Scatter(
                    x=sorted_data,
                    y=ecdf,
                    name=group,
                    mode='lines',
                    line=dict(color=config.colors[i % len(config.colors)]),
                    legendgroup=group,
                    showlegend=False
                ),
                row=2, col=1
            )

        # 4. Q-Q Plot (bottom-right)
        for i, group in enumerate(groups):
            group_data = data[data[group_var] == group][dependent_var]
            qq = scipy.stats.probplot(group_data, dist="norm")

            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    mode='markers',
                    name=group,
                    marker=dict(color=config.colors[i % len(config.colors)]),
                    legendgroup=group,
                    showlegend=False
                ),
                row=2, col=2
            )

            z = np.polyfit(qq[0][0], qq[0][1], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=qq[0][0],
                    y=p(qq[0][0]),
                    mode='lines',
                    name=group,
                    line=dict(color=config.colors[i % len(config.colors)], dash='dash'),
                    legendgroup=group,
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=config.height,
            width=config.width,
            title=dict(
                text="Distribution Analysis",
                x=0.5,
                font=dict(size=config.title_font_size)
            ),
            template=f"plotly_{config.theme}",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            boxmode='group',
            violinmode='group'
        )

        fig.update_xaxes(title_text=group_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
        fig.update_xaxes(title_text=dependent_var, row=1, col=2, title_font=dict(size=config.axis_font_size))
        fig.update_xaxes(title_text=dependent_var, row=2, col=1, title_font=dict(size=config.axis_font_size))
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))

        fig.update_yaxes(title_text=dependent_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
        fig.update_yaxes(title_text="Count", row=1, col=2, title_font=dict(size=config.axis_font_size))
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1, title_font=dict(size=config.axis_font_size))
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))

        return fig


# ---------------------------------------------------------------------------
# Plotly figure I/O helpers
# ---------------------------------------------------------------------------


def figures_to_html(figs, filename: str = "dashboard.html") -> None:
    """Write a list of Plotly figures to a single stacked HTML dashboard.

    Each figure's ``<body>`` fragment is extracted and concatenated inside
    a minimal HTML shell, so the file opens as a single scroll-through
    dashboard.

    Parameters
    ----------
    figs : iterable of plotly.graph_objects.Figure
        Figures to include, in the order they should appear.
    filename : str, optional
        Output HTML path. Default ``"dashboard.html"``.

    Returns
    -------
    None
        The dashboard is written to ``filename``; nothing is returned.
    """
    # from https://stackoverflow.com/questions/45577255/plot-multiple-figures-as-subplots
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")


def save_plotly_fig(fig, fname_prefix: str, image_format: str = "jpg") -> list[str]:
    """Save a plotly figure to HTML, JSON, and a raster image at one prefix.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to export.
    fname_prefix : str
        Common prefix; ``.html``, ``.json``, and the chosen image
        extension are appended.
    image_format : str, optional
        Raster image format (e.g. ``'jpg'``, ``'png'``). Default
        ``'jpg'``.

    Returns
    -------
    list of str
        Paths of all written files.
    """
    import plotly
    uFiles = [f"{fname_prefix}.{ext}" for ext in ["html", "json", image_format]]
    for uFile in uFiles:
        ext = uFile.split(".")[-1]
        if ext == "html":
            plotly.offline.plot(fig, filename=uFile, auto_open=False)
        elif ext == "json":
            plotly.io.write_json(fig, uFile)
        else:
            fig.write_image(uFile, width=2400, height=1400, scale=4)
    return uFiles


def save_fig(fig, fname_prefix: str, formats=("html", "json", "png")) -> list[str]:
    """Write a plotly figure to ``<prefix>.{html,json,png,...}`` in one call.

    Lightweight alternative to :func:`save_plotly_fig`: caller picks the
    formats, raster output uses ``fig.write_image`` with ``scale=2``. Returns
    the list of paths actually written. Raster formats (png/jpg/svg) require
    ``kaleido``; missing-engine errors are surfaced rather than silently
    swallowed.
    """
    import plotly.io as pio

    written = []
    for ext in formats:
        path = f"{fname_prefix}.{ext}"
        if ext == "html":
            fig.write_html(path)
        elif ext == "json":
            pio.write_json(fig, path)
        else:
            fig.write_image(path, scale=2)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Statistical Process Control (SPC) charts
# ---------------------------------------------------------------------------

_DEFAULT_FIG_WIDTH = 1200
_DEFAULT_FIG_HEIGHT = 800
_DEFAULT_HIST_FIGSIZE = (20, 10)
_DEFAULT_QUANTILE_RANGE = (0.10, 0.90)
_SPC_LINE_STYLES = (
    {'color': 'cornflowerblue', 'width': 1, 'dash': 'solid'},
    {'color': 'chocolate',      'width': 1, 'dash': 'solid'},
    {'color': 'darkcyan',       'width': 1, 'dash': 'solid'},
)


def hist_plot(df: pd.DataFrame, col: str, quantile_range=None) -> list[float]:
  """Plot a histogram with KDE and vertical lines at the requested quantiles.

  Parameters
  ----------
  df : pandas.DataFrame
      Source data.
  col : str
      Column whose distribution is plotted.
  quantile_range : list of float or None, optional
      Quantiles in [0, 1] to mark on the chart. Defaults to
      ``[0.10, 0.90]``.

  Returns
  -------
  list of float
      The quantile values plotted, in the same order as ``quantile_range``.
  """
  if quantile_range is None:
    quantile_range = list(_DEFAULT_QUANTILE_RANGE)
  fig, ax = plt.subplots(figsize=_DEFAULT_HIST_FIGSIZE)
  sns.histplot(ax=ax, data=df.reset_index(), x=col, kde=True)

  quan = [df[col].quantile(q) for q in quantile_range]
  # Place labels near the top of the axes so they stay in-frame regardless
  # of the y-axis scale (previously hardcoded y=1).
  y_top = ax.get_ylim()[1]
  for x_loc, q in zip(quan, quantile_range):
      ax.axvline(x=x_loc, color='blue', linestyle='--')
      ax.text(x_loc + 0.02, y_top * 0.95, f'quantile {q * 100}%: {round(x_loc, 1)}')

  return quan


def _add_series_traces(fig, df, x_col: str, cols: list[str], prefix: str, row: int):
  """Add one scatter trace per column to a Plotly subplot; return the last trace."""
  import plotly.graph_objects as go
  last_trace = None
  for y in cols:
    trace = go.Scatter(x=df[x_col], y=df[y], mode='markers', name=f'{prefix}{y}')
    fig.add_trace(trace, row=row, col=1)
    last_trace = trace
  return last_trace


def _limit_line_style(limit_col: str, line: str) -> tuple[str, str]:
  """Return ``(color, legend_name)`` for a single limit line."""
  if 'AVG' in line:
    return 'blue', 'avg'
  if limit_col == 'ctrl_lmt':
    return 'red', 'control_limit'
  if limit_col == 'sigma_lmt':
    return 'brown', '3Sigma_limit'
  return 'blue', 'avg'


def _add_limit_lines(fig, limits_sub, x_range, row: int, is_first_panel: bool):
  """Add horizontal limit lines (LL/AVG/UL) to a Plotly subplot."""
  import plotly.graph_objects as go
  for limit_col in limits_sub:
    series = limits_sub[limit_col]
    for con2, line in enumerate(series.index):
      linecolor, name = _limit_line_style(limit_col, line)
      show = is_first_panel and (con2 == 0) and ('AVG' not in line)
      fig.add_trace(
          go.Scatter(
              name=name,
              x=list(x_range),
              y=[series.loc[line], series.loc[line]],
              mode="lines",
              line={'color': linecolor, 'width': 1, 'dash': 'dash'},
              showlegend=show,
          ),
          row=row, col=1,
      )


def plot_I_MR(df_I: pd.DataFrame, limits, x_col: str = 'TimeStamp', fig=None):
  """Plot interactive I-MR (Individual + Moving-Range) control charts.

  Parameters
  ----------
  df_I : pandas.DataFrame
      Time-indexed values; must contain ``x_col`` plus the numeric columns
      to chart. The Moving-Range series is computed internally as
      ``df_I.diff().abs()``.
  limits : pandas.Series or pandas.DataFrame
      Control limits (typically from :func:`i_mr_ctrl_limits`).
  x_col : str, optional
      Name of the time column. Defaults to ``'TimeStamp'``.
  fig : plotly.graph_objects.Figure, optional
      Existing figure to draw on; a new one is created if ``None``.

  Returns
  -------
  plotly.graph_objects.Figure
      Figure containing one I and one MR subplot per numeric column.
  """
  from plotly.subplots import make_subplots

  df_MR = df_I.drop([x_col], axis=1).diff().abs()
  df_MR[x_col] = df_I[x_col]

  if isinstance(limits, pd.Series):
    limits = limits.to_frame()

  if fig is None:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Individual Chart", "Moving Range Chart"),
        shared_xaxes=True,
    )

  cols = [c for c in df_I.columns if c != x_col and c != 'TimeStamp']

  for panel_idx, (df_plot, prefix) in enumerate([(df_I, 'I_'), (df_MR, 'MR_')]):
    last_trace = _add_series_traces(fig, df_plot, x_col, cols, prefix, row=panel_idx + 1)
    x_range = (min(last_trace.x), max(last_trace.x))
    limits_sub = limits[limits.index.str.contains(prefix)]
    _add_limit_lines(fig, limits_sub, x_range, row=panel_idx + 1, is_first_panel=(panel_idx == 0))

  fig.update_layout(
      width=_DEFAULT_FIG_WIDTH,
      height=_DEFAULT_FIG_HEIGHT,
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
  )
  y_title = ', '.join(df_I.columns.drop(['TimeStamp', x_col], errors='ignore').tolist())
  fig.update_yaxes(title_text=y_title)
  return fig


def plot_series_overlay(
    df_plot: pd.DataFrame,
    x_col: str = 'TimeStamp',
    name_prefix: str = 'Series',
    baseline_suffix: str | None = None,
):
  """Overlay one or more numeric series against a common x column.

  Renders every non-``x_col`` column as a Plotly scatter trace on a single
  subplot. Trace names are ``f"{name_prefix}_{suffix}"``, where ``suffix``
  is the substring after the last ``_`` in the column name; if
  ``baseline_suffix`` matches that substring, the trace is renamed to
  the raw ``baseline_suffix`` value (useful for highlighting a reference
  series inside a group of derived series).

  Parameters
  ----------
  df_plot : pandas.DataFrame
      Wide-format frame with a time (or other) x column plus one or more
      numeric series to overlay.
  x_col : str, optional
      Name of the x column. Defaults to ``'TimeStamp'``.
  name_prefix : str, optional
      Prefix for derived-trace legend names. Defaults to ``'Series'``.
  baseline_suffix : str or None, optional
      Column-name suffix (after the last ``_``) that should be rendered
      with its bare suffix as the legend label (no prefix). Default
      ``None``.

  Returns
  -------
  plotly.graph_objects.Figure
      Single-row figure with overlaid line traces for each non-x column.
  """
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots

  cols0 = [c for c in df_plot.columns if c != x_col and c != 'TimeStamp']
  fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

  mode = 'markers' if x_col == 'TimeStamp' else 'markers+lines'
  for idx, col in enumerate(cols0):
    suffix = col.split('_')[-1]
    if baseline_suffix is not None and suffix == baseline_suffix:
      trace_name = suffix
    else:
      trace_name = f"{name_prefix}_{suffix}"
    fig.add_trace(
        go.Scatter(
            x=df_plot[x_col], y=df_plot[col],
            mode=mode,
            line=_SPC_LINE_STYLES[idx % len(_SPC_LINE_STYLES)],
            name=trace_name,
            showlegend=True,
        ),
        row=1, col=1,
    )

  fig.update_layout(
      width=_DEFAULT_FIG_WIDTH,
      height=_DEFAULT_FIG_HEIGHT,
      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
  )
  y_title = ', '.join(cols0) if cols0 else ''
  fig.update_yaxes(title_text=y_title)
  return fig

