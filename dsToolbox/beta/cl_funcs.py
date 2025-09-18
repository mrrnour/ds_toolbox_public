"""
Enhanced Control Charts and Statistical Process Control Module.

This module provides comprehensive functionality for statistical process control (SPC),
control charts, and quality control analysis. It includes sigma limits calculation,
I-MR charts, visualization tools, and recovery analysis.

Classes:
    SigmaLimitCalculator: Statistical process control limits using sigma method
    ControlLimitCalculator: Process control limits using control chart methodology  
    ControlChartVisualizer: Visualization tools for control charts
    ProcessRecoveryAnalyzer: Recovery and process analysis tools

Author: Data Science Toolbox
Version: 2.0 (Refactored)
"""

import os
import sys
import datetime
import math
import warnings
from typing import List, Tuple, Union, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pylab as pl
import seaborn as sns

# Configure pandas display options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Configure seaborn
sns.set_style("darkgrid")
sns.set(rc={'figure.figsize': (20, 10)})

# Optional imports with graceful fallback
try:
    import plotly.express as px
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plotting features will be disabled.")


class SigmaLimitCalculator:
    """
    Statistical Process Control using Sigma Limits methodology.
    
    This class provides methods for calculating control limits based on standard
    deviation (sigma) methodology, commonly used in statistical process control.
    """
    
    @staticmethod
    def calculate_sigma_limits(data: Union[pd.Series, np.ndarray], coefficient: float = 3) -> Tuple[float, float, float]:
        """
        Calculate sigma-based control limits for process data.
        
        Computes Lower Control Limit (LCL), mean, and Upper Control Limit (UCL)
        based on standard deviation methodology.
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Process data for control limit calculation
        coefficient : float, default 3
            Sigma coefficient for control limits (typically 2 or 3)
            
        Returns
        -------
        tuple
            (LCL, mean, UCL) - Lower limit, center line, upper limit
            
        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> lcl, mean, ucl = SigmaLimitCalculator.calculate_sigma_limits(data)
        >>> print(f"LCL: {lcl:.2f}, Mean: {mean:.2f}, UCL: {ucl:.2f}")
        """
        try:
            if data is None or len(data) == 0:
                raise ValueError("Data cannot be None or empty")
                
            data_array = np.array(data)
            if not np.isfinite(data_array).all():
                warnings.warn("Data contains non-finite values. They will be excluded from calculations.")
                data_array = data_array[np.isfinite(data_array)]
                
            if len(data_array) == 0:
                raise ValueError("No valid data points after removing non-finite values")
                
            data_mean = np.mean(data_array)
            data_std = np.std(data_array, ddof=1)  # Use sample standard deviation
            
            lcl = data_mean - coefficient * data_std
            ucl = data_mean + coefficient * data_std
            
            return (lcl, data_mean, ucl)
            
        except Exception as e:
            raise ValueError(f"Error calculating sigma limits: {str(e)}")
    
    @staticmethod
    def calculate_grouped_sigma_limits(df: pd.DataFrame, column: str, 
                                     group_by_columns: Optional[List[str]] = None,
                                     coefficient: float = 3) -> pd.DataFrame:
        """
        Calculate sigma limits for grouped data.
        
        Computes control limits for each group in the dataset, useful for
        multi-stream or multi-machine process control.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        column : str
            Column name for which to calculate control limits
        group_by_columns : list of str, optional
            Columns to group by. If None, calculates for entire dataset
        coefficient : float, default 3
            Sigma coefficient for control limits
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: LL_{column}, AVG_{column}, UL_{column}
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'machine': ['A', 'A', 'B', 'B'],
        ...     'measurement': [10, 12, 15, 18]
        ... })
        >>> limits = SigmaLimitCalculator.calculate_grouped_sigma_limits(
        ...     df, 'measurement', ['machine']
        ... )
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            if group_by_columns is not None:
                missing_cols = [col for col in group_by_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Group by columns not found: {missing_cols}")
                    
                cls = df.groupby(group_by_columns)[column].apply(
                    lambda x: SigmaLimitCalculator.calculate_sigma_limits(x, coefficient)
                )
            else:
                cls = SigmaLimitCalculator.calculate_sigma_limits(df[column], coefficient)
                cls = pd.Series([cls])  # Convert to series for consistent handling
                
            cls_df = pd.DataFrame(
                cls.tolist(), 
                index=cls.index, 
                columns=[f'LL_{column}', f'AVG_{column}', f'UL_{column}']
            ).rename_axis(cls.index.name)
            
            return cls_df
            
        except Exception as e:
            raise ValueError(f"Error calculating grouped sigma limits: {str(e)}")
    
    @staticmethod
    def calculate_multiple_columns_sigma_limits(df: pd.DataFrame, columns: List[str],
                                              group_by_columns: List[str],
                                              coefficient: float = 3) -> pd.DataFrame:
        """
        Calculate sigma limits for multiple columns with grouping.
        
        Efficiently computes control limits for multiple process variables
        grouped by specified columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        columns : list of str
            Column names for which to calculate control limits
        group_by_columns : list of str
            Columns to group by
        coefficient : float, default 3
            Sigma coefficient for control limits
            
        Returns
        -------
        pd.DataFrame
            DataFrame with control limits for all specified columns
        """
        try:
            if not columns:
                raise ValueError("Columns list cannot be empty")
                
            cls_all = pd.DataFrame()
            
            for col in columns:
                cls = SigmaLimitCalculator.calculate_grouped_sigma_limits(
                    df, column=col, group_by_columns=group_by_columns, coefficient=coefficient
                )
                cls_all = pd.concat([cls_all, cls], axis=1)
                
            cls_all = cls_all.reset_index()
            return cls_all
            
        except Exception as e:
            raise ValueError(f"Error calculating multiple columns sigma limits: {str(e)}")
    
    @staticmethod
    def calculate_i_mr_sigma_limits(df: pd.DataFrame, columns: List[str],
                                   group_by_column: str) -> pd.DataFrame:
        """
        Calculate Individual and Moving Range (I-MR) sigma limits.
        
        Computes control limits for both individual values and moving ranges,
        essential for continuous process monitoring.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        columns : list of str
            Column names for process measurements
        group_by_column : str
            Column to group by (e.g., 'machine', 'line')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with I and MR control limits for all columns
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'machine': ['A']*5 + ['B']*5,
        ...     'temp': [100, 102, 98, 101, 99, 105, 103, 107, 104, 106],
        ...     'pressure': [50, 52, 48, 51, 49, 55, 53, 57, 54, 56]
        ... })
        >>> limits = SigmaLimitCalculator.calculate_i_mr_sigma_limits(
        ...     df, ['temp', 'pressure'], 'machine'
        ... )
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if group_by_column not in df.columns:
                raise ValueError(f"Group by column '{group_by_column}' not found")
                
            # Calculate Individual chart control limits
            i_chart_cls = SigmaLimitCalculator.calculate_multiple_columns_sigma_limits(
                df, columns=columns, group_by_columns=[group_by_column]
            ).set_index(group_by_column)
            
            # Calculate Moving Range data
            mr_data = df.copy()
            mr_data[columns] = mr_data.groupby(group_by_column)[columns].diff(axis=0).abs()
            
            # Calculate Moving Range chart control limits
            mr_chart_cls = SigmaLimitCalculator.calculate_multiple_columns_sigma_limits(
                mr_data, columns=columns, group_by_columns=[group_by_column]
            ).set_index(group_by_column)
            
            # Rename columns with prefixes
            i_chart_cls.columns = [f'I_{col}' for col in i_chart_cls.columns]
            mr_chart_cls.columns = [f'MR_{col}' for col in mr_chart_cls.columns]
            
            # Combine results
            sigma_limits = pd.concat([i_chart_cls, mr_chart_cls], axis=1).reset_index()
            
            return sigma_limits
            
        except Exception as e:
            raise ValueError(f"Error calculating I-MR sigma limits: {str(e)}")


class ControlLimitCalculator:
    """
    Process Control Limits using Control Chart methodology.
    
    This class provides methods for calculating control limits using traditional
    control chart formulas with appropriate constants for different chart types.
    """
    
    @staticmethod
    def calculate_control_limits(data: Union[pd.Series, np.ndarray]) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate I-MR control limits using control chart methodology.
        
        Computes Individual (I) and Moving Range (MR) control limits using
        standard control chart constants (2.66 for I-chart, 3.267 for MR-chart).
        
        Parameters
        ----------
        data : pd.Series or np.ndarray
            Process data for control limit calculation
            
        Returns
        -------
        tuple
            (I_LCL, I_avg, I_UCL, MR_LCL, MR_avg, MR_UCL)
            
        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.Series([100, 102, 98, 101, 99])
        >>> limits = ControlLimitCalculator.calculate_control_limits(data)
        >>> i_lcl, i_avg, i_ucl, mr_lcl, mr_avg, mr_ucl = limits
        """
        try:
            if data is None or len(data) == 0:
                raise ValueError("Data cannot be None or empty")
                
            data_array = np.array(data)
            if not np.isfinite(data_array).all():
                warnings.warn("Data contains non-finite values. They will be excluded from calculations.")
                data_array = data_array[np.isfinite(data_array)]
                
            if len(data_array) < 2:
                raise ValueError("Need at least 2 data points for control limit calculation")
                
            # Calculate Individual chart limits
            x_avg = np.mean(data_array)
            
            # Calculate moving ranges
            moving_ranges = np.abs(np.diff(data_array))
            if len(moving_ranges) == 0:
                raise ValueError("Cannot calculate moving ranges with less than 2 data points")
                
            mr_avg = np.mean(moving_ranges)
            
            # Control chart constants
            # For Individual chart: ±2.66 * MR_avg
            # For MR chart: 3.267 * MR_avg (LCL is always 0)
            x_lcl = x_avg - mr_avg * 2.66
            x_ucl = x_avg + mr_avg * 2.66
            
            mr_lcl = 0  # MR LCL is always 0
            mr_ucl = mr_avg * 3.267
            
            return (x_lcl, x_avg, x_ucl, mr_lcl, mr_avg, mr_ucl)
            
        except Exception as e:
            raise ValueError(f"Error calculating control limits: {str(e)}")
    
    @staticmethod
    def calculate_grouped_control_limits(df: pd.DataFrame, column: str,
                                       group_by_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate control limits for grouped data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        column : str
            Column name for which to calculate control limits
        group_by_columns : list of str, optional
            Columns to group by. If None, calculates for entire dataset
            
        Returns
        -------
        pd.DataFrame
            DataFrame with I and MR control limits
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            if group_by_columns is not None:
                missing_cols = [col for col in group_by_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Group by columns not found: {missing_cols}")
                    
                cls = df.groupby(group_by_columns)[column].apply(
                    ControlLimitCalculator.calculate_control_limits
                )
            else:
                cls = ControlLimitCalculator.calculate_control_limits(df[column])
                cls = pd.Series([cls])
                
            cls_df = pd.DataFrame(
                cls.tolist(),
                index=cls.index,
                columns=[
                    f'I_LL_{column}', f'I_AVG_{column}', f'I_UL_{column}',
                    f'MR_LL_{column}', f'MR_AVG_{column}', f'MR_UL_{column}'
                ]
            ).rename_axis(cls.index.name)
            
            return cls_df
            
        except Exception as e:
            raise ValueError(f"Error calculating grouped control limits: {str(e)}")
    
    @staticmethod
    def calculate_i_mr_control_limits(df: pd.DataFrame, columns: List[str],
                                    group_by_column: str) -> pd.DataFrame:
        """
        Calculate I-MR control limits for multiple columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        columns : list of str
            Column names for process measurements
        group_by_column : str
            Column to group by
            
        Returns
        -------
        pd.DataFrame
            DataFrame with I-MR control limits for all columns
            
        Note
        ----
        This function has a parameter 'coef' in the original that is not used.
        It's maintained for backward compatibility but doesn't affect calculations.
        """
        try:
            if not columns:
                raise ValueError("Columns list cannot be empty")
                
            cls_all = pd.DataFrame()
            
            for col in columns:
                cls = ControlLimitCalculator.calculate_grouped_control_limits(
                    df, column=col, group_by_columns=[group_by_column]
                )
                cls_all = pd.concat([cls_all, cls], axis=1)
                
            cls_all = cls_all.reset_index()
            return cls_all
            
        except Exception as e:
            raise ValueError(f"Error calculating I-MR control limits: {str(e)}")


class ControlChartVisualizer:
    """
    Visualization tools for control charts and statistical process control.
    
    This class provides comprehensive plotting capabilities for control charts,
    histograms, and process analysis visualizations.
    """
    
    @staticmethod
    def create_histogram_plot(df: pd.DataFrame, column: str,
                            quantile_range: List[float] = [0.10, 0.90],
                            figsize: Tuple[int, int] = (20, 10)) -> List[float]:
        """
        Create histogram plot with quantile markers.
        
        Generates a histogram with kernel density estimation and quantile markers
        to visualize data distribution and process capability.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing process data
        column : str
            Column name to plot
        quantile_range : list of float, default [0.10, 0.90]
            Quantiles to mark on the plot
        figsize : tuple, default (20, 10)
            Figure size (width, height)
            
        Returns
        -------
        list
            Quantile values calculated for the specified range
            
        Examples
        --------
        >>> df = pd.DataFrame({'temperature': [98, 99, 100, 101, 102]})
        >>> quantiles = ControlChartVisualizer.create_histogram_plot(
        ...     df, 'temperature', [0.1, 0.9]
        ... )
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            if not quantile_range or len(quantile_range) != 2:
                raise ValueError("quantile_range must be a list of exactly 2 values")
                
            if not all(0 <= q <= 1 for q in quantile_range):
                raise ValueError("Quantile values must be between 0 and 1")
                
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create histogram with KDE
            sns.histplot(
                ax=ax,
                data=df.reset_index(),
                x=column,
                kde=True
            )
            
            # Calculate and plot quantiles
            quantiles = [df[column].quantile(q) for q in quantile_range]
            
            for x_loc, q in zip(quantiles, quantile_range):
                ax.axvline(x=x_loc, color='blue', linestyle='--')
                ax.text(x_loc + 0.02, 1, f'quantile {q*100}%: {round(x_loc, 1)}')
            
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
            
            return quantiles
            
        except Exception as e:
            raise ValueError(f"Error creating histogram plot: {str(e)}")
    
    @staticmethod
    def create_i_mr_chart(df_i: pd.DataFrame, limits: pd.DataFrame,
                         x_column: str = 'TimeStamp',
                         fig=None) -> Any:
        """
        Create Interactive Individual and Moving Range (I-MR) control charts.
        
        Generates interactive Plotly charts showing both Individual and Moving Range
        charts with control limits and sigma limits overlaid.
        
        Parameters
        ----------
        df_i : pd.DataFrame
            DataFrame containing Individual values data
        limits : pd.DataFrame or pd.Series
            Control limits data with appropriate column naming
        x_column : str, default 'TimeStamp'
            Column name for x-axis (typically time)
        fig : plotly.graph_objects.Figure, optional
            Existing figure to add traces to
            
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure with I-MR charts
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'TimeStamp': pd.date_range('2023-01-01', periods=10),
        ...     'temperature': [100, 102, 98, 101, 99, 103, 97, 104, 96, 105]
        ... })
        >>> limits = pd.DataFrame({
        ...     'I_LL_temperature': [90], 'I_AVG_temperature': [100], 'I_UL_temperature': [110],
        ...     'MR_LL_temperature': [0], 'MR_AVG_temperature': [2], 'MR_UL_temperature': [6]
        ... })
        >>> fig = ControlChartVisualizer.create_i_mr_chart(df, limits)
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for interactive charts. Install with: pip install plotly")
                
            if df_i is None or df_i.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if x_column not in df_i.columns:
                raise ValueError(f"X-axis column '{x_column}' not found in DataFrame")
            
            # Prepare Moving Range data
            df_mr = df_i.drop([x_column], axis=1).diff(axis=0).abs()
            df_mr[x_column] = df_i[x_column]
            
            # Convert limits to DataFrame if Series
            if isinstance(limits, pd.Series):
                limits = limits.to_frame().T
                
            # Create subplots if figure not provided
            if fig is None:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Individual Chart", "Moving Range Chart"),
                    shared_xaxes=True
                )
            
            # Get data columns (exclude time columns)
            data_cols = [col for col in df_i.columns 
                        if col not in [x_column, 'TimeStamp']]
            
            # Plot Individual and Moving Range data
            for chart_idx, df_plot in enumerate([df_i, df_mr]):
                prefix = 'I_' if chart_idx == 0 else 'MR_'
                
                # Plot data points
                for y_col in data_cols:
                    trace = go.Scatter(
                        x=df_plot[x_column],
                        y=df_plot[y_col],
                        mode='markers',
                        name=f'{prefix}{y_col}'
                    )
                    fig.add_trace(trace, row=chart_idx + 1, col=1)
                
                # Plot control limits
                if not limits.empty:
                    limit_cols = [col for col in limits.columns if col.startswith(prefix)]
                    
                    for limit_col in limit_cols:
                        for _, limit_value in limits[limit_col].items():
                            # Determine line color and name based on limit type
                            if 'AVG' in limit_col:
                                line_color = 'blue'
                                name = 'avg'
                            elif 'ctrl_lmt' in str(limits.columns):
                                line_color = 'red'
                                name = 'control_limit'
                            else:
                                line_color = 'brown'
                                name = '3Sigma_limit'
                            
                            fig.add_trace(
                                go.Scatter(
                                    name=name,
                                    x=[df_plot[x_column].min(), df_plot[x_column].max()],
                                    y=[limit_value, limit_value],
                                    mode="lines",
                                    line={
                                        'color': line_color,
                                        'width': 1,
                                        'dash': 'dash'
                                    },
                                    showlegend=('AVG' not in limit_col) and (chart_idx == 0)
                                ),
                                row=chart_idx + 1, col=1
                            )
            
            # Update layout
            fig.update_layout(
                width=1200,
                height=800,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            y_title = ', '.join([col for col in df_i.columns 
                               if col not in ['TimeStamp', x_column]])
            fig.update_yaxes(title_text=y_title)
            
            return fig
            
        except Exception as e:
            raise ValueError(f"Error creating I-MR chart: {str(e)}")
    
    @staticmethod
    def create_recovery_plot(df_plot: pd.DataFrame,
                           x_column: str = 'TimeStamp') -> Any:
        """
        Create recovery analysis plot for process monitoring.
        
        Generates interactive plots for recovery analysis, typically used
        in mining or manufacturing processes to track recovery rates.
        
        Parameters
        ----------
        df_plot : pd.DataFrame
            DataFrame containing recovery data
        x_column : str, default 'TimeStamp'
            Column name for x-axis
            
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive Plotly figure showing recovery trends
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for interactive charts. Install with: pip install plotly")
                
            if df_plot is None or df_plot.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            if x_column not in df_plot.columns:
                raise ValueError(f"X-axis column '{x_column}' not found in DataFrame")
            
            # Get data columns
            data_cols = [col for col in df_plot.columns 
                        if col not in [x_column, 'TimeStamp']]
            
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
            
            # Define line styles
            line_styles = [
                {'color': 'cornflowerblue', 'width': 1, 'dash': 'solid'},
                {'color': 'chocolate', 'width': 1, 'dash': 'solid'},
                {'color': 'darkcyan', 'width': 1, 'dash': 'solid'}
            ]
            
            # Determine plot mode
            mode = 'markers' if x_column == 'TimeStamp' else 'markers+lines'
            
            # Add traces
            for idx, col in enumerate(data_cols):
                line_style = line_styles[idx % len(line_styles)]
                suffix = col.split('_')[-1] if '_' in col else col
                
                display_name = f"Recovery_{suffix}" if suffix != 'asMine' else 'asMine'
                
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[x_column],
                        y=df_plot[col],
                        mode=mode,
                        line=line_style,
                        name=display_name,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                width=1200,
                height=800,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            if data_cols:
                fig.update_yaxes(title_text=data_cols[0])
            
            return fig
            
        except Exception as e:
            raise ValueError(f"Error creating recovery plot: {str(e)}")


class ProcessRecoveryAnalyzer:
    """
    Process Recovery and Mining Analysis Tools.
    
    This class provides specialized tools for analyzing process recovery,
    particularly useful in mining, manufacturing, and chemical processes.
    """
    
    @staticmethod
    def calculate_recovery_metrics(df: pd.DataFrame, 
                                 input_column: str,
                                 output_column: str) -> Dict[str, float]:
        """
        Calculate basic recovery metrics.
        
        Parameters
        ----------
        df : pd.DataFrame
            Process data
        input_column : str
            Column representing input values
        output_column : str
            Column representing output values
            
        Returns
        -------
        dict
            Dictionary containing recovery metrics
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
                
            required_cols = [input_column, output_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Required columns not found: {missing_cols}")
            
            # Calculate basic recovery metrics
            total_input = df[input_column].sum()
            total_output = df[output_column].sum()
            
            if total_input == 0:
                raise ValueError("Total input cannot be zero for recovery calculation")
            
            recovery_rate = (total_output / total_input) * 100
            loss_rate = 100 - recovery_rate
            
            metrics = {
                'total_input': total_input,
                'total_output': total_output,
                'recovery_rate_percent': recovery_rate,
                'loss_rate_percent': loss_rate,
                'efficiency': recovery_rate / 100  # As decimal
            }
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Error calculating recovery metrics: {str(e)}")


# Backward compatibility functions with deprecation warnings
def sigma_limit(df, coef=3):
    """
    DEPRECATED: Use SigmaLimitCalculator.calculate_sigma_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "sigma_limit() is deprecated. Use SigmaLimitCalculator.calculate_sigma_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SigmaLimitCalculator.calculate_sigma_limits(df, coef)


def sigma_limit_grpby(df, col, grpby_col=[], coef=3):
    """
    DEPRECATED: Use SigmaLimitCalculator.calculate_grouped_sigma_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "sigma_limit_grpby() is deprecated. Use SigmaLimitCalculator.calculate_grouped_sigma_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    group_cols = grpby_col if grpby_col != [] else None
    return SigmaLimitCalculator.calculate_grouped_sigma_limits(df, col, group_cols, coef)


def sigma_limit_cols_grpby(df, cols, grpby_col, coef=3):
    """
    DEPRECATED: Use SigmaLimitCalculator.calculate_multiple_columns_sigma_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "sigma_limit_cols_grpby() is deprecated. Use SigmaLimitCalculator.calculate_multiple_columns_sigma_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SigmaLimitCalculator.calculate_multiple_columns_sigma_limits(df, cols, grpby_col, coef)


def i_mr_sigma_limits(df, cols, grpby_col):
    """
    DEPRECATED: Use SigmaLimitCalculator.calculate_i_mr_sigma_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "i_mr_sigma_limits() is deprecated. Use SigmaLimitCalculator.calculate_i_mr_sigma_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SigmaLimitCalculator.calculate_i_mr_sigma_limits(df, cols, grpby_col)


def control_limit(df):
    """
    DEPRECATED: Use ControlLimitCalculator.calculate_control_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "control_limit() is deprecated. Use ControlLimitCalculator.calculate_control_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ControlLimitCalculator.calculate_control_limits(df)


def control_limit_grpby(df, col, grpby_col=[]):
    """
    DEPRECATED: Use ControlLimitCalculator.calculate_grouped_control_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "control_limit_grpby() is deprecated. Use ControlLimitCalculator.calculate_grouped_control_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    group_cols = grpby_col if grpby_col != [] else None
    return ControlLimitCalculator.calculate_grouped_control_limits(df, col, group_cols)


def i_mr_ctrl_limits(df, cols, grpby_col, coef):
    """
    DEPRECATED: Use ControlLimitCalculator.calculate_i_mr_control_limits() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "i_mr_ctrl_limits() is deprecated. Use ControlLimitCalculator.calculate_i_mr_control_limits() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ControlLimitCalculator.calculate_i_mr_control_limits(df, cols, grpby_col)


def hist_plot(df, col, quantile_range=[.10, .90]):
    """
    DEPRECATED: Use ControlChartVisualizer.create_histogram_plot() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "hist_plot() is deprecated. Use ControlChartVisualizer.create_histogram_plot() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ControlChartVisualizer.create_histogram_plot(df, col, quantile_range)


def plot_I_MR(df_I, limits, x_col='TimeStamp', fig=None):
    """
    DEPRECATED: Use ControlChartVisualizer.create_i_mr_chart() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "plot_I_MR() is deprecated. Use ControlChartVisualizer.create_i_mr_chart() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ControlChartVisualizer.create_i_mr_chart(df_I, limits, x_col, fig)


def plot_recovery_asMine(df_plot, x_col='TimeStamp'):
    """
    DEPRECATED: Use ControlChartVisualizer.create_recovery_plot() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "plot_recovery_asMine() is deprecated. Use ControlChartVisualizer.create_recovery_plot() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return ControlChartVisualizer.create_recovery_plot(df_plot, x_col)


# Function mapping for reference
FUNCTION_MAPPING = {
    # Sigma Limits Functions
    'sigma_limit': 'SigmaLimitCalculator.calculate_sigma_limits',
    'sigma_limit_grpby': 'SigmaLimitCalculator.calculate_grouped_sigma_limits',
    'sigma_limit_cols_grpby': 'SigmaLimitCalculator.calculate_multiple_columns_sigma_limits',
    'i_mr_sigma_limits': 'SigmaLimitCalculator.calculate_i_mr_sigma_limits',
    
    # Control Limits Functions
    'control_limit': 'ControlLimitCalculator.calculate_control_limits',
    'control_limit_grpby': 'ControlLimitCalculator.calculate_grouped_control_limits',
    'i_mr_ctrl_limits': 'ControlLimitCalculator.calculate_i_mr_control_limits',
    
    # Visualization Functions
    'hist_plot': 'ControlChartVisualizer.create_histogram_plot',
    'plot_I_MR': 'ControlChartVisualizer.create_i_mr_chart',
    'plot_recovery_asMine': 'ControlChartVisualizer.create_recovery_plot',
}

__all__ = [
    'SigmaLimitCalculator',
    'ControlLimitCalculator', 
    'ControlChartVisualizer',
    'ProcessRecoveryAnalyzer',
    # Backward compatibility
    'sigma_limit',
    'sigma_limit_grpby',
    'sigma_limit_cols_grpby',
    'i_mr_sigma_limits',
    'control_limit',
    'control_limit_grpby',
    'i_mr_ctrl_limits',
    'hist_plot',
    'plot_I_MR',
    'plot_recovery_asMine',
    'FUNCTION_MAPPING'
]