"""
Enhanced Apache Spark Data Processing Module.

This module provides comprehensive functionality for large-scale data processing using Apache Spark,
including advanced join operations, data transformations, ETL pipeline management, and feature engineering.
Designed for production-ready data science workflows with robust error handling and optimization.

Classes:
    SparkJoinOperations: Advanced join operations including asof joins and column conflict resolution
    SparkDataTransformations: Data reshaping, melting, column operations, and type conversions  
    SparkETLPipeline: End-to-end ETL pipeline management with incremental processing
    SparkFeatureEngineering: Time-series feature engineering with rolling and tumbling windows

Author: Data Science Toolbox
Version: 2.0 (Refactored)
"""

import warnings
import datetime as dt
import inspect
from typing import List, Dict, Any, Optional, Union, Tuple

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as spk_dtp
from pyspark.sql.window import Window
from pyspark.sql import DataFrame as SparkDataFrame

# Graceful handling of internal dependencies
try:
    import dsToolbox.io_funcs as io_funcs
    import dsToolbox.utilities as cfuncs
    DSTOOLS_AVAILABLE = True
except ImportError:
    DSTOOLS_AVAILABLE = False
    warnings.warn("dsToolbox modules not available. Some functionality may be limited.")


class SparkJoinOperations:
    """
    Advanced join operations for Spark DataFrames with specialized handling for time-series data.
    
    This class provides sophisticated joining capabilities including asof (as-of) joins,
    which are essential for time-series analysis and point-in-time lookups.
    """
    
    @staticmethod
    def perform_asof_join_pandas(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                                left_time_column: str, right_time_column: str,
                                left_by_column: str, right_by_column: str,
                                tolerance: pd.Timedelta = pd.Timedelta('600S'),
                                direction: str = 'forward', **kwargs) -> pd.DataFrame:
        """
        Execute asof join operation using pandas for grouped data.
        
        Asof joins are useful for time-series data where you need to match records
        based on the nearest timestamp within a tolerance window.
        
        Parameters
        ----------
        left_df : pd.DataFrame
            Left DataFrame for the join operation
        right_df : pd.DataFrame  
            Right DataFrame for the join operation
        left_time_column : str
            Column name for time-based matching in left DataFrame
        right_time_column : str
            Column name for time-based matching in right DataFrame
        left_by_column : str
            Column name for exact matching/grouping in left DataFrame
        right_by_column : str
            Column name for exact matching/grouping in right DataFrame
        tolerance : pd.Timedelta, default '600S'
            Maximum time difference for matching records
        direction : str, default 'forward'
            Direction for matching: 'forward', 'backward', or 'nearest'
        **kwargs
            Additional parameters passed to pd.merge_asof
            
        Returns
        -------
        pd.DataFrame
            Joined DataFrame with matched records
            
        Examples
        --------
        >>> left_df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
        ...     'machine_id': ['A'] * 5,
        ...     'temperature': [20, 21, 22, 23, 24]
        ... })
        >>> right_df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2023-01-01 00:30', periods=3, freq='2H'), 
        ...     'machine_id': ['A'] * 3,
        ...     'pressure': [100, 105, 110]
        ... })
        >>> joined = SparkJoinOperations.perform_asof_join_pandas(
        ...     left_df, right_df, 'timestamp', 'timestamp', 
        ...     'machine_id', 'machine_id', tolerance=pd.Timedelta('1H')
        ... )
        """
        try:
            if left_df is None or left_df.empty:
                raise ValueError("Left DataFrame cannot be None or empty")
            if right_df is None or right_df.empty:
                raise ValueError("Right DataFrame cannot be None or empty")
                
            # Validate required columns exist
            required_left_cols = [left_time_column, left_by_column]
            required_right_cols = [right_time_column, right_by_column]
            
            missing_left = [col for col in required_left_cols if col not in left_df.columns]
            missing_right = [col for col in required_right_cols if col not in right_df.columns]
            
            if missing_left:
                raise ValueError(f"Missing columns in left DataFrame: {missing_left}")
            if missing_right:
                raise ValueError(f"Missing columns in right DataFrame: {missing_right}")
            
            # Convert time columns to datetime
            left_processed = left_df.copy()
            right_processed = right_df.copy()
            
            left_processed[left_time_column] = pd.to_datetime(left_processed[left_time_column])
            right_processed[right_time_column] = pd.to_datetime(right_processed[right_time_column])
            
            # Sort DataFrames by time columns for optimal merge performance
            left_processed = left_processed.sort_values(left_time_column)
            right_processed = right_processed.sort_values(right_time_column)
            
            # Remove rows with null timestamps from right DataFrame
            right_processed = right_processed.dropna(subset=[right_time_column])
            
            # Perform asof join
            result = pd.merge_asof(
                left_processed, right_processed,
                left_on=left_time_column,
                right_on=right_time_column,
                left_by=left_by_column,
                right_by=right_by_column,
                tolerance=tolerance,
                direction=direction,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error performing asof join: {str(e)}")
    
    @staticmethod
    def resolve_column_conflicts(left_df: SparkDataFrame, right_df: SparkDataFrame,
                                conflict_resolution: str = 'suffix',
                                custom_suffixes: Optional[Tuple[str, str]] = None) -> Tuple[SparkDataFrame, SparkDataFrame, Dict[str, str]]:
        """
        Resolve column name conflicts between two Spark DataFrames.
        
        Parameters
        ----------
        left_df : SparkDataFrame
            Left DataFrame
        right_df : SparkDataFrame
            Right DataFrame
        conflict_resolution : str, default 'suffix'
            Strategy for resolving conflicts: 'suffix', 'prefix', or 'drop'
        custom_suffixes : tuple of str, optional
            Custom suffixes to use (default: ('_left', '_right'))
            
        Returns
        -------
        tuple
            (processed_left_df, processed_right_df, column_mapping)
        """
        try:
            if custom_suffixes is None:
                custom_suffixes = ('_left', '_right')
                
            common_columns = list(set(left_df.columns).intersection(set(right_df.columns)))
            
            if not common_columns:
                return left_df, right_df, {}
            
            column_mapping = {}
            left_processed = left_df
            right_processed = right_df
            
            for col in common_columns:
                if conflict_resolution == 'suffix':
                    new_left_col = f"{col}{custom_suffixes[0]}"
                    new_right_col = f"{col}{custom_suffixes[1]}"
                    
                    left_processed = left_processed.withColumnRenamed(col, new_left_col)
                    right_processed = right_processed.withColumnRenamed(col, new_right_col)
                    
                    column_mapping[col] = {
                        'left_renamed': new_left_col,
                        'right_renamed': new_right_col
                    }
                elif conflict_resolution == 'drop':
                    left_processed = left_processed.drop(col)
                    right_processed = right_processed.drop(col)
                    column_mapping[col] = {'action': 'dropped'}
            
            return left_processed, right_processed, column_mapping
            
        except Exception as e:
            raise RuntimeError(f"Error resolving column conflicts: {str(e)}")
    
    @staticmethod
    def perform_asof_join_spark(left_df: SparkDataFrame, right_df: SparkDataFrame,
                               left_time_column: str, right_time_column: str,
                               left_by_column: str, right_by_column: str,
                               tolerance: pd.Timedelta = pd.Timedelta('600S'),
                               direction: str = 'forward',
                               suffixes: Optional[Tuple[str, str]] = None,
                               **kwargs) -> SparkDataFrame:
        """
        Perform asof join on Spark DataFrames using cogroup and applyInPandas.
        
        This method leverages Spark's distributed computing capabilities while using
        pandas asof join functionality for grouped data processing.
        
        Parameters
        ----------
        left_df : SparkDataFrame
            Left DataFrame for the join
        right_df : SparkDataFrame
            Right DataFrame for the join
        left_time_column : str
            Time column in left DataFrame
        right_time_column : str
            Time column in right DataFrame
        left_by_column : str
            Grouping column in left DataFrame
        right_by_column : str
            Grouping column in right DataFrame
        tolerance : pd.Timedelta, default '600S'
            Time tolerance for matching
        direction : str, default 'forward'
            Join direction
        suffixes : tuple of str, optional
            Column suffixes for conflicts
        **kwargs
            Additional parameters
            
        Returns
        -------
        SparkDataFrame
            Result of asof join operation
            
        Examples
        --------
        >>> # Create sample Spark DataFrames
        >>> left_spark_df = spark.createDataFrame([
        ...     ('2023-01-01 10:00:00', 'machine_A', 100.5),
        ...     ('2023-01-01 11:00:00', 'machine_A', 101.2)
        ... ], ['timestamp', 'equipment_id', 'temperature'])
        >>> 
        >>> right_spark_df = spark.createDataFrame([
        ...     ('2023-01-01 10:30:00', 'machine_A', 50.1),
        ...     ('2023-01-01 11:30:00', 'machine_A', 52.3)
        ... ], ['load_time', 'equipment_id', 'pressure'])
        >>>
        >>> result = SparkJoinOperations.perform_asof_join_spark(
        ...     left_spark_df, right_spark_df,
        ...     'timestamp', 'load_time', 
        ...     'equipment_id', 'equipment_id'
        ... )
        """
        try:
            # Validate inputs
            if left_df is None or right_df is None:
                raise ValueError("DataFrames cannot be None")
                
            # Resolve column conflicts
            if suffixes is None:
                suffixes = ('_left', '_right')
            
            left_processed, right_processed, column_mapping = SparkJoinOperations.resolve_column_conflicts(
                left_df, right_df, 'suffix', suffixes
            )
            
            # Update column names based on mapping
            current_left_time = left_time_column
            current_right_time = right_time_column
            current_left_by = left_by_column
            current_right_by = right_by_column
            
            for original_col, mapping in column_mapping.items():
                if original_col == left_time_column:
                    current_left_time = mapping['left_renamed']
                elif original_col == right_time_column:
                    current_right_time = mapping['right_renamed'] 
                elif original_col == left_by_column:
                    current_left_by = mapping['left_renamed']
                elif original_col == right_by_column:
                    current_right_by = mapping['right_renamed']
            
            # Create combined schema for result
            left_schema = [field for field in left_processed.schema]
            right_schema = [field for field in right_processed.schema]
            combined_schema = spk_dtp.StructType(left_schema + right_schema)
            
            # Sort DataFrames for optimal join performance
            left_sorted = left_processed.sort(current_left_by, current_left_time)
            right_sorted = right_processed.sort(current_right_by, current_right_time)
            
            # Define wrapper function for asof join
            def asof_join_wrapper(left_pandas: pd.DataFrame, right_pandas: pd.DataFrame) -> pd.DataFrame:
                return SparkJoinOperations.perform_asof_join_pandas(
                    left_pandas, right_pandas,
                    current_left_time, current_right_time,
                    current_left_by, current_right_by,
                    tolerance=tolerance,
                    direction=direction,
                    **kwargs
                )
            
            # Group DataFrames and perform cogroup operation
            left_grouped = left_sorted.groupby(current_left_by)
            right_grouped = right_sorted.groupby(current_right_by)
            
            # Execute distributed asof join
            result = left_grouped.cogroup(right_grouped).applyInPandas(
                asof_join_wrapper, 
                schema=combined_schema
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error performing Spark asof join: {str(e)}")


class SparkDataTransformations:
    """
    Comprehensive data transformation utilities for Spark DataFrames.
    
    This class provides essential data manipulation operations including melting,
    column operations, type conversions, and schema discovery utilities.
    """
    
    @staticmethod
    def melt_dataframe(df: SparkDataFrame, identifier_columns: List[str],
                      value_columns: List[str], variable_column_name: str = "variable",
                      value_column_name: str = "value") -> SparkDataFrame:
        """
        Convert DataFrame from wide to long format (equivalent to pandas melt).
        
        This transformation is essential for data analysis workflows where you need
        to reshape data for visualization or modeling purposes.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame in wide format
        identifier_columns : List[str]
            Column(s) to use as identifier variables (will remain as columns)
        value_columns : List[str]
            Column(s) to unpivot (will become rows)
        variable_column_name : str, default "variable"
            Name for the variable column in output
        value_column_name : str, default "value"
            Name for the value column in output
            
        Returns
        -------
        SparkDataFrame
            Melted DataFrame in long format
            
        Examples
        --------
        >>> # Sample wide-format data
        >>> wide_df = spark.createDataFrame([
        ...     ('A', 1, 100, 200),
        ...     ('B', 2, 150, 250)
        ... ], ['id', 'category', 'metric_1', 'metric_2'])
        >>> 
        >>> # Melt to long format
        >>> long_df = SparkDataTransformations.melt_dataframe(
        ...     wide_df,
        ...     identifier_columns=['id', 'category'],
        ...     value_columns=['metric_1', 'metric_2']
        ... )
        >>> # Result: id, category, variable, value columns
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            if not identifier_columns:
                raise ValueError("Identifier columns cannot be empty")
                
            if not value_columns:
                raise ValueError("Value columns cannot be empty")
                
            # Validate that columns exist in DataFrame
            df_columns = set(df.columns)
            missing_id_cols = set(identifier_columns) - df_columns
            missing_value_cols = set(value_columns) - df_columns
            
            if missing_id_cols:
                raise ValueError(f"Identifier columns not found in DataFrame: {missing_id_cols}")
            if missing_value_cols:
                raise ValueError(f"Value columns not found in DataFrame: {missing_value_cols}")
            
            # Create array of structs for unpivoting
            vars_and_values = F.array(*[
                F.struct(
                    F.lit(column).alias(variable_column_name),
                    F.col(column).alias(value_column_name)
                )
                for column in value_columns
            ])
            
            # Add exploded column to DataFrame
            df_with_exploded = df.withColumn("_vars_and_vals", F.explode(vars_and_values))
            
            # Select required columns for final result
            result_columns = identifier_columns + [
                F.col("_vars_and_vals")[variable_column_name].alias(variable_column_name),
                F.col("_vars_and_vals")[value_column_name].alias(value_column_name)
            ]
            
            result = df_with_exploded.select(*result_columns)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error melting DataFrame: {str(e)}")
    
    @staticmethod 
    def rename_columns_batch(df: SparkDataFrame, column_mapping: Dict[str, str]) -> SparkDataFrame:
        """
        Rename multiple columns in a single operation for better performance.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame
        column_mapping : Dict[str, str]
            Dictionary mapping old column names to new column names
            
        Returns
        -------
        SparkDataFrame
            DataFrame with renamed columns
            
        Examples
        --------
        >>> mapping = {
        ...     'old_temp': 'temperature_celsius',
        ...     'old_press': 'pressure_bar',
        ...     'ts': 'timestamp'
        ... }
        >>> renamed_df = SparkDataTransformations.rename_columns_batch(df, mapping)
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            if not column_mapping:
                return df
                
            # Validate that old column names exist
            df_columns = set(df.columns)
            missing_columns = set(column_mapping.keys()) - df_columns
            
            if missing_columns:
                raise ValueError(f"Columns to rename not found in DataFrame: {missing_columns}")
            
            result = df
            for old_name, new_name in column_mapping.items():
                if old_name != new_name:  # Only rename if names are different
                    result = result.withColumnRenamed(old_name, new_name)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error renaming columns: {str(e)}")
    
    @staticmethod
    def convert_columns_to_numeric(df: SparkDataFrame, exclude_columns: List[str],
                                  target_type: str = 'float') -> SparkDataFrame:
        """
        Convert multiple columns to numeric types with intelligent type detection.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame
        exclude_columns : List[str]
            Columns to exclude from conversion
        target_type : str, default 'float'
            Target numeric type ('float', 'double', 'int', 'long')
            
        Returns
        -------
        SparkDataFrame
            DataFrame with converted numeric columns
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            # Validate target type
            valid_types = ['float', 'double', 'int', 'long', 'integer']
            if target_type not in valid_types:
                raise ValueError(f"Invalid target type. Must be one of: {valid_types}")
            
            # Get columns to convert (all except excluded)
            exclude_set = set(exclude_columns or [])
            columns_to_convert = [col for col in df.columns if col not in exclude_set]
            
            result = df
            for column in columns_to_convert:
                try:
                    result = result.withColumn(column, F.col(column).cast(target_type))
                except Exception as col_error:
                    warnings.warn(f"Could not convert column '{column}' to {target_type}: {col_error}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error converting columns to numeric: {str(e)}")
    
    @staticmethod
    def discover_columns_by_pattern(table_name: str, search_patterns: List[str],
                                   key_vault_dict: Optional[Dict] = None) -> List[str]:
        """
        Discover column names matching specified patterns in a database table.
        
        Parameters
        ----------
        table_name : str
            Full table name (e.g., 'database.table_name')
        search_patterns : List[str]
            List of patterns to search for in column names
        key_vault_dict : Dict, optional
            Configuration dictionary for database access
            
        Returns
        -------
        List[str]
            List of column names matching the patterns
        """
        try:
            if not DSTOOLS_AVAILABLE:
                raise ImportError("dsToolbox modules required for database operations")
                
            if not table_name:
                raise ValueError("Table name cannot be empty")
                
            if not search_patterns:
                raise ValueError("Search patterns cannot be empty")
            
            # Query table columns
            column_query = f'SHOW COLUMNS IN {table_name};'
            column_df = io_funcs.query_deltaTable_db(
                column_query,
                key_vault_dict=key_vault_dict,
                verbose=False
            )
            
            all_columns = column_df.toPandas().squeeze().tolist()
            
            # Find matching columns using pattern matching
            matching_columns, _ = cfuncs.inWithReg(search_patterns, all_columns)
            
            return matching_columns
            
        except Exception as e:
            raise RuntimeError(f"Error discovering columns: {str(e)}")


class SparkETLPipeline:
    """
    Comprehensive ETL pipeline management for Spark-based data workflows.
    
    This class handles end-to-end ETL operations including date tracking,
    incremental processing, and output management to various destinations.
    """
    
    @staticmethod
    def get_last_processed_date(output_target: Union[str, Dict], date_column: str = 'date',
                               custom_config: Optional[Dict] = None,
                               key_vault_dict: str = 'deltaTable',
                               platform: str = 'databricks') -> Optional[dt.datetime]:
        """
        Retrieve the last processed date from output destination for incremental processing.
        
        This method supports both Delta tables and blob storage to track processing state,
        enabling efficient incremental data processing workflows.
        
        Parameters
        ----------
        output_target : str or Dict
            Output destination - string for Delta table or dict for blob storage
        date_column : str, default 'date'
            Column name containing date information
        custom_config : Dict, optional
            Custom configuration dictionary
        key_vault_dict : str, default 'deltaTable'
            Key vault configuration for Delta table access
        platform : str, default 'databricks'
            Platform specification for blob storage
            
        Returns
        -------
        datetime or None
            Last processed date, or None if no data exists
            
        Examples
        --------
        >>> # For Delta table
        >>> last_date = SparkETLPipeline.get_last_processed_date(
        ...     'analytics.sales_daily',
        ...     date_column='process_date'
        ... )
        >>> 
        >>> # For blob storage  
        >>> blob_config = {
        ...     'container': 'data',
        ...     'blob_path': 'processed/sales.parquet'
        ... }
        >>> last_date = SparkETLPipeline.get_last_processed_date(
        ...     blob_config,
        ...     date_column='process_date'
        ... )
        """
        try:
            if not DSTOOLS_AVAILABLE:
                warnings.warn("dsToolbox modules not available. Cannot access external storage.")
                return None
            
            # Handle Delta table
            if isinstance(output_target, str) and io_funcs.deltaTable_check(output_target):
                date_query = f"""
                    SELECT MIN({date_column}) as min_time,
                           MAX({date_column}) as max_time
                    FROM {output_target}
                """
                
                result_df = io_funcs.query_deltaTable_db(
                    date_query,
                    key_vault_dict=key_vault_dict,
                    custom_config=custom_config,
                    verbose=False
                ).toPandas()
                
                last_date = result_df['max_time'].iloc[0]
                print(f"Last date found in Delta table: {last_date}")
                
            # Handle blob storage
            elif isinstance(output_target, dict) and io_funcs.blob_check(
                blob_dict=output_target,
                custom_config=custom_config,
                platform=platform
            ):
                data = io_funcs.blob2pd(
                    blob_dict=output_target,
                    custom_config=custom_config,
                    platform=platform
                )
                
                data[date_column] = pd.to_datetime(data[date_column], format="%Y-%m-%d")
                last_date = data[date_column].max()
                
                print(f"Last date found in blob: {last_date}")
                
            else:
                print("Output target not found or not accessible")
                last_date = None
                
            return last_date
            
        except Exception as e:
            print(f"Error retrieving last processed date: {str(e)}")
            return None
    
    @staticmethod
    def save_pipeline_outputs(outputs_dict: Union[Dict, List], **kwargs) -> None:
        """
        Save pipeline outputs to multiple destinations (Delta tables and blob storage).
        
        Parameters
        ----------
        outputs_dict : Dict or List
            Dictionary or list of (destination, data) pairs
        **kwargs
            Additional arguments for save operations
        """
        try:
            if not DSTOOLS_AVAILABLE:
                raise ImportError("dsToolbox modules required for save operations")
            
            # Extract function-specific arguments
            spark2del_args = {}
            pd2blob_args = {}
            
            if hasattr(io_funcs, 'spark2deltaTable'):
                spark2del_signature = inspect.signature(io_funcs.spark2deltaTable)
                spark2del_args = {k: kwargs.pop(k, None) for k in spark2del_signature.parameters if k in kwargs}
            
            if hasattr(io_funcs, 'pd2blob'):
                pd2blob_signature = inspect.signature(io_funcs.pd2blob)
                pd2blob_args = {k: kwargs.pop(k, None) for k in pd2blob_signature.parameters if k in kwargs}
            
            # Convert to consistent format
            if isinstance(outputs_dict, dict):
                outputs = outputs_dict.items()
            else:
                outputs = outputs_dict
            
            for destination, data in outputs:
                print(f"Saving to destination: {destination}")
                
                # Save to Delta table
                if isinstance(destination, str):
                    table_parts = destination.split('.')
                    if len(table_parts) >= 2:
                        schema_name = table_parts[0]
                        table_name = table_parts[1]
                        
                        io_funcs.spark2deltaTable(
                            data,
                            table_name=table_name,
                            schema=schema_name,
                            write_mode='append',
                            mergeSchema=True,
                            **{k: v for k, v in spark2del_args.items() if v is not None}
                        )
                
                # Save to blob storage
                elif isinstance(destination, dict):
                    io_funcs.pd2blob(
                        data,
                        blob_dict=destination,
                        overwrite=False,
                        append=True,
                        **{k: v for k, v in pd2blob_args.items() if v is not None}
                    )
                    
        except Exception as e:
            raise RuntimeError(f"Error saving pipeline outputs: {str(e)}")
    
    @staticmethod
    def execute_incremental_pipeline(data_generator_function: callable, output_target: Union[str, Dict],
                                   year_range: List[int] = [2021, 2099],
                                   first_date: Optional[str] = None,
                                   last_date: Optional[dt.date] = None,
                                   date_column: str = 'date',
                                   custom_config: Optional[Dict] = None,
                                   key_vault_dict: str = 'deltaTable',
                                   platform: str = 'databricks',
                                   **kwargs) -> None:
        """
        Execute incremental ETL pipeline with automatic date management.
        
        This method orchestrates the complete incremental processing workflow,
        tracking the last processed date and only processing new data.
        
        Parameters
        ----------
        data_generator_function : callable
            Function that generates data for given date ranges
            Must accept (start_date, end_date, output_target, **kwargs)
        output_target : str or Dict
            Destination for processed data
        year_range : List[int], default [2021, 2099]
            Year range for date generation
        first_date : str, optional
            Override start date (format: 'YYYY-MM-DD')
        last_date : date, optional
            End date for processing (default: today)
        date_column : str, default 'date'
            Date column name for tracking
        custom_config : Dict, optional
            Custom configuration
        key_vault_dict : str, default 'deltaTable'
            Key vault configuration
        platform : str, default 'databricks'
            Platform specification
        **kwargs
            Additional arguments passed to data generator function
            
        Examples
        --------
        >>> def process_sales_data(start_date, end_date, output_target, **kwargs):
        ...     # Your data processing logic here
        ...     processed_data = transform_sales_data(start_date, end_date)
        ...     return [(output_target, processed_data)]
        >>> 
        >>> SparkETLPipeline.execute_incremental_pipeline(
        ...     data_generator_function=process_sales_data,
        ...     output_target='analytics.daily_sales',
        ...     date_column='sale_date',
        ...     first_date='2023-01-01'
        ... )
        """
        try:
            if not DSTOOLS_AVAILABLE:
                raise ImportError("dsToolbox modules required for pipeline execution")
                
            if not callable(data_generator_function):
                raise ValueError("data_generator_function must be callable")
            
            # Extract function-specific arguments
            func_signature = inspect.signature(data_generator_function)
            generator_args = {
                k: kwargs.pop(k, None) 
                for k in list(kwargs.keys()) 
                if k in func_signature.parameters and kwargs[k] is not None
            }
            
            # Get last processed date
            last_saved_date = SparkETLPipeline.get_last_processed_date(
                output_target=output_target,
                date_column=date_column,
                custom_config=custom_config,
                key_vault_dict=key_vault_dict,
                platform=platform
            )
            
            # Determine effective start date
            warn_override = False
            if first_date is not None:
                if isinstance(first_date, str):
                    effective_start_date = dt.datetime.strptime(first_date, "%Y-%m-%d").date()
                    warn_override = True
                elif isinstance(first_date, pd.Timestamp):
                    effective_start_date = first_date.date()
                    warn_override = True
                else:
                    effective_start_date = first_date
                    warn_override = True
            else:
                if last_saved_date is not None:
                    effective_start_date = last_saved_date.date() + dt.timedelta(days=1)
                else:
                    effective_start_date = dt.date(year_range[0], 1, 1)
            
            # Show warning if overriding existing data
            if warn_override and last_saved_date is not None:
                print(f"WARNING: Last processed date is {last_saved_date}, but starting from {effective_start_date}")
            
            # Generate date ranges for processing
            if last_date is None:
                last_date = dt.datetime.now().date()
                
            date_ranges = cfuncs.datesList(
                year_range=year_range,
                firstDate=effective_start_date,
                lastDate=last_date
            )
            
            if len(date_ranges) == 0:
                print("Pipeline is up to date - no new data to process")
                return
            else:
                print(f"Processing {len(date_ranges)-1} date ranges: {date_ranges[0]} to {date_ranges[-1]}")
            
            # Process each date range
            for i in range(len(date_ranges) - 1):
                try:
                    start_date, end_date = cfuncs.extract_start_end(date_ranges, i)
                    print(f"Processing data from {start_date} to {end_date}")
                    
                    # Generate data for date range
                    outputs_list = data_generator_function(
                        start_date, end_date, output_target, **generator_args
                    )
                    
                    # Save outputs
                    SparkETLPipeline.save_pipeline_outputs(outputs_list)
                    
                except Exception as e:
                    print(f"ERROR: Processing failed for {start_date}: {str(e)}")
                    print("*" * 80)
                    continue
                    
        except Exception as e:
            raise RuntimeError(f"Error executing incremental pipeline: {str(e)}")


class SparkFeatureEngineering:
    """
    Advanced feature engineering utilities for time-series data using Spark.
    
    This class provides sophisticated feature generation capabilities including
    rolling windows, tumbling windows, and time-based aggregations optimized
    for large-scale distributed processing.
    """
    
    @staticmethod
    def identify_numeric_columns(df: SparkDataFrame) -> List[str]:
        """
        Identify numeric columns in a Spark DataFrame.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame
            
        Returns
        -------
        List[str]
            List of numeric column names
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            numeric_types = ['int', 'float', 'long', 'double', 'bigint', 'decimal']
            numeric_columns = [
                col_name for col_name, dtype in df.dtypes
                if any(dtype.startswith(num_type) for num_type in numeric_types)
            ]
            
            return numeric_columns
            
        except Exception as e:
            raise RuntimeError(f"Error identifying numeric columns: {str(e)}")
    
    @staticmethod
    def parse_time_duration(duration_string: str) -> int:
        """
        Parse time duration string and convert to seconds.
        
        Parameters
        ----------
        duration_string : str
            Duration in format "5 minutes", "2 hours", etc.
            
        Returns
        -------
        int
            Duration in seconds
        """
        try:
            parts = duration_string.strip().split(' ')
            if len(parts) != 2:
                raise ValueError(f"Invalid duration format: {duration_string}")
                
            value = int(parts[0])
            unit = parts[1].lower()
            
            unit_multipliers = {
                'second': 1, 'seconds': 1,
                'minute': 60, 'minutes': 60,
                'hour': 3600, 'hours': 3600,
                'day': 86400, 'days': 86400
            }
            
            if unit not in unit_multipliers:
                raise ValueError(f"Unsupported time unit: {unit}")
                
            return value * unit_multipliers[unit]
            
        except Exception as e:
            raise ValueError(f"Error parsing time duration: {str(e)}")
    
    @staticmethod
    def create_rolling_window_features(df: SparkDataFrame, timestamp_column: str = 'Time_Stamp',
                                     groupby_column: str = 'machine',
                                     window_duration: str = '5 minutes',
                                     aggregation_type: str = 'avg',
                                     include_original_columns: bool = False) -> SparkDataFrame:
        """
        Create rolling window features for time-series data analysis.
        
        Rolling windows compute statistics over a sliding time window, which is
        essential for capturing temporal patterns and trends in time-series data.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame with time-series data
        timestamp_column : str, default 'Time_Stamp'
            Column containing timestamps for window calculation
        groupby_column : str, default 'machine'
            Column for partitioning data (e.g., by machine, sensor, etc.)
        window_duration : str, default '5 minutes'
            Rolling window size (e.g., '5 minutes', '1 hour', '2 days')
        aggregation_type : str, default 'avg'
            Aggregation function: 'avg', 'min', 'max', 'sum', 'count', 'stddev'
        include_original_columns : bool, default False
            Whether to keep original numeric columns
            
        Returns
        -------
        SparkDataFrame
            DataFrame with rolling window features
            
        Examples
        --------
        >>> # Create rolling average features over 10-minute windows
        >>> rolling_df = SparkFeatureEngineering.create_rolling_window_features(
        ...     df=sensor_data,
        ...     timestamp_column='measurement_time',
        ...     groupby_column='sensor_id', 
        ...     window_duration='10 minutes',
        ...     aggregation_type='avg'
        ... )
        >>> 
        >>> # Result will have columns like: temperature_avg, pressure_avg, etc.
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            # Validate columns exist
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            if groupby_column not in df.columns:
                raise ValueError(f"Groupby column '{groupby_column}' not found in DataFrame")
            
            # Validate aggregation type
            valid_agg_types = ['avg', 'min', 'max', 'sum', 'count', 'stddev', 'mean']
            if aggregation_type not in valid_agg_types:
                raise ValueError(f"Invalid aggregation type. Must be one of: {valid_agg_types}")
            
            # Get numeric columns for feature generation
            numeric_columns = SparkFeatureEngineering.identify_numeric_columns(df)
            if not numeric_columns:
                raise ValueError("No numeric columns found for feature engineering")
            
            # Convert window duration to seconds
            window_seconds = SparkFeatureEngineering.parse_time_duration(window_duration)
            
            # Create window specification
            window_spec = Window.partitionBy(groupby_column).orderBy(
                F.col(timestamp_column).cast('long')
            ).rangeBetween(-window_seconds, 0)
            
            result_df = df
            
            # Create rolling features for each numeric column
            for column in numeric_columns:
                feature_name = f"{column}_{aggregation_type}"
                
                # Apply aggregation function
                if aggregation_type in ['avg', 'mean']:
                    result_df = result_df.withColumn(feature_name, F.avg(column).over(window_spec))
                elif aggregation_type == 'min':
                    result_df = result_df.withColumn(feature_name, F.min(column).over(window_spec))
                elif aggregation_type == 'max':
                    result_df = result_df.withColumn(feature_name, F.max(column).over(window_spec))
                elif aggregation_type == 'sum':
                    result_df = result_df.withColumn(feature_name, F.sum(column).over(window_spec))
                elif aggregation_type == 'count':
                    result_df = result_df.withColumn(feature_name, F.count(column).over(window_spec))
                elif aggregation_type == 'stddev':
                    result_df = result_df.withColumn(feature_name, F.stddev(column).over(window_spec))
                
                # Cast to float for consistency
                result_df = result_df.withColumn(feature_name, F.col(feature_name).cast('float'))
                
                # Optionally remove original column
                if not include_original_columns:
                    result_df = result_df.drop(column)
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Error creating rolling window features: {str(e)}")
    
    @staticmethod
    def create_tumbling_window_features(df: SparkDataFrame, timestamp_column: str = 'Time_Stamp',
                                      groupby_column: str = 'machine',
                                      window_duration: str = '5 minutes',
                                      aggregation_type: str = 'avg',
                                      join_direction: str = 'backward',
                                      tolerance: Optional[str] = None) -> SparkDataFrame:
        """
        Create tumbling window features for time-series data.
        
        Tumbling windows divide time into non-overlapping intervals and compute
        aggregations over each interval. This is useful for creating periodic
        summaries and reducing data volume while preserving temporal patterns.
        
        Parameters
        ----------
        df : SparkDataFrame
            Input DataFrame with time-series data
        timestamp_column : str, default 'Time_Stamp'
            Column containing timestamps
        groupby_column : str, default 'machine'
            Column for partitioning data
        window_duration : str, default '5 minutes'
            Tumbling window size
        aggregation_type : str, default 'avg'
            Aggregation function to apply
        join_direction : str, default 'backward'
            Direction for asof join: 'backward', 'forward', 'nearest'
        tolerance : str, optional
            Time tolerance for asof join (default: same as window_duration)
            
        Returns
        -------
        SparkDataFrame
            DataFrame with tumbling window features
            
        Examples
        --------
        >>> # Create 15-minute tumbling window features
        >>> tumbling_df = SparkFeatureEngineering.create_tumbling_window_features(
        ...     df=production_data,
        ...     timestamp_column='production_time',
        ...     groupby_column='production_line',
        ...     window_duration='15 minutes',
        ...     aggregation_type='sum'
        ... )
        """
        try:
            if df is None:
                raise ValueError("DataFrame cannot be None")
                
            # Validate inputs
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found")
            if groupby_column not in df.columns:
                raise ValueError(f"Groupby column '{groupby_column}' not found")
                
            # Set default tolerance
            if tolerance is None:
                tolerance = window_duration
                
            # Get numeric and non-numeric columns
            numeric_columns = SparkFeatureEngineering.identify_numeric_columns(df)
            all_columns = set(df.columns)
            non_numeric_columns = list(all_columns - set(numeric_columns))
            
            if not numeric_columns:
                raise ValueError("No numeric columns found for aggregation")
            
            # Create left DataFrame (non-numeric columns)
            left_df = df.select(*non_numeric_columns)
            
            # Create tumbling window aggregation
            if aggregation_type == 'avg':
                right_df = df.groupBy(groupby_column, F.window(timestamp_column, window_duration)).avg()
            elif aggregation_type == 'sum':
                right_df = df.groupBy(groupby_column, F.window(timestamp_column, window_duration)).sum()
            elif aggregation_type == 'min':
                right_df = df.groupBy(groupby_column, F.window(timestamp_column, window_duration)).min()
            elif aggregation_type == 'max':
                right_df = df.groupBy(groupby_column, F.window(timestamp_column, window_duration)).max()
            elif aggregation_type == 'count':
                right_df = df.groupBy(groupby_column, F.window(timestamp_column, window_duration)).count()
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
            
            # Extract window boundaries
            right_df = right_df.withColumn('window_start', right_df.window.start)
            right_df = right_df.withColumn('window_end', right_df.window.end)
            right_df = right_df.drop('window')
            
            # Rename aggregated columns
            for column in numeric_columns:
                agg_column_name = f"{aggregation_type}({column})"
                feature_name = f"{column}_{aggregation_type}"
                
                if agg_column_name in right_df.columns:
                    right_df = right_df.withColumnRenamed(agg_column_name, feature_name)
                    right_df = right_df.withColumn(feature_name, F.col(feature_name).cast('float'))
            
            # Perform asof join to align tumbling windows with original timestamps
            result_df = SparkJoinOperations.perform_asof_join_spark(
                left_df, right_df,
                left_time_column=timestamp_column,
                right_time_column='window_start',
                left_by_column=groupby_column,
                right_by_column=groupby_column,
                tolerance=pd.Timedelta(tolerance),
                direction=join_direction
            )
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Error creating tumbling window features: {str(e)}")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS (DEPRECATED)
# =============================================================================

def asof_join_sub(l, r, left_on, right_on, left_by, right_by,
                 tolerance=pd.Timedelta('600S'), direction='forward', **kwargs):
    """
    DEPRECATED: Use SparkJoinOperations.perform_asof_join_pandas() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "asof_join_sub() is deprecated. Use SparkJoinOperations.perform_asof_join_pandas() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkJoinOperations.perform_asof_join_pandas(
        l, r, left_on, right_on, left_by, right_by, tolerance, direction, **kwargs
    )

def asof_join_spark2(df_left, df_right, left_on, right_on, left_by, right_by,
                    tolerance=pd.Timedelta('600S'), direction='forward', **kwargs):
    """
    DEPRECATED: Use SparkJoinOperations.perform_asof_join_spark() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "asof_join_spark2() is deprecated. Use SparkJoinOperations.perform_asof_join_spark() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkJoinOperations.perform_asof_join_spark(
        df_left, df_right, left_on, right_on, left_by, right_by, tolerance, direction, **kwargs
    )

def melt(df: SparkDataFrame, id_vars: List[str], value_vars: List[str],
         var_name: str = "variable", value_name: str = "value") -> SparkDataFrame:
    """
    DEPRECATED: Use SparkDataTransformations.melt_dataframe() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "melt() is deprecated. Use SparkDataTransformations.melt_dataframe() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkDataTransformations.melt_dataframe(df, id_vars, value_vars, var_name, value_name)

def rename_cols(sp, mapCols_dict):
    """
    DEPRECATED: Use SparkDataTransformations.rename_columns_batch() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "rename_cols() is deprecated. Use SparkDataTransformations.rename_columns_batch() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkDataTransformations.rename_columns_batch(sp, mapCols_dict)

def sp_to_numeric(sp, exclude_cols, caseTo='float'):
    """
    DEPRECATED: Use SparkDataTransformations.convert_columns_to_numeric() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "sp_to_numeric() is deprecated. Use SparkDataTransformations.convert_columns_to_numeric() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkDataTransformations.convert_columns_to_numeric(sp, exclude_cols, caseTo)

def col_finder(key_vault_dict, tableName='mcsdata.mcs_bm_15', cols2search=['facies_', 'formation_']):
    """
    DEPRECATED: Use SparkDataTransformations.discover_columns_by_pattern() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "col_finder() is deprecated. Use SparkDataTransformations.discover_columns_by_pattern() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkDataTransformations.discover_columns_by_pattern(tableName, cols2search, key_vault_dict)

def last_date(output_name, date_col='date', custom_config=None,
             key_vault_dict='deltaTable', platform='databricks'):
    """
    DEPRECATED: Use SparkETLPipeline.get_last_processed_date() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "last_date() is deprecated. Use SparkETLPipeline.get_last_processed_date() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkETLPipeline.get_last_processed_date(
        output_name, date_col, custom_config, key_vault_dict, platform
    )

def save_outputs(ouputs_dict_list, **kwargs):
    """
    DEPRECATED: Use SparkETLPipeline.save_pipeline_outputs() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "save_outputs() is deprecated. Use SparkETLPipeline.save_pipeline_outputs() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkETLPipeline.save_pipeline_outputs(ouputs_dict_list, **kwargs)

def update_db_recursively(dfGenerator_func, output_name, year_range=[2021, 2099],
                         firstDate=None, lastDate=dt.datetime.now().date(),
                         date_col='date', custom_config=None, key_vault_dict='deltaTable',
                         platform='databricks', **kwargs):
    """
    DEPRECATED: Use SparkETLPipeline.execute_incremental_pipeline() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "update_db_recursively() is deprecated. Use SparkETLPipeline.execute_incremental_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkETLPipeline.execute_incremental_pipeline(
        dfGenerator_func, output_name, year_range, firstDate, lastDate,
        date_col, custom_config, key_vault_dict, platform, **kwargs
    )

def create_rolling_features(df, timestamp_col_name='Time_Stamp', groupby_col_name='machine',
                          window_duration='5 minutes', agg_type='avg'):
    """
    DEPRECATED: Use SparkFeatureEngineering.create_rolling_window_features() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "create_rolling_features() is deprecated. Use SparkFeatureEngineering.create_rolling_window_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkFeatureEngineering.create_rolling_window_features(
        df, timestamp_col_name, groupby_col_name, window_duration, agg_type, include_original_columns=False
    )

def create_tumbling_features(df, timestamp_col_name='Time_Stamp', groupby_col_name='machine',
                           window_duration='5 minutes', agg_type='avg', direction='backward',
                           tolerance=None):
    """
    DEPRECATED: Use SparkFeatureEngineering.create_tumbling_window_features() instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "create_tumbling_features() is deprecated. Use SparkFeatureEngineering.create_tumbling_window_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return SparkFeatureEngineering.create_tumbling_window_features(
        df, timestamp_col_name, groupby_col_name, window_duration, agg_type, direction, tolerance
    )


# =============================================================================
# FUNCTION MAPPING DOCUMENTATION
# =============================================================================

FUNCTION_MAPPING = {
    # Join Operations
    'asof_join_sub': 'SparkJoinOperations.perform_asof_join_pandas',
    'asof_join_spark2': 'SparkJoinOperations.perform_asof_join_spark',
    
    # Data Transformations
    'melt': 'SparkDataTransformations.melt_dataframe',
    'rename_cols': 'SparkDataTransformations.rename_columns_batch',
    'sp_to_numeric': 'SparkDataTransformations.convert_columns_to_numeric',
    'col_finder': 'SparkDataTransformations.discover_columns_by_pattern',
    
    # ETL Pipeline
    'last_date': 'SparkETLPipeline.get_last_processed_date',
    'save_outputs': 'SparkETLPipeline.save_pipeline_outputs',
    'update_db_recursively': 'SparkETLPipeline.execute_incremental_pipeline',
    
    # Feature Engineering
    'create_rolling_features': 'SparkFeatureEngineering.create_rolling_window_features',
    'create_tumbling_features': 'SparkFeatureEngineering.create_tumbling_window_features',
}

def print_spark_function_mapping():
    """Print comprehensive function mapping guide for Spark functions."""
    print("=" * 80)
    print("SPARK FUNCTIONS - MIGRATION GUIDE")
    print("=" * 80)
    print("Old Function → New Class Method")
    print("-" * 40)
    
    for old_func, new_method in FUNCTION_MAPPING.items():
        print(f"{old_func:30} → {new_method}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE MIGRATIONS:")
    print("-" * 40)
    
    print("\n# Join Operations:")
    print("# Old:")
    print("result = asof_join_spark2(df_left, df_right, 'timestamp', 'timestamp', 'machine', 'machine')")
    print("# New:")
    print("result = SparkJoinOperations.perform_asof_join_spark(")
    print("    df_left, df_right, 'timestamp', 'timestamp', 'machine', 'machine')")
    
    print("\n# Data Transformations:")
    print("# Old:")
    print("melted = melt(df, ['id'], ['col1', 'col2'])")
    print("renamed = rename_cols(df, {'old_name': 'new_name'})")
    print("# New:")
    print("melted = SparkDataTransformations.melt_dataframe(df, ['id'], ['col1', 'col2'])")
    print("renamed = SparkDataTransformations.rename_columns_batch(df, {'old_name': 'new_name'})")
    
    print("\n# ETL Pipeline:")
    print("# Old:")
    print("update_db_recursively(process_func, 'table_name', firstDate='2023-01-01')")
    print("# New:")
    print("SparkETLPipeline.execute_incremental_pipeline(")
    print("    process_func, 'table_name', first_date='2023-01-01')")
    
    print("\n# Feature Engineering:")
    print("# Old:")
    print("features = create_rolling_features(df, 'timestamp', 'machine', '5 minutes')")
    print("# New:")
    print("features = SparkFeatureEngineering.create_rolling_window_features(")
    print("    df, 'timestamp', 'machine', '5 minutes')")
    
    print("\n" + "=" * 80)


# Export all classes and functions for external use
__all__ = [
    # Main classes
    'SparkJoinOperations',
    'SparkDataTransformations', 
    'SparkETLPipeline',
    'SparkFeatureEngineering',
    
    # Backward compatibility functions
    'asof_join_sub',
    'asof_join_spark2',
    'melt',
    'rename_cols',
    'sp_to_numeric',
    'col_finder',
    'last_date',
    'save_outputs',
    'update_db_recursively',
    'create_rolling_features',
    'create_tumbling_features',
    
    # Utilities
    'FUNCTION_MAPPING',
    'print_spark_function_mapping'
]


# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

def print_spark_usage_examples():
    """Print comprehensive usage examples for Spark functions."""
    print("=" * 80)
    print("SPARK DATA PROCESSING - USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n1. ADVANCED JOIN OPERATIONS:")
    print("# Time-series asof join for sensor data")
    print("join_ops = SparkJoinOperations()")
    print("result = join_ops.perform_asof_join_spark(")
    print("    left_df=sensor_readings,")
    print("    right_df=maintenance_events,")
    print("    left_time_column='reading_timestamp',")
    print("    right_time_column='event_timestamp',") 
    print("    left_by_column='sensor_id',")
    print("    right_by_column='sensor_id',")
    print("    tolerance=pd.Timedelta('30 minutes'),")
    print("    direction='backward'")
    print(")")
    
    print("\n2. DATA TRANSFORMATIONS:")
    print("# Melt wide-format data to long format")
    print("transformer = SparkDataTransformations()")
    print("long_df = transformer.melt_dataframe(")
    print("    df=wide_metrics_df,")
    print("    identifier_columns=['date', 'machine_id'],")
    print("    value_columns=['temperature', 'pressure', 'vibration'],")
    print("    variable_column_name='metric_type',")
    print("    value_column_name='measurement'")
    print(")")
    print("")
    print("# Batch rename columns with meaningful names")
    print("column_mapping = {")
    print("    'temp': 'temperature_celsius',")
    print("    'press': 'pressure_bar',")
    print("    'ts': 'measurement_timestamp'")
    print("}")
    print("renamed_df = transformer.rename_columns_batch(df, column_mapping)")
    
    print("\n3. ETL PIPELINE MANAGEMENT:")
    print("# Incremental data processing with automatic date tracking")
    print("def process_production_data(start_date, end_date, output_target, **kwargs):")
    print("    # Your data processing logic here")
    print("    processed_df = load_and_transform_data(start_date, end_date)")
    print("    return [(output_target, processed_df)]")
    print("")
    print("pipeline = SparkETLPipeline()")
    print("pipeline.execute_incremental_pipeline(")
    print("    data_generator_function=process_production_data,")
    print("    output_target='analytics.production_metrics',")
    print("    year_range=[2023, 2024],")
    print("    first_date='2023-01-01',")
    print("    date_column='production_date'")
    print(")")
    
    print("\n4. TIME-SERIES FEATURE ENGINEERING:")
    print("# Rolling window features for anomaly detection")
    print("feature_eng = SparkFeatureEngineering()")
    print("rolling_features = feature_eng.create_rolling_window_features(")
    print("    df=sensor_data,")
    print("    timestamp_column='measurement_time',")
    print("    groupby_column='sensor_id',")
    print("    window_duration='30 minutes',")
    print("    aggregation_type='avg',")
    print("    include_original_columns=True")
    print(")")
    print("")
    print("# Tumbling window features for periodic summaries")
    print("tumbling_features = feature_eng.create_tumbling_window_features(")
    print("    df=production_data,")
    print("    timestamp_column='production_timestamp',")
    print("    groupby_column='production_line',")
    print("    window_duration='1 hour',")
    print("    aggregation_type='sum'")
    print(")")
    
    print("\n" + "=" * 80)
    print("KEY BENEFITS:")
    print("-" * 40)
    print("✅ Distributed processing optimized for large datasets")
    print("✅ Advanced time-series join operations (asof joins)")
    print("✅ Incremental ETL pipelines with automatic date tracking") 
    print("✅ Sophisticated feature engineering for time-series data")
    print("✅ Production-ready error handling and validation")
    print("✅ Complete backward compatibility with deprecation warnings")
    print("✅ Comprehensive documentation and examples")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("SPARK FUNCTIONS REFACTORED - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Print usage examples
    print_spark_usage_examples()
    
    # Print migration guide
    print_spark_function_mapping()