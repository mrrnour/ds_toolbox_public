"""
Database I/O Functions - Refactored Version
==========================================

This module provides database connectivity and I/O operations for MSSQL Server and Incorta,
along with utilities for data processing workflows and file operations.

Classes:
    - DatabaseConnectionManager: Handles database connections and credentials
    - DatabaseQueryExecutor: Executes queries and manages database operations
    - DataWorkflowManager: Manages data processing workflows and output specifications

Utility Functions:
    - File operations (parquet handling)
    - Date utilities  
    - Output management functions

Author: Refactored from original procedural code
"""

import os
import sys
import datetime as dt
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Configuration setup
DEFAULT_CONFIG_FILE = os.path.join('dsToolbox', 'config.yml')

try:
    import src.common_funcs as cfuncs
except ImportError:
    warnings.warn("src.common_funcs not available. Some functionality may be limited.")
    cfuncs = None


class DatabaseConnectionManager:
    """
    Manages database connections and credentials for MSSQL Server and Incorta.
    
    This class handles configuration loading, connection establishment, and
    credential management for various database systems.
    
    Attributes:
        config (dict): Configuration dictionary loaded from YAML file
        db_server (str): Current database server name
        
    Example:
        >>> conn_manager = DatabaseConnectionManager('config.yml')
        >>> connection, config = conn_manager.get_mssql_connection_pyodbc('server1')
        >>> engine, params = conn_manager.get_mssql_connection_sqlalchemy('server1')
    """
    
    def __init__(self, config_file_path: str):
        """
        Initialize the database connection manager.
        
        Parameters:
            config_file_path (str): Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ImportError: If required dependencies are missing
            ValueError: If configuration file is invalid
        """
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
            
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for configuration loading. Install with: pip install PyYAML")
        
        try:
            with open(config_file_path, 'r') as stream:
                self.config = yaml.safe_load(stream)
                
            if not self.config:
                raise ValueError("Configuration file is empty or invalid")
                
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
        
        self.db_server = None
    
    def get_mssql_connection_pyodbc(self, database_server_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Create a PyODBC connection to MSSQL Server.
        
        This method establishes a connection using Windows Trusted Authentication
        via the SQL Server Native Client driver.
        
        Parameters:
            database_server_id (str): Database server identifier from configuration
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Connection object and configuration dictionary
            
        Raises:
            ImportError: If PyODBC is not available
            KeyError: If server ID not found in configuration
            Exception: If connection fails
            
        Example:
            >>> manager = DatabaseConnectionManager('config.yml')
            >>> conn, config = manager.get_mssql_connection_pyodbc('production_server')
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT @@VERSION")
        """
        try:
            import pyodbc
        except ImportError:
            raise ImportError("PyODBC is required for MSSQL connections. Install with: pip install pyodbc")
        
        if 'sql_servers' not in self.config:
            raise KeyError("'sql_servers' section not found in configuration")
            
        if database_server_id not in self.config['sql_servers']:
            available_servers = list(self.config['sql_servers'].keys())
            raise KeyError(f"Server ID '{database_server_id}' not found. Available servers: {available_servers}")
        
        server_config = self.config['sql_servers'][database_server_id]
        
        if 'db_server' not in server_config:
            raise KeyError(f"'db_server' not specified for server ID '{database_server_id}'")
        
        self.db_server = server_config['db_server']
        
        try:
            connection_string = (
                "DRIVER={SQL Server Native Client 11.0};"
                f"SERVER={self.db_server};"
                "Trusted_Connection=yes;"
            )
            
            connection = pyodbc.connect(connection_string)
            print(f"Connected to {self.db_server}")
            
            return connection, server_config
            
        except pyodbc.Error as e:
            raise Exception(f"Failed to connect to MSSQL server '{self.db_server}': {e}")
    
    def get_mssql_connection_sqlalchemy(self, database_server_id: str) -> Tuple[Any, str]:
        """
        Create a SQLAlchemy engine for MSSQL Server.
        
        This method creates a SQLAlchemy engine using ODBC driver for
        operations like pandas DataFrame to_sql().
        
        Parameters:
            database_server_id (str): Database server identifier from configuration
            
        Returns:
            Tuple[Any, str]: SQLAlchemy engine and connection parameters string
            
        Raises:
            ImportError: If SQLAlchemy is not available
            KeyError: If server ID not found in configuration
            Exception: If engine creation fails
            
        Example:
            >>> manager = DatabaseConnectionManager('config.yml')
            >>> engine, params = manager.get_mssql_connection_sqlalchemy('dev_server')
            >>> df.to_sql('table_name', engine, if_exists='append')
        """
        try:
            from sqlalchemy import create_engine
            import urllib.parse
        except ImportError:
            raise ImportError("SQLAlchemy is required for engine creation. Install with: pip install sqlalchemy")
        
        if 'sql_servers' not in self.config:
            raise KeyError("'sql_servers' section not found in configuration")
            
        if database_server_id not in self.config['sql_servers']:
            available_servers = list(self.config['sql_servers'].keys())
            raise KeyError(f"Server ID '{database_server_id}' not found. Available servers: {available_servers}")
        
        server_config = self.config['sql_servers'][database_server_id]
        
        if 'db_server' not in server_config:
            raise KeyError(f"'db_server' not specified for server ID '{database_server_id}'")
        
        self.db_server = server_config['db_server']
        
        try:
            connection_params = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={self.db_server};'
                f'Trusted_Connection=yes;'
            )
            
            encoded_params = urllib.parse.quote_plus(connection_params)
            connection_string = f'mssql+pyodbc:///?odbc_connect={encoded_params}'
            
            engine = create_engine(connection_string)
            
            return engine, encoded_params
            
        except Exception as e:
            raise Exception(f"Failed to create SQLAlchemy engine for '{self.db_server}': {e}")
    
    def get_incorta_connection(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Create a connection to Incorta server.
        
        Note: This functionality is under development and not fully tested.
        
        Returns:
            Tuple[Any, Dict[str, Any]]: Connection object and configuration dictionary
            
        Raises:
            ImportError: If PyODBC is not available
            KeyError: If Incorta configuration not found
            Exception: If connection fails
            
        Example:
            >>> manager = DatabaseConnectionManager('config.yml')
            >>> conn, config = manager.get_incorta_connection()
        """
        warnings.warn("Incorta connector is under development and not fully tested")
        
        try:
            import pyodbc
        except ImportError:
            raise ImportError("PyODBC is required for Incorta connections. Install with: pip install pyodbc")
        
        if 'incorta_server' not in self.config:
            raise KeyError("'incorta_server' section not found in configuration")
        
        incorta_config = self.config['incorta_server']
        
        required_fields = ['host', 'port', 'database', 'user', 'password']
        missing_fields = [field for field in required_fields if field not in incorta_config]
        
        if missing_fields:
            raise KeyError(f"Missing required Incorta configuration fields: {missing_fields}")
        
        connection_params = {
            "host": incorta_config['host'],
            "port": incorta_config['port'],
            "database": incorta_config['database'],
            "user": incorta_config['user'],
            "password": incorta_config['password']
        }
        
        try:
            connection = pyodbc.connect(**connection_params)
            return connection, connection_params
            
        except pyodbc.Error as e:
            raise Exception(f"Failed to connect to Incorta server: {e}")


class DatabaseQueryExecutor:
    """
    Executes database queries and manages database operations.
    
    This class provides methods for executing SQL queries, checking table existence,
    and writing data to databases.
    
    Example:
        >>> executor = DatabaseQueryExecutor()
        >>> df = executor.execute_mssql_query("SELECT * FROM table", "server1")
        >>> exists = executor.check_table_exists("schema.table", "server1")
    """
    
    def __init__(self, config_file_path: str = DEFAULT_CONFIG_FILE):
        """
        Initialize the query executor.
        
        Parameters:
            config_file_path (str): Path to configuration file
        """
        self.config_file = config_file_path
        self.connection_manager = DatabaseConnectionManager(config_file_path)
    
    def execute_mssql_query(self, sql_query: str, database_server_id: str, 
                           return_dataframe: bool = True) -> Union[pd.DataFrame, Any]:
        """
        Execute a SQL query on MSSQL Server.
        
        This method executes SQL queries and optionally returns results as a DataFrame.
        Supports both SELECT queries (returning data) and action queries (INSERT, UPDATE, DELETE).
        
        Parameters:
            sql_query (str): SQL query to execute
            database_server_id (str): Database server identifier
            return_dataframe (bool): Whether to return results as DataFrame (True) 
                                   or execute without return (False)
        
        Returns:
            Union[pd.DataFrame, Any]: Query results as DataFrame or execution result
            
        Raises:
            ValueError: If query or server ID is invalid
            Exception: If query execution fails
            
        Example:
            >>> executor = DatabaseQueryExecutor()
            >>> # Select query returning DataFrame
            >>> df = executor.execute_mssql_query("SELECT * FROM users", "prod_server")
            >>> 
            >>> # Action query without return
            >>> executor.execute_mssql_query("UPDATE users SET status='active'", 
            ...                              "prod_server", return_dataframe=False)
        """
        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")
            
        if not database_server_id or not database_server_id.strip():
            raise ValueError("Database server ID cannot be empty")
        
        connection = None
        try:
            connection, _ = self.connection_manager.get_mssql_connection_pyodbc(database_server_id)
            
            if return_dataframe:
                result = pd.read_sql_query(sql_query, connection)
            else:
                result = connection.execute(sql_query)
                connection.commit()  # Commit for action queries
            
            return result
            
        except Exception as e:
            if connection:
                try:
                    connection.rollback()
                except:
                    pass  # Rollback might fail for SELECT queries
            
            error_msg = f"Error executing SQL query on MSSQL Server: {str(e)}"
            raise Exception(error_msg)
            
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    def execute_incorta_query(self, sql_query: str, 
                             return_dataframe: bool = True) -> Union[pd.DataFrame, Any]:
        """
        Execute a SQL query on Incorta server.
        
        Note: This functionality is under development and not fully tested.
        
        Parameters:
            sql_query (str): SQL query to execute
            return_dataframe (bool): Whether to return DataFrame
            
        Returns:
            Union[pd.DataFrame, Any]: Query results
            
        Raises:
            ValueError: If query is invalid
            Exception: If query execution fails
        """
        warnings.warn("Incorta query execution is under development and not fully tested")
        
        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")
        
        connection = None
        try:
            connection, _ = self.connection_manager.get_incorta_connection()
            
            if return_dataframe:
                result = pd.read_sql_query(sql_query, connection)
            else:
                result = connection.execute(sql_query)
                connection.commit()
            
            return result
            
        except Exception as e:
            if connection:
                try:
                    connection.rollback()
                except:
                    pass
            
            error_msg = f"Error executing SQL query on Incorta: {str(e)}"
            raise Exception(error_msg)
            
        finally:
            if connection:
                try:
                    connection.close()
                except:
                    pass
    
    def write_dataframe_to_mssql(self, dataframe: pd.DataFrame, table_name: str, 
                                database_server_id: str, **kwargs) -> None:
        """
        Write a pandas DataFrame to MSSQL Server table.
        
        This method uses SQLAlchemy engine to write DataFrame data to MSSQL tables
        with configurable options for schema, chunking, and write behavior.
        
        Parameters:
            dataframe (pd.DataFrame): DataFrame to write to database
            table_name (str): Name of the target table
            database_server_id (str): Database server identifier
            **kwargs: Additional arguments passed to pandas.to_sql()
                     Common options:
                     - schema (str): Database schema name
                     - if_exists (str): 'fail', 'replace', 'append'
                     - index (bool): Whether to write row index
                     - chunksize (int): Rows per chunk for large datasets
                     - method (str): Insert method ('multi' for batch inserts)
        
        Raises:
            ValueError: If DataFrame is empty or parameters are invalid
            Exception: If write operation fails
            
        Example:
            >>> executor = DatabaseQueryExecutor()
            >>> df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            >>> executor.write_dataframe_to_mssql(
            ...     df, 'my_table', 'prod_server',
            ...     schema='dbo', if_exists='append', index=False
            ... )
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty")
            
        if not database_server_id or not database_server_id.strip():
            raise ValueError("Database server ID cannot be empty")
        
        try:
            engine, _ = self.connection_manager.get_mssql_connection_sqlalchemy(database_server_id)
            
            # Set default parameters
            default_params = {
                'con': engine,
                'index': False,
                'if_exists': 'append',
                'chunksize': 200,
                'method': 'multi'
            }
            
            # Merge with user-provided kwargs
            write_params = {**default_params, **kwargs}
            write_params['name'] = table_name
            
            dataframe.to_sql(**write_params)
            
        except Exception as e:
            error_msg = f"Error writing DataFrame to MSSQL table '{table_name}': {str(e)}"
            raise Exception(error_msg)
    
    def check_table_exists(self, table_name: str, database_server_id: str) -> bool:
        """
        Check if a table exists in the MSSQL database.
        
        This method queries the information_schema to determine table existence.
        Supports both schema.table and database.schema.table naming formats.
        
        Parameters:
            table_name (str): Table name in format 'schema.table' or 'database.schema.table'
            database_server_id (str): Database server identifier
            
        Returns:
            bool: True if table exists, False otherwise
            
        Raises:
            ValueError: If table name format is invalid
            Exception: If database query fails
            
        Example:
            >>> executor = DatabaseQueryExecutor()
            >>> exists = executor.check_table_exists('dbo.users', 'prod_server')
            >>> if exists:
            ...     print("Table exists")
        """
        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty")
        
        # Parse table name components
        table_parts = table_name.strip().split(".")
        
        if len(table_parts) == 3:
            database, schema, table = table_parts
            information_schema = f"{database}.information_schema.tables"
            
        elif len(table_parts) == 2:
            schema, table = table_parts
            information_schema = "information_schema.tables"
            
        else:
            raise ValueError(
                f"Invalid table name format: '{table_name}'. "
                f"Use 'schema.table' or 'database.schema.table'"
            )
        
        query = f"""
            SELECT COUNT(*)
            FROM {information_schema}
            WHERE table_name = '{table}'
            AND TABLE_SCHEMA = '{schema}'
        """
        
        try:
            result_df = self.execute_mssql_query(query, database_server_id)
            table_count = result_df.iloc[0, 0]
            
            return table_count == 1
            
        except Exception as e:
            error_msg = f"Error checking table existence for '{table_name}': {str(e)}"
            raise Exception(error_msg)
    
    def get_latest_date_from_table(self, table_name: str, database_server_id: str, 
                                  date_column: str, logger: Optional[Any] = None) -> Optional[dt.datetime]:
        """
        Retrieve the most recent date from a specified column in a database table.
        
        This method queries the specified table to find the minimum and maximum
        dates in the given date column, returning the maximum (most recent) date.
        
        Parameters:
            table_name (str): Name of the database table
            database_server_id (str): Database server identifier  
            date_column (str): Name of the date column to query
            logger (Optional[Any]): Logger instance for custom logging
            
        Returns:
            Optional[dt.datetime]: Most recent date found, or None if table doesn't exist
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If query execution fails
            
        Example:
            >>> executor = DatabaseQueryExecutor()
            >>> latest_date = executor.get_latest_date_from_table(
            ...     'sales.transactions', 'prod_server', 'transaction_date'
            ... )
            >>> if latest_date:
            ...     print(f"Latest transaction: {latest_date}")
        """
        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty")
            
        if not date_column or not date_column.strip():
            raise ValueError("Date column name cannot be empty")
        
        # Check if table exists first
        if not self.check_table_exists(table_name, database_server_id):
            message = f"Table '{table_name}' does not exist"
            if cfuncs and logger:
                cfuncs.custom_print(message, logger)
            else:
                print(message)
            return None
        
        # Query for min and max dates
        query = f"""
            SELECT MIN({date_column}) as min_time,
                   MAX({date_column}) as max_time
            FROM {table_name}
        """
        
        try:
            date_results = self.execute_mssql_query(query, database_server_id)
            most_recent_date = date_results['max_time'].iloc[0]
            
            message = f"Latest date in {table_name}: {most_recent_date}"
            if cfuncs and logger:
                cfuncs.custom_print(message, logger)
            else:
                print(message)
            
            return most_recent_date
            
        except Exception as e:
            error_msg = f"Error retrieving latest date from '{table_name}': {str(e)}"
            raise Exception(error_msg)


class DataWorkflowManager:
    """
    Manages data processing workflows and output specifications.
    
    This class handles the coordination of data processing workflows, including
    date range management, output specification processing, and recursive data updates.
    
    Example:
        >>> workflow = DataWorkflowManager()
        >>> updated_specs, dates = workflow.update_output_specifications(output_specs)
        >>> workflow.run_data_processing_recursively(specs, generator_func, dates)
    """
    
    def __init__(self, config_file_path: str = DEFAULT_CONFIG_FILE):
        """
        Initialize the workflow manager.
        
        Parameters:
            config_file_path (str): Path to configuration file
        """
        self.config_file = config_file_path
        self.query_executor = DatabaseQueryExecutor(config_file_path)
    
    def update_output_specifications(self, output_specifications: Union[List[Dict], Dict],
                                    date_range_years: List[int] = [2021, 2099],
                                    month_step: int = 1,
                                    first_date: Optional[Union[str, dt.datetime, pd.Timestamp]] = None,
                                    last_date: dt.date = dt.datetime.now().date(),
                                    logger: Optional[Any] = None) -> Tuple[List[Dict], List[dt.date]]:
        """
        Update output specifications with last saved dates and generate run dates.
        
        This method checks the last saved date for each output specification and
        generates a list of date ranges for processing based on what needs to be updated.
        
        Parameters:
            output_specifications (Union[List[Dict], Dict]): Output configuration(s)
            date_range_years (List[int]): Start and end years for date range
            month_step (int): Step size in months for date generation
            first_date (Optional[Union[str, dt.datetime, pd.Timestamp]]): Start date override
            last_date (dt.date): End date for processing
            logger (Optional[Any]): Logger instance for custom logging
            
        Returns:
            Tuple[List[Dict], List[dt.date]]: Updated specifications and run dates
            
        Raises:
            ValueError: If specifications are invalid
            Exception: If date processing fails
            
        Example:
            >>> workflow = DataWorkflowManager()
            >>> specs = [{'format': 'MS_db', 'output_location': 'db.schema.table', 
            ...           'db_server_id': 'server1', 'date_col': 'created_date'}]
            >>> updated_specs, run_dates = workflow.update_output_specifications(specs)
            >>> print(f"Generated {len(run_dates)} date ranges for processing")
        """
        # Convert single dict to list for uniform processing
        if isinstance(output_specifications, dict):
            output_specifications = [output_specifications]
        
        if not output_specifications:
            raise ValueError("Output specifications cannot be empty")
        
        updated_specifications = output_specifications.copy()
        all_last_dates = []
        
        # Process each output specification
        for index, output_spec in enumerate(output_specifications):
            if not isinstance(output_spec, dict):
                raise ValueError(f"Output specification at index {index} must be a dictionary")
            
            # Get last saved date for this output
            try:
                last_saved_date = self._get_last_date_from_output(output_spec, logger)
                updated_specifications[index]['last_date'] = last_saved_date
                all_last_dates.append(last_saved_date)
                
            except Exception as e:
                error_msg = f"Error processing output specification {index}: {str(e)}"
                if cfuncs and logger:
                    cfuncs.custom_print(error_msg, logger)
                else:
                    print(error_msg)
                continue
        
        # Generate run dates based on first specification
        if updated_specifications:
            run_dates = self._generate_run_dates(
                all_last_dates[0] if all_last_dates else None,
                first_date, last_date, date_range_years, month_step, logger
            )
        else:
            run_dates = []
        
        # Check for inconsistent last dates across specifications
        unique_last_dates = set(filter(None, all_last_dates))  # Remove None values
        if len(unique_last_dates) > 1:
            warning_msg = (
                f"Warning: Different last dates found across specifications: {unique_last_dates}. "
                f"Run dates based on first specification."
            )
            if cfuncs and logger:
                cfuncs.custom_print(warning_msg, logger)
            else:
                print(warning_msg)
        
        return updated_specifications, run_dates
    
    def save_workflow_outputs(self, output_data_dict: Dict[str, Any], 
                             output_specifications: Union[List[Dict], Dict],
                             logger: Optional[Any] = None) -> bool:
        """
        Save DataFrames from output dictionary to specified locations and formats.
        
        This method processes the output data dictionary and saves DataFrames to
        their specified destinations (database tables or files) according to the
        output specifications.
        
        Parameters:
            output_data_dict (Dict[str, Any]): Dictionary containing:
                - 'output_df_keys': List of DataFrame identifiers
                - 'dfs': List of DataFrames to save
            output_specifications (Union[List[Dict], Dict]): Output configurations
            logger (Optional[Any]): Logger instance for custom logging
            
        Returns:
            bool: True if all saves successful, False otherwise
            
        Raises:
            ValueError: If output specifications don't match data dictionary
            Exception: If save operations fail
            
        Example:
            >>> workflow = DataWorkflowManager()
            >>> output_dict = {
            ...     'output_df_keys': [['table1'], ['table2']], 
            ...     'dfs': [df1, df2]
            ... }
            >>> specs = [{'output_df_key': 'table1', 'format': 'MS_db', ...}]
            >>> success = workflow.save_workflow_outputs(output_dict, specs)
        """
        if not isinstance(output_data_dict, dict):
            raise ValueError("output_data_dict must be a dictionary")
        
        required_keys = ['output_df_keys', 'dfs']
        missing_keys = [key for key in required_keys if key not in output_data_dict]
        if missing_keys:
            raise ValueError(f"Missing required keys in output_data_dict: {missing_keys}")
        
        # Convert single dict to list
        if isinstance(output_specifications, dict):
            output_specifications = [output_specifications]
        
        # Flatten the list of DataFrame keys
        if cfuncs:
            flattened_keys = cfuncs.flattenList(output_data_dict['output_df_keys'])
        else:
            flattened_keys = [key for sublist in output_data_dict['output_df_keys'] for key in sublist]
        
        # Check for orphaned DataFrames (in dict but not in specs)
        spec_keys = set(spec['output_df_key'] for spec in output_specifications)
        orphaned_dfs = set(flattened_keys) - spec_keys
        
        if orphaned_dfs:
            error_msg = (
                f"DataFrames not specified in output configurations: {orphaned_dfs}. "
                f"Match output specifications with generator function return values."
            )
            if cfuncs and logger:
                cfuncs.custom_print(error_msg, logger)
            else:
                print(error_msg)
            return False
        
        # Check for orphaned specifications (in specs but not in dict)
        orphaned_specs = spec_keys - set(flattened_keys)
        if orphaned_specs:
            error_msg = (
                f"Output specifications without corresponding DataFrames: {orphaned_specs}. "
                f"Match output specifications with generator function return values."
            )
            if cfuncs and logger:
                cfuncs.custom_print(error_msg, logger)
            else:
                print(error_msg)
            return False
        
        # Process each DataFrame and its corresponding keys
        try:
            for df_keys, dataframe in zip(output_data_dict['output_df_keys'], output_data_dict['dfs']):
                if dataframe.empty:
                    message = "DataFrame is empty, skipping save operation"
                    if cfuncs and logger:
                        cfuncs.custom_print(message, logger)
                    else:
                        print(message)
                    continue
                
                for df_key in df_keys:
                    # Find matching output specification
                    matching_specs = [spec for spec in output_specifications if spec['output_df_key'] == df_key]
                    
                    if not matching_specs:
                        continue  # Already checked above, but safety check
                    
                    output_spec = matching_specs[0]
                    
                    try:
                        self._save_single_output(dataframe, output_spec, logger)
                        
                    except Exception as e:
                        error_msg = f"Error saving {df_key}: {str(e)}"
                        if cfuncs and logger:
                            cfuncs.custom_print(error_msg, logger)
                        else:
                            print(error_msg)
                        return False
            
            # Success message
            success_msg = "All outputs saved successfully"
            if cfuncs and logger:
                cfuncs.custom_print('-' * 50, logger)
                cfuncs.custom_print(success_msg, logger)
            else:
                print('-' * 50)
                print(success_msg)
            
            return True
            
        except Exception as e:
            error_msg = f"Error in save workflow: {str(e)}"
            if cfuncs and logger:
                cfuncs.custom_print(error_msg, logger)
            else:
                print(error_msg)
            return False
    
    def run_data_processing_recursively(self, output_specifications: List[Dict],
                                       data_generator_function: Callable,
                                       date_range_years: List[int] = [2021, 2099],
                                       month_step: int = 1,
                                       first_date: Optional[Union[str, dt.datetime]] = None,
                                       last_date: dt.date = dt.datetime.now().date(),
                                       logger: Optional[Any] = None,
                                       **kwargs) -> None:
        """
        Execute data generation function recursively over date ranges and save outputs.
        
        This method orchestrates the complete workflow of updating output specifications,
        generating date ranges, executing the data processing function for each date range,
        and saving the results.
        
        Parameters:
            output_specifications (List[Dict]): List of output configurations
            data_generator_function (Callable): Function that generates data for date ranges
            date_range_years (List[int]): Start and end years for processing
            month_step (int): Step size in months
            first_date (Optional[Union[str, dt.datetime]]): Start date override
            last_date (dt.date): End date for processing
            logger (Optional[Any]): Logger instance
            **kwargs: Additional arguments passed to data generator function
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If processing fails
            
        Example:
            >>> def my_generator(start_date, end_date, logger=None):
            ...     # Generate data for date range
            ...     df = create_data(start_date, end_date)
            ...     return {'output_df_keys': [['my_table']], 'dfs': [df]}
            >>> 
            >>> workflow = DataWorkflowManager()
            >>> specs = [{'output_df_key': 'my_table', 'format': 'MS_db', ...}]
            >>> workflow.run_data_processing_recursively(specs, my_generator)
        """
        import inspect
        
        if not callable(data_generator_function):
            raise ValueError("data_generator_function must be callable")
        
        if not output_specifications:
            raise ValueError("output_specifications cannot be empty")
        
        # Update output specifications and get run dates
        message = "Updating output specifications..."
        if cfuncs and logger:
            cfuncs.custom_print(message, logger)
        else:
            print(message)
        
        try:
            updated_specs, run_dates = self.update_output_specifications(
                output_specifications, date_range_years, month_step,
                first_date, last_date, logger
            )
        except Exception as e:
            error_msg = f"Error updating specifications: {str(e)}"
            raise Exception(error_msg)
        
        if not run_dates or len(run_dates) < 2:
            message = "No date ranges to process. Data is up to date."
            if cfuncs and logger:
                cfuncs.custom_print(message, logger)
            else:
                print(message)
            return
        
        # Extract function parameters for data generator
        try:
            function_signature = inspect.signature(data_generator_function)
            function_params = list(function_signature.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in function_params}
        except Exception as e:
            filtered_kwargs = kwargs  # Fallback to all kwargs
        
        separator = '/' * 50
        if cfuncs and logger:
            cfuncs.custom_print(separator, logger)
        else:
            print(separator)
        
        # Process each date range
        try:
            for i in range(len(run_dates) - 1):
                if cfuncs:
                    start_date, end_date = cfuncs.extract_start_end(run_dates, i)
                else:
                    start_date, end_date = run_dates[i], run_dates[i + 1]
                
                process_msg = f"Processing {data_generator_function.__name__} for period {start_date} to {end_date}"
                if cfuncs and logger:
                    cfuncs.custom_print(process_msg, logger)
                else:
                    print(process_msg)
                
                # Execute data generator function
                try:
                    output_data = data_generator_function(
                        start_date, end_date, logger=logger, **filtered_kwargs
                    )
                    
                    if not isinstance(output_data, dict):
                        raise ValueError("Data generator function must return a dictionary")
                    
                    # Save the outputs
                    save_success = self.save_workflow_outputs(output_data, updated_specs, logger)
                    
                    if not save_success:
                        raise Exception("Failed to save workflow outputs")
                    
                except Exception as e:
                    error_msg = f"Error processing date range {start_date} to {end_date}: {str(e)}"
                    if cfuncs and logger:
                        cfuncs.custom_print(f"*** {error_msg} ***", logger)
                        cfuncs.custom_print('*' * 70, logger)
                    else:
                        print(f"*** {error_msg} ***")
                        print('*' * 70)
                    raise
        
        except Exception as e:
            final_error = f"Recursive processing failed for {data_generator_function.__name__}: {str(e)}"
            if cfuncs and logger:
                cfuncs.custom_print(f"*** {final_error} ***", logger)
                cfuncs.custom_print('*' * 70, logger)
            else:
                print(f"*** {final_error} ***")
                print('*' * 70)
            raise Exception(final_error)
    
    def _get_last_date_from_output(self, output_spec: Dict, logger: Optional[Any] = None) -> Optional[dt.datetime]:
        """Get last date from output specification (database or file)."""
        if 'date_col' not in output_spec:
            raise ValueError("'date_col' not specified in output specification")
        
        date_column = output_spec['date_col']
        
        if output_spec['format'] == 'MS_db':
            if 'db_server_id' not in output_spec:
                raise ValueError("'db_server_id' not specified for MS_db format")
                
            return self.query_executor.get_latest_date_from_table(
                output_spec['output_location'],
                output_spec['db_server_id'],
                date_column,
                logger
            )
            
        elif output_spec['format'] == 'parquet':
            return get_latest_date_from_parquet(
                output_spec['output_location'], date_column, logger
            )
            
        else:
            return None
    
    def _generate_run_dates(self, last_saved_date: Optional[dt.datetime],
                           first_date: Optional[Union[str, dt.datetime, pd.Timestamp]],
                           last_date: dt.date, date_range_years: List[int],
                           month_step: int, logger: Optional[Any]) -> List[dt.date]:
        """Generate list of run dates based on parameters."""
        
        # Determine effective first date
        effective_first_date = None
        warning_needed = False
        
        if first_date is not None:
            if isinstance(first_date, str):
                if cfuncs and logger:
                    cfuncs.custom_print(f"Using provided first date: {first_date}", logger)
                effective_first_date = dt.datetime.strptime(first_date, "%Y-%m-%d").date()
            elif isinstance(first_date, pd.Timestamp):
                effective_first_date = first_date.date()
                warning_needed = True
            elif isinstance(first_date, dt.datetime):
                effective_first_date = first_date.date()
            elif isinstance(first_date, dt.date):
                effective_first_date = first_date
        else:
            if last_saved_date is not None:
                effective_first_date = (last_saved_date + dt.timedelta(days=1)).date()
            else:
                effective_first_date = None
        
        # Show warning if needed
        if warning_needed and last_saved_date is not None:
            warning_msg = (
                f"Last saved date: {last_saved_date}, but starting from provided date: {first_date}"
            )
            if cfuncs and logger:
                cfuncs.custom_print(warning_msg, logger)
            else:
                print(warning_msg)
        
        # Generate date list
        if cfuncs:
            run_dates = cfuncs.datesList(
                range_date__year=date_range_years,
                month_step=month_step,
                firstDate=effective_first_date,
                lastDate=last_date
            )
        else:
            # Fallback simple date generation
            run_dates = []
            if effective_first_date and effective_first_date <= last_date:
                current_date = effective_first_date
                while current_date <= last_date:
                    run_dates.append(current_date)
                    # Simple monthly increment
                    if current_date.month + month_step <= 12:
                        current_date = current_date.replace(month=current_date.month + month_step)
                    else:
                        current_date = current_date.replace(
                            year=current_date.year + 1,
                            month=current_date.month + month_step - 12
                        )
        
        # Log results
        if not run_dates:
            message = "Database/file is up to date"
            if cfuncs and logger:
                cfuncs.custom_print(message, logger)
            else:
                print(message)
        else:
            message = f"Generated date ranges: {run_dates}"
            if cfuncs and logger:
                cfuncs.custom_print(message, logger)
            else:
                print(message)
        
        return run_dates
    
    def _save_single_output(self, dataframe: pd.DataFrame, output_spec: Dict,
                           logger: Optional[Any] = None) -> None:
        """Save a single DataFrame according to its specification."""
        output_format = output_spec['format']
        output_location = output_spec['output_location']
        overwrite_flag = output_spec.get('overwrite', False)
        
        save_msg = f"Saving {output_spec['output_df_key']} to {output_location}"
        if cfuncs and logger:
            cfuncs.custom_print(save_msg, logger)
        else:
            print(save_msg)
        
        if output_format == 'MS_db':
            self._save_to_database(dataframe, output_spec, overwrite_flag)
            
        elif output_format == 'parquet':
            self._save_to_parquet(dataframe, output_location, overwrite_flag)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_to_database(self, dataframe: pd.DataFrame, output_spec: Dict, overwrite: bool) -> None:
        """Save DataFrame to database."""
        if 'db_server_id' not in output_spec:
            raise ValueError("'db_server_id' not specified for database output")
        
        output_location = output_spec['output_location']
        location_parts = output_location.split('.')
        
        if len(location_parts) < 3:
            raise ValueError(f"Invalid database location format: {output_location}")
        
        table_name = location_parts[2]
        schema_name = f"{location_parts[0]}.{location_parts[1]}"
        
        self.query_executor.write_dataframe_to_mssql(
            dataframe,
            table_name=table_name,
            database_server_id=output_spec['db_server_id'],
            schema=schema_name,
            chunksize=200,
            method='multi',
            index=False,
            if_exists='replace' if overwrite else 'append'
        )
    
    def _save_to_parquet(self, dataframe: pd.DataFrame, file_path: str, overwrite: bool) -> None:
        """Save DataFrame to Parquet file."""
        if not overwrite and os.path.isfile(file_path):
            # Append to existing file
            existing_df = pd.read_parquet(file_path)
            combined_df = pd.concat([existing_df, dataframe], axis=0)
            combined_df.to_parquet(file_path, index=False)
        else:
            # Create new file or overwrite
            dataframe.to_parquet(file_path, index=False)


# ============================================================================
# UTILITY FUNCTIONS (Simple functions for file operations and date handling)
# ============================================================================

def get_latest_date_from_parquet(file_path: str, date_column: str, 
                                logger: Optional[Any] = None) -> Optional[dt.datetime]:
    """
    Retrieve the most recent date from a specified column in a Parquet file.
    
    Parameters:
        file_path (str): Path to the Parquet file
        date_column (str): Name of the date column to check
        logger (Optional[Any]): Logger instance for custom logging
        
    Returns:
        Optional[dt.datetime]: Most recent date found, or None if file doesn't exist
        
    Example:
        >>> latest_date = get_latest_date_from_parquet('data.parquet', 'created_date')
        >>> if latest_date:
        ...     print(f"Latest record: {latest_date}")
    """
    if not os.path.isfile(file_path):
        message = f"File '{file_path}' does not exist"
        if cfuncs and logger:
            cfuncs.custom_print(message, logger)
        else:
            print(message)
        return None
    
    try:
        dataframe = pd.read_parquet(file_path)
        
        if date_column not in dataframe.columns:
            raise ValueError(f"Date column '{date_column}' not found in file")
        
        latest_date = dataframe[date_column].max()
        
        message = f"Latest date in {file_path}: {latest_date}"
        if cfuncs and logger:
            cfuncs.custom_print(message, logger)
        else:
            print(message)
        
        return latest_date
        
    except Exception as e:
        error_msg = f"Error reading Parquet file '{file_path}': {str(e)}"
        if cfuncs and logger:
            cfuncs.custom_print(error_msg, logger)
        else:
            print(error_msg)
        return None


def load_parquet_data_between_dates(file_path: str, date_column: str,
                                   start_date: str = '2019-01-01',
                                   end_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Load data from a Parquet file filtered by date range.
    
    Parameters:
        file_path (str): Path to the Parquet file
        date_column (str): Name of the date column for filtering
        start_date (str): Start date in 'YYYY-MM-DD' format (inclusive)
        end_date (str): End date in 'YYYY-MM-DD' format (exclusive)
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing data within date range
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If date format is invalid or column doesn't exist
        
    Example:
        >>> df = load_parquet_data_between_dates(
        ...     'transactions.parquet', 'transaction_date',
        ...     '2023-01-01', '2023-12-31'
        ... )
        >>> print(f"Loaded {len(df)} records")
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist")
    
    try:
        start_datetime = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = dt.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")
    
    try:
        dataframe = pd.read_parquet(file_path)
        
        if date_column not in dataframe.columns:
            raise ValueError(f"Date column '{date_column}' not found in file")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(dataframe[date_column]):
            dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        
        # Filter by date range
        filtered_df = dataframe[
            (dataframe[date_column] >= start_datetime) & 
            (dataframe[date_column] < end_datetime)
        ]
        
        return filtered_df
        
    except Exception as e:
        raise Exception(f"Error loading and filtering Parquet file: {str(e)}")


def get_last_date_from_output_spec(output_specification: Dict, 
                                  config_file: str = DEFAULT_CONFIG_FILE,
                                  logger: Optional[Any] = None) -> Optional[dt.datetime]:
    """
    Get the last date from an output specification (database or file).
    
    This utility function determines the output type and retrieves the most
    recent date from the appropriate source.
    
    Parameters:
        output_specification (Dict): Output specification containing format and location info
        config_file (str): Path to configuration file
        logger (Optional[Any]): Logger instance
        
    Returns:
        Optional[dt.datetime]: Most recent date found, or None if not found
        
    Example:
        >>> spec = {'format': 'MS_db', 'output_location': 'db.schema.table',
        ...         'db_server_id': 'server1', 'date_col': 'created_date'}
        >>> last_date = get_last_date_from_output_spec(spec)
    """
    if 'date_col' not in output_specification:
        raise ValueError("'date_col' not specified in output specification")
    
    date_column = output_specification['date_col']
    output_format = output_specification.get('format')
    
    if output_format == 'MS_db':
        if 'db_server_id' not in output_specification:
            raise ValueError("'db_server_id' not specified for MS_db format")
        
        executor = DatabaseQueryExecutor(config_file)
        return executor.get_latest_date_from_table(
            output_specification['output_location'],
            output_specification['db_server_id'],
            date_column,
            logger
        )
        
    elif output_format == 'parquet':
        return get_latest_date_from_parquet(
            output_specification['output_location'], date_column, logger
        )
        
    else:
        message = f"Unsupported output format: {output_format}"
        if cfuncs and logger:
            cfuncs.custom_print(message, logger)
        else:
            print(message)
        return None


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def mSql_query(sql_query, db_server_id, config_file=DEFAULT_CONFIG_FILE, return_df=True):
    """Legacy function for backward compatibility."""
    executor = DatabaseQueryExecutor(config_file)
    return executor.execute_mssql_query(sql_query, db_server_id, return_df)


def incorta_query(sql_query, config_file=DEFAULT_CONFIG_FILE, return_df=True):
    """Legacy function for backward compatibility."""
    executor = DatabaseQueryExecutor(config_file)
    return executor.execute_incorta_query(sql_query, return_df)


def df2MSQL(df, table_name, db_server_id, config_file=DEFAULT_CONFIG_FILE, **kwargs):
    """Legacy function for backward compatibility."""
    executor = DatabaseQueryExecutor(config_file)
    executor.write_dataframe_to_mssql(df, table_name, db_server_id, **kwargs)


def table_exists(connection, table_name):
    """Legacy function for backward compatibility - direct connection version."""
    table_parts = table_name.split(".")
    if len(table_parts) == 3:
        database, schema, table = table_parts
        information_schema = f"{database}.information_schema.tables"
    elif len(table_parts) == 2:
        schema, table = table_parts
        information_schema = "information_schema.tables"
    else:
        raise ValueError(f"Invalid table name format: {table_name}")
    
    with connection.cursor() as cursor:
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM {information_schema}
                WHERE table_schema = %s AND table_name = %s
            );
        """, (schema, table))
        return cursor.fetchone()[0]


def MSql_table_check(tablename, db_server_id, config_file=DEFAULT_CONFIG_FILE):
    """Legacy function for backward compatibility."""
    executor = DatabaseQueryExecutor(config_file)
    return executor.check_table_exists(tablename, db_server_id)


def get_last_date_from_mssql_table(table_name, db_server_id, date_column, 
                                   config_file=DEFAULT_CONFIG_FILE, logger=None):
    """Legacy function for backward compatibility."""
    executor = DatabaseQueryExecutor(config_file)
    return executor.get_latest_date_from_table(table_name, db_server_id, date_column, logger)


def last_date_MSql(db_name, db_server_id, date_col, config_file=DEFAULT_CONFIG_FILE, logger=None):
    """Legacy function for backward compatibility."""
    return get_last_date_from_mssql_table(db_name, db_server_id, date_col, config_file, logger)


def last_date_parquet(file_name, date_col, logger=None):
    """Legacy function for backward compatibility."""
    return get_latest_date_from_parquet(file_name, date_col, logger)


def last_date(output_dict, logger=None, **kwargs):
    """Legacy function for backward compatibility."""
    return get_last_date_from_output_spec(output_dict, logger=logger, **kwargs)


def load_parquet_between_dates(ufile, date_col, start_date='2019-01-01', end_date='2020-01-01'):
    """Legacy function for backward compatibility."""
    return load_parquet_data_between_dates(ufile, date_col, start_date, end_date)


def update_output_specS(output_specS, range_date__year=[2021,2099], month_step=1,
                       firstDate=None, lastDate=dt.datetime.now().date(), logger=None):
    """Legacy function for backward compatibility."""
    workflow = DataWorkflowManager()
    return workflow.update_output_specifications(
        output_specS, range_date__year, month_step, firstDate, lastDate, logger
    )


def save_outputs(output_dict, output_specS, logger=None):
    """Legacy function for backward compatibility."""
    workflow = DataWorkflowManager()
    success = workflow.save_workflow_outputs(output_dict, output_specS, logger)
    return 1 if success else 0


def run_recursively(output_specS, dfGenerator_func, range_date__year=[2021,2099],
                   month_step=1, firstDate=None, lastDate=dt.datetime.now().date(),
                   logger=None, **kwargs):
    """Legacy function for backward compatibility."""
    workflow = DataWorkflowManager()
    workflow.run_data_processing_recursively(
        output_specS, dfGenerator_func, range_date__year, month_step,
        firstDate, lastDate, logger, **kwargs
    )


# Legacy class for backward compatibility
class cred_setup_mssql:
    """Legacy class for backward compatibility."""
    def __init__(self, config_file):
        self.connection_manager = DatabaseConnectionManager(config_file)
        self.config = self.connection_manager.config
        
    def MSSQL_connector__pyodbc(self, db_server_id):
        return self.connection_manager.get_mssql_connection_pyodbc(db_server_id)
    
    def MSSQL_connector__sqlalchemy(self, db_server_id):
        return self.connection_manager.get_mssql_connection_sqlalchemy(db_server_id)
    
    def incorta_connector(self):
        return self.connection_manager.get_incorta_connection()


# Function mapping for reference
FUNCTION_MAPPING = {
    'cred_setup_mssql': 'DatabaseConnectionManager',
    'mSql_query': 'DatabaseQueryExecutor.execute_mssql_query()',
    'incorta_query': 'DatabaseQueryExecutor.execute_incorta_query()',
    'df2MSQL': 'DatabaseQueryExecutor.write_dataframe_to_mssql()',
    'MSql_table_check': 'DatabaseQueryExecutor.check_table_exists()',
    'get_last_date_from_mssql_table': 'DatabaseQueryExecutor.get_latest_date_from_table()',
    'last_date_MSql': 'get_latest_date_from_parquet()',
    'last_date_parquet': 'get_latest_date_from_parquet()',
    'last_date': 'get_last_date_from_output_spec()',
    'load_parquet_between_dates': 'load_parquet_data_between_dates()',
    'update_output_specS': 'DataWorkflowManager.update_output_specifications()',
    'save_outputs': 'DataWorkflowManager.save_workflow_outputs()',
    'run_recursively': 'DataWorkflowManager.run_data_processing_recursively()',
}


def print_function_mapping():
    """Print the mapping of old functions to new implementations."""
    print("Function Mapping - Old to New:")
    print("=" * 70)
    for old_func, new_impl in FUNCTION_MAPPING.items():
        print(f"{old_func:35} -> {new_impl}")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Print function mapping for reference
    print_function_mapping()
    
    # Example usage
    print("\nExample Usage:")
    print("# Object-oriented approach:")
    print("conn_manager = DatabaseConnectionManager('config.yml')")
    print("executor = DatabaseQueryExecutor('config.yml')")
    print("workflow = DataWorkflowManager('config.yml')")
    print("\n# Backward compatible approach:")
    print("df = mSql_query('SELECT * FROM table', 'server_id')")
    print("success = save_outputs(output_dict, output_specs)")