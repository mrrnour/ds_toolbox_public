"""
I/O Functions - Pragmatic Refactored Version
==========================================

This module provides comprehensive I/O operations for Azure cloud services including
Synapse, Delta Tables, Blob Storage, and PI Server data processing with a balanced
object-oriented approach that groups related functionality without over-engineering.

Classes:
    - ConfigurationManager: Handles configuration loading and credential management
    - DatabaseOperator: Synapse and Delta Table query operations
    - BlobStorageManager: Azure blob storage I/O operations
    - DataProcessor: PI server and data processing utilities

Utility Functions:
    - Platform utilities (Spark, Databricks)
    - Data transformation helpers
    - Query processing utilities

Author: Refactored from original procedural code
"""

import os
import sys
import re
import io
import datetime as dt
from importlib import resources as res
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import warnings

import pandas as pd
import numpy as np
import yaml

try:
    import dsToolbox.common_funcs as cfuncs
    import dsToolbox.default_values as par
except ImportError:
    warnings.warn("dsToolbox modules not available. Some functionality may be limited.")
    cfuncs = None
    par = None


class ConfigurationManager:
    """
    Manages configuration loading and credential management for Azure services.
    
    This class handles YAML configuration files, environment variables, and
    Azure Key Vault credentials for various Azure services.
    
    Attributes:
        config (dict): Main configuration dictionary
        key_vault_dictionaries (dict): Key vault configuration mappings
        local_access_config (dict): Local access configuration
        synapse_credentials (dict): Synapse connection credentials
        azure_ml_app_id (str): Azure ML application ID
        pi_server_config (dict): PI server configuration
        
    Example:
        >>> config_mgr = ConfigurationManager()
        >>> config = config_mgr.load_configuration('custom_config.yml')
        >>> creds = config_mgr.get_credential_strings('azure_synapse')
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = None
        self.key_vault_dictionaries = None
        self.local_access_config = None
        self.synapse_credentials = None
        self.azure_ml_app_id = None
        self.pi_server_config = None
    
    def load_configuration(self, custom_config: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
        """
        Load configuration from various sources.
        
        This method loads configuration from YAML files, dictionaries, or default
        embedded configuration with proper validation and transformation.
        
        Parameters:
            custom_config (Optional[Union[str, Dict]]): Configuration source.
                                                      Can be:
                                                      - None: Use default embedded config
                                                      - str: Path to YAML configuration file
                                                      - dict: Configuration dictionary
        
        Returns:
            Dict[str, Any]: Complete configuration dictionary containing:
                           - key_vault_dictS: Key vault configurations
                           - KV_access_local: Local key vault access config
                           - synapse_cred_dict: Synapse credentials
                           - azure_ml_appID: Azure ML application ID
                           - pi_server: PI server configuration
                           
        Raises:
            FileNotFoundError: If configuration file path doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration format is invalid
            
        Example:
            >>> config_mgr = ConfigurationManager()
            >>> 
            >>> # Load default configuration
            >>> config = config_mgr.load_configuration()
            >>> 
            >>> # Load from file
            >>> config = config_mgr.load_configuration('my_config.yml')
            >>> 
            >>> # Load from dictionary
            >>> custom_dict = {'key_vault_name': 'my-vault', ...}
            >>> config = config_mgr.load_configuration(custom_dict)
        """
        try:
            if custom_config is None:
                # Load default embedded configuration
                try:
                    with res.open_binary('dsToolbox', 'config.yml') as fp:
                        config_yaml = yaml.load(fp, Loader=yaml.Loader)
                except Exception as e:
                    raise FileNotFoundError(f"Default configuration file not found: {e}")
                    
            elif isinstance(custom_config, dict):
                # Handle dictionary configuration with transformation
                config_yaml = self._transform_config_dict(custom_config.copy())
                
            elif isinstance(custom_config, str):
                # Load from file path
                if not os.path.exists(custom_config):
                    raise FileNotFoundError(f"Configuration file not found: {custom_config}")
                
                try:
                    from pathlib import Path
                    config_yaml = yaml.safe_load(Path(custom_config).read_text())
                except yaml.YAMLError as e:
                    raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading configuration file: {e}")
            else:
                raise ValueError("custom_config must be None, string path, or dictionary")
            
            if not config_yaml:
                raise ValueError("Configuration is empty or invalid")
            
            # Extract and store configuration components
            self.config = config_yaml
            self.key_vault_dictionaries = config_yaml.get('key_vault_dictS', {})
            self.local_access_config = config_yaml.get('KV_access_local')
            self.synapse_credentials = config_yaml.get('synapse_cred_dict')
            self.azure_ml_app_id = config_yaml.get('azure_ml_appID')
            self.pi_server_config = config_yaml.get('pi_server')
            
            return config_yaml
            
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            raise Exception(error_msg)
    
    def _transform_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Transform simplified config dictionary to full format."""
        # Handle legacy format transformation
        required_legacy_keys = ['key_vault_name', 'secret_name', 'storage_account']
        
        if (all(key in config_dict for key in required_legacy_keys) and 
            'key_vault_dictS' not in config_dict):
            
            # Transform legacy format
            config_dict['key_vault_dictS'] = {}
            config_dict['key_vault_dictS'][config_dict['storage_account']] = {
                'key_vault_name': config_dict['key_vault_name'],
                'secret_name': config_dict['secret_name']
            }
            
            # Remove legacy keys
            for key in required_legacy_keys:
                config_dict.pop(key, None)
        
        return config_dict
    
    def get_credential_strings(self, key_vault_identifier: str, 
                              custom_config: Optional[Union[str, Dict]] = None,
                              platform: str = 'databricks') -> 'AzureCredentialManager':
        """
        Get credential manager for specified Azure service.
        
        Parameters:
            key_vault_identifier (str): Key vault dictionary identifier
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform ('databricks', 'local', 'vm_docker')
            
        Returns:
            AzureCredentialManager: Configured credential manager instance
            
        Raises:
            KeyError: If key vault identifier not found in configuration
            
        Example:
            >>> config_mgr = ConfigurationManager()
            >>> cred_mgr = config_mgr.get_credential_strings('azure_synapse')
            >>> url, props, _ = cred_mgr.get_synapse_connection()
        """
        if not self.config:
            self.load_configuration(custom_config)
        
        if key_vault_identifier not in self.key_vault_dictionaries:
            available_keys = list(self.key_vault_dictionaries.keys())
            raise KeyError(f"Key vault identifier '{key_vault_identifier}' not found. "
                          f"Available: {available_keys}")
        
        return AzureCredentialManager(
            key_vault_identifier, 
            self.config, 
            platform
        )
    
    def get_spark_session(self) -> Tuple[Any, Any]:
        """
        Get or create Spark session and SQL context.
        
        Returns:
            Tuple[Any, Any]: Spark session and SQL context
            
        Raises:
            ImportError: If PySpark is not available
            Exception: If Spark session creation fails
            
        Example:
            >>> config_mgr = ConfigurationManager()
            >>> spark, sql_context = config_mgr.get_spark_session()
            >>> df = spark.sql("SELECT * FROM my_table")
        """
        try:
            import pyspark
            spark = pyspark.sql.SparkSession.builder.getOrCreate()
            sql_context = pyspark.SQLContext(spark.sparkContext)
            return spark, sql_context
            
        except ImportError:
            raise ImportError("PySpark is required for Spark operations. Install with: pip install pyspark")
        except Exception as e:
            raise Exception(f"Error creating Spark session: {str(e)}")
    
    def get_databricks_utilities(self) -> Any:
        """
        Get Databricks utilities (dbutils).
        
        Returns:
            Any: Databricks utilities object
            
        Raises:
            ImportError: If not running in Databricks environment
            Exception: If dbutils is not available
            
        Example:
            >>> config_mgr = ConfigurationManager()
            >>> dbutils = config_mgr.get_databricks_utilities()
            >>> dbutils.fs.ls('/mnt/data/')
        """
        try:
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
            return dbutils
            
        except (ImportError, KeyError) as e:
            raise ImportError("dbutils not available. This function requires Databricks environment.")
        except Exception as e:
            raise Exception(f"Error accessing dbutils: {str(e)}")


class AzureCredentialManager:
    """
    Manages Azure credentials and connection strings for various Azure services.
    
    This class handles credential retrieval from Azure Key Vault and provides
    connection strings for Synapse, Blob Storage, and other Azure services.
    """
    
    def __init__(self, key_vault_identifier: str, config: Dict[str, Any], platform: str = 'databricks'):
        """
        Initialize Azure credential manager.
        
        Parameters:
            key_vault_identifier (str): Key vault dictionary identifier
            config (Dict[str, Any]): Full configuration dictionary
            platform (str): Target platform
        """
        self.key_vault_identifier = key_vault_identifier
        self.config = config
        self.platform = platform
        
        # Extract credential configuration
        key_vault_dict = config['key_vault_dictS'][key_vault_identifier]
        self.key_vault_name = key_vault_dict.get('key_vault_name')
        self.secret_name = key_vault_dict.get('secret_name')
        
        # Additional configurations
        self.azure_ml_app_id = config.get('azure_ml_appID')
        self.local_access_config = config.get('KV_access_local')
        self.synapse_credentials = config.get('synapse_cred_dict')
        self.pi_server_config = config.get('pi_server')
        
        # Retrieve password/secret
        self.password = self._fetch_key_vault_secret()
    
    def _fetch_key_vault_secret(self) -> str:
        """Fetch secret from Azure Key Vault with fallback to local access."""
        try:
            # Try Azure Key Vault access first
            return fetch_azure_key_vault_secret(
                self.key_vault_name, 
                self.secret_name, 
                self.platform, 
                self.local_access_config
            )
        except Exception as e:
            error_msg = f"Error fetching secret from Key Vault: {str(e)}"
            raise Exception(error_msg)
    
    def get_synapse_connection(self) -> Tuple[str, Dict[str, str], str]:
        """
        Get Synapse connection parameters.
        
        Returns:
            Tuple[str, Dict[str, str], str]: URL, properties dictionary, connection string
        """
        if not self.synapse_credentials:
            raise ValueError("Synapse credentials not configured")
        
        server = self.synapse_credentials.get('server')
        database = self.synapse_credentials.get('database')
        
        if not server or not database:
            raise ValueError("Synapse server and database must be configured")
        
        url = f"jdbc:sqlserver://{server}:1433;database={database}"
        properties = {
            "user": self.synapse_credentials.get('username', ''),
            "password": self.password,
            "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
        }
        
        connection_string = (
            f"Driver={{ODBC Driver 17 for SQL Server}};"
            f"Server=tcp:{server},1433;"
            f"Database={database};"
            f"Uid={properties['user']};"
            f"Pwd={self.password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        
        return url, properties, connection_string
    
    def get_blob_connection(self, blob_name: str, container_name: str) -> Tuple[str, str, str]:
        """
        Get blob storage connection parameters.
        
        Parameters:
            blob_name (str): Name of the blob
            container_name (str): Name of the container
            
        Returns:
            Tuple[str, str, str]: Blob host, blob path, connection string
        """
        storage_account = self.key_vault_identifier
        
        blob_host = f"fs.azure.account.key.{storage_account}.blob.core.windows.net"
        blob_path = f"abfss://{container_name}@{storage_account}.dfs.core.windows.net/{blob_name}"
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={storage_account};"
            f"AccountKey={self.password};"
            f"EndpointSuffix=core.windows.net"
        )
        
        return blob_host, blob_path, connection_string


class DatabaseOperator:
    """
    Handles database operations for Azure Synapse and Delta Tables.
    
    This class provides methods for executing queries on Azure Synapse and
    Delta Table databases with support for different platforms.
    
    Example:
        >>> db_ops = DatabaseOperator()
        >>> df = db_ops.execute_synapse_query("SELECT * FROM my_table", "azure_synapse")
        >>> exists = db_ops.check_delta_table_exists("my_database.my_table")
    """
    
    def __init__(self):
        """Initialize the database operator."""
        self.config_manager = ConfigurationManager()
        self.config_manager.load_configuration()
    
    def execute_synapse_query(self, query: str, key_vault_identifier: str = 'azure_synapse',
                             platform: str = 'databricks', 
                             custom_config: Optional[Union[str, Dict]] = None,
                             verbose: bool = True) -> Union[Any, pd.DataFrame]:
        """
        Execute a SQL query on Azure Synapse.
        
        This method executes SQL queries on Azure Synapse with automatic platform
        detection and appropriate return types (Spark DataFrame or Pandas DataFrame).
        
        Parameters:
            query (str): SQL query string to execute
            key_vault_identifier (str): Key vault configuration identifier
            platform (str): Target platform ('databricks', 'local', 'vm_docker')
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            verbose (bool): Whether to print detailed information
            
        Returns:
            Union[Any, pd.DataFrame]: Spark DataFrame (databricks) or Pandas DataFrame (local)
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If query execution fails
            
        Example:
            >>> db_ops = DatabaseOperator()
            >>> 
            >>> # Execute on Databricks (returns Spark DataFrame)
            >>> spark_df = db_ops.execute_synapse_query(
            ...     "SELECT * FROM sales WHERE date >= '2023-01-01'",
            ...     platform='databricks'
            ... )
            >>> 
            >>> # Execute locally (returns Pandas DataFrame)  
            >>> pandas_df = db_ops.execute_synapse_query(
            ...     "SELECT COUNT(*) as total FROM customers",
            ...     platform='local'
            ... )
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Load custom configuration if provided
        if custom_config:
            self.config_manager.load_configuration(custom_config)
        
        if platform == 'databricks':
            return self._execute_synapse_query_databricks(
                query, key_vault_identifier, verbose
            )
        elif platform in ['local', 'vm_docker']:
            return self._execute_synapse_query_local(
                query, key_vault_identifier, verbose
            )
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def _execute_synapse_query_databricks(self, query: str, key_vault_identifier: str,
                                         verbose: bool) -> Any:
        """Execute Synapse query on Databricks platform."""
        try:
            # Get credentials
            cred_manager = self.config_manager.get_credential_strings(
                key_vault_identifier, platform='databricks'
            )
            url, properties, _ = cred_manager.get_synapse_connection()
            
            # Format query for JDBC
            formatted_query = self._format_jdbc_query(query)
            
            if verbose:
                print(f"Executing Synapse query on Databricks:\n{formatted_query}")
            
            # Execute query
            spark, _ = self.config_manager.get_spark_session()
            result_df = spark.read.jdbc(table=formatted_query, url=url, properties=properties)
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Error executing Synapse query on Databricks: {str(e)}")
    
    def _execute_synapse_query_local(self, query: str, key_vault_identifier: str,
                                    verbose: bool) -> pd.DataFrame:
        """Execute Synapse query on local platform."""
        try:
            # Get credentials
            cred_manager = self.config_manager.get_credential_strings(
                key_vault_identifier, platform='local'
            )
            _, _, connection_string = cred_manager.get_synapse_connection()
            
            # Clean query for direct execution
            cleaned_query = clean_sql_query(query)
            
            if verbose:
                print(f"Executing Synapse query locally:\n{cleaned_query}")
            
            # Execute query with pyodbc
            try:
                import pyodbc
            except ImportError:
                raise ImportError("pyodbc is required for local Synapse queries. Install with: pip install pyodbc")
            
            with pyodbc.connect(connection_string) as connection:
                result_df = pd.read_sql(cleaned_query, connection)
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Error executing Synapse query locally: {str(e)}")
    
    def execute_delta_table_query(self, query: str, key_vault_identifier: str = 'deltaTable',
                                 custom_config: Optional[Union[str, Dict]] = None,
                                 verbose: bool = True) -> Any:
        """
        Execute a SQL query on Delta Tables.
        
        Parameters:
            query (str): SQL query string to execute
            key_vault_identifier (str): Key vault configuration identifier
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            verbose (bool): Whether to print detailed information
            
        Returns:
            Any: Spark DataFrame with query results
            
        Raises:
            ValueError: If query is empty
            Exception: If query execution fails
            
        Example:
            >>> db_ops = DatabaseOperator()
            >>> df = db_ops.execute_delta_table_query(
            ...     "SELECT * FROM delta_table WHERE status = 'active'"
            ... )
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            if verbose:
                print(f"Executing Delta Table query:\n{query}")
            
            # Get Spark session and execute query
            spark, _ = self.config_manager.get_spark_session()
            result_df = spark.sql(query)
            
            return result_df
            
        except Exception as e:
            raise Exception(f"Error executing Delta Table query: {str(e)}")
    
    def check_delta_table_exists(self, table_name: str, 
                                custom_config: Optional[Union[str, Dict]] = None) -> bool:
        """
        Check if a Delta Table exists.
        
        Parameters:
            table_name (str): Name of the Delta Table to check
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            
        Returns:
            bool: True if table exists, False otherwise
            
        Example:
            >>> db_ops = DatabaseOperator()
            >>> exists = db_ops.check_delta_table_exists('analytics.customer_data')
            >>> if exists:
            ...     print("Table is available for queries")
        """
        try:
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            # Get Spark session
            spark, _ = self.config_manager.get_spark_session()
            
            # Check if table exists using Spark catalog
            try:
                spark.catalog.tableExists(table_name)
                print(f"Delta Table '{table_name}' exists!")
                return True
            except Exception:
                return False
                
        except Exception as e:
            warnings.warn(f"Error checking Delta Table existence: {str(e)}")
            return False
    
    def _format_jdbc_query(self, query: str) -> str:
        """Format query for JDBC execution."""
        query = query.strip()
        if not (query.startswith('(') and query.endswith(')')):
            query = f'({query}) query'
        elif not query.endswith(' query'):
            query = f'{query} query'
        return query


class BlobStorageManager:
    """
    Manages Azure Blob Storage operations including read/write for various data formats.
    
    This class provides comprehensive blob storage operations supporting CSV, Parquet,
    and Excel files with both Spark DataFrame and Pandas DataFrame interfaces.
    
    Example:
        >>> blob_mgr = BlobStorageManager()
        >>> blob_dict = {'storage_account': 'myaccount', 'container': 'data', 'blob': 'file.csv'}
        >>> df = blob_mgr.read_blob_to_pandas(blob_dict)
        >>> blob_mgr.write_dataframe_to_blob(df, blob_dict)
    """
    
    def __init__(self):
        """Initialize the blob storage manager."""
        self.config_manager = ConfigurationManager()
        self.config_manager.load_configuration()
    
    def read_blob_to_spark(self, blob_specification: Dict[str, str],
                          custom_config: Optional[Union[str, Dict]] = None,
                          platform: str = 'databricks') -> Any:
        """
        Read blob file as Spark DataFrame.
        
        This method reads CSV or Parquet files from Azure Blob Storage and returns
        a Spark DataFrame with automatic format detection and appropriate options.
        
        Parameters:
            blob_specification (Dict[str, str]): Blob location specification containing:
                                               - 'storage_account': Storage account name
                                               - 'container': Container name
                                               - 'blob': Blob name with extension
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform (default: 'databricks')
            
        Returns:
            Any: Spark DataFrame containing the blob data
            
        Raises:
            ValueError: If blob specification is invalid
            Exception: If blob reading fails
            
        Example:
            >>> blob_mgr = BlobStorageManager()
            >>> blob_dict = {
            ...     'storage_account': 'mystorageaccount',
            ...     'container': 'datacontainer',
            ...     'blob': 'sales_data.parquet'
            ... }
            >>> spark_df = blob_mgr.read_blob_to_spark(blob_dict)
            >>> spark_df.show(10)
        """
        # Validate blob specification
        required_keys = ['storage_account', 'container', 'blob']
        missing_keys = [key for key in required_keys if key not in blob_specification]
        if missing_keys:
            raise ValueError(f"Missing required blob specification keys: {missing_keys}")
        
        try:
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            storage_account = blob_specification['storage_account']
            container = blob_specification['container']
            blob = blob_specification['blob']
            
            # Get credentials and connection info
            cred_manager = self.config_manager.get_credential_strings(
                storage_account, platform=platform
            )
            blob_host, blob_path, _ = cred_manager.get_blob_connection(blob, container)
            
            # Configure Spark for blob access
            spark, _ = self.config_manager.get_spark_session()
            spark.conf.set(blob_host, cred_manager.password)
            
            # Determine file format and read appropriately
            file_extension = blob.split('.')[-1].lower()
            
            if file_extension == 'csv':
                spark_df = (spark.read.format('csv')
                           .option('header', 'true')
                           .option('inferSchema', 'true')
                           .load(blob_path))
            elif file_extension == 'parquet':
                spark_df = spark.read.format("parquet").load(blob_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return spark_df
            
        except Exception as e:
            error_msg = f"Error reading blob to Spark DataFrame: {str(e)}"
            raise Exception(error_msg)
    
    def write_spark_to_blob(self, dataframe: Any, blob_specification: Dict[str, str],
                           write_mode: str = "append", 
                           custom_config: Optional[Union[str, Dict]] = None,
                           platform: str = 'databricks') -> None:
        """
        Write Spark DataFrame to Azure Blob Storage.
        
        Parameters:
            dataframe (Any): Spark DataFrame to write
            blob_specification (Dict[str, str]): Blob location specification
            write_mode (str): Write mode ('append', 'overwrite', 'error', 'ignore')
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If write operation fails
            
        Example:
            >>> blob_mgr = BlobStorageManager()
            >>> blob_dict = {'storage_account': 'myaccount', 'container': 'output', 'blob': 'results.parquet'}
            >>> blob_mgr.write_spark_to_blob(spark_df, blob_dict, write_mode='overwrite')
        """
        # Validate inputs
        if dataframe is None:
            raise ValueError("DataFrame cannot be None")
        
        required_keys = ['storage_account', 'container', 'blob']
        missing_keys = [key for key in required_keys if key not in blob_specification]
        if missing_keys:
            raise ValueError(f"Missing required blob specification keys: {missing_keys}")
        
        valid_modes = ['append', 'overwrite', 'error', 'ignore']
        if write_mode not in valid_modes:
            raise ValueError(f"write_mode must be one of {valid_modes}")
        
        try:
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            storage_account = blob_specification['storage_account']
            container = blob_specification['container']
            blob = blob_specification['blob']
            
            # Get credentials and connection info
            cred_manager = self.config_manager.get_credential_strings(
                storage_account, platform=platform
            )
            blob_host, blob_path, _ = cred_manager.get_blob_connection(blob, container)
            
            # Configure Spark for blob access
            spark, _ = self.config_manager.get_spark_session()
            spark.conf.set(blob_host, cred_manager.password)
            
            # Determine file format
            file_extension = blob.split('.')[-1].lower()
            
            # Write DataFrame
            print(f"Writing Spark DataFrame to blob: {blob_path}")
            (dataframe.write
             .format(file_extension)
             .mode(write_mode)
             .save(blob_path))
            
        except Exception as e:
            error_msg = f"Error writing Spark DataFrame to blob: {str(e)}"
            raise Exception(error_msg)
    
    def read_blob_to_pandas(self, blob_specification: Dict[str, str],
                           verbose: bool = True,
                           custom_config: Optional[Union[str, Dict]] = None,
                           platform: str = 'databricks',
                           load_to_memory: bool = False,
                           **kwargs) -> pd.DataFrame:
        """
        Read blob file as Pandas DataFrame.
        
        This method reads CSV, Parquet, or Excel files from Azure Blob Storage
        and returns a Pandas DataFrame with support for various read options.
        
        Parameters:
            blob_specification (Dict[str, str]): Blob location specification
            verbose (bool): Whether to print detailed information
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform
            load_to_memory (bool): Whether to load directly to memory vs temp files
            **kwargs: Additional arguments passed to pandas read functions
            
        Returns:
            pd.DataFrame: Pandas DataFrame containing the blob data
            
        Raises:
            ValueError: If blob specification is invalid
            Exception: If blob reading fails
            
        Example:
            >>> blob_mgr = BlobStorageManager()
            >>> blob_dict = {
            ...     'storage_account': 'myaccount',
            ...     'container': 'data', 
            ...     'blob': 'dataset.csv'
            ... }
            >>> df = blob_mgr.read_blob_to_pandas(
            ...     blob_dict, 
            ...     sep=',', 
            ...     parse_dates=['date_column']
            ... )
        """
        # Validate blob specification
        required_keys = ['storage_account', 'container', 'blob']
        missing_keys = [key for key in required_keys if key not in blob_specification]
        if missing_keys:
            raise ValueError(f"Missing required blob specification keys: {missing_keys}")
        
        try:
            import inspect
            from azure.storage.blob import BlobServiceClient
            
            # Separate kwargs for different pandas functions
            csv_args = list(inspect.signature(pd.read_csv).parameters)
            csv_kwargs = {k: v for k, v in kwargs.items() if k in csv_args}
            
            parquet_args = list(inspect.signature(pd.read_parquet).parameters)
            parquet_kwargs = {k: v for k, v in kwargs.items() if k in parquet_args}
            
            excel_args = list(inspect.signature(pd.read_excel).parameters)
            excel_kwargs = {k: v for k, v in kwargs.items() if k in excel_args}
            
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            storage_account = blob_specification['storage_account']
            container = blob_specification['container']
            blob = blob_specification['blob']
            
            # Get credentials and connection info
            cred_manager = self.config_manager.get_credential_strings(
                storage_account, platform=platform
            )
            _, _, connection_string = cred_manager.get_blob_connection(blob, container)
            
            # Create blob client
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob)
            
            file_extension = blob.split('.')[-1].lower()
            
            if verbose:
                print(f"Reading from storage_account: '{storage_account}', "
                      f"container: '{container}', blob: '{blob}'")
            
            if load_to_memory:
                # Load directly to memory
                with io.BytesIO() as blob_buffer:
                    blob_client.download_blob().readinto(blob_buffer)
                    blob_buffer.seek(0)
                    
                    if file_extension == 'csv':
                        dataframe = pd.read_csv(blob_buffer, **csv_kwargs)
                    elif file_extension == 'parquet':
                        dataframe = pd.read_parquet(blob_buffer, **parquet_kwargs)
                    elif file_extension in ['xlsx', 'xls']:
                        dataframe = pd.read_excel(blob_buffer, **excel_kwargs)
                    else:
                        raise ValueError(f"Unsupported file format: {file_extension}")
            else:
                # Use temporary file
                import tempfile
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                    try:
                        # Download blob to temporary file
                        with open(temp_file.name, 'wb') as download_file:
                            blob_client.download_blob().readinto(download_file)
                        
                        # Read from temporary file
                        if file_extension == 'csv':
                            dataframe = pd.read_csv(temp_file.name, **csv_kwargs)
                        elif file_extension == 'parquet':
                            dataframe = pd.read_parquet(temp_file.name, **parquet_kwargs)
                        elif file_extension in ['xlsx', 'xls']:
                            dataframe = pd.read_excel(temp_file.name, **excel_kwargs)
                        else:
                            raise ValueError(f"Unsupported file format: {file_extension}")
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
            
            return dataframe
            
        except Exception as e:
            error_msg = f"Error reading blob to Pandas DataFrame: {str(e)}"
            raise Exception(error_msg)
    
    def write_pandas_to_blob(self, dataframe: pd.DataFrame, 
                            blob_specification: Dict[str, str],
                            overwrite: bool = True, append: bool = False,
                            custom_config: Optional[Union[str, Dict]] = None,
                            platform: str = 'databricks',
                            **kwargs) -> None:
        """
        Write Pandas DataFrame to Azure Blob Storage.
        
        Parameters:
            dataframe (pd.DataFrame): Pandas DataFrame to write
            blob_specification (Dict[str, str]): Blob location specification
            overwrite (bool): Whether to overwrite existing blob
            append (bool): Whether to append to existing blob (CSV only)
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform
            **kwargs: Additional arguments passed to pandas write functions
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If write operation fails
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        required_keys = ['storage_account', 'container', 'blob']
        missing_keys = [key for key in required_keys if key not in blob_specification]
        if missing_keys:
            raise ValueError(f"Missing required blob specification keys: {missing_keys}")
        
        try:
            from azure.storage.blob import BlobServiceClient
            import tempfile
            
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            storage_account = blob_specification['storage_account']
            container = blob_specification['container']
            blob = blob_specification['blob']
            
            # Get credentials and connection info
            cred_manager = self.config_manager.get_credential_strings(
                storage_account, platform=platform
            )
            _, _, connection_string = cred_manager.get_blob_connection(blob, container)
            
            # Create blob client
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob)
            
            file_extension = blob.split('.')[-1].lower()
            
            # Handle append mode for CSV
            if append and file_extension == 'csv':
                try:
                    existing_df = self.read_blob_to_pandas(blob_specification, verbose=False)
                    dataframe = pd.concat([existing_df, dataframe], ignore_index=True)
                except:
                    pass  # If blob doesn't exist, just write the new data
            
            # Write to temporary file then upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_file:
                try:
                    if file_extension == 'csv':
                        dataframe.to_csv(temp_file.name, index=False, **kwargs)
                    elif file_extension == 'parquet':
                        dataframe.to_parquet(temp_file.name, index=False, **kwargs)
                    elif file_extension in ['xlsx', 'xls']:
                        dataframe.to_excel(temp_file.name, index=False, **kwargs)
                    else:
                        raise ValueError(f"Unsupported file format: {file_extension}")
                    
                    # Upload file to blob
                    with open(temp_file.name, 'rb') as upload_file:
                        blob_client.upload_blob(upload_file, overwrite=overwrite)
                        
                    print(f"Successfully wrote DataFrame to blob: {blob}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            error_msg = f"Error writing Pandas DataFrame to blob: {str(e)}"
            raise Exception(error_msg)
    
    def check_blob_exists(self, blob_specification: Dict[str, str],
                         custom_config: Optional[Union[str, Dict]] = None,
                         platform: str = 'databricks') -> bool:
        """
        Check if a blob exists in Azure Storage.
        
        Parameters:
            blob_specification (Dict[str, str]): Blob location specification
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            platform (str): Target platform
            
        Returns:
            bool: True if blob exists, False otherwise
        """
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Load custom configuration if provided
            if custom_config:
                self.config_manager.load_configuration(custom_config)
            
            storage_account = blob_specification['storage_account']
            container = blob_specification['container']
            blob = blob_specification['blob']
            
            # Get credentials and connection info
            cred_manager = self.config_manager.get_credential_strings(
                storage_account, platform=platform
            )
            _, _, connection_string = cred_manager.get_blob_connection(blob, container)
            
            # Create blob client and check existence
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container=container, blob=blob)
            
            return blob_client.exists()
            
        except Exception:
            return False


class DataProcessor:
    """
    Handles data processing operations including PI server data and various transformations.
    
    This class provides utilities for processing time-series data from PI servers
    and other data transformation operations.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.config_manager = ConfigurationManager()
        self.config_manager.load_configuration()
    
    def process_pi_server_interpolated_data(self, tag_names: List[str],
                                          start_time: str, end_time: str,
                                          interval: str = '1h',
                                          custom_config: Optional[Union[str, Dict]] = None) -> pd.DataFrame:
        """
        Retrieve interpolated data from PI server.
        
        Parameters:
            tag_names (List[str]): List of PI tag names to retrieve
            start_time (str): Start time in ISO format
            end_time (str): End time in ISO format
            interval (str): Data interval (e.g., '1h', '15m', '1d')
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            
        Returns:
            pd.DataFrame: DataFrame with interpolated PI data
            
        Example:
            >>> processor = DataProcessor()
            >>> tags = ['TAG001', 'TAG002', 'TAG003']
            >>> df = processor.process_pi_server_interpolated_data(
            ...     tags, '2023-01-01T00:00:00', '2023-01-02T00:00:00', '1h'
            ... )
        """
        # Implementation would depend on PI server client library
        # This is a placeholder for the actual PI server integration
        raise NotImplementedError("PI server integration requires specific PI client libraries")
    
    def process_pi_server_raw_data(self, tag_names: List[str],
                                 start_time: str, end_time: str,
                                 custom_config: Optional[Union[str, Dict]] = None) -> pd.DataFrame:
        """
        Retrieve raw data from PI server.
        
        Parameters:
            tag_names (List[str]): List of PI tag names to retrieve
            start_time (str): Start time in ISO format  
            end_time (str): End time in ISO format
            custom_config (Optional[Union[str, Dict]]): Custom configuration override
            
        Returns:
            pd.DataFrame: DataFrame with raw PI data
        """
        # Implementation would depend on PI server client library
        # This is a placeholder for the actual PI server integration
        raise NotImplementedError("PI server integration requires specific PI client libraries")


# ============================================================================
# UTILITY FUNCTIONS (Simple functions for common operations)
# ============================================================================

def clean_sql_query(query: str, start_time: Optional[str] = None, 
                   end_time: Optional[str] = None) -> str:
    """
    Clean and format SQL query string.
    
    This function removes extra whitespace, handles query formatting,
    and optionally applies time-based filtering.
    
    Parameters:
        query (str): SQL query string to clean
        start_time (Optional[str]): Start time for time-based filtering
        end_time (Optional[str]): End time for time-based filtering
        
    Returns:
        str: Cleaned and formatted SQL query
        
    Example:
        >>> clean_query = clean_sql_query("SELECT * FROM table WHERE id > 100")
        >>> time_filtered = clean_sql_query(
        ...     "SELECT * FROM events", 
        ...     start_time='2023-01-01', 
        ...     end_time='2023-12-31'
        ... )
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Basic cleaning
    cleaned = re.sub(r'\s+', ' ', query.strip())
    
    # Remove query wrapper if present
    if cleaned.lower().endswith(' query') and cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = cleaned[1:-6].strip()  # Remove '(' and ') query'
    
    # Apply time filtering if provided
    if start_time and end_time:
        if 'WHERE' in cleaned.upper():
            cleaned += f" AND timestamp BETWEEN '{start_time}' AND '{end_time}'"
        else:
            cleaned += f" WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'"
    
    return cleaned


def fetch_azure_key_vault_secret(key_vault_name: str, secret_name: str,
                                platform: str = 'databricks',
                                local_access_config: Optional[Dict] = None) -> str:
    """
    Fetch secret from Azure Key Vault with platform-specific authentication.
    
    Parameters:
        key_vault_name (str): Name of the Azure Key Vault
        secret_name (str): Name of the secret to retrieve
        platform (str): Target platform ('databricks', 'local', 'vm_docker')
        local_access_config (Optional[Dict]): Local access configuration
        
    Returns:
        str: Secret value from Key Vault
        
    Raises:
        ImportError: If required Azure libraries are not available
        Exception: If secret retrieval fails
    """
    try:
        if platform == 'databricks':
            # Use Databricks secrets
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
            return dbutils.secrets.get(scope=key_vault_name, key=secret_name)
            
        else:
            # Use Azure SDK for local/VM access
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise ImportError("Azure SDK libraries required. Install with: pip install azure-keyvault-secrets azure-identity")
            
            # Try environment variables first
            try:
                credential = DefaultAzureCredential()
                vault_url = f"https://{key_vault_name}.vault.azure.net/"
                client = SecretClient(vault_url=vault_url, credential=credential)
                secret = client.get_secret(secret_name)
                return secret.value
                
            except Exception as e:
                print(f"Environment authentication failed: {str(e)}")
                
                # Fallback to local configuration
                if not local_access_config:
                    raise Exception(
                        "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET "
                        "environment variables or provide KV_access_local in config"
                    )
                
                from azure.identity import ClientSecretCredential
                
                credential = ClientSecretCredential(
                    tenant_id=local_access_config['tenant_id'],
                    client_id=local_access_config['client_id'],
                    client_secret=local_access_config['client_secret']
                )
                
                vault_url = f"https://{key_vault_name}.vault.azure.net/"
                client = SecretClient(vault_url=vault_url, credential=credential)
                secret = client.get_secret(secret_name)
                return secret.value
                
    except Exception as e:
        raise Exception(f"Error fetching secret from Key Vault: {str(e)}")


def get_azure_secret_uri(key_vault_name: str, secret_name: str, credential: Any) -> str:
    """
    Get secret from Azure Key Vault using URI approach.
    
    Parameters:
        key_vault_name (str): Key vault name
        secret_name (str): Secret name
        credential (Any): Azure credential object
        
    Returns:
        str: Secret value
    """
    try:
        from azure.keyvault.secrets import SecretClient
        
        vault_url = f"https://{key_vault_name}.vault.azure.net/"
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret(secret_name)
        return secret.value
        
    except Exception as e:
        raise Exception(f"Error getting secret from Key Vault URI: {str(e)}")


def execute_query_template(template_name: str, parameters: Dict[str, Any],
                          custom_config: Optional[Union[str, Dict]] = None) -> str:
    """
    Execute SQL query template with parameter substitution.
    
    Parameters:
        template_name (str): Name of the query template
        parameters (Dict[str, Any]): Parameters for template substitution
        custom_config (Optional[Union[str, Dict]]): Custom configuration
        
    Returns:
        str: Executed query string with parameters substituted
        
    Example:
        >>> params = {'table_name': 'customers', 'min_age': 18}
        >>> query = execute_query_template('customer_query', params)
    """
    try:
        # Load configuration
        if custom_config:
            config_mgr = ConfigurationManager()
            config = config_mgr.load_configuration(custom_config)
        else:
            # Use default configuration
            with res.open_binary('dsToolbox', 'config.yml') as fp:
                config = yaml.load(fp, Loader=yaml.Loader)
        
        # Get template
        templates = config.get('sql_templates', {})
        if template_name not in templates:
            raise KeyError(f"Query template '{template_name}' not found")
        
        template = templates[template_name]
        
        # Substitute parameters
        executed_query = template.format(**parameters)
        
        return executed_query
        
    except Exception as e:
        raise Exception(f"Error executing query template: {str(e)}")


def copy_dbfs_to_blob(source_path: str, blob_specification: Dict[str, str],
                     custom_config: Optional[Union[str, Dict]] = None) -> None:
    """
    Copy file from DBFS to Azure Blob Storage.
    
    Parameters:
        source_path (str): Source path in DBFS
        blob_specification (Dict[str, str]): Target blob specification
        custom_config (Optional[Union[str, Dict]]): Custom configuration
    """
    try:
        # Get Databricks utilities
        config_mgr = ConfigurationManager()
        dbutils = config_mgr.get_databricks_utilities()
        
        # Copy file
        dbutils.fs.cp(source_path, f"abfss://{blob_specification['container']}@{blob_specification['storage_account']}.dfs.core.windows.net/{blob_specification['blob']}")
        
        print(f"Successfully copied {source_path} to blob storage")
        
    except Exception as e:
        raise Exception(f"Error copying DBFS to blob: {str(e)}")


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def get_spark():
    """Legacy function for backward compatibility."""
    config_mgr = ConfigurationManager()
    return config_mgr.get_spark_session()


def get_dbutils():
    """Legacy function for backward compatibility."""
    config_mgr = ConfigurationManager()
    return config_mgr.get_databricks_utilities()


def load_config(custom_config=None):
    """Legacy function for backward compatibility."""
    config_mgr = ConfigurationManager()
    config = config_mgr.load_configuration(custom_config)
    
    key_vault_dictS = config.get('key_vault_dictS')
    KV_access_local = config.get('KV_access_local')
    synapse_cred_dict = config.get('synapse_cred_dict')
    azure_ml_appID = config.get('azure_ml_appID')
    pi_server_dict = config.get('pi_server')
    
    return config, key_vault_dictS, KV_access_local, synapse_cred_dict, azure_ml_appID, pi_server_dict


class cred_strings:
    """Legacy class for backward compatibility."""
    def __init__(self, key_vault_dict, custom_config=None, platform='databricks'):
        config_mgr = ConfigurationManager()
        if custom_config:
            config_mgr.load_configuration(custom_config)
        self.cred_manager = config_mgr.get_credential_strings(key_vault_dict, custom_config, platform)
        
        # Expose legacy attributes
        self.key_vault_name = self.cred_manager.key_vault_name
        self.secret_name = self.cred_manager.secret_name
        self.password = self.cred_manager.password
        self.platform = platform
    
    def synapse_connector(self):
        return self.cred_manager.get_synapse_connection()
    
    def blob_connector(self, blob, container):
        return self.cred_manager.get_blob_connection(blob, container)


def clean_query(q, start_time=None, end_time=None):
    """Legacy function for backward compatibility."""
    return clean_sql_query(q, start_time, end_time)


def fetch_key_value(key_vault_name, secret_name, platform='databricks', KV_access_local=None):
    """Legacy function for backward compatibility."""
    return fetch_azure_key_vault_secret(key_vault_name, secret_name, platform, KV_access_local)


def get_secret_KVUri(key_vault_name, secret_name, credential):
    """Legacy function for backward compatibility."""
    return get_azure_secret_uri(key_vault_name, secret_name, credential)


def query_synapse(query, platform='databricks', key_vault_dict='azure_synapse', custom_config=None, verbose=True):
    """Legacy function for backward compatibility."""
    db_ops = DatabaseOperator()
    return db_ops.execute_synapse_query(query, key_vault_dict, platform, custom_config, verbose)


def query_synapse_db(query, key_vault_dict='azure_synapse', custom_config=None, verbose=True):
    """Legacy function for backward compatibility."""
    db_ops = DatabaseOperator()
    return db_ops.execute_synapse_query(query, key_vault_dict, 'databricks', custom_config, verbose)


def query_synapse_local(query, key_vault_dict='azure_synapse', custom_config=None, verbose=True):
    """Legacy function for backward compatibility."""
    db_ops = DatabaseOperator()
    return db_ops.execute_synapse_query(query, key_vault_dict, 'local', custom_config, verbose)


def query_deltaTable_db(query, key_vault_dict='deltaTable', custom_config=None, verbose=True):
    """Legacy function for backward compatibility."""
    db_ops = DatabaseOperator()
    return db_ops.execute_delta_table_query(query, key_vault_dict, custom_config, verbose)


def deltaTable_check(delta_tableName, key_vault_dict='deltaTable', custom_config=None):
    """Legacy function for backward compatibility."""
    db_ops = DatabaseOperator()
    return db_ops.check_delta_table_exists(delta_tableName, custom_config)


def blob2spark(blob_dict, custom_config=None, platform='databricks'):
    """Legacy function for backward compatibility."""
    blob_mgr = BlobStorageManager()
    return blob_mgr.read_blob_to_spark(blob_dict, custom_config, platform)


def spark2blob(df, blob_dict, write_mode="append", custom_config=None, platform='databricks'):
    """Legacy function for backward compatibility."""
    blob_mgr = BlobStorageManager()
    blob_mgr.write_spark_to_blob(df, blob_dict, write_mode, custom_config, platform)


def blob2pd(blob_dict, verbose=True, custom_config=None, platform='databricks', load_to_memory=False, **kwargs):
    """Legacy function for backward compatibility."""
    blob_mgr = BlobStorageManager()
    return blob_mgr.read_blob_to_pandas(blob_dict, verbose, custom_config, platform, load_to_memory, **kwargs)


def pd2blob(data, blob_dict, overwrite=True, append=False, custom_config=None, platform='databricks', **kwargs):
    """Legacy function for backward compatibility."""
    blob_mgr = BlobStorageManager()
    blob_mgr.write_pandas_to_blob(data, blob_dict, overwrite, append, custom_config, platform, **kwargs)


def blob_check(blob_dict, custom_config=None, platform='databricks'):
    """Legacy function for backward compatibility."""
    blob_mgr = BlobStorageManager()
    return blob_mgr.check_blob_exists(blob_dict, custom_config, platform)


def dbfs2blob(ufile, blob_dict, custom_config=None):
    """Legacy function for backward compatibility."""
    copy_dbfs_to_blob(ufile, blob_dict, custom_config)


def query_template_run(query_temp_name, custom_config=None, **kwargs):
    """Legacy function for backward compatibility."""
    return execute_query_template(query_temp_name, kwargs, custom_config)


# Function mapping for reference
FUNCTION_MAPPING = {
    'get_spark': 'ConfigurationManager.get_spark_session()',
    'get_dbutils': 'ConfigurationManager.get_databricks_utilities()',
    'load_config': 'ConfigurationManager.load_configuration()',
    'cred_strings': 'AzureCredentialManager',
    'clean_query': 'clean_sql_query()',
    'fetch_key_value': 'fetch_azure_key_vault_secret()',
    'get_secret_KVUri': 'get_azure_secret_uri()',
    'query_synapse': 'DatabaseOperator.execute_synapse_query()',
    'query_synapse_db': 'DatabaseOperator.execute_synapse_query() [databricks]',
    'query_synapse_local': 'DatabaseOperator.execute_synapse_query() [local]',
    'query_deltaTable_db': 'DatabaseOperator.execute_delta_table_query()',
    'deltaTable_check': 'DatabaseOperator.check_delta_table_exists()',
    'blob2spark': 'BlobStorageManager.read_blob_to_spark()',
    'spark2blob': 'BlobStorageManager.write_spark_to_blob()',
    'blob2pd': 'BlobStorageManager.read_blob_to_pandas()',
    'pd2blob': 'BlobStorageManager.write_pandas_to_blob()',
    'blob_check': 'BlobStorageManager.check_blob_exists()',
    'dbfs2blob': 'copy_dbfs_to_blob()',
    'query_template_run': 'execute_query_template()',
}


def print_function_mapping():
    """Print the mapping of old functions to new implementations."""
    print("Function Mapping - Old to New:")
    print("=" * 70)
    for old_func, new_impl in FUNCTION_MAPPING.items():
        print(f"{old_func:25} -> {new_impl}")
    print("=" * 70)


# Global configuration for backward compatibility
try:
    io_config_dict, _, _, _, _, _ = load_config(custom_config=None)
except:
    io_config_dict = {}

# Example usage
if __name__ == "__main__":
    # Print function mapping for reference
    print_function_mapping()
    
    # Example usage
    print("\nExample Usage:")
    print("# Object-oriented approach:")
    print("config_mgr = ConfigurationManager()")
    print("db_ops = DatabaseOperator()")
    print("blob_mgr = BlobStorageManager()")
    print("df = db_ops.execute_synapse_query('SELECT * FROM table', 'azure_synapse')")
    print("\n# Backward compatible approach:")
    print("df = query_synapse('SELECT * FROM table')")
    print("blob_df = blob2pd(blob_dict)")
    print("pd2blob(dataframe, blob_dict)")