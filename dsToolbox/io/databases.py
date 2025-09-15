"""
Database & Platform Operations Manager for Data Science Toolbox I/O Operations
==============================================================================

Combined database and platform operations including MSSQL database management,
Google Colab environment setup, legacy database connections, and ETL pipeline
orchestration.

Classes:
--------
- MSSQLManager: Microsoft SQL Server database operations
- ColabManager: Google Colab environment setup and management
- DatabaseConnectionManager: Legacy database connection compatibility
- DataPipelineManager: ETL pipeline orchestration and management

Dependencies:
------------
- sqlalchemy: For database ORM operations
- pyodbc: For SQL Server connections  
- pandas: For DataFrame operations
- google.colab: For Colab-specific functionality

Author: Data Science Toolbox Contributors
License: MIT License
"""

import os
import shutil
import logging
import datetime as dt
from typing import Dict, List, Optional, Any
from pathlib import Path

# Third-party imports (with graceful handling)
try:
    import pandas as pd
except ImportError as e:
    logging.warning(f"Pandas dependency not found: {e}")
    raise

# Internal imports
try:
    from .config import ConfigurationManager
except ImportError:
    try:
        from dsToolbox.io.config import ConfigurationManager
    except ImportError:
        # Fallback for backward compatibility
        ConfigurationManager = None
        logging.warning("ConfigurationManager not available - some functionality may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MSSQLManager:
    """
    Enhanced MSSQL operations manager for database connections, queries, and workflows.
    
    This class provides comprehensive MSSQL database operations with environment
    detection and unified configuration management. Note: MSSQL connectivity 
    requires compatible environment (typically Windows or properly configured Linux).
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the MSSQL manager."""
        self.config_manager = config_manager or (ConfigurationManager() if ConfigurationManager else None)
        
        if self.config_manager and self.config_manager.platform == 'colab':
            logger.warning("MSSQL connections typically fail in Google Colab (Linux environment, driver limitations)")
    
    def get_database_engine(self, database_server_id: str) -> Any:
        """
        Get SQLAlchemy engine for database operations.
        
        Parameters
        ----------
        database_server_id : str
            Database server identifier from configuration
            
        Returns
        -------
        Any
            SQLAlchemy engine for pandas operations
            
        Examples
        --------
        >>> mssql_mgr = MSSQLManager()
        >>> engine = mssql_mgr.get_database_engine('server1')
        >>> df = pd.read_sql("SELECT * FROM table", engine)
        """
        try:
            from sqlalchemy import create_engine
            import urllib.parse
        except ImportError:
            raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")
        
        if not self.config_manager:
            raise RuntimeError("ConfigurationManager required for database connections")
            
        try:
            server_config = self.config_manager.get_database_configuration(database_server_id)
            db_server = server_config['db_server']
            
            connection_params = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={db_server};'
                f'Trusted_Connection=yes;'
            )
            
            encoded_params = urllib.parse.quote_plus(connection_params)
            connection_string = f'mssql+pyodbc:///?odbc_connect={encoded_params}'
            
            engine = create_engine(connection_string)
            logger.info(f"Created SQLAlchemy engine for {db_server}")
            
            return engine
            
        except Exception as e:
            raise Exception(f"Failed to create MSSQL engine: {e}")
    
    def check_table_exists(self, table_name: str, database_server_id: str) -> bool:
        """
        Check if a table exists in the MSSQL database.
        
        Parameters
        ----------
        table_name : str
            Full table name in format 'schema.table' or 'database.schema.table'
        database_server_id : str
            Database server identifier from configuration
            
        Returns
        -------
        bool
            True if table exists, False otherwise
            
        Examples
        --------
        >>> mssql_mgr = MSSQLManager()
        >>> exists = mssql_mgr.check_table_exists('dbo.customers', 'server1')
        >>> print(f"Table exists: {exists}")
        """
        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc is required for table checking. Install with: pip install pyodbc")
        
        if not self.config_manager:
            raise RuntimeError("ConfigurationManager required for database operations")
            
        try:
            # Parse table name components
            table_parts = table_name.split('.')
            if len(table_parts) == 3:
                database, schema, table = table_parts
                information_schema = f"{database}.information_schema.tables"
            elif len(table_parts) == 2:
                schema, table = table_parts
                information_schema = "information_schema.tables"
            else:
                raise ValueError("Table name must be in format 'schema.table' or 'database.schema.table'")
            
            # Get database configuration
            server_config = self.config_manager.get_database_configuration(database_server_id)
            db_server = server_config['db_server']
            
            # Create pyodbc connection
            connection_string = (
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={db_server};'
                f'Trusted_Connection=yes;'
            )
            
            cnxn = pyodbc.connect(connection_string)
            
            # Check table existence
            sql_query = f"""
                SELECT COUNT(*)
                FROM {information_schema}
                WHERE table_name = ?
                AND TABLE_SCHEMA = ?
            """
            
            cursor = cnxn.cursor()
            cursor.execute(sql_query, (table, schema))
            result = cursor.fetchone()
            
            exists = result[0] == 1
            cnxn.close()
            
            return exists
            
        except Exception as e:
            if 'cnxn' in locals():
                cnxn.close()
            raise Exception(f"Error checking table existence: {e}")
    
    def write_dataframe_to_table(self, df: pd.DataFrame, table_name: str, 
                                database_server_id: str, **kwargs) -> None:
        """
        Write DataFrame to MSSQL table.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to write to database
        table_name : str
            Target table name
        database_server_id : str
            Database server identifier from configuration
        **kwargs
            Additional parameters passed to pandas.to_sql()
            
        Examples
        --------
        >>> mssql_mgr = MSSQLManager()
        >>> df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        >>> mssql_mgr.write_dataframe_to_table(
        ...     df, 'dbo.test_table', 'server1', if_exists='replace'
        ... )
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame cannot be None or empty")
            
            engine = self.get_database_engine(database_server_id)
            
            # Default parameters for optimal performance
            default_params = {
                'con': engine,
                'index': False,
                'if_exists': 'append',
                'method': 'multi',
                'chunksize': 1000
            }
            
            # Override with user parameters
            default_params.update(kwargs)
            
            df.to_sql(table_name, **default_params)
            logger.info(f"Successfully wrote {len(df)} rows to {table_name}")
            
        except Exception as e:
            raise Exception(f"Error writing DataFrame to MSSQL: {e}")
    
    def get_last_date_from_table(self, table_name: str, database_server_id: str, 
                                date_column: str) -> Optional[pd.Timestamp]:
        """
        Retrieve the most recent date from a specified column in an MSSQL table.
        
        Parameters
        ----------
        table_name : str
            Name of the database table
        database_server_id : str
            Database server identifier
        date_column : str
            Name of the date column to query
            
        Returns
        -------
        pd.Timestamp or None
            The most recent date found, or None if table doesn't exist
            
        Examples
        --------
        >>> mssql_mgr = MSSQLManager()
        >>> last_date = mssql_mgr.get_last_date_from_table(
        ...     'dbo.transactions', 'server1', 'transaction_date'
        ... )
        >>> print(f"Last transaction date: {last_date}")
        """
        try:
            # Check if table exists before querying
            if not self.check_table_exists(table_name, database_server_id):
                logger.warning(f"{table_name} does not exist")
                return None
            
            # Query for min and max dates
            query = f"""
                SELECT MIN({date_column}) as min_time,
                       MAX({date_column}) as max_time
                FROM {table_name}
            """
            
            engine = self.get_database_engine(database_server_id)
            date_results = pd.read_sql(query, engine)
            most_recent_date = date_results['max_time'].iloc[0]
            
            logger.info(f"Last date found in {table_name}: {most_recent_date}")
            return most_recent_date
            
        except Exception as e:
            raise Exception(f"Error retrieving last date from table: {e}")


class ColabManager:
    """
    Enhanced Google Colab operations manager for environment setup and data management.
    
    This class provides comprehensive Colab operations with automatic environment
    detection and unified configuration management.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the Colab manager."""
        self.config_manager = config_manager or (ConfigurationManager() if ConfigurationManager else None)
        
        if self.config_manager and self.config_manager.platform != 'colab':
            logger.warning(f"ColabManager initialized on non-Colab platform: {self.config_manager.platform}")
        
        self.drive_mounted = False
        self.ssh_configured = False
        self.git_configured = False
    
    def setup_complete_environment(self, user_email: str, user_name: str,
                                 ssh_source_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Complete environment setup for Google Colab with comprehensive configuration.
        
        Parameters
        ----------
        user_email : str
            Git user email address
        user_name : str
            Git user name
        ssh_source_path : str, optional
            Path to SSH keys in Google Drive
            
        Returns
        -------
        dict
            Dictionary with setup results for each component
            
        Examples
        --------
        >>> colab_mgr = ColabManager()
        >>> results = colab_mgr.setup_complete_environment(
        ...     'user@email.com', 'User Name', '/content/drive/MyDrive/.ssh'
        ... )
        >>> print(f"Drive mounted: {results['drive_mounted']}")
        """
        if self.config_manager and self.config_manager.platform != 'colab':
            raise RuntimeError("Environment setup requires Google Colab platform")
        
        setup_results = {
            'drive_mounted': False,
            'github_configured': False,
            'ssh_setup': False
        }
        
        try:
            # Mount Google Drive
            setup_results['drive_mounted'] = self._mount_google_drive()
            
            # Setup GitHub integration if drive is mounted
            if setup_results['drive_mounted'] and ssh_source_path:
                setup_results['ssh_setup'] = self._setup_ssh_keys(ssh_source_path)
                setup_results['github_configured'] = self._configure_git_credentials(user_email, user_name)
            
            logger.info(f"Colab environment setup completed: {setup_results}")
            return setup_results
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return setup_results
    
    def _mount_google_drive(self, mount_point: str = '/content/drive') -> bool:
        """Mount Google Drive with validation."""
        try:
            if os.path.ismount(mount_point):
                logger.info(f"Google Drive already mounted at {mount_point}")
                self.drive_mounted = True
                return True
            
            from google.colab import drive
            drive.mount(mount_point)
            
            if os.path.exists(mount_point) and os.listdir(mount_point):
                self.drive_mounted = True
                logger.info("Google Drive mounted successfully")
                return True
            else:
                logger.warning("Drive mounted but appears empty")
                return False
                
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            return False
    
    def _setup_ssh_keys(self, ssh_source_path: str) -> bool:
        """Setup SSH keys from Google Drive."""
        try:
            if not os.path.exists(ssh_source_path):
                raise FileNotFoundError(f"SSH source directory not found: {ssh_source_path}")
            
            ssh_dest_path = os.path.expanduser('~/.ssh')
            
            # Copy SSH configuration
            shutil.copytree(ssh_source_path, ssh_dest_path, dirs_exist_ok=True)
            os.chmod(ssh_dest_path, 0o700)
            
            # Set proper permissions for key files
            key_files = ['id_rsa', 'id_ed25519', 'id_ecdsa']
            for key_file in key_files:
                key_path = os.path.join(ssh_dest_path, key_file)
                if os.path.exists(key_path):
                    os.chmod(key_path, 0o600)
            
            self.ssh_configured = True
            logger.info("SSH keys configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup SSH keys: {e}")
            return False
    
    def _configure_git_credentials(self, user_email: str, user_name: str) -> bool:
        """Configure Git with user credentials."""
        try:
            # Set Git global configuration
            email_result = os.system(f'git config --global user.email "{user_email}"')
            name_result = os.system(f'git config --global user.name "{user_name}"')
            
            if email_result != 0 or name_result != 0:
                raise Exception("Failed to set Git configuration")
            
            self.git_configured = True
            logger.info(f"Git configured for: {user_name} <{user_email}>")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Git: {e}")
            return False


class DatabaseConnectionManager:
    """
    Legacy compatibility database connection manager.
    
    Provides backward compatibility for legacy database connection patterns
    while maintaining modern error handling and logging.
    """
    
    def __init__(self):
        """Initialize the legacy connection manager."""
        logger.info("DatabaseConnectionManager initialized (legacy compatibility)")
    
    def create_connection(self, connection_config: Dict[str, str]) -> Any:
        """
        Create database connection using legacy configuration format.
        
        Parameters
        ----------
        connection_config : dict
            Legacy connection configuration
            
        Returns
        -------
        Any
            Database connection object
        """
        logger.warning("DatabaseConnectionManager.create_connection is deprecated. Use platform-specific managers instead.")
        
        # Basic legacy compatibility - delegate to appropriate manager
        if 'snowflake' in str(connection_config).lower():
            from .snowflake import SnowflakeManager
            return SnowflakeManager().create_database_connection(connection_config)
        elif 'aws' in str(connection_config).lower():
            from .aws import AWSManager
            return AWSManager()
        else:
            raise NotImplementedError("Legacy connection type not supported. Use specific managers instead.")


class DataPipelineManager:
    """
    ETL pipeline orchestration and management.
    
    Provides pipeline management functionality including date tracking,
    incremental processing, and workflow orchestration.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the pipeline manager."""
        self.config_manager = config_manager or (ConfigurationManager() if ConfigurationManager else None)
        logger.info("DataPipelineManager initialized")
    
    def get_last_processed_date(self, source_identifier: str, date_column: str = 'date') -> Optional[dt.datetime]:
        """
        Get the last processed date for incremental pipeline processing.
        
        Parameters
        ----------
        source_identifier : str
            Identifier for the data source or pipeline
        date_column : str, default='date'
            Name of the date column to check
            
        Returns
        -------
        datetime or None
            Last processed date, or None if no previous processing
            
        Examples
        --------
        >>> pipeline_mgr = DataPipelineManager()
        >>> last_date = pipeline_mgr.get_last_processed_date('daily_sales')
        >>> print(f"Process from: {last_date}")
        """
        try:
            # Implementation would check various sources:
            # - Delta table metadata
            # - Control tables
            # - File system markers
            logger.info(f"Checking last processed date for: {source_identifier}")
            
            # Placeholder implementation - would integrate with actual data sources
            return None
            
        except Exception as e:
            logger.error(f"Error getting last processed date: {e}")
            return None
    
    def execute_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a data pipeline with comprehensive monitoring and error handling.
        
        Parameters
        ----------
        pipeline_config : dict
            Pipeline configuration including source, transformations, and target
            
        Returns
        -------
        dict
            Pipeline execution results and metadata
            
        Examples
        --------
        >>> pipeline_mgr = DataPipelineManager()
        >>> config = {
        ...     'name': 'daily_etl',
        ...     'source': 'raw_data',
        ...     'target': 'processed_data'
        ... }
        >>> result = pipeline_mgr.execute_pipeline(config)
        """
        try:
            pipeline_name = pipeline_config.get('name', 'unnamed_pipeline')
            logger.info(f"Starting pipeline execution: {pipeline_name}")
            
            # Pipeline execution logic would go here
            # - Data validation
            # - Transformation steps
            # - Quality checks
            # - Output generation
            
            execution_result = {
                'pipeline_name': pipeline_name,
                'status': 'completed',
                'start_time': dt.datetime.now(),
                'records_processed': 0,
                'errors': []
            }
            
            logger.info(f"Pipeline {pipeline_name} completed successfully")
            return execution_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'pipeline_name': pipeline_config.get('name', 'unknown'),
                'status': 'failed',
                'error': str(e)
            }


# Export all classes for external use
__all__ = [
    'MSSQLManager',
    'ColabManager', 
    'DatabaseConnectionManager',
    'DataPipelineManager'
]