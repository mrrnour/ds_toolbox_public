"""
Snowflake Database Manager for Data Science Toolbox I/O Operations
=================================================================

Comprehensive Snowflake database operations including connection management,
query execution, data transfer, and table management with robust error handling
and data type conversion.

Classes:
--------
- SnowflakeManager: Complete Snowflake database operations manager

Dependencies:
------------
- snowflake-connector-python: Required for Snowflake connections
- pandas: For DataFrame operations
- ConfigurationManager: From dsToolbox.io.config

Author: Data Science Toolbox Contributors
License: MIT License
"""

import os
import re
import datetime as dt
import calendar
import logging
from typing import Dict, List, Union, Optional, Any
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

# Import utility functions
try:
    from dsToolbox.utilities import TextProcessor, SQLProcessor
except ImportError:
    # Graceful fallback if data utilities are not available
    TextProcessor = None
    SQLProcessor = None
    logging.warning("Data utilities not available - some functionality may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeManager:
    """
    Comprehensive Snowflake database operations manager.
    
    This class provides complete Snowflake database operations including
    connection management, query execution, data transfer, and table
    management with robust error handling and data type conversion.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize Snowflake manager with configuration.
        
        Parameters
        ----------
        config_manager : ConfigurationManager, optional
            Configuration manager instance
        """
        self.config_manager = config_manager or (ConfigurationManager() if ConfigurationManager else None)
        self._connection_cache = {}
        
        logger.info("SnowflakeManager initialized")
    
    def create_database_connection(self, snowflake_config: Dict[str, str]) -> Any:
        """
        Create Snowflake database connection with comprehensive configuration.
        
        This method establishes a connection to Snowflake using provided
        credentials and connection parameters. Includes connection pooling
        and error handling for production use.
        
        Parameters
        ----------
        snowflake_config : dict
            Snowflake connection configuration containing:
            - user: Snowflake username
            - password: Snowflake password  
            - account: Snowflake account identifier
            - database: Target database name
            - warehouse: Compute warehouse name
            - schema: Optional schema name
            - role: Optional role name
            
        Returns
        -------
        Any
            Snowflake connection object
            
        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed
        Exception
            If connection fails
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> config = {
        ...     'user': 'username',
        ...     'password': 'password',
        ...     'account': 'account.region',
        ...     'database': 'DATABASE',
        ...     'warehouse': 'WAREHOUSE'
        ... }
        >>> conn = sf_mgr.create_database_connection(config)
        """
        # Validate required configuration parameters
        required_params = ['user', 'password', 'account', 'database', 'warehouse']
        missing_params = [param for param in required_params if param not in snowflake_config]
        
        if missing_params:
            raise ValueError(f"Missing required Snowflake configuration parameters: {missing_params}")
        
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "Snowflake connector is required. "
                "Install with: pip install snowflake-connector-python"
            )
        
        try:
            connection_params = {
                'user': snowflake_config['user'],
                'password': snowflake_config['password'],
                'account': snowflake_config['account'],
                'database': snowflake_config['database'],
                'warehouse': snowflake_config['warehouse']
            }
            
            # Add optional parameters
            if 'schema' in snowflake_config:
                connection_params['schema'] = snowflake_config['schema']
            if 'role' in snowflake_config:
                connection_params['role'] = snowflake_config['role']
            
            connection = snowflake.connector.connect(**connection_params)
            
            logger.info(f"Successfully connected to Snowflake: {snowflake_config['account']}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise Exception(f"Snowflake connection failed: {e}")
    
    def execute_sql_query(self, sql_command: Union[str, Path], 
                         snowflake_config: Dict[str, str],
                         print_query: bool = False) -> None:
        """
        Execute SQL command or SQL file in Snowflake with transaction management.
        
        This method executes SQL commands or processes SQL files containing
        multiple statements. Includes transaction management, error handling,
        and query logging capabilities.
        
        Parameters
        ----------
        sql_command : str or Path
            SQL query string or path to SQL file containing queries
        snowflake_config : dict
            Snowflake connection configuration dictionary
        print_query : bool, default=False
            Whether to print SQL queries before execution
            
        Raises
        ------
        ValueError
            If SQL command is empty
        FileNotFoundError
            If SQL file path does not exist
        Exception
            If query execution fails
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> config = {'user': 'user', 'password': 'pass', ...}
        >>> sf_mgr.execute_sql_query("CREATE TABLE test AS SELECT 1", config)
        >>> sf_mgr.execute_sql_query("queries.sql", config, print_query=True)
        """
        if not sql_command or (isinstance(sql_command, str) and not sql_command.strip()):
            raise ValueError("SQL command cannot be empty")
        
        connection = None
        cursor = None
        
        try:
            # Create connection
            connection = self.create_database_connection(snowflake_config)
            cursor = connection.cursor()
            
            # Check if input is a file path
            if isinstance(sql_command, (str, Path)) and os.path.isfile(str(sql_command)):
                logger.info(f"Executing SQL file: {sql_command}")
                
                # Use SQL processor if available
                if SQLProcessor is not None:
                    sql_statements = SQLProcessor.parse_sql_file(str(sql_command))
                else:
                    # Fallback parsing
                    with open(sql_command, 'r', encoding='utf-8') as f:
                        sql_content = f.read()
                    sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                # Execute each statement
                for statement in sql_statements:
                    if statement.strip():
                        if print_query:
                            logger.info(f"Executing SQL statement:\n{statement}")
                        cursor.execute(statement)
                        
            else:
                # Execute single query
                logger.info("Executing SQL query in Snowflake")
                sql_query = str(sql_command)
                
                if print_query:
                    logger.info(f"Query:\n{sql_query}")
                
                cursor.execute(sql_query)
            
            # Commit transaction
            connection.commit()
            logger.info("SQL execution completed successfully")
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            if connection:
                connection.rollback()
            raise Exception(f"Snowflake SQL execution error: {e}")
        
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def upload_dataframe_to_table(self, dataframe: pd.DataFrame,
                                 snowflake_table_name: str,
                                 snowflake_schema: str,
                                 workspace_database: str,
                                 snowflake_config: Dict[str, str]) -> None:
        """
        Upload pandas DataFrame to Snowflake table with comprehensive data type handling.
        
        This method uploads a pandas DataFrame to Snowflake by creating a staging
        CSV file, uploading it to Snowflake's internal stage, and copying the data
        into the target table. Includes automatic data type conversion and schema creation.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to upload to Snowflake
        snowflake_table_name : str
            Target table name in Snowflake
        snowflake_schema : str
            Target schema name in Snowflake
        workspace_database : str
            Target database name in Snowflake
        snowflake_config : dict
            Snowflake connection configuration
            
        Raises
        ------
        ValueError
            If DataFrame is empty or parameters are invalid
        Exception
            If upload process fails
            
        Examples
        --------
        >>> df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        >>> sf_mgr = SnowflakeManager()
        >>> sf_mgr.upload_dataframe_to_table(
        ...     df, 'my_table', 'my_schema', 'my_db', config
        ... )
        """
        # Input validation
        if dataframe is None or dataframe.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        if not all([snowflake_table_name, snowflake_schema, workspace_database]):
            raise ValueError("Table name, schema, and database cannot be empty")
        
        # Clean table and column names
        if TextProcessor is not None:
            clean_table_name = TextProcessor.normalize_text(
                snowflake_table_name, remove_spaces=True, lowercase=False
            )
        else:
            clean_table_name = re.sub(r'[^A-Za-z0-9_]', '_', snowflake_table_name)
        
        connection = None
        temporary_csv_file = None
        
        try:
            # Create temporary CSV file
            temporary_csv_file = f"{clean_table_name}_upload_temp.csv"
            dataframe.to_csv(temporary_csv_file, sep=',', index=False)
            logger.info(f"Created temporary CSV file: {temporary_csv_file}")
            
            # Create connection
            connection = self.create_database_connection(snowflake_config)
            
            # Create table schema
            self._create_snowflake_table_schema(
                connection, dataframe, clean_table_name, 
                snowflake_schema, workspace_database
            )
            
            # Upload data via staging
            self._upload_data_via_staging(
                connection, temporary_csv_file, clean_table_name,
                snowflake_schema, workspace_database
            )
            
            logger.info(f"Successfully uploaded DataFrame to {workspace_database}.{snowflake_schema}.{clean_table_name}")
            
        except Exception as e:
            logger.error(f"DataFrame upload failed: {e}")
            raise Exception(f"Failed to upload DataFrame to Snowflake: {e}")
        
        finally:
            # Cleanup
            if temporary_csv_file and os.path.exists(temporary_csv_file):
                try:
                    os.remove(temporary_csv_file)
                    logger.info(f"Cleaned up temporary file: {temporary_csv_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
            
            if connection:
                connection.close()
    
    def query_to_dataframe(self, snowflake_query: str, 
                          snowflake_config: Dict[str, str]) -> pd.DataFrame:
        """
        Execute Snowflake query and return results as pandas DataFrame.
        
        This method executes a SQL query in Snowflake and returns the results
        as a pandas DataFrame with proper data type handling and error management.
        
        Parameters
        ----------
        snowflake_query : str
            SQL query to execute in Snowflake
        snowflake_config : dict
            Snowflake connection configuration
            
        Returns
        -------
        pd.DataFrame
            Query results as DataFrame
            
        Raises
        ------
        ValueError
            If query is empty
        Exception
            If query execution fails
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> query = "SELECT * FROM my_database.my_schema.my_table LIMIT 1000"
        >>> df = sf_mgr.query_to_dataframe(query, config)
        >>> print(f"Retrieved {len(df)} rows")
        """
        if not snowflake_query or not snowflake_query.strip():
            raise ValueError("Snowflake query cannot be empty")
        
        connection = None
        
        try:
            # Create connection
            connection = self.create_database_connection(snowflake_config)
            
            # Clean and validate query
            if TextProcessor is not None:
                clean_query = TextProcessor.clean_sql_query(snowflake_query.strip())
            else:
                clean_query = snowflake_query.strip()
            
            logger.info(f"Executing Snowflake query:\n{clean_query}")
            
            # Execute query and fetch results
            result_dataframe = pd.read_sql(clean_query, connection)
            
            logger.info(f"Successfully retrieved {len(result_dataframe)} rows from Snowflake")
            return result_dataframe
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise Exception(f"Snowflake query failed: {e}")
        
        finally:
            if connection:
                connection.close()
    
    def check_table_exists(self, snowflake_config: Dict[str, str], 
                          full_table_name: str) -> bool:
        """
        Check if a table exists in Snowflake with comprehensive validation.
        
        This method checks for table existence using Snowflake's information_schema,
        supporting both schema.table and database.schema.table formats.
        
        Parameters
        ----------
        snowflake_config : dict
            Snowflake connection configuration
        full_table_name : str
            Full table name in format: database.schema.table or schema.table
            
        Returns
        -------
        bool
            True if table exists, False otherwise
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> exists = sf_mgr.check_table_exists(config, 'MY_DB.MY_SCHEMA.MY_TABLE')
        >>> print(f"Table exists: {exists}")
        """
        if not full_table_name or not full_table_name.strip():
            raise ValueError("Table name cannot be empty")
        
        try:
            # Parse table name components
            table_parts = full_table_name.upper().split('.')
            
            if len(table_parts) == 3:
                workspace_database, snowflake_schema, table_name = table_parts
            elif len(table_parts) == 2:
                snowflake_schema, table_name = table_parts
                workspace_database = snowflake_config.get('database', '').upper()
            else:
                raise ValueError(f"Invalid table name format: {full_table_name}")
            
            # Set database context and check existence
            database_use_command = f"USE DATABASE {workspace_database};"
            self.execute_sql_query(database_use_command, snowflake_config)
            
            # Query information schema
            existence_query = f"""
                SELECT row_count
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                AND table_name = '{table_name}'
                AND table_schema = '{snowflake_schema}';
            """
            
            result_df = self.query_to_dataframe(existence_query, snowflake_config)
            table_exists = len(result_df) > 0
            
            logger.info(f"Table {full_table_name} {'exists' if table_exists else 'does not exist'}")
            return table_exists
            
        except Exception as e:
            logger.warning(f"Error checking table existence: {e}")
            return False
    
    def find_maximum_date_in_table(self, table_name: str,
                                  snowflake_config: Dict[str, str],
                                  database_platform: str = 'SnowFlake',
                                  date_column: str = 'AUTO') -> Optional[dt.datetime]:
        """
        Find the maximum date value in a Snowflake table with automatic column detection.
        
        This method finds the most recent date in a table by automatically detecting
        common date column names or using a specified column. Supports multiple
        date formats and database platforms.
        
        Parameters
        ----------
        table_name : str
            Full table name (database.schema.table)
        snowflake_config : dict
            Snowflake connection configuration
        database_platform : str, default='SnowFlake'
            Database platform ('SnowFlake' or 'Redshift')
        date_column : str, default='AUTO'
            Date column name or 'AUTO' for automatic detection
            
        Returns
        -------
        datetime or None
            Maximum date found in the table, None if no date column found
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> max_date = sf_mgr.find_maximum_date_in_table('DB.SCHEMA.TABLE', config)
        >>> print(f"Latest data: {max_date}")
        """
        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty")
        
        try:
            # Auto-detect date column if needed
            if date_column == 'AUTO':
                date_column = self._detect_date_column(table_name, snowflake_config, database_platform)
                
                if not date_column:
                    logger.warning(f"No date column found in table {table_name}")
                    return None
            
            # Query maximum date
            max_date_query = f"SELECT MAX({date_column}) as max_date FROM {table_name};"
            result_df = self.query_to_dataframe(max_date_query, snowflake_config)
            
            if result_df.empty or pd.isna(result_df.iloc[0, 0]):
                logger.warning(f"No date values found in column {date_column}")
                return None
            
            max_date_raw = str(result_df.iloc[0, 0])
            
            # Parse date with multiple format support
            max_date = self._parse_flexible_date(max_date_raw)
            
            logger.info(f"Maximum date in {table_name}.{date_column}: {max_date}")
            return max_date
            
        except Exception as e:
            logger.error(f"Error finding maximum date: {e}")
            raise Exception(f"Failed to find maximum date in {table_name}: {e}")
    
    def get_table_statistics(self, table_name: str,
                           snowflake_config: Dict[str, str],
                           database_platform: str = 'SnowFlake',
                           date_column: str = 'AUTO') -> pd.Series:
        """
        Get comprehensive table statistics including row count and latest date.
        
        This method retrieves key table statistics including total row count
        and the most recent date value for monitoring data freshness and completeness.
        
        Parameters
        ----------
        table_name : str
            Full table name (database.schema.table)
        snowflake_config : dict
            Snowflake connection configuration
        database_platform : str, default='SnowFlake'
            Database platform identifier
        date_column : str, default='AUTO'
            Date column name for latest date calculation
            
        Returns
        -------
        pd.Series
            Series with 'row_count' and 'last_update' statistics
            
        Examples
        --------
        >>> sf_mgr = SnowflakeManager()
        >>> stats = sf_mgr.get_table_statistics('DB.SCHEMA.TABLE', config)
        >>> print(f"Rows: {stats['row_count']}, Last update: {stats['last_update']}")
        """
        if not table_name or not table_name.strip():
            raise ValueError("Table name cannot be empty")
        
        try:
            # Get row count
            row_count_query = f"SELECT COUNT(1) as row_count FROM {table_name};"
            row_count_result = self.query_to_dataframe(row_count_query, snowflake_config)
            total_rows = int(row_count_result.iloc[0, 0])
            
            # Get latest date
            latest_date = self.find_maximum_date_in_table(
                table_name, snowflake_config, database_platform, date_column
            )
            
            # Create statistics series
            statistics = pd.Series(
                [total_rows, latest_date],
                index=['row_count', 'last_update'],
                name=table_name
            )
            
            logger.info(f"Table statistics for {table_name}: {total_rows} rows, latest: {latest_date}")
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            raise Exception(f"Failed to get statistics for {table_name}: {e}")
    
    def _create_snowflake_table_schema(self, connection: Any, dataframe: pd.DataFrame,
                                     table_name: str, schema_name: str, database_name: str) -> None:
        """Create Snowflake table schema based on DataFrame structure."""
        cursor = connection.cursor()
        
        try:
            # Set schema context
            cursor.execute(f'USE SCHEMA {database_name}.{schema_name};')
            
            logger.info(f"Creating table schema for {table_name}")
            
            # Generate column definitions
            headers = dataframe.columns.tolist()
            data_types = self._convert_pandas_to_snowflake_types(dataframe)
            
            # Clean column names
            clean_headers = []
            for header in headers:
                clean_header = re.sub(r'[^A-Za-z0-9_]', '_', str(header))
                if clean_header[0].isdigit():
                    clean_header = f"_{clean_header}"
                clean_headers.append(clean_header)
            
            # Build CREATE TABLE statement
            column_definitions = []
            for header, data_type in zip(clean_headers, data_types):
                column_definitions.append(f"{header} {data_type}")
            
            create_table_sql = f"""
                CREATE OR REPLACE TABLE {table_name} (
                    {','.join(column_definitions)}
                );
            """
            
            logger.info(f"Table creation SQL:\n{create_table_sql}")
            cursor.execute(create_table_sql)
            
        finally:
            cursor.close()
    
    def _upload_data_via_staging(self, connection: Any, csv_file: str,
                               table_name: str, schema_name: str, database_name: str) -> None:
        """Upload data to Snowflake using staging area."""
        cursor = connection.cursor()
        
        try:
            # Create staging area
            stage_name = 'dataframe_upload_stage'
            create_stage_sql = f"""
                CREATE OR REPLACE STAGE {stage_name}
                FILE_FORMAT = (
                    TYPE = 'CSV' 
                    FIELD_DELIMITER = ',' 
                    SKIP_HEADER = 1
                    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                    ESCAPE_UNENCLOSED_FIELD = NONE
                )
            """
            cursor.execute(create_stage_sql)
            
            # Upload file to stage
            put_command = f"PUT file://{csv_file} @{stage_name} AUTO_COMPRESS=TRUE"
            cursor.execute(put_command)
            
            # Copy from stage to table
            full_table_name = f"{database_name}.{schema_name}.{table_name}"
            copy_command = f"""
                COPY INTO {full_table_name}
                FROM @{stage_name}/{csv_file}
                FILE_FORMAT = (
                    TYPE = CSV
                    FIELD_DELIMITER = ','
                    SKIP_HEADER = 1
                    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                )
                ON_ERROR = 'CONTINUE'
            """
            cursor.execute(copy_command)
            
            logger.info(f"Data successfully copied to {full_table_name}")
            
        finally:
            cursor.close()
    
    def _convert_pandas_to_snowflake_types(self, dataframe: pd.DataFrame) -> List[str]:
        """Convert pandas data types to Snowflake SQL types."""
        type_mapping = {
            'object': 'VARCHAR(255)',
            'string': 'VARCHAR(255)',
            'datetime': 'DATE',
            'datetime64[ns]': 'TIMESTAMP',
            'bool': 'BOOLEAN',
            'boolean': 'BOOLEAN',
            'category': 'VARCHAR(255)',
            'timedelta': 'TIMESTAMP',
            'int8': 'SMALLINT',
            'int16': 'SMALLINT', 
            'int32': 'INTEGER',
            'int64': 'BIGINT',
            'uint8': 'SMALLINT',
            'uint16': 'INTEGER',
            'uint32': 'BIGINT',
            'uint64': 'BIGINT',
            'float16': 'FLOAT',
            'float32': 'FLOAT',
            'float64': 'DOUBLE',
            'Int8': 'SMALLINT',
            'Int16': 'SMALLINT',
            'Int32': 'INTEGER', 
            'Int64': 'BIGINT'
        }
        
        snowflake_types = []
        for dtype in dataframe.dtypes.astype(str):
            snowflake_type = type_mapping.get(dtype, 'VARCHAR(255)')
            snowflake_types.append(snowflake_type)
        
        return snowflake_types
    
    def _detect_date_column(self, table_name: str, snowflake_config: Dict[str, str], 
                           database_platform: str) -> Optional[str]:
        """Automatically detect date column in table."""
        try:
            if database_platform == 'SnowFlake':
                columns_query = f"SHOW COLUMNS IN TABLE {table_name};"
                columns_df = self.query_to_dataframe(columns_query, snowflake_config)
                column_names = '__'.join(columns_df['column_name'].str.upper())
            else:
                # Redshift format (if needed in future)
                table_parts = table_name.split('.')
                schema_name = table_parts[-2] if len(table_parts) > 1 else 'public'
                table_only = table_parts[-1]
                
                columns_query = f"""
                    SELECT column_name 
                    FROM information_schema.columns
                    WHERE table_name = '{table_only}' 
                    AND table_schema = '{schema_name}'
                """
                columns_df = self.query_to_dataframe(columns_query, snowflake_config)
                column_names = '__'.join(columns_df['column_name'].str.upper())
            
            # Priority order for date column detection
            date_column_patterns = [
                'SNAPSHOT_DATE',
                'RUNDATE_B4', 
                'RUNDATE',
                'PARTITION_DATETIME',
                'SYS_UPDATED_ON',
                'UPDATED_AT',
                'CREATED_AT'
            ]
            
            for pattern in date_column_patterns:
                if pattern in column_names:
                    return pattern
                    
            return None
            
        except Exception as e:
            logger.warning(f"Failed to detect date column: {e}")
            return None
    
    def _parse_flexible_date(self, date_string: str) -> dt.datetime:
        """Parse date string with multiple format support."""
        date_str = str(date_string).strip()
        
        # Count separators to determine format
        dash_count = date_str.count('-')
        
        try:
            if dash_count == 0:
                # Format: YYYYMMDD
                return dt.datetime.strptime(date_str, '%Y%m%d')
            elif dash_count == 1:
                # Format: YYYY-MM (add last day of month)
                year, month = date_str.split('-')
                last_day = calendar.monthrange(int(year), int(month))[1]
                return dt.datetime.strptime(f"{date_str}-{last_day}", '%Y-%m-%d')
            elif dash_count == 2:
                # Format: YYYY-MM-DD
                return dt.datetime.strptime(date_str, '%Y-%m-%d')
            else:
                # Try parsing as full datetime
                return pd.to_datetime(date_str)
                
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_string}': {e}")
            raise ValueError(f"Cannot parse date: {date_string}")


# Export all classes for external use
__all__ = [
    'SnowflakeManager'
]