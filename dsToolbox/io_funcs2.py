import yaml
import pyodbc
import urllib.parse
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from typing import Dict, Tuple, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Unified database connection manager for SQL Server, PostgreSQL, and Incorta.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize with configuration file.
        
        Args:
            config_file (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file: str) -> Dict:
        """Load and validate configuration file."""
        try:
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
    
    def get_mssql_connection_pyodbc(self, server_id: str) -> Tuple[pyodbc.Connection, Dict]:
        """
        Create SQL Server connection using pyodbc.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[pyodbc.Connection, Dict]: Connection object and config
        """
        try:
            mssql_config = self.config['sql_servers'][server_id]
            db_server = mssql_config['db_server']
            
            # Use modern driver
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={db_server};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"  # For SSL issues
            )
            
            # Add database if specified
            if 'database' in mssql_config:
                connection_string += f"DATABASE={mssql_config['database']};"
            
            cnxn = pyodbc.connect(connection_string)
            logger.info(f"Connected to SQL Server: {db_server}")
            
            return cnxn, mssql_config
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise
        except pyodbc.Error as e:
            logger.error(f"SQL Server connection error: {e}")
            raise
    
    def get_mssql_engine_sqlalchemy(self, server_id: str) -> Tuple[Any, str]:
        """
        Create SQL Server engine using SQLAlchemy.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Engine, str]: SQLAlchemy engine and connection parameters
        """
        try:
            mssql_config = self.config['sql_servers'][server_id]
            db_server = mssql_config['db_server']
            
            # Build connection string
            driver_params = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={db_server};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
            
            # Add database if specified
            if 'database' in mssql_config:
                driver_params += f"DATABASE={mssql_config['database']};"
            
            db_params = urllib.parse.quote_plus(driver_params)
            engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params}')
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION"))
                version = result.fetchone()[0]
                logger.info(f"Connected to SQL Server: {version[:50]}...")
            
            return engine, db_params
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            logger.error(f"SQLAlchemy SQL Server connection error: {e}")
            raise
    
    def get_postgresql_connection_psycopg2(self, server_id: str) -> Optional[psycopg2.extensions.connection]:
        """
        Create PostgreSQL connection using psycopg2.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            psycopg2.extensions.connection: Connection object or None
        """
        try:
            pg_config = self.config['postgresql_servers'][server_id]
            conn_params = {
                "host": pg_config['host'],
                "port": pg_config.get('port', 5432),
                "database": pg_config['database'],
                "user": pg_config['user'],
                "password": pg_config['password']
            }
            
            conn = psycopg2.connect(**conn_params)
            
            # Test connection
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()[0]
                logger.info(f"Connected to PostgreSQL: {db_version[:50]}...")
            
            return conn
            
        except KeyError as e:
            logger.error(f"Missing PostgreSQL configuration key: {e}")
            return None
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL connection error: {e}")
            return None
    
    def get_postgresql_engine_sqlalchemy(self, server_id: str) -> Optional[Any]:
        """
        Create PostgreSQL engine using SQLAlchemy.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Engine: SQLAlchemy engine or None
        """
        try:
            pg_config = self.config['postgresql_servers'][server_id]
            
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config.get('port', 5432)}"
                f"/{pg_config['database']}"
            )
            
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                db_version = result.fetchone()[0]
                logger.info(f"Connected to PostgreSQL: {db_version[:50]}...")
            
            return engine
            
        except KeyError as e:
            logger.error(f"Missing PostgreSQL configuration key: {e}")
            return None
        except Exception as e:
            logger.error(f"SQLAlchemy PostgreSQL connection error: {e}")
            return None
    
    def get_incorta_connection(self) -> Tuple[Optional[pyodbc.Connection], Optional[Dict]]:
        """
        Create Incorta connection.
        
        Returns:
            Tuple[Connection, Dict]: Connection object and parameters, or (None, None)
        """
        try:
            incorta_config = self.config['incorta_server']
            
            connection_string = (
                f"DRIVER={{Incorta ODBC Driver}};"  # Adjust driver name as needed
                f"HOST={incorta_config['host']};"
                f"PORT={incorta_config['port']};"
                f"DATABASE={incorta_config['database']};"
                f"UID={incorta_config['user']};"
                f"PWD={incorta_config['password']};"
            )
            
            cnxn = pyodbc.connect(connection_string)
            logger.info(f"Connected to Incorta: {incorta_config['host']}")
            
            return cnxn, incorta_config
            
        except KeyError as e:
            logger.error(f"Missing Incorta configuration key: {e}")
            return None, None
        except pyodbc.Error as e:
            logger.error(f"Incorta connection error: {e}")
            return None, None
    
    @contextmanager
    def get_connection_context(self, db_type: str, server_id: str, use_sqlalchemy: bool = False):
        """
        Context manager for database connections with automatic cleanup.
        
        Args:
            db_type (str): 'mssql', 'postgresql', or 'incorta'
            server_id (str): Server identifier from config
            use_sqlalchemy (bool): Whether to use SQLAlchemy (for mssql/postgresql)
        
        Yields:
            Connection or Engine object
        """
        connection = None
        try:
            if db_type == 'mssql':
                if use_sqlalchemy:
                    connection, _ = self.get_mssql_engine_sqlalchemy(server_id)
                else:
                    connection, _ = self.get_mssql_connection_pyodbc(server_id)
            elif db_type == 'postgresql':
                if use_sqlalchemy:
                    connection = self.get_postgresql_engine_sqlalchemy(server_id)
                else:
                    connection = self.get_postgresql_connection_psycopg2(server_id)
            elif db_type == 'incorta':
                connection, _ = self.get_incorta_connection()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            yield connection
            
        finally:
            if connection:
                try:
                    if hasattr(connection, 'close'):
                        connection.close()
                        logger.info(f"Connection to {db_type} closed")
                    elif hasattr(connection, 'dispose'):
                        connection.dispose()
                        logger.info(f"Engine for {db_type} disposed")
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example configuration structure
    example_config = {
        'sql_servers': {
            'dev': {
                'db_server': 'localhost\\SQLEXPRESS',
                'database': 'TestDB'  # Optional
            },
            'prod': {
                'db_server': 'prod-sql-server.company.com',
                'database': 'ProductionDB'
            }
        },
        'postgresql_servers': {
            'dev': {
                'host': 'localhost',
                'port': 5432,
                'database': 'testdb',
                'user': 'testuser',
                'password': 'testpass'
            }
        },
        'incorta_server': {
            'host': 'incorta.company.com',
            'port': 1433,
            'database': 'incorta_db',
            'user': 'incorta_user',
            'password': 'incorta_pass'
        }
    }
    
    # Usage examples:
    try:
        # db_manager = DatabaseConnectionManager('config.yaml')
        
        # Using context manager (recommended)
        # with db_manager.get_connection_context('mssql', 'dev') as conn:
        #     cursor = conn.cursor()
        #     cursor.execute("SELECT GETDATE()")
        #     result = cursor.fetchone()
        #     print(f"Current time: {result[0]}")
        
        # Direct connection (remember to close)
        # conn, config = db_manager.get_mssql_connection_pyodbc('dev')
        # # ... use connection ...
        # conn.close()
        
        print("Database connection manager is ready to use!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")