"""
Configuration Management for Data Science Toolbox I/O Operations
================================================================

Universal configuration manager with platform detection and file handling for
data science workflows across different platforms (local, Colab, Databricks, etc.).

Classes:
--------
- ConfigurationManager: Universal configuration handling with platform detection

Functions:
----------
- detect_execution_platform: Automatic platform detection utility

Author: Data Science Toolbox Contributors
License: MIT License
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Third-party imports (with graceful handling)
try:
    import yaml
except ImportError as e:
    logging.warning(f"YAML dependency not found: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_execution_platform() -> str:
    """
    Detect current execution platform for environment-specific operations.
    
    Returns
    -------
    str
        Platform identifier: 'colab', 'databricks', 'vm_docker', or 'local'
        
    Examples
    --------
    >>> platform = detect_execution_platform()
    >>> print(f"Running on: {platform}")
    """
    try:
        # Check for Google Colab
        import google.colab
        return 'colab'
    except ImportError:
        pass
    
    try:
        # Check for Databricks
        import IPython
        if "dbutils" in IPython.get_ipython().user_ns:
            return 'databricks'
    except (ImportError, AttributeError):
        pass
    
    # Check for Docker environment
    if os.path.exists('/.dockerenv'):
        return 'vm_docker'
    
    return 'local'


class ConfigurationManager:
    """
    Universal configuration manager with platform detection and file handling.
    
    This class provides comprehensive configuration management with automatic
    platform detection, YAML file loading, and environment-specific adaptations
    for data science workflows across different platforms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager with optional config file.
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        """
        self.platform = detect_execution_platform()
        self.config = {}
        
        if config_path:
            self.load_configuration_file(config_path)
        
        logger.info(f"ConfigurationManager initialized for platform: {self.platform}")
    
    def load_configuration_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with comprehensive error handling.
        
        This method loads configuration from YAML files with robust error
        handling and validation. Supports nested configuration structures
        commonly used in data science projects.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
            
        Returns
        -------
        dict
            Loaded configuration dictionary
            
        Raises
        ------
        FileNotFoundError
            If configuration file does not exist
        yaml.YAMLError
            If configuration file contains invalid YAML
            
        Examples
        --------
        >>> config_mgr = ConfigurationManager()
        >>> config = config_mgr.load_configuration_file('config.yaml')
        >>> print(f"Loaded {len(config)} configuration sections")
        """
        if not config_path or not config_path.strip():
            raise ValueError("Configuration path cannot be empty")
        
        expanded_path = os.path.expanduser(config_path)
        
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Configuration file not found: {expanded_path}")
        
        try:
            with open(expanded_path, 'r', encoding='utf-8') as config_file:
                loaded_config = yaml.safe_load(config_file) or {}
                
            # Validate configuration structure
            if not isinstance(loaded_config, dict):
                raise ValueError("Configuration file must contain a dictionary structure")
            
            self.config = loaded_config
            logger.info(f"Successfully loaded configuration from {config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in configuration file {config_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def get_database_configuration(self, database_identifier: str) -> Dict[str, Any]:
        """
        Get database configuration by identifier.
        
        Parameters
        ----------
        database_identifier : str
            Database configuration identifier
            
        Returns
        -------
        dict
            Database configuration dictionary
            
        Raises
        ------
        KeyError
            If database configuration not found
        """
        if not self.config:
            raise ValueError("No configuration loaded. Use load_configuration_file() first.")
        
        # Check multiple possible configuration keys
        for config_key in ['databases', 'db_configs', 'connections']:
            if config_key in self.config and database_identifier in self.config[config_key]:
                return self.config[config_key][database_identifier]
        
        # Check direct configuration
        if database_identifier in self.config:
            return self.config[database_identifier]
        
        raise KeyError(f"Database configuration '{database_identifier}' not found")
    
    def get_spark_session(self) -> Tuple[Any, Any]:
        """
        Get or create Spark session with proper error handling.
        
        Returns
        -------
        tuple
            (spark_session, sql_context) - Spark session and SQL context
            
        Raises
        ------
        ImportError
            If PySpark is not available
        RuntimeError
            If Spark session cannot be created
        """
        try:
            import pyspark
            from pyspark.sql import SparkSession
            from pyspark import SQLContext
            
            spark = SparkSession.builder.appName("DataScienceToolbox").getOrCreate()
            sql_context = SQLContext(spark.sparkContext)
            
            logger.info("Spark session created successfully")
            return spark, sql_context
            
        except ImportError:
            raise ImportError(
                "PySpark is required for Spark operations. "
                "Install with: pip install pyspark"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Spark session: {e}")
    
    def get_databricks_utilities(self) -> Any:
        """
        Get Databricks utilities with platform validation.
        
        Returns
        -------
        Any
            Databricks utilities object
            
        Raises
        ------
        RuntimeError
            If not running in Databricks environment
        ImportError
            If Databricks utilities not available
        """
        if self.platform != 'databricks':
            raise RuntimeError(
                f"Databricks utilities require Databricks platform. "
                f"Current platform: {self.platform}"
            )
        
        try:
            import IPython
            dbutils = IPython.get_ipython().user_ns.get("dbutils")
            
            if dbutils is None:
                raise ImportError("dbutils not found in IPython namespace")
            
            return dbutils
            
        except (ImportError, KeyError, AttributeError) as e:
            raise ImportError(f"Databricks utilities not available: {e}")


# Export all classes and functions for external use
__all__ = [
    'ConfigurationManager',
    'detect_execution_platform'
]