"""
Data Science Toolbox I/O Operations - Modular Architecture
==========================================================

Comprehensive I/O operations organized into focused modules for better 
performance, maintainability, and dependency management.

This package provides:
- Platform detection and configuration management
- Snowflake database operations
- AWS services (S3, Athena)  
- Azure services (Synapse, Blob Storage)
- Microsoft SQL Server operations
- Google Colab environment management
- Kaggle dataset operations
- Legacy database connections
- ETL pipeline orchestration

Key Benefits of Modular Design:
- Lazy loading: Only import what you need
- Faster startup: Avoid loading unused dependencies
- Better dependency isolation
- Clearer code organization

Usage Examples:
--------------
# New modular imports (recommended):
from dsToolbox.io.snowflake import SnowflakeManager
from dsToolbox.io.aws import AWSManager

# Or use convenience imports:
from dsToolbox.io import SnowflakeManager, AWSManager

# Backward compatibility (still works):
from dsToolbox.io_funcs import SnowflakeManager  # Will show deprecation warning

Author: Data Science Toolbox Contributors
License: MIT License
"""

import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Core configuration (always available)
from .config import ConfigurationManager, detect_execution_platform

# Initialize the main exports list
__all__ = ['ConfigurationManager', 'detect_execution_platform']

# Conditional imports with graceful error handling
# This provides performance benefits by only loading dependencies when needed

# Snowflake Manager
try:
    from .snowflake import SnowflakeManager
    __all__.append('SnowflakeManager')
except ImportError as e:
    logger.warning(f"Snowflake functionality not available: {e}")
    SnowflakeManager = None

# AWS Manager
try:
    from .aws import AWSManager
    __all__.append('AWSManager')
except ImportError as e:
    logger.warning(f"AWS functionality not available: {e}")
    AWSManager = None

# Azure Manager
try:
    from .azure import AzureManager
    __all__.append('AzureManager')
except ImportError as e:
    logger.warning(f"Azure functionality not available: {e}")
    AzureManager = None

# Database Managers (MSSQL, Colab, Legacy, Pipeline)
try:
    from .databases import MSSQLManager, ColabManager, DatabaseConnectionManager, DataPipelineManager
    __all__.extend(['MSSQLManager', 'ColabManager', 'DatabaseConnectionManager', 'DataPipelineManager'])
except ImportError as e:
    logger.warning(f"Database functionality not available: {e}")
    MSSQLManager = ColabManager = DatabaseConnectionManager = DataPipelineManager = None

try:
    from .kaggle import KaggleManager
    __all__.append('KaggleManager')
except ImportError as e:
    logger.warning(f"Kaggle functionality not available: {e}")
    KaggleManager = None

# Legacy support functions for backward compatibility
def _show_migration_warning(old_import: str, new_import: str):
    """Show migration warning for deprecated imports."""
    warnings.warn(
        f"Importing {old_import} from dsToolbox.io_funcs is deprecated. "
        f"Use '{new_import}' instead for better performance and fewer dependencies.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy import compatibility (with warnings)
def _create_legacy_exports():
    """Create legacy export names with deprecation warnings."""
    legacy_exports = {}
    
    # Only add legacy exports for available modules
    if SnowflakeManager is not None:
        class LegacySnowflakeManager(SnowflakeManager):
            def __init__(self, *args, **kwargs):
                _show_migration_warning('SnowflakeManager', 'from dsToolbox.io.snowflake import SnowflakeManager')
                super().__init__(*args, **kwargs)
        
        legacy_exports['SnowflakeManager'] = LegacySnowflakeManager
    
    if AWSManager is not None:
        class LegacyAWSManager(AWSManager):
            def __init__(self, *args, **kwargs):
                _show_migration_warning('AWSManager', 'from dsToolbox.io.aws import AWSManager') 
                super().__init__(*args, **kwargs)
        
        legacy_exports['AWSManager'] = LegacyAWSManager
    
    return legacy_exports

# Create legacy exports for backward compatibility
_legacy_exports = _create_legacy_exports()

# Add legacy exports to module namespace for compatibility
globals().update(_legacy_exports)

# Add legacy exports to __all__ for discoverability  
__all__.extend(_legacy_exports.keys())

# Performance monitoring function
def get_import_stats():
    """
    Get statistics about which I/O modules are loaded.
    
    Returns
    -------
    dict
        Dictionary showing loaded modules and their availability
    """
    stats = {
        'core': {
            'ConfigurationManager': ConfigurationManager is not None,
            'detect_execution_platform': detect_execution_platform is not None
        },
        'cloud_services': {
            'SnowflakeManager': SnowflakeManager is not None,
            'AWSManager': AWSManager is not None
        },
        'platform_managers': {
            # Will be populated as we add more modules
        }
    }
    return stats

# Add utility function to exports
__all__.append('get_import_stats')

# Module initialization message
logger.info("DS Toolbox I/O module initialized with modular architecture")