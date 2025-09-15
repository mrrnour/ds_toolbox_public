# 🧰 DS Toolbox - Comprehensive Data Science Utilities

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Azure](https://img.shields.io/badge/microsoft%20azure-0089D0?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)](https://spark.apache.org/)

<div align="center">
  <img src="https://raw.githubusercontent.com/mrrnour/ds_toolbox_public/main/assets/ds_toolbox_banner.png" alt="DS Toolbox Banner" width="600"/>
</div>

---

**A comprehensive, production-ready Python toolkit for data science and machine learning workflows**

DS Toolbox provides specialized, refactored modules for end-to-end data science pipelines across multiple domains including I/O operations, NLP/LLM processing, machine learning utilities, distributed computing with Spark, and cloud integrations.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                🧰 DS TOOLBOX ARCHITECTURE 🧰               │
    ├─────────────────────────────────────────────────────────────┤
    │  📁 I/O & Cloud Integration  │  🧠 NLP & LLM Processing      │
    │  ├─ Azure (Synapse, Blob)    │  ├─ Text Preprocessing        │
    │  ├─ MSSQL & PostgreSQL       │  ├─ Similarity Analysis       │
    │  ├─ Google Colab & Kaggle    │  ├─ LangChain Integration     │
    │  └─ Multi-Platform Support   │  └─ Model Caching            │
    │                               │                               │
    │  ⚙️  Machine Learning Suite   │  🔥 Big Data & Distributed    │
    │  ├─ Model Training/Evaluation │  ├─ Apache Spark Functions   │
    │  ├─ SHAP Analysis & Segments  │  ├─ Time-Series Operations   │
    │  ├─ Feature Engineering       │  ├─ Join Operations          │
    │  └─ Performance Metrics       │  └─ ETL Workflows            │
    │                               │                               │
    │  🔍 RAG & Vector Stores       │  🛠️ Common Utilities          │
    │  ├─ Document Processing       │  ├─ Text Processing          │
    │  ├─ Web Scraping & Chunking   │  ├─ DataFrame Operations     │
    │  ├─ Embedding Management      │  ├─ Date/Time Utilities      │
    │  └─ Knowledge Retrieval       │  └─ File System Tools        │
    └─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

- **🏗️ Production-Ready Architecture**: Refactored from procedural to object-oriented design with comprehensive class organization
- **🔄 Multi-Platform I/O**: Seamless integration with Azure, MSSQL, Google Colab, Kaggle, and local environments
- **🧠 Advanced ML & NLP**: Complete SHAP analysis, customer segmentation, text similarity, and LLM integration
- **⚡ Distributed Computing**: Apache Spark utilities for big data processing and time-series operations  
- **🎯 Domain-Specific Tools**: Specialized modules for RAG systems, control charts, and statistical analysis
- **📊 Comprehensive Metrics**: Built-in evaluation tools with visualization and performance tracking
- **🔧 Developer-Friendly**: Type hints, comprehensive documentation, and backward compatibility

## 📦 Module Overview

### Core Modules (`dsToolbox/`)

| Module | Size | Classes | Description |
|--------|------|---------|-------------|
| **`ml_funcs.py`** | 3.5k lines | 8 classes | Complete ML pipeline with SHAP analysis, XGBoost rule extraction, decision tree interpretation, model training/evaluation, and performance metrics |
| **`io_funcs.py`** | 3.1k lines | 9 classes | Comprehensive I/O operations across Snowflake, Azure, AWS, MSSQL, Colab, Kaggle platforms + ETL pipeline management with automated workflows |
| **`nlp_llm_funcs.py`** | 1.1k lines | 4 classes | Advanced NLP processing, text similarity, and LLM integration with caching |
| **`utilities.py`** | 2.5k lines | 9 classes | Essential utilities for text processing, SQL parsing, advanced data encoding, DataFrame operations, and file management |
| **`advanced_analytics.py`** | 850+ lines | 5 classes | Advanced analytics for dimensionality reduction, multicollinearity detection, canonical analysis, clustering, and partition analysis |
| **`rag_funcs.py`** | 2.6k lines | Multiple | RAG system implementation with document processing and vector stores |
| **`spark_funcs.py`** | 1.5k lines | 4 classes | Advanced Spark operations with asof joins, ETL pipelines, feature engineering, and distributed time-series processing |
| **`cl_funcs.py`** | 261 lines | 4 classes | Statistical Process Control (SPC) with sigma/control limits, I-MR charts, process recovery analysis, and interactive visualization |

## 🏗️ Architecture Highlights

### Object-Oriented Design Patterns

**Machine Learning Pipeline:**
```python
# Refactored class-based architecture
from dsToolbox.ml_funcs import (
    ModelTemplateManager,      # Classifier/regressor templates
    ModelTrainer,              # Cross-validation and training
    ModelEvaluator,           # Metrics and scoring
    ModelPerformanceAnalyzer, # Visualization and analysis
    SHAPAnalyzer,             # Advanced interpretability & customer segmentation
    XGBoostRuleExtractor,     # Rule extraction from XGBoost models
    DecisionTreeInterpreter   # Decision tree code generation
)

from dsToolbox.utilities import (
    TextProcessor,            # Text normalization and cleaning
    SQLProcessor,             # SQL parsing and statement processing  
    EncodingUtilities,        # Advanced dummy encoding and sparse labels
    ProductUtilities,         # Product analysis and business logic
    DataFrameUtilities,       # DataFrame manipulation and optimization
    DataVisualization         # Sankey diagrams, 3D plots, word clouds
)

from dsToolbox.advanced_analytics import (
    DimensionalityReducer,    # t-SNE visualization and dimensionality reduction
    MulticollinearityDetector, # VIF analysis and multicollinearity detection
    CanonicalAnalyzer,        # Canonical correlation analysis with preprocessing
    ClusteringAnalyzer,       # K-means clustering with optimization metrics
    PartitionAnalyzer         # Combinatorial partition analysis and mapping
)

from dsToolbox.io_funcs import (
    SnowflakeManager,         # Snowflake database operations and data workflows
    AWSManager,               # Amazon Web Services (S3, Athena, Redshift)
    AzureManager,             # Azure operations (Synapse, Blob Storage)
    MSSQLManager,             # Microsoft SQL Server operations with enhanced table management
    ColabManager,             # Google Colab environment management
    KaggleManager,            # Kaggle dataset operations and competitions
    DataPipelineManager       # ETL workflows, Parquet operations, and automated data pipelines
)

from dsToolbox.cl_funcs import (
    SigmaLimitCalculator,     # Statistical process control using sigma limits
    ControlLimitCalculator,   # Process control limits with control chart methodology  
    ControlChartVisualizer,   # Interactive I-MR charts and SPC visualization
    ProcessRecoveryAnalyzer   # Recovery analysis for mining/manufacturing processes
)

from dsToolbox.spark_funcs import (
    SparkJoinOperations,      # Advanced asof joins and column conflict resolution
    SparkDataTransformations, # DataFrame melting, column operations, type conversions
    SparkETLPipeline,        # Incremental ETL with automatic date tracking
    SparkFeatureEngineering  # Rolling/tumbling windows for time-series features
)

# Example: Complete ML workflow
template_mgr = ModelTemplateManager()
trainer = ModelTrainer()
evaluator = ModelEvaluator()
performance = ModelPerformanceAnalyzer()
shap_analyzer = SHAPAnalyzer()

# Get models and train
models = template_mgr.get_classification_models()
results = trainer.compare_models(models, X, y, cv_folds=5)
metrics = evaluator.calculate_model_scores(best_model, X_test, y_test)
performance.plot_model_comparison(results)

# Advanced SHAP analysis with customer segmentation
segments, freq_table, filtered_freq, impact_rank = shap_analyzer.create_shap_based_customer_segments(
    shap_contributions=shap_data,
    feature_values=X_test,
    target_class_name='churn_probability',
    min_segment_frequency=100
)
```

**Multi-Platform I/O Operations:**
```python
# Unified configuration management
from dsToolbox.io_funcs import (
    ConfigurationManager,    # Universal config handling
    AzureManager,           # Azure Synapse, Blob Storage
    MSSQLManager,           # SQL Server operations
    ColabManager,           # Google Colab integration
    KaggleManager           # Kaggle dataset management
)

# Platform-agnostic data pipeline
config_mgr = ConfigurationManager('config.yml')
azure_mgr = AzureManager(config_mgr)
mssql_mgr = MSSQLManager(config_mgr)

# Seamless data flow across platforms
engine = mssql_mgr.get_engine('production')
df = pd.read_sql("SELECT * FROM customer_data", engine)
blob_spec = {'storage_account': 'analytics', 'container': 'processed', 'blob': 'customer_features.parquet'}
azure_mgr.write_pandas_to_blob(df, blob_spec)
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mrrnour/ds_toolbox_public.git
cd ds_toolbox_public/dsToolbox

# Create environment
conda create --name ds_toolbox python=3.9 -y
conda activate ds_toolbox

# Install dependencies
pip install -r requirements.txt
```

### 🎯 **Complete Tutorial Available!**

For comprehensive examples covering all features, see our complete tutorial:

**📚 [examples/quick_start_tutorial.py](examples/quick_start_tutorial.py)**

> 🎯 **Pro Tip**: Run `python examples/quick_start_tutorial.py` for an interactive demonstration of all features!

This comprehensive tutorial demonstrates:
- ✅ **Enhanced ML Pipeline** with model templates & interpretability
- ✅ **Advanced NLP Processing** with LangChain integration  
- ✅ **Modular I/O Architecture** (Snowflake, AWS, Azure, Kaggle)
- ✅ **Utilities** with enhanced functionality
- ✅ **Spark Integration** for distributed processing
- ✅ **End-to-End Workflows** with real-world examples

### Quick Import Patterns

```python
# Enhanced ML Pipeline
from dsToolbox.ml_funcs import ModelTrainer, SHAPAnalyzer, XGBoostRuleExtractor

# Advanced Analytics (NEW in v2.0)
from dsToolbox.advanced_analytics import (
    DimensionalityReducer, MulticollinearityDetector, 
    CanonicalAnalyzer, ClusteringAnalyzer, PartitionAnalyzer
)

# Modular I/O (NEW in v2.0)
from dsToolbox.io.snowflake import SnowflakeManager
from dsToolbox.io.aws import AWSManager  
from dsToolbox.io.azure import AzureManager

# Utilities (renamed from common_funcs)
from dsToolbox.utilities import TextProcessor, DataFrameUtilities

# Spark Integration  
from dsToolbox.spark_funcs import SparkJoinOperations, SparkETLPipeline

# Run the complete tutorial
python examples/quick_start_tutorial.py
```

**📝 Note**: All detailed usage examples have been moved to the comprehensive tutorial above. The sections below focus on architecture and module descriptions.

---

## 🏗️ Architecture Highlights

### Object-Oriented Design Patterns

**Machine Learning Pipeline:**
```python
# Refactored class-based architecture
from dsToolbox.ml_funcs import (
    ModelTemplateManager,      # Classifier/regressor templates
    ModelTrainer,              # Cross-validation and training
    ModelEvaluator,           # Metrics and scoring
    ModelPerformanceAnalyzer, # Visualization and analysis
    SHAPAnalyzer,             # Advanced interpretability & customer segmentation
    XGBoostRuleExtractor,     # Rule extraction from XGBoost models
    DecisionTreeInterpreter   # Decision tree code generation
)

from dsToolbox.utilities import (
    TextProcessor,            # Text normalization and cleaning
    SQLProcessor,             # SQL parsing and statement processing  
    EncodingUtilities,        # Advanced dummy encoding and sparse labels
    ProductUtilities,         # Product analysis and business logic
    DataFrameUtilities,       # DataFrame manipulation and optimization
    DataVisualization         # Sankey diagrams, 3D plots, word clouds
)

from dsToolbox.advanced_analytics import (
    DimensionalityReducer,    # t-SNE visualization and dimensionality reduction
    MulticollinearityDetector, # VIF analysis and multicollinearity detection
    CanonicalAnalyzer,        # Canonical correlation analysis with preprocessing
    ClusteringAnalyzer,       # K-means clustering with optimization metrics
    PartitionAnalyzer         # Combinatorial partition analysis and mapping
)

from dsToolbox.io.snowflake import SnowflakeManager     # Modular Snowflake operations
from dsToolbox.io.aws import AWSManager                 # Modular AWS operations  
from dsToolbox.io.azure import AzureManager             # Modular Azure operations
from dsToolbox.io.kaggle import KaggleManager           # Modular Kaggle operations

from dsToolbox.cl_funcs import (
    SigmaLimitCalculator,     # Statistical process control using sigma limits
    ControlLimitCalculator,   # Process control limits with control chart methodology  
    ControlChartVisualizer,   # Interactive I-MR charts and SPC visualization
    ProcessRecoveryAnalyzer   # Recovery analysis for mining/manufacturing processes
)

from dsToolbox.spark_funcs import (
    SparkJoinOperations,      # Advanced asof joins and column conflict resolution
    SparkDataTransformations, # DataFrame melting, column operations, type conversions
    SparkETLPipeline,        # Incremental ETL with automatic date tracking
    SparkFeatureEngineering  # Rolling/tumbling windows for time-series features
)
```

**Key Architectural Benefits:**
- **🔧 Modular Design**: Each module has focused responsibility  
- **⚡ Performance**: Optimized for large-scale data processing
- **🔄 Backward Compatibility**: Seamless migration from procedural code
- **📊 Production Ready**: Enterprise-grade error handling and logging
- **🎯 Type Safety**: Comprehensive type hints throughout

## 🆕 Enhanced I/O Operations Module (v2.0)

### Advanced Distributed Data Processing Classes
    direction='backward',
    suffixes=('_sensor', '_maintenance')
)

# 2. Data transformation pipeline
transformer = SparkDataTransformations()

# Convert wide sensor data to long format for analysis
long_format = transformer.melt_dataframe(
    df=df_sensor_data,
    identifier_columns=['equipment_id', 'timestamp'],
    value_columns=['temperature', 'pressure', 'vibration'],
    variable_column_name='metric_type',
    value_column_name='sensor_value'
)

# Batch rename columns for consistency
column_mapping = {
    'equip_temp': 'temperature_celsius',
    'equip_press': 'pressure_bar',
    'ts': 'measurement_timestamp'
}
renamed_df = transformer.rename_columns_batch(df_sensor_data, column_mapping)

# 3. Incremental ETL Pipeline
def process_daily_sensor_data(start_date, end_date, output_target, **kwargs):
    """Process sensor data for given date range."""
    daily_data = spark.sql(f"""
        SELECT equipment_id, measurement_timestamp, 
               AVG(temperature) as avg_temp,
               MAX(pressure) as max_pressure,
               COUNT(*) as reading_count
        FROM sensor_readings 
        WHERE DATE(measurement_timestamp) BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY equipment_id, DATE(measurement_timestamp)
    """)
    return [(output_target, daily_data)]

# Execute incremental pipeline with automatic date management
SparkETLPipeline.execute_incremental_pipeline(
    data_generator_function=process_daily_sensor_data,
    output_target='analytics.daily_sensor_metrics',
    year_range=[2023, 2024],
    first_date='2023-01-01',
    date_column='measurement_timestamp'
)

# 4. Time-series feature engineering
feature_eng = SparkFeatureEngineering()

# Create rolling window features for anomaly detection
rolling_features = feature_eng.create_rolling_window_features(
    df=df_sensor_data,
    timestamp_column='measurement_timestamp',
    groupby_column='equipment_id',
    window_duration='30 minutes',
    aggregation_type='avg',
    include_original_columns=True
)

# Create tumbling window summaries for reporting
tumbling_features = feature_eng.create_tumbling_window_features(
    df=df_sensor_data,
    timestamp_column='measurement_timestamp', 
    groupby_column='production_line',
    window_duration='1 hour',
    aggregation_type='sum',
    join_direction='backward'
)

# Write results to distributed storage
rolling_features.write.mode('overwrite').parquet('s3://output/rolling_features/')
tumbling_features.write.mode('overwrite').parquet('s3://output/hourly_summaries/')
```

## 🆕 Enhanced I/O Operations Module (v2.0)

### New Database & Cloud Integration Classes

The I/O module has been significantly enhanced with comprehensive database and cloud service integrations:

#### Snowflake Database Operations
- **`SnowflakeManager.create_database_connection()`**: Robust Snowflake connection with credential management
- **`SnowflakeManager.execute_sql_query()`**: Execute SQL commands/files with transaction management
- **`SnowflakeManager.upload_dataframe_to_table()`**: Upload pandas DataFrame to Snowflake with auto schema creation
- **`SnowflakeManager.query_to_dataframe()`**: Execute queries and return pandas DataFrame results
- **`SnowflakeManager.check_table_exists()`**: Validate table existence with comprehensive error handling
- **`SnowflakeManager.find_maximum_date_in_table()`**: Auto-detect date columns and find latest dates
- **`SnowflakeManager.get_table_statistics()`**: Get row counts and data freshness metrics

#### AWS Services Integration
- **`AWSManager.upload_file_to_s3()`**: Upload files to S3 with error handling and progress tracking
- **`AWSManager.download_file_from_s3()`**: Download files from S3 with directory creation
- **`AWSManager.clean_s3_folder()`**: Bulk delete S3 objects with folder cleanup
- **`AWSManager.execute_athena_query()`**: Execute Athena queries with result tracking

#### Enhanced Platform Management
- **`ColabManager.setup_complete_environment()`**: Complete Colab setup with Git and SSH integration
- **`KaggleManager.configure_api_credentials()`**: Secure Kaggle API credential management
- **`KaggleManager.download_competition_data()`**: Enhanced dataset downloads with filtering options

## 🆕 Advanced Analytics Module (v2.0)

### New Statistical Analysis & Machine Learning Classes

The advanced analytics module introduces 5 specialized classes for advanced statistical analysis, dimensionality reduction, and machine learning:

#### DimensionalityReducer - t-SNE Visualization
- **`create_tsne_visualization()`**: Interactive t-SNE visualization with Plotly integration
- **Smart Parameter Handling**: Automatic perplexity adjustment based on dataset size
- **Flexible Input Types**: Support for numpy arrays, pandas DataFrames, and lists
- **Customizable Visualization**: Color schemes, marker sizes, and plot dimensions
- **Data Preprocessing**: Optional normalization and standardization

```python
from dsToolbox.advanced_analytics import DimensionalityReducer

# Create interactive t-SNE visualization
reducer = DimensionalityReducer()
plot, embeddings = reducer.create_tsne_visualization(
    data=feature_matrix,
    class_labels=target_labels,
    n_dimensions=2,
    perplexity_value=30.0,
    plot_title="Customer Segmentation Analysis",
    apply_normalization=True
)
plot.show()  # Interactive Plotly visualization
```

#### MulticollinearityDetector - VIF Analysis
- **`calculate_variance_inflation_factors()`**: Calculate VIF scores for multicollinearity detection
- **`analyze_multicollinearity_patterns()`**: Comprehensive analysis with recommendations
- **Hierarchical Clustering**: Order variables by similarity for better interpretation
- **Actionable Insights**: Specific recommendations for feature removal

```python
from dsToolbox.advanced_analytics import MulticollinearityDetector

# Detect multicollinearity patterns
detector = MulticollinearityDetector()
vif_results, problematic_features, analysis = detector.analyze_multicollinearity_patterns(
    feature_matrix=X_train
)
print(analysis)  # Detailed analysis with recommendations
```

#### CanonicalAnalyzer - Canonical Correlation Analysis
- **`perform_canonical_correlation_analysis()`**: Full CCA implementation with preprocessing
- **Multiple Preprocessing Options**: Standard scaling, min-max scaling, robust scaling
- **Variable Clustering**: Optional hierarchical clustering for variable ordering
- **Comprehensive Results**: Canonical correlations, weights, loadings, and scores

```python
from dsToolbox.advanced_analytics import CanonicalAnalyzer

# Perform canonical correlation analysis
analyzer = CanonicalAnalyzer()
results = analyzer.perform_canonical_correlation_analysis(
    variable_set_1=marketing_metrics,
    variable_set_2=customer_behavior,
    n_canonical_components=3,
    preprocessing_method='standard',
    cluster_variables_1=True
)
```

#### ClusteringAnalyzer - Optimal K-means Analysis
- **`perform_optimal_clustering_analysis()`**: Find optimal number of clusters
- **Multiple Evaluation Metrics**: Silhouette score, inertia, Calinski-Harabasz index
- **Comprehensive Results**: Cluster assignments, centroids, and evaluation metrics
- **Flexible Scaling**: Multiple scaling strategies for preprocessing

```python
from dsToolbox.advanced_analytics import ClusteringAnalyzer

# Find optimal clustering solution
analyzer = ClusteringAnalyzer()
results = analyzer.perform_optimal_clustering_analysis(
    dataset=customer_features,
    cluster_range=range(2, 10),
    scaling_strategy='standard',
    evaluation_metric='silhouette'
)
optimal_k = results['optimal_clusters']
```

#### PartitionAnalyzer - Combinatorial Analysis
- **`generate_all_set_partitions()`**: Generate all possible set partitions
- **`create_element_grouping_map()`**: Convert partitions to element-group mappings
- **Memory Efficient**: Handles large sets with optimized algorithms
- **Flexible Constraints**: Min/max subset count limitations

```python
from dsToolbox.advanced_analytics import PartitionAnalyzer

# Generate all possible groupings
analyzer = PartitionAnalyzer()
partitions = analyzer.generate_all_set_partitions(
    elements=['feature_A', 'feature_B', 'feature_C'],
    min_subset_count=1,
    max_subset_count=2
)
```

For comprehensive I/O usage examples and tutorials, see `examples/quick_start_tutorial.py`.

## 🆕 Enhanced Spark Operations Module (v2.0)

### Advanced Distributed Data Processing Classes

The Spark module has been completely refactored from procedural functions to a comprehensive object-oriented architecture for production-ready distributed data processing:

#### SparkJoinOperations - Advanced Join Operations
- **`SparkJoinOperations.perform_asof_join_pandas()`**: High-performance asof joins using pandas for grouped data processing
- **`SparkJoinOperations.resolve_column_conflicts()`**: Intelligent column conflict resolution with multiple strategies
- **`SparkJoinOperations.perform_asof_join_spark()`**: Distributed asof joins using Spark's cogroup and applyInPandas

#### SparkDataTransformations - Data Reshaping & Processing
- **`SparkDataTransformations.melt_dataframe()`**: Convert wide to long format with optimized performance
- **`SparkDataTransformations.rename_columns_batch()`**: Efficient batch column renaming operations
- **`SparkDataTransformations.convert_columns_to_numeric()`**: Intelligent type conversion with error handling
- **`SparkDataTransformations.discover_columns_by_pattern()`**: Pattern-based column discovery in large schemas

#### SparkETLPipeline - Production ETL Management
- **`SparkETLPipeline.get_last_processed_date()`**: Multi-source date tracking (Delta tables, blob storage)
- **`SparkETLPipeline.save_pipeline_outputs()`**: Unified output management across storage systems
- **`SparkETLPipeline.execute_incremental_pipeline()`**: Automated incremental processing with date management

#### SparkFeatureEngineering - Time-Series Feature Creation
- **`SparkFeatureEngineering.create_rolling_window_features()`**: Distributed rolling window aggregations
- **`SparkFeatureEngineering.create_tumbling_window_features()`**: Non-overlapping window summaries with asof joins
- **`SparkFeatureEngineering.identify_numeric_columns()`**: Smart column type detection for large schemas
- **`SparkFeatureEngineering.parse_time_duration()`**: Flexible time duration parsing for window specifications

For comprehensive Spark usage examples and production workflows, see `examples/quick_start_tutorial.py`.

**Key Enhanced Spark Features:**
- **Distributed Asof Joins**: Production-ready time-series joins optimized for large datasets
- **Intelligent ETL Management**: Automatic date tracking across Delta tables and blob storage
- **Advanced Feature Engineering**: Rolling and tumbling windows with multiple aggregation types
- **Column Conflict Resolution**: Smart handling of schema conflicts during joins
- **Incremental Processing**: Only process new data since last successful run
- **Multi-format Output**: Support for Delta tables, Parquet files, and cloud storage
- **Error Recovery**: Robust error handling with retry mechanisms and graceful degradation
- **100% Backward Compatibility**: All legacy functions work with deprecation warnings

## 🆕 Enhanced Utilities Module (v2.0)

### New SQL Processing & Data Encoding Classes

The utilities module has been significantly enhanced with new specialized classes for SQL processing, advanced data encoding, and business utilities:

#### SQL Processing Utilities
- **`SQLProcessor.validate_file_path()`**: Path validation with tilde expansion and existence checks
- **`SQLProcessor.split_sql_statements()`**: Robust SQL statement parsing with comment handling
- **`SQLProcessor.parse_sql_file()`**: Complete SQL file processing with error handling

#### Advanced Data Encoding
- **`EncodingUtilities.create_optimized_dummy_encoding()`**: Memory-efficient dummy variable creation for large datasets
- **`EncodingUtilities.create_sparse_label_encoding()`**: Sparse label encoding with value filling for category-value relationships
- **`EncodingUtilities.fill_dataframe_with_column_names()`**: Replace values with column names for readable sparse matrices

#### Product & Business Analysis
- **`ProductUtilities.clean_product_descriptions()`**: Product metadata cleaning and alignment with reference data
- **`ProductUtilities.condense_dataframe_columns()`**: Column condensing for human-readable summaries
- **`ProductUtilities.extract_current_products()`**: Business-specific product extraction with LOB logic
- **`ProductUtilities.sort_prediction_probabilities()`**: Probability ranking for recommendation systems

For comprehensive utilities usage examples, see `examples/quick_start_tutorial.py`.

## 🆕 New ML Interpretability Features

### Recently Added Functions (v2.0)

The machine learning module has been enhanced with advanced model interpretability and rule extraction capabilities:

#### SHAP Analysis Functions
- **`calculate_shap_contributions()`**: Batch SHAP analysis with comprehensive feature importance and correlation analysis
- **`explain_local_prediction()`**: Individual prediction explanations with detailed SHAP breakdowns
- **`create_shap_based_customer_segments()`**: Advanced customer segmentation based on SHAP contribution patterns

#### XGBoost Rule Extraction
- **`extract_decision_rules()`**: Extract interpretable rules from XGBoost model trees
- **`combine_decision_rules()`**: Combine and simplify extracted rules for business interpretation
- **`generate_rule_summary()`**: Create human-readable summaries of model decision logic

#### Decision Tree Interpretation
- **`generate_tree_code()`**: Convert decision tree models into executable Python code
- **`create_tree_visualization()`**: Generate visual representations of decision paths
- **`extract_tree_rules()`**: Extract logical rules from trained decision trees

For comprehensive ML interpretability examples and advanced workflows, see `examples/quick_start_tutorial.py`.

For advanced feature examples including SHAP analysis, XGBoost rule extraction, text similarity, and RAG systems, see `examples/quick_start_tutorial.py`.

## 🔧 Configuration

### Configuration File (`config.yml`)
```yaml
# MSSQL Server configurations
mssql_servers:
  analytics_db:
    db_server: analytics-sql.company.com
    database: CustomerAnalytics
    trusted_connection: true
    trust_server_certificate: true
  
  production_db:
    db_server: prod-sql.company.com
    database: ProductionDB
    trusted_connection: true

# Azure configurations
azure_storage:
  analytics:
    storage_account: analyticsstore
    container_names: [processed-data, raw-data, models]

# Synapse connection
synapse_cred_dict:
  hostname: analytics-synapse.sql.azuresynapse.net
  database: analytics_dw
  username: analytics_user
  port: 1433

# Key Vault configurations
key_vault_dictS:
  analytics_secrets:
    key_vault_name: analytics-kv
    secret_name: synapse-password
```

For statistical process control and control chart examples, see `examples/quick_start_tutorial.py`.

For comprehensive use case examples including customer churn analysis, document knowledge bases, and production workflows, see `examples/quick_start_tutorial.py`.

## 🔄 Migration & Backward Compatibility

DS Toolbox maintains backward compatibility while providing migration paths:

```python
# Legacy function mapping (newly added functions included)
FUNCTION_MAPPING = {
    # Advanced Analytics Functions (NEW)
    'plot_tsne': 'DimensionalityReducer.create_tsne_visualization()',
    'generate_all_partitions': 'PartitionAnalyzer.generate_all_set_partitions()',
    'partition_to_mapping': 'PartitionAnalyzer.create_element_grouping_map()',
    'calculate_vif': 'MulticollinearityDetector.calculate_variance_inflation_factors()',
    'detect_multicollinearity': 'MulticollinearityDetector.analyze_multicollinearity_patterns()',
    'cca_analysis': 'CanonicalAnalyzer.perform_canonical_correlation_analysis()',
    'cluster_analysis': 'ClusteringAnalyzer.perform_optimal_clustering_analysis()',
    
    # SHAP Analysis Functions (ML)
    'shap_batch': 'SHAPAnalyzer.calculate_shap_contributions()',
    'shap_localInterpretability': 'SHAPAnalyzer.explain_local_prediction()',
    'segmentor': 'SHAPAnalyzer.create_shap_based_customer_segments()',
    'top_n_driver': 'SHAPAnalyzer.calculate_top_feature_drivers()',
    'shap_plots_batch': 'SHAPAnalyzer.generate_shap_plots()',
    
    # XGBoost Rule Extraction Functions (ML)
    'leafs2rules': 'XGBoostRuleExtractor.extract_decision_rules()',
    'rulesCombiner': 'XGBoostRuleExtractor.combine_decision_rules()',
    'xgb_rules': 'XGBoostRuleExtractor.generate_rule_summary()',
    
    # Decision Tree Functions (ML)
    'tree_to_code': 'DecisionTreeInterpreter.generate_tree_code()',
    'string_parser': 'DecisionTreeInterpreter.parse_tree_structure()',
    
    # Spark Operations Functions (Spark)
    'asof_join_sub': 'SparkJoinOperations.perform_asof_join_pandas()',
    'asof_join_spark2': 'SparkJoinOperations.perform_asof_join_spark()',
    'melt': 'SparkDataTransformations.melt_dataframe()',
    'rename_cols': 'SparkDataTransformations.rename_columns_batch()',
    'sp_to_numeric': 'SparkDataTransformations.convert_columns_to_numeric()',
    'col_finder': 'SparkDataTransformations.discover_columns_by_pattern()',
    'last_date': 'SparkETLPipeline.get_last_processed_date()',
    'save_outputs': 'SparkETLPipeline.save_pipeline_outputs()',
    'update_db_recursively': 'SparkETLPipeline.execute_incremental_pipeline()',
    'create_rolling_features': 'SparkFeatureEngineering.create_rolling_window_features()',
    'create_tumbling_features': 'SparkFeatureEngineering.create_tumbling_window_features()',
    
    # Snowflake Database Functions (I/O)
    'conn2snowFlake': 'SnowflakeManager.create_database_connection()',
    'runSQL_snowFlake': 'SnowflakeManager.execute_sql_query()',
    'df2snowFlake': 'SnowflakeManager.upload_dataframe_to_table()',
    'snowFlake2df': 'SnowflakeManager.query_to_dataframe()',
    'chkTblinsnowFlake': 'SnowflakeManager.check_table_exists()',
    'findMaxDate': 'SnowflakeManager.find_maximum_date_in_table()',
    'tbls_dateNrows': 'SnowflakeManager.get_table_statistics()',
    'df2snowFlake_dtype_sub': '[Internal helper - now handled within SnowflakeManager]',
    
    # AWS Services Functions (I/O)
    'sage2s3': 'AWSManager.upload_file_to_s3()',
    's32sage': 'AWSManager.download_file_from_s3()',
    'sweeper_S3': 'AWSManager.clean_s3_folder()',
    'runSQL_athena': 'AWSManager.execute_athena_query()',
    'athena2df': 'AWSManager.query_athena_to_dataframe()',
    'sage2athena': 'AWSManager.upload_dataframe_to_athena()',
    'redshift2Athena': 'AWSManager.transfer_redshift_to_athena()',
    'track_athena_response_sub': '[Internal helper - now handled within AWSManager]',
    'sage2athena_dtype_sub': '[Internal helper - now handled within AWSManager]',
    
    # Platform Detection & Configuration (I/O)
    'detect_platform': 'detect_execution_platform()',
    'ConfigurationManager': 'ConfigurationManager (enhanced with platform detection)',
    'DatabaseConnectionManager': 'DatabaseConnectionManager (backward compatible)',
    
    # SQL Processing Functions (Common)
    'valid_path': 'SQLProcessor.validate_file_path()',
    'split_sql_expressions_sub': 'SQLProcessor.split_sql_statements()',
    'parse_sql_file_sub': 'SQLProcessor.parse_sql_file()',
    
    # Advanced Data Encoding Functions (Common)
    'get_dummies2': 'EncodingUtilities.create_optimized_dummy_encoding()',
    'sparseLabel': 'EncodingUtilities.create_sparse_label_encoding()',
    'fill_with_colnames': 'EncodingUtilities.fill_dataframe_with_column_names()',
    'get_dummies2_sub': '[Internal helper - now handled within EncodingUtilities]',
    'sparseLabel_sub': '[Internal helper - now handled within EncodingUtilities]',
    
    # Product & Business Utilities (Common)
    'joinNonZero': 'ProductUtilities.join_non_zero_values()',
    'prodDesc_clean': 'ProductUtilities.clean_product_descriptions()',
    'condense_cols': 'ProductUtilities.condense_dataframe_columns()',
    'current_prds': 'ProductUtilities.extract_current_products()',
    'sortPrds': 'ProductUtilities.sort_prediction_probabilities()',
    
    # Text Processing Functions (Common)
    'normalize_text': 'TextProcessor.normalize_text()',
    'sanitize_filename': 'TextProcessor.sanitize_filename()',
    'clean_column_names': 'TextProcessor.clean_column_names()',
    'find_fuzzy_matches': 'TextProcessor.find_fuzzy_matches()',
    'clean_sql_query': 'TextProcessor.clean_sql_query()',
    
    # DataFrame Operations (Common)
    'movecol': 'DataFrameUtilities.reorder_columns()',
    'cellWeight': 'DataFrameUtilities.calculate_cell_proportions()',
    'reduce_mem_usage': 'DataFrameUtilities.reduce_memory_usage()',
    'null_per_column': 'DataFrameUtilities.analyze_missing_values()',
    
    # List Operations (Common)
    'inWithReg': 'ListUtilities.search_with_regex()',
    'flattenList': 'ListUtilities.flatten_nested_list()',
    'unique_list': 'ListUtilities.get_unique_ordered()',
    'remove_extra_none': 'ListUtilities.remove_nested_none_values()',
    
    # Existing ML mappings
    'ml_comparison': 'ModelTrainer.compare_models()',
    'model_eval': 'ModelEvaluator.calculate_model_scores()',
    
    # Direct function calls still work with deprecation warnings
}

# Print migration guide with new functions
from dsToolbox.ml_funcs import print_function_mapping
print_function_mapping()  # Shows complete mapping including new functions
```

## 📊 Performance & Scalability

### Memory Optimization
- **Chunked Processing**: Large datasets processed in configurable chunks
- **Model Caching**: Heavy NLP models cached to avoid reloading
- **Spark Integration**: Distributed processing for big data workloads

### Platform Scalability
- **Local Development**: Full functionality on local machines
- **Cloud Environments**: Optimized for Azure, Databricks, and Colab
- **Containerized Deployment**: Docker-ready with environment detection

## 🛠️ Development & Contributing

### Project Structure
```
ds_toolbox_public/
├── dsToolbox/
│   ├── ml_funcs.py           # Machine learning pipeline
│   ├── io_funcs.py           # Multi-platform I/O
│   ├── nlp_llm_funcs.py      # NLP and LLM utilities
│   ├── common_funcs.py       # Shared utilities
│   ├── rag_funcs.py          # RAG implementation
│   ├── spark_funcs.py        # Spark operations
│   ├── cl_funcs.py           # Control charts
│   └── doc/                  # Documentation and examples
├── requirements.txt          # Dependencies
└── README.md                # This file
```

### Code Quality Standards
- **Type Hints**: Full type annotation for better IDE support
- **PEP 257 Documentation**: Comprehensive docstrings with examples
- **Error Handling**: Graceful handling of missing dependencies
- **Logging**: Structured logging throughout modules

## 📈 Metrics & Monitoring

Built-in performance tracking with comprehensive model evaluation metrics, ROC curves, precision-recall curves, and model comparison visualizations. For detailed usage examples, see `examples/quick_start_tutorial.py`.

## 🎯 Roadmap

- [ ] **Auto-ML Integration**: Automated model selection and hyperparameter tuning
- [ ] **Real-time Streaming**: Kafka and streaming data processing utilities
- [ ] **MLOps Integration**: Model versioning and deployment utilities  
- [ ] **Advanced RAG**: Multi-modal document processing and retrieval
- [ ] **Edge Deployment**: Optimized functions for edge computing scenarios

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📞 Support

- **Documentation**: Check `/dsToolbox/doc/` for detailed examples
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions for questions and feature requests

---

<div align="center">
  <strong>Built for Data Scientists, by Data Scientists</strong>
  <br/>
  <em>Streamlining workflows from data ingestion to model deployment</em>
</div>