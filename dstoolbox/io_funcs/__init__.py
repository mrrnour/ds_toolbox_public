"""io_funcs package: Azure (Synapse, Blob, KV), Databricks Delta, PI Web API, MSSQL/Postgres, SQL templates, Colab/Kaggle.

Single credentials seam: ``data_sources.get(target_id)`` returns a typed
``DataSource``. Layer 2 ``db`` helpers open live SQLAlchemy/pyodbc/psycopg2
clients from a DataSource. See CONTEXT.md and docs/adr/0001-data-sources-seam.md.
"""

from . import data_sources, db
from .blob import (
    blob2pd,
    blob2spark,
    blob_check,
    pd2blob,
    pd2blob_batch,
    spark2blob,
    xls2blob,
)
from .bootstrap import (
    clean_query,
    get_dbutils,
    get_spark,
    io_config_dict,
    load_config,
)
from .colab import (
    copy_kaggle_json_to_colab,
    download_and_extract_dataset,
    setup_github_colab,
)
from .data_sources import (
    ADLSDataSource,
    BlobDataSource,
    DataSourceError,
    MSSQLDataSource,
    PIDataSource,
    PostgresDataSource,
    SynapseDataSource,
    UnknownAuthKindError,
    UnknownKindError,
    UnknownTargetError,
)
from .delta import (
    dbfs2blob,
    delta_table_check,
    query_delta_table_db,
    spark2delta_table,
)
from .mssql import (
    df2mssql,
    get_last_date_from_mssql_table,
    last_date,
    last_date_parquet,
    load_parquet_between_dates,
    mssql_query,
    mssql_table_check,
    run_recursively,
    save_outputs,
    update_output_specs,
)
from .pi import (
    get_web_ids,
    pi2pd_interpolate,
    pi2pd_raw_data,
    pi2pd_seconds,
)
from .synapse import (
    query_synapse,
    query_synapse_db,
    query_synapse_local,
)
from .templates import (
    query_template_reader,
    query_template_run,
)
