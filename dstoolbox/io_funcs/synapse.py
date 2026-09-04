"""Azure Synapse SQL query helpers (Databricks JDBC + local pyodbc)."""

import pandas as pd

from . import data_sources
from .bootstrap import clean_query, get_spark


def query_synapse(
    query: str, runtime: str = "databricks", target_id: str = "azure_synapse", verbose: bool = True
):
    """Run a query against Azure Synapse.

    Dispatches on ``runtime``: returns a Spark DataFrame on Databricks
    (via JDBC) and a pandas DataFrame locally (via pyodbc).
    """
    if runtime == "databricks":
        return query_synapse_db(query, target_id=target_id, verbose=verbose)
    if runtime in ("local", "vm_docker"):
        return query_synapse_local(query, target_id=target_id, runtime=runtime, verbose=verbose)
    raise ValueError(f"query_synapse: unsupported runtime {runtime!r}")


def query_synapse_db(query: str, target_id: str = "azure_synapse", verbose: bool = True):
    """Run a query against Azure Synapse via Spark JDBC; returns a Spark DataFrame."""
    ds = data_sources.get(target_id, runtime="databricks")
    wrapped = (
        f"({query}) query" if query.strip()[-5:] != "query" or query.strip()[0] != "(" else query
    )
    if verbose:
        print("pulling data from azure_synapse:\n", wrapped)
    spark, _ = get_spark()
    return spark.read.jdbc(table=wrapped, url=ds.jdbc_url, properties=ds.jdbc_properties)


def query_synapse_local(
    query: str, target_id: str = "azure_synapse", runtime: str = "local", verbose: bool = True
) -> pd.DataFrame:
    """Run a query against Azure Synapse via local pyodbc; returns a pandas DataFrame."""
    import pyodbc

    ds = data_sources.get(target_id, runtime=runtime)
    q = clean_query(query)
    if verbose:
        print("pulling data from azure_synapse:\n", q)
    with pyodbc.connect(ds.odbc_connection_string) as conn:
        return pd.read_sql(q, conn)
