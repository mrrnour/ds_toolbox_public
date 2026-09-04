"""Databricks Delta Table helpers + DBFS-to-blob copy."""

from . import data_sources
from .bootstrap import get_dbutils, get_spark


def query_delta_table_db(query: str, target_id: str = "deltaTable", verbose: bool = True):
    """Run a query against a Delta table; returns a Spark DataFrame."""
    if verbose:
        print("pulling data from Delta table:\n", query)
    ds = data_sources.get(target_id, runtime="databricks")
    spark, _ = get_spark()
    spark.conf.set(ds.spark_conf_key, ds.account_key)
    return spark.sql(query)


def dbfs2blob(ufile: str, blob_dict: dict):
    """Copy a DBFS file to Azure blob storage.

    ``blob_dict`` shape: ``{'target_id': ..., 'container': ..., 'blob': ...}``.
    """
    target_id = blob_dict["target_id"]
    container = blob_dict["container"]
    blob = blob_dict["blob"]
    ds = data_sources.get(target_id, runtime="databricks")

    spark, _ = get_spark()
    spark.conf.set(ds.spark_conf_key, ds.account_key)

    uri = ds.wasbs_uri(container, blob)
    dbutils = get_dbutils()
    dbutils.fs.cp(ufile.replace("/dbfs", "dbfs:"), uri)
    print(f"{ufile} saved in {uri}")


def spark2delta_table(
    df,
    table_name: str,
    schema: str = "xxx_analytics",
    write_mode: str = "append",
    partitionby=None,
    **options,
):
    """Write a Spark DataFrame to a Delta table."""
    spark, _ = get_spark()
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
    writer = df.write.mode(write_mode)
    if partitionby is not None:
        cols = partitionby if isinstance(partitionby, list) else [partitionby]
        writer = writer.partitionBy(*cols)
    writer.options(**options).saveAsTable(f"{schema}.{table_name}")


def delta_table_check(delta_table_name: str) -> bool:
    """Return True iff the Delta table exists."""
    spark, _ = get_spark()
    exists = spark._jsparkSession.catalog().tableExists(delta_table_name)
    if exists:
        print(f"table {delta_table_name} exists!")
    return exists
