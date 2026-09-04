"""Spark/dbutils accessors, config loading, and lightweight query cleanup."""

from importlib import resources as res
from pathlib import Path
from typing import Any

import yaml


def get_spark():
    """Return ``(SparkSession, SQLContext)`` for the active session."""
    import pyspark

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    sql_context = pyspark.SQLContext(spark.sparkContext)
    return spark, sql_context


def get_dbutils():
    """Return the Databricks ``dbutils`` handle from the active IPython kernel."""
    import IPython

    return IPython.get_ipython().user_ns["dbutils"]


def load_config(custom_config: dict[str, Any] | str | None = None) -> dict[str, Any]:
    """Load the dstoolbox configuration.

    Parameters
    ----------
    custom_config : None, dict, or str, optional
        * ``None``: read the bundled ``dstoolbox/config.yml`` resource.
        * ``dict``: use as the config directly.
        * ``str``: path to a YAML file to load.

    Returns
    -------
    dict
        Parsed configuration. The canonical schema lives under
        ``data_sources:`` — see CONTEXT.md.
    """
    if custom_config is None:
        with res.open_binary("dstoolbox", "config.yml") as fp:
            return yaml.safe_load(fp) or {}
    if isinstance(custom_config, dict):
        return custom_config
    return yaml.safe_load(Path(custom_config).read_text()) or {}


def clean_query(q: str) -> str:
    """Strip an outer ``(... ) query`` wrapper from a SQL string, if present."""
    q = q.strip().lstrip("(")
    q = q.rstrip("query")
    q = q.strip().rstrip(")")
    return q


# Module-level config load (preserved for back-compat callers).
io_config_dict = load_config(custom_config=None)
