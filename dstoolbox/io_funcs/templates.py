"""SQL template loading + execution: replace placeholders, dispatch by host."""

from importlib import resources as res

import yaml

from dstoolbox import utils

from .. import default_values as par
from .delta import query_delta_table_db
from .synapse import query_synapse_db, query_synapse_local


def query_template_reader(query_str: str, replace_dict=None) -> str:
    """Substitute placeholders in a SQL template string.

    ``replace_dict`` defaults to ``{'start___date', 'end___date'}`` filled
    from :mod:`default_values`. Substitution only runs if the timestamps
    validate via ``utils.check_timestamps``.
    """
    if replace_dict is None:
        replace_dict = {
            "start___date": par.start_date,
            "end___date": par.end_date,
        }
    query = query_str
    if utils.check_timestamps(replace_dict.get("start___date"), replace_dict.get("end___date")):
        for key, value in replace_dict.items():
            query = query.replace(key, value)
    return query


def query_template_run(
    query_temp_name: str,
    replace_dict=None,
    custom_sql_template_yml=None,
    runtime: str = "databricks",
):
    """Run a named SQL template against Synapse or Delta.

    The template's ``db`` field is interpreted as a ``target_id`` under
    ``data_sources:``. ``runtime`` controls Synapse dispatch (Spark JDBC
    on databricks vs. pyodbc on local/vm_docker).
    """
    if replace_dict is None:
        replace_dict = {
            "start___date": par.start_date,
            "end___date": par.end_date,
        }

    if custom_sql_template_yml is None:
        with res.open_binary("dstoolbox", "sql_template.yml") as fp:
            templates = yaml.safe_load(fp)
    else:
        from pathlib import Path

        templates = yaml.safe_load(Path(custom_sql_template_yml).read_text())

    template = templates[query_temp_name]
    host, query_str = template["db"], template["query"]

    query = query_template_reader(query_str, replace_dict=replace_dict)

    if host == "azure_synapse":
        if runtime == "databricks":
            return query_synapse_db(query, target_id=host, verbose=True)
        if runtime in ("local", "vm_docker"):
            return query_synapse_local(
                query,
                target_id=host,
                runtime=runtime,
                verbose=True,
            )
        raise ValueError(f"query_template_run: unsupported runtime {runtime!r}")

    return query_delta_table_db(query, target_id=host)
