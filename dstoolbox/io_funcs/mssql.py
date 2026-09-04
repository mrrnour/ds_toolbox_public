"""MSSQL helpers + output-spec runner.

All connections flow through ``data_sources.get(target_id)`` + ``db.engine`` /
``db.pyodbc_connection``. See CONTEXT.md.
"""

import datetime as dt
import logging
import os

import pandas as pd

from .. import utils
from . import data_sources
from . import db as _db
from .exceptions import MSSQLError, OutputSpecError

logger_mod = logging.getLogger(__name__)


def _log_or_print(logger, message):
    """Emit ``message`` via ``logger.info`` when supplied, else the module logger."""
    (logger if logger is not None else logger_mod).info(message)


def mssql_query(sql_query: str, target_id: str, return_df: bool = True):
    """Execute a SQL query against an MSSQL target; optionally return a DataFrame.

    Parameters
    ----------
    sql_query : str
        The SQL statement to execute.
    target_id : str
        Key into ``data_sources`` (e.g. ``"my_mssql"``) used to open a
        pyodbc connection via ``db.pyodbc_connection``.
    return_df : bool, default True
        If True, wrap the result set in a pandas DataFrame via
        ``pd.read_sql_query``. If False, execute the statement and return
        the underlying cursor.

    Returns
    -------
    pandas.DataFrame or pyodbc cursor
        DataFrame when ``return_df`` is True, otherwise the executed cursor.

    Raises
    ------
    MSSQLError
        If the query fails; the underlying pyodbc error is chained.
    """
    conn = _db.pyodbc_connection(data_sources.get(target_id))
    try:
        if return_df:
            return pd.read_sql_query(sql_query, conn)
        return conn.execute(sql_query)
    except Exception as e:
        conn.rollback()
        raise MSSQLError(f"Error running SQL in MSSQL: {e}") from e


def df2mssql(df: pd.DataFrame, table_name: str, target_id: str, **kwargs) -> None:
    """Write a pandas DataFrame to an MSSQL table via SQLAlchemy ``to_sql``.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to write.
    table_name : str
        Destination table name (without the schema; pass ``schema=...`` in
        ``kwargs`` if needed).
    target_id : str
        Key into ``data_sources``; used to build the SQLAlchemy engine.
    **kwargs
        Forwarded verbatim to ``DataFrame.to_sql`` (e.g. ``schema``,
        ``if_exists``, ``chunksize``, ``method``, ``index``, ``dtype``).

    Raises
    ------
    MSSQLError
        If ``to_sql`` fails; the underlying error is chained.
    """
    engine = _db.engine(data_sources.get(target_id))
    try:
        df.to_sql(table_name, con=engine, **kwargs)
    except Exception as e:
        raise MSSQLError(f"Error writing DataFrame into MSSQL table {table_name!r}: {e}") from e


def mssql_table_check(tablename: str, target_id: str) -> bool:
    """Return True iff a table exists in the MSSQL target.

    Parameters
    ----------
    tablename : str
        Fully-qualified table name, either ``"db.schema.table"`` or
        ``"schema.table"``.
    target_id : str
        Key into ``data_sources`` (opens a pyodbc connection).

    Returns
    -------
    bool
        True when the table is reported by ``information_schema.tables``.

    Raises
    ------
    ValueError
        If ``tablename`` doesn't have 2 or 3 dot-separated parts.
    MSSQLError
        If the lookup query fails.
    """
    conn = _db.pyodbc_connection(data_sources.get(target_id))
    try:
        parts = tablename.split(".")
        if len(parts) == 3:
            database, schema, table = parts
            information_schema = f"{database}.information_schema.tables"
        elif len(parts) == 2:
            schema, table = parts
            information_schema = "information_schema.tables"
        else:
            raise ValueError(f"Invalid table name format: {tablename!r}")

        sql_query = (
            f"SELECT COUNT(*) FROM {information_schema} "
            f"WHERE table_name = '{table}' AND TABLE_SCHEMA = '{schema}'"
        )
        out = pd.read_sql_query(sql_query, conn)
        return bool(out.iloc[0, 0] == 1)
    except Exception as e:
        conn.rollback()
        raise MSSQLError(f"Error checking table existence for {tablename!r}: {e}") from e
    finally:
        conn.close()


def get_last_date_from_mssql_table(table_name: str, target_id: str, date_column: str, logger=None):
    """Return the most recent value of ``date_column`` in an MSSQL table, or None.

    Parameters
    ----------
    table_name : str
        Fully-qualified table name (``"db.schema.table"`` or ``"schema.table"``).
    target_id : str
        Key into ``data_sources``.
    date_column : str
        Column whose ``MAX(...)`` is returned.
    logger : logging.Logger, optional
        Logger for status messages; falls back to the module logger.

    Returns
    -------
    Any or None
        The max value of ``date_column``, or ``None`` when the table
        does not exist.
    """
    if not mssql_table_check(table_name, target_id):
        _log_or_print(logger, f"{table_name} does not exist")
        return None
    query = (
        f"SELECT MIN({date_column}) AS min_time, MAX({date_column}) AS max_time "
        f"FROM {table_name}"
    )
    results = mssql_query(query, target_id)
    most_recent = results["max_time"].iloc[0]
    print(f"The last date found in {table_name}: {most_recent}")
    return most_recent


def last_date_parquet(file_name: str, date_col: str, logger=None):
    """Return the most recent value of ``date_col`` in a parquet file, or None.

    Parameters
    ----------
    file_name : str
        Path to a parquet file.
    date_col : str
        Column whose ``max()`` is returned.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    Any or None
        The max value of ``date_col``, or ``None`` when the file does not exist.
    """
    if not os.path.isfile(file_name):
        _log_or_print(logger, f"{file_name} does not exist")
        return None
    df = pd.read_parquet(file_name)
    last = df[date_col].max()
    _log_or_print(logger, f"The last date found in {file_name}:{last}")
    return last


def last_date(output_dict: dict, logger=None):
    """Dispatch to the format-appropriate last-date helper for a given output spec.

    Parameters
    ----------
    output_dict : dict
        Output spec containing ``format``, ``output_location``, ``date_col``
        and (when ``format == "MS_db"``) ``target_id``.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    Any or None
        Max value of the date column, or ``None`` for unknown formats or
        missing targets.
    """
    fmt = output_dict["format"]
    date_col = output_dict["date_col"]
    location = output_dict["output_location"]
    if fmt == "MS_db":
        return get_last_date_from_mssql_table(
            location,
            output_dict["target_id"],
            date_col,
            logger=logger,
        )
    if fmt == "parquet":
        return last_date_parquet(location, date_col, logger=logger)
    return None


def load_parquet_between_dates(
    ufile: str, date_col: str, start_date: str = "2019-01-01", end_date: str = "2020-01-01"
) -> pd.DataFrame:
    """Read a parquet file and return rows whose ``date_col`` falls in [start, end).

    Parameters
    ----------
    ufile : str
        Path to a parquet file.
    date_col : str
        Column used for the filter.
    start_date, end_date : str, optional
        Inclusive lower / exclusive upper bounds in ``YYYY-MM-DD`` form.

    Returns
    -------
    pandas.DataFrame
        Rows where ``start_date <= df[date_col] < end_date``.
    """
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    df = pd.read_parquet(ufile)
    return df[(df[date_col] >= start) & (df[date_col] < end)]


def update_output_specs(
    output_specs, year_range=None, month_step: int = 1, firstDate=None, lastDate=None, logger=None
):
    """Annotate output specs with their last saved date and build a run-date list.

    For each spec, the helper looks up the most recent date already saved
    (via :func:`last_date`) and stashes it on the spec under ``"last_date"``.
    From the first spec's last-saved date it then builds a list of monthly
    run boundaries via :func:`dstoolbox.utils.monthly_first_dates`.

    Parameters
    ----------
    output_specs : dict or list of dict
        A single output spec or a list of them. Each spec must be shaped
        as accepted by :func:`last_date`. A single-dict input is wrapped
        into a one-element list.
    year_range : list of int, optional
        ``[start_year, stop_year]`` forwarded to
        :func:`~dstoolbox.utils.monthly_first_dates`. Defaults to
        ``[2021, 2099]``.
    month_step : int, default 1
        Month stride passed through to ``monthly_first_dates``.
    firstDate : str, datetime, pd.Timestamp, or None, optional
        Lower bound. When ``None``, uses the day after the first spec's
        last-saved date.
    lastDate : datetime.date or None, optional
        Upper bound. Defaults to today.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    (list of dict, list of str)
        The (possibly-wrapped) spec list with ``"last_date"`` populated,
        and the generated list of run boundary dates.
    """
    if year_range is None:
        year_range = [2021, 2099]
    if lastDate is None:
        lastDate = dt.datetime.now().date()

    if isinstance(output_specs, dict):
        output_specs = [output_specs]

    last_saved_dates = []
    run_dates = []
    for i, spec in enumerate(output_specs):
        last_saved_date = last_date(spec, logger=logger)
        spec["last_date"] = last_saved_date
        last_saved_dates.append(last_saved_date)

        if i == 0:
            warn_text = False
            if firstDate is not None:
                if isinstance(firstDate, str):
                    _log_or_print(logger, firstDate)
                    first_resolved = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
                elif isinstance(firstDate, pd._libs.tslibs.timestamps.Timestamp):
                    first_resolved = firstDate.date()
                    warn_text = True
                else:
                    first_resolved = firstDate
            else:
                first_resolved = (
                    None if last_saved_date is None else last_saved_date + dt.timedelta(days=1)
                )

            if warn_text and last_saved_date is not None:
                _log_or_print(
                    logger,
                    f"The last date is {last_saved_date}; however, the "
                    f"function starts from given first date: {firstDate}",
                )

            run_dates = utils.monthly_first_dates(
                year_range=year_range,
                month_step=month_step,
                firstDate=first_resolved,
                lastDate=lastDate,
            )

            if not run_dates:
                _log_or_print(logger, "Database|file is updated")
            else:
                _log_or_print(logger, "Date list updated to :\n" + str(run_dates))

    if len(set(last_saved_dates)) > 1:
        _log_or_print(
            logger,
            "Warning! There are different last_date values across output_specs; "
            "run_dates were built from the first spec:\t",
        )
        _log_or_print(logger, last_saved_dates)

    return output_specs, run_dates


def save_outputs(output_dict: dict, output_specs, logger=None) -> int:
    """Save DataFrames per the output specs.

    Parameters
    ----------
    output_dict : dict
        Must contain ``"output_df_keys"`` (list of lists of keys, one per
        DataFrame) and ``"dfs"`` (the DataFrames in the same order).
    output_specs : dict or list of dict
        Per-key destination specs. Each entry must contain
        ``output_df_key``, ``format`` (``"MS_db"`` or ``"parquet"``),
        ``output_location``, ``overwrite``, and for ``"MS_db"`` also
        ``target_id``. A single dict is wrapped into a one-element list.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    int
        Always ``1`` on completion (kept for backward compatibility).

    Raises
    ------
    OutputSpecError
        When the set of keys in ``output_dict`` and ``output_specs`` differ.
    """
    flatten = utils.flatten_list(output_dict["output_df_keys"])
    if isinstance(output_specs, dict):
        output_specs = [output_specs]

    spec_keys = {s["output_df_key"] for s in output_specs}
    orphan_dfs = set(flatten) - spec_keys
    if orphan_dfs:
        raise OutputSpecError(
            f"Dataframes not saved: {orphan_dfs}. "
            "Match output_list and return values of df_generator_func."
        )

    orphan_outputs = spec_keys - set(flatten)
    if orphan_outputs:
        raise OutputSpecError(
            f"Outputs do not exist in df_generator_func: {orphan_outputs}. "
            "Match output_list and return values of df_generator_func."
        )

    for key_dfs, df in zip(output_dict["output_df_keys"], output_dict["dfs"], strict=False):
        if df.size == 0:
            _log_or_print(logger, "Dataframe is empty")
            continue
        for key_df in key_dfs:
            spec = next(s for s in output_specs if s["output_df_key"] == key_df)
            fmt = spec["format"]
            location = spec["output_location"]
            overwrite = spec["overwrite"]
            _log_or_print(
                logger,
                f"saving output {spec['output_df_key']} in {location}...",
            )

            if fmt == "MS_db":
                db, schema_part, table = location.split(".")
                df2mssql(
                    df,
                    table_name=table,
                    target_id=spec["target_id"],
                    schema=f"{db}.{schema_part}",
                    chunksize=200,
                    method="multi",
                    index=False,
                    if_exists="replace" if overwrite else "append",
                )
            elif fmt == "parquet":
                if not overwrite and os.path.isfile(location):
                    df_current = pd.read_parquet(location)
                    df = pd.concat([df_current, df], axis=0)
                df.to_parquet(location, index=False)

    _log_or_print(logger, "-" * 50)
    return 1


def run_recursively(
    output_specs,
    df_generator_func,
    year_range=None,
    month_step: int = 1,
    firstDate=None,
    lastDate=None,
    logger=None,
    **kwargs,
):
    """Run ``df_generator_func`` over each month-slice and persist results.

    Parameters
    ----------
    output_specs : dict or list of dict
        Destination specs, forwarded to :func:`update_output_specs` and
        :func:`save_outputs`.
    df_generator_func : callable
        Called as ``df_generator_func(start_date, end_date, logger=logger,
        **matched_kwargs)`` per month slice; must return a dict shaped
        like the ``output_dict`` accepted by :func:`save_outputs`.
    year_range : list of int, optional
        ``[start_year, stop_year]``; defaults to ``[2021, 2099]``.
    month_step : int, default 1
        Month stride between slice boundaries.
    firstDate, lastDate : optional
        Passed through to :func:`update_output_specs`.
    logger : logging.Logger, optional
        Logger for status messages.
    **kwargs
        Additional keyword arguments. Any whose names match a parameter
        of ``df_generator_func`` are forwarded to it; the rest are ignored.
    """
    import inspect

    if year_range is None:
        year_range = [2021, 2099]
    if lastDate is None:
        lastDate = dt.datetime.now().date()

    _log_or_print(logger, "Updating the outputs list...\n")
    output_specs2, run_dates = update_output_specs(
        output_specs,
        year_range=year_range,
        month_step=month_step,
        firstDate=firstDate,
        lastDate=lastDate,
        logger=logger,
    )

    gen_args = list(inspect.signature(df_generator_func).parameters)
    gen_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in gen_args}

    _log_or_print(logger, "/" * 50 + "\n")

    try:
        for ii in range(len(run_dates) - 1):
            start_date, end_date = utils.extract_start_end(run_dates, ii)
            _log_or_print(
                logger,
                f"Running {df_generator_func.__name__} for "
                f"the period {start_date} to {end_date}...",
            )
            output_dict = df_generator_func(start_date, end_date, logger=logger, **gen_kwargs)
            save_outputs(output_dict, output_specs2, logger=logger)
    except Exception as e:
        _log_or_print(
            logger,
            f"***Running function {df_generator_func.__name__} failed: " f"\n\t\t {e}",
        )
