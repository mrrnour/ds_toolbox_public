"""Layer 2 conveniences: open a live database client from a DataSource.

These helpers live only where there is one obvious right client — relational
databases. Blob and PI deliberately do not have helpers here; callers use the
DataSource materials directly with whichever SDK fits.

See docs/adr/0001-data-sources-seam.md.
"""

from __future__ import annotations

from typing import Any

from .data_sources import (
    MSSQLDataSource,
    PostgresDataSource,
    SynapseDataSource,
)


def engine(ds: Any, **engine_kwargs: Any) -> Any:
    """Build a SQLAlchemy engine from a DataSource.

    Supports ``MSSQLDataSource``, ``SynapseDataSource``, ``PostgresDataSource``.
    """
    from sqlalchemy import create_engine

    if isinstance(ds, (MSSQLDataSource, SynapseDataSource)):
        from urllib.parse import quote_plus

        params = quote_plus(ds.odbc_connection_string)
        return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", **engine_kwargs)

    if isinstance(ds, PostgresDataSource):
        return create_engine(ds.sqlalchemy_url, **engine_kwargs)

    raise TypeError(f"db.engine: no SQLAlchemy engine path for {type(ds).__name__}")


def pyodbc_connection(ds: Any) -> Any:
    """Open a pyodbc connection from a DataSource.

    Supports ``MSSQLDataSource`` and ``SynapseDataSource``.
    """
    if not isinstance(ds, (MSSQLDataSource, SynapseDataSource)):
        raise TypeError(
            f"db.pyodbc_connection: expected MSSQL/Synapse DataSource, " f"got {type(ds).__name__}"
        )
    import pyodbc

    return pyodbc.connect(ds.odbc_connection_string)


def psycopg2_connection(ds: Any) -> Any:
    """Open a psycopg2 connection from a ``PostgresDataSource``."""
    if not isinstance(ds, PostgresDataSource):
        raise TypeError(
            f"db.psycopg2_connection: expected PostgresDataSource, " f"got {type(ds).__name__}"
        )
    import psycopg2

    kwargs = dict(
        host=ds.host,
        port=ds.port,
        database=ds.database,
        user=ds.user,
        password=ds.password,
    )
    if ds.sslmode:
        kwargs["sslmode"] = ds.sslmode
    return psycopg2.connect(**kwargs)
