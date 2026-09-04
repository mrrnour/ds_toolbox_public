"""Tests for the data_sources seam.

Pure dispatch + dataclass construction. No real Azure/MSSQL needed —
Key Vault resolution and the PI OAuth POST are monkeypatched.
"""

import pytest

from dstoolbox.io_funcs import data_sources as ds_module
from dstoolbox.io_funcs.data_sources import (
    ADLSDataSource,
    BlobDataSource,
    MSSQLDataSource,
    PIDataSource,
    PostgresDataSource,
    SynapseDataSource,
    UnknownAuthKindError,
    UnknownKindError,
    UnknownTargetError,
)


def _cfg(targets):
    return {"data_sources": targets}


# ---------- Happy paths per kind ----------


def test_mssql_windows_trusted():
    cfg = _cfg(
        {
            "prod_sql": {
                "kind": "mssql",
                "auth": "windows_trusted",
                "db_server": "PROD-SQL01",
                "database": "Sales",
            }
        }
    )
    ds = ds_module.get("prod_sql", config=cfg)
    assert isinstance(ds, MSSQLDataSource)
    assert ds.target_id == "prod_sql"
    assert ds.kind == "mssql"
    assert ds.db_server == "PROD-SQL01"
    assert ds.database == "Sales"
    assert ds.trusted_connection is True
    assert ds.password is None
    cs = ds.odbc_connection_string
    assert "SERVER=PROD-SQL01" in cs
    assert "DATABASE=Sales" in cs
    assert "Trusted_Connection=yes" in cs


def test_mssql_defaults_minimal_config():
    cfg = _cfg({"t": {"kind": "mssql", "auth": "windows_trusted", "db_server": "X"}})
    ds = ds_module.get("t", config=cfg)
    assert ds.database is None
    assert ds.trusted_connection is True
    assert ds.trust_server_certificate is True
    assert "DATABASE=" not in ds.odbc_connection_string


def test_synapse_azure_keyvault(monkeypatch):
    cfg = _cfg(
        {
            "azure_synapse": {
                "kind": "synapse",
                "auth": "azure_keyvault",
                "key_vault": "kv-an",
                "secret": "synapse-pw",
                "hostname": "mysynapse.sql.azuresynapse.net",
                "database": "warehouse",
                "port": 1433,
                "username": "svc_user",
            }
        }
    )
    monkeypatch.setattr(
        ds_module,
        "_resolve_keyvault_secret",
        lambda kv, s, runtime: "PW",
    )
    ds = ds_module.get("azure_synapse", config=cfg)
    assert isinstance(ds, SynapseDataSource)
    assert ds.kind == "synapse"
    assert ds.jdbc_url == (
        "jdbc:sqlserver://mysynapse.sql.azuresynapse.net:1433;database=warehouse"
    )
    assert ds.jdbc_properties["user"] == "svc_user"
    assert ds.jdbc_properties["password"] == "PW"
    assert "PW" in ds.odbc_connection_string


def test_blob_azure_keyvault(monkeypatch):
    cfg = _cfg(
        {
            "example_blob": {
                "kind": "blob",
                "auth": "azure_keyvault",
                "key_vault": "kv-example",
                "secret": "blob-key",
                "storage_account": "sadatascienceexample",
            }
        }
    )
    monkeypatch.setattr(
        ds_module,
        "_resolve_keyvault_secret",
        lambda kv, s, runtime: "FAKE_KEY",
    )
    ds = ds_module.get("example_blob", config=cfg)
    assert isinstance(ds, BlobDataSource)
    assert ds.account_key == "FAKE_KEY"
    assert ds.storage_account == "sadatascienceexample"
    assert ds.spark_conf_key == ("fs.azure.account.key.sadatascienceexample.blob.core.windows.net")
    assert ds.wasbs_uri("mycontainer", "path/to/file.parquet") == (
        "wasbs://mycontainer@sadatascienceexample.blob.core.windows.net/path/to/file.parquet"
    )
    assert "FAKE_KEY" in ds.account_connection_string


def test_adls_azure_keyvault(monkeypatch):
    cfg = _cfg(
        {
            "lake": {
                "kind": "adls",
                "auth": "azure_keyvault",
                "key_vault": "kv",
                "secret": "k",
                "storage_account": "datalake01",
            }
        }
    )
    monkeypatch.setattr(
        ds_module,
        "_resolve_keyvault_secret",
        lambda kv, s, runtime: "K",
    )
    ds = ds_module.get("lake", config=cfg)
    assert isinstance(ds, ADLSDataSource)
    assert ds.spark_conf_key == "fs.azure.account.key.datalake01.dfs.core.windows.net"
    assert ds.abfss_uri("c", "p/f") == ("abfss://c@datalake01.dfs.core.windows.net/p/f")


def test_pi_oauth_post(monkeypatch):
    cfg = _cfg(
        {
            "pi": {
                "kind": "pi",
                "auth": "azure_keyvault",
                "key_vault": "kv",
                "secret": "pi-secret",
                "url": "https://pi.example/oauth/token",
                "grant_type": "client_credentials",
                "client_id": "cid",
                "scope": "scope-x",
            }
        }
    )
    monkeypatch.setattr(
        ds_module,
        "_resolve_keyvault_secret",
        lambda kv, s, runtime: "OAUTH_SECRET",
    )

    posted = {}

    class FakeResponse:
        def json(self):
            return {"access_token": "BEARER123"}

    def fake_post(url, data):
        posted["url"] = url
        posted["data"] = data
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    ds = ds_module.get("pi", config=cfg)
    assert isinstance(ds, PIDataSource)
    assert ds.bearer_token == "BEARER123"
    assert posted["url"] == "https://pi.example/oauth/token"
    assert posted["data"]["client_secret"] == "OAUTH_SECRET"
    assert posted["data"]["client_id"] == "cid"


def test_postgres_inline_password():
    cfg = _cfg(
        {
            "local_pg": {
                "kind": "postgres",
                "auth": "inline_password",
                "host": "localhost",
                "port": 5432,
                "user": "alice",
                "password": "s3cret",
                "database": "appdb",
            }
        }
    )
    ds = ds_module.get("local_pg", config=cfg)
    assert isinstance(ds, PostgresDataSource)
    assert ds.password == "s3cret"
    assert "alice" in ds.sqlalchemy_url
    assert "appdb" in ds.sqlalchemy_url


# ---------- Repr never leaks secrets ----------


def test_password_not_in_repr_mssql():
    ds = MSSQLDataSource(
        target_id="t",
        db_server="x",
        trusted_connection=False,
        username="u",
        password="SHHH",
    )
    assert "SHHH" not in repr(ds)


def test_account_key_not_in_repr_blob():
    ds = BlobDataSource(target_id="t", storage_account="acct", account_key="SHHH")
    assert "SHHH" not in repr(ds)


def test_bearer_not_in_repr_pi():
    ds = PIDataSource(target_id="t", url="https://x", bearer_token="SHHH")
    assert "SHHH" not in repr(ds)


# ---------- Error paths ----------


def test_unknown_target():
    with pytest.raises(UnknownTargetError, match="missing"):
        ds_module.get("missing", config=_cfg({}))


def test_unknown_auth_kind():
    cfg = _cfg({"t": {"kind": "mssql", "auth": "magic", "db_server": "x"}})
    with pytest.raises(UnknownAuthKindError, match="magic"):
        ds_module.get("t", config=cfg)


def test_unknown_kind():
    cfg = _cfg({"t": {"kind": "alien", "auth": "windows_trusted"}})
    with pytest.raises(UnknownKindError, match="alien"):
        ds_module.get("t", config=cfg)


def test_azure_keyvault_missing_fields():
    cfg = _cfg({"t": {"kind": "blob", "auth": "azure_keyvault", "storage_account": "x"}})
    with pytest.raises(Exception, match="key_vault"):
        ds_module.get("t", config=cfg)


# ---------- Runtime override ----------


def test_runtime_override_passed_to_resolver(monkeypatch):
    cfg = _cfg(
        {
            "b": {
                "kind": "blob",
                "auth": "azure_keyvault",
                "key_vault": "kv",
                "secret": "s",
                "storage_account": "acct",
            }
        }
    )
    seen = {}

    def fake_resolve(kv, secret, runtime):
        seen["runtime"] = runtime
        return "K"

    monkeypatch.setattr(ds_module, "_resolve_keyvault_secret", fake_resolve)
    ds_module.get("b", config=cfg, runtime="local")
    assert seen["runtime"] == "local"


def test_runtime_defaults_to_databricks(monkeypatch):
    cfg = _cfg(
        {
            "b": {
                "kind": "blob",
                "auth": "azure_keyvault",
                "key_vault": "kv",
                "secret": "s",
                "storage_account": "acct",
            }
        }
    )
    seen = {}
    monkeypatch.delenv("DSTOOLBOX_RUNTIME", raising=False)
    monkeypatch.setattr(
        ds_module,
        "_resolve_keyvault_secret",
        lambda kv, s, runtime: seen.setdefault("runtime", runtime) or "K",
    )
    ds_module.get("b", config=cfg)
    assert seen["runtime"] == "databricks"
