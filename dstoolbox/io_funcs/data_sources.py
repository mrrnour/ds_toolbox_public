"""The data_sources seam: resolve a target_id to a typed DataSource.

Single public entry point: ``data_sources.get(target_id)``. See CONTEXT.md and
docs/adr/0001-data-sources-seam.md for the architecture decisions behind this
module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

# === Errors ===


class DataSourceError(Exception):
    """Base for data_sources seam errors."""


class UnknownTargetError(DataSourceError):
    """Raised when target_id is not defined in config."""


class UnknownAuthKindError(DataSourceError):
    """Raised when entry['auth'] has no registered adapter."""


class UnknownKindError(DataSourceError):
    """Raised when entry['kind'] has no registered builder."""


# === DataSource dataclasses ===


@dataclass(frozen=True)
class MSSQLDataSource:
    kind: ClassVar[str] = "mssql"
    target_id: str
    db_server: str
    database: str | None = None
    trusted_connection: bool = True
    trust_server_certificate: bool = True
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    driver: str = "ODBC Driver 17 for SQL Server"

    @property
    def odbc_connection_string(self) -> str:
        parts = [f"DRIVER={{{self.driver}}}", f"SERVER={self.db_server}"]
        if self.database:
            parts.append(f"DATABASE={self.database}")
        if self.trusted_connection:
            parts.append("Trusted_Connection=yes")
        elif self.username and self.password:
            parts.append(f"UID={self.username}")
            parts.append(f"PWD={self.password}")
        if self.trust_server_certificate:
            parts.append("TrustServerCertificate=yes")
        return ";".join(parts) + ";"


@dataclass(frozen=True)
class SynapseDataSource:
    kind: ClassVar[str] = "synapse"
    target_id: str
    hostname: str
    database: str
    port: int
    username: str
    password: str = field(repr=False)
    driver: str = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    driver_odbc: str = "{ODBC Driver 17 for SQL Server}"

    @property
    def jdbc_url(self) -> str:
        return f"jdbc:sqlserver://{self.hostname}:{self.port};database={self.database}"

    @property
    def jdbc_properties(self) -> dict[str, str]:
        return {"user": self.username, "password": self.password, "driver": self.driver}

    @property
    def odbc_connection_string(self) -> str:
        return (
            f"DRIVER={self.driver_odbc};SERVER={self.hostname};PORT={self.port};"
            f"DATABASE={self.database};UID={self.username};PWD={self.password};"
            "MARS_Connection=yes"
        )


@dataclass(frozen=True)
class BlobDataSource:
    kind: ClassVar[str] = "blob"
    target_id: str
    storage_account: str
    account_key: str = field(repr=False)

    @property
    def account_connection_string(self) -> str:
        return (
            f"DefaultEndpointsProtocol=https;AccountName={self.storage_account};"
            f"AccountKey={self.account_key};EndpointSuffix=core.windows.net"
        )

    @property
    def spark_conf_key(self) -> str:
        return f"fs.azure.account.key.{self.storage_account}.blob.core.windows.net"

    def wasbs_uri(self, container: str, blob: str) -> str:
        return f"wasbs://{container}@{self.storage_account}.blob.core.windows.net/{blob}"


@dataclass(frozen=True)
class ADLSDataSource:
    kind: ClassVar[str] = "adls"
    target_id: str
    storage_account: str
    account_key: str = field(repr=False)

    @property
    def spark_conf_key(self) -> str:
        return f"fs.azure.account.key.{self.storage_account}.dfs.core.windows.net"

    def abfss_uri(self, container: str, path: str) -> str:
        return f"abfss://{container}@{self.storage_account}.dfs.core.windows.net/{path}"


@dataclass(frozen=True)
class PIDataSource:
    kind: ClassVar[str] = "pi"
    target_id: str
    url: str
    bearer_token: str = field(repr=False)


@dataclass(frozen=True)
class PostgresDataSource:
    kind: ClassVar[str] = "postgres"
    target_id: str
    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str
    sslmode: str | None = None

    @property
    def sqlalchemy_url(self) -> str:
        from urllib.parse import quote_plus

        base = (
            f"postgresql://{quote_plus(self.user)}:{quote_plus(self.password)}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        if self.sslmode:
            base += f"?sslmode={self.sslmode}"
        return base


# === Auth adapters: entry -> Optional[secret string] ===


def _fetch_via_azure_keyvault(entry: dict[str, Any]) -> str:
    kv_name = entry.get("key_vault")
    secret_name = entry.get("secret")
    if not kv_name or not secret_name:
        raise DataSourceError("azure_keyvault auth requires 'key_vault' and 'secret' fields")
    runtime = entry.get("runtime") or os.environ.get("DSTOOLBOX_RUNTIME", "databricks")
    return _resolve_keyvault_secret(kv_name, secret_name, runtime)


def _fetch_via_windows_trusted(entry: dict[str, Any]) -> None:
    return None


def _fetch_via_inline_password(entry: dict[str, Any]) -> str:
    pwd = entry.get("password")
    if pwd is None:
        raise DataSourceError("inline_password auth requires 'password' field")
    return pwd


_AUTH_ADAPTERS: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "azure_keyvault": _fetch_via_azure_keyvault,
    "windows_trusted": _fetch_via_windows_trusted,
    "inline_password": _fetch_via_inline_password,
}


def _resolve_keyvault_secret(kv_name: str, secret_name: str, runtime: str) -> str:
    """Fetch a secret value from Azure Key Vault, dispatching on runtime.

    Indirected through a module-level function so tests can monkeypatch it.
    """
    if runtime == "databricks":
        from .bootstrap import get_dbutils

        return get_dbutils().secrets.get(scope=kv_name, key=secret_name)

    if runtime == "aml":
        ml_app_id = os.environ.get("AZURE_ML_APP_ID")
        if not ml_app_id:
            raise DataSourceError("aml runtime requires AZURE_ML_APP_ID env var")
        from azure.identity import ManagedIdentityCredential

        credential = ManagedIdentityCredential(client_id=ml_app_id)
        credential.get_token("https://vault.azure.net/.default")
        return _kv_get(kv_name, secret_name, credential)

    if runtime in ("local", "vm_docker"):
        from azure.identity import DefaultAzureCredential

        return _kv_get(kv_name, secret_name, DefaultAzureCredential())

    raise DataSourceError(f"unknown runtime: {runtime!r}")


def _kv_get(kv_name: str, secret_name: str, credential: Any) -> str:
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(
        vault_url=f"https://{kv_name}.vault.azure.net",
        credential=credential,
    )
    return client.get_secret(secret_name).value


# === Kind builders: (entry, secret, target_id) -> DataSource ===


def _build_mssql(entry, secret, target_id):
    return MSSQLDataSource(
        target_id=target_id,
        db_server=entry["db_server"],
        database=entry.get("database"),
        trusted_connection=entry.get("trusted_connection", True),
        trust_server_certificate=entry.get("trust_server_certificate", True),
        username=entry.get("username"),
        password=secret,
    )


def _build_synapse(entry, secret, target_id):
    return SynapseDataSource(
        target_id=target_id,
        hostname=entry["hostname"],
        database=entry["database"],
        port=entry["port"],
        username=entry["username"],
        password=secret,
        driver=entry.get("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver"),
        driver_odbc=entry.get("driver_odbc", "{ODBC Driver 17 for SQL Server}"),
    )


def _build_blob(entry, secret, target_id):
    return BlobDataSource(
        target_id=target_id,
        storage_account=entry["storage_account"],
        account_key=secret,
    )


def _build_adls(entry, secret, target_id):
    return ADLSDataSource(
        target_id=target_id,
        storage_account=entry["storage_account"],
        account_key=secret,
    )


def _build_pi(entry, secret, target_id):
    import requests

    url = entry["url"]
    payload = {
        "grant_type": entry["grant_type"],
        "client_id": entry["client_id"],
        "scope": entry["scope"],
        "client_secret": secret,
    }
    response = requests.post(url, data=payload)
    token = response.json().get("access_token")
    if not token:
        raise DataSourceError(f"PI OAuth response did not include access_token for {target_id!r}")
    return PIDataSource(target_id=target_id, url=url, bearer_token=token)


def _build_postgres(entry, secret, target_id):
    return PostgresDataSource(
        target_id=target_id,
        host=entry["host"],
        port=entry.get("port", 5432),
        user=entry["user"],
        password=secret,
        database=entry["database"],
        sslmode=entry.get("sslmode"),
    )


_KIND_BUILDERS: dict[str, Callable[..., Any]] = {
    "mssql": _build_mssql,
    "synapse": _build_synapse,
    "blob": _build_blob,
    "adls": _build_adls,
    "pi": _build_pi,
    "postgres": _build_postgres,
}


# === Public entry point ===


def get(
    target_id: str,
    runtime: str | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Resolve ``target_id`` to a typed ``DataSource``.

    Parameters
    ----------
    target_id : str
        Key under ``data_sources:`` in config.yml.
    runtime : str, optional
        Override the runtime for Azure Key Vault resolution. One of
        ``databricks``, ``aml``, ``local``, ``vm_docker``. Falls back to the
        ``DSTOOLBOX_RUNTIME`` env var, then to ``databricks``.
    config : dict, optional
        Parsed config dict (mainly for tests). Default: load the bundled
        ``config.yml``.

    Returns
    -------
    DataSource
        A frozen dataclass of the right kind: ``MSSQLDataSource``,
        ``SynapseDataSource``, ``BlobDataSource``, ``ADLSDataSource``,
        ``PIDataSource``, or ``PostgresDataSource``.
    """
    if config is None:
        config = _load_bundled_config()

    entries = config.get("data_sources") or {}
    if target_id not in entries:
        raise UnknownTargetError(
            f"target_id {target_id!r} not found under 'data_sources:' in config"
        )

    entry = dict(entries[target_id])
    if runtime is not None:
        entry["runtime"] = runtime

    auth = entry.get("auth")
    kind = entry.get("kind")

    if auth not in _AUTH_ADAPTERS:
        raise UnknownAuthKindError(
            f"unknown auth kind {auth!r} for target {target_id!r}; "
            f"valid: {sorted(_AUTH_ADAPTERS)}"
        )
    if kind not in _KIND_BUILDERS:
        raise UnknownKindError(
            f"unknown kind {kind!r} for target {target_id!r}; " f"valid: {sorted(_KIND_BUILDERS)}"
        )

    secret = _AUTH_ADAPTERS[auth](entry)
    return _KIND_BUILDERS[kind](entry, secret, target_id)


def _load_bundled_config() -> dict[str, Any]:
    from importlib import resources as res

    import yaml

    with res.open_binary("dstoolbox", "config.yml") as fp:
        return yaml.safe_load(fp) or {}
