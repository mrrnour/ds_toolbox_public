# dstoolbox

A grab-bag of Python utilities for day-to-day data science and ML — I/O against Azure/MSSQL/Synapse/Delta/PI, ML training/scoring, NLP/LLM tagging, RAG pipelines, Spark ETL, SPC, and pandas/plotting helpers. This file captures the language we use to talk about the architecture; refactors should read it before introducing new names.

## Language

### I/O and credentials seam

**DataSource**:
An authenticated, addressable handle to a piece of data infrastructure (an MSSQL server, a blob container, a Synapse warehouse, the PI Web API, an ADLS Gen2 account). Lives as a typed dataclass per kind — `MSSQLDataSource`, `SynapseDataSource`, `BlobDataSource`, `PIDataSource`, `ADLSDataSource`. Carries connection materials (server, database, account URL, bearer token, etc.) but does **not** open clients itself.
_Avoid_: "connection" (implies a live client), "credentials" (too narrow — DataSources carry addressing info too), "backend" (overloaded), "target" (too vague).

**Target id**:
The user-facing string that identifies a `DataSource` (e.g. `"azure_synapse"`, `"onprem_sales_db"`). Single global namespace under `data_sources:` in `config.yml`.
_Avoid_: "connection id", "vault key", "server id" (each was used in a single legacy module — superseded).

**auth kind**:
The *how* of secret resolution for a `DataSource` entry. Currently two values: `azure_keyvault`, `windows_trusted`. Each maps 1:1 to an auth adapter behind the `data_sources` seam.
_Avoid_: "platform" (overloaded — used historically for `databricks`/`aml`/`local`, which is the *runtime* context, not the auth mechanism).

**Auth adapter**:
The concrete thing behind the `data_sources` seam that knows how to resolve one `auth kind` into a populated `DataSource`. `AzureKeyVaultAdapter` reads a secret from Key Vault. `WindowsTrustedAuthAdapter` does no secret resolution and produces a trusted-auth connection string.
_Avoid_: "credential provider", "connector class".

**Layer 2 conveniences**:
Thin helpers that open a live client from a `DataSource`. Currently `db.engine(ds)` and `db.pyodbc_connection(ds)`. Layer 2 exists only where there is one obvious right client. Blob and PI do **not** get Layer 2 helpers — callers use the DataSource materials directly with whichever SDK fits.

## Relationships

- A **DataSource** is identified by exactly one **target id**.
- A **DataSource** entry declares exactly one **auth kind**.
- An **auth kind** is handled by exactly one **Auth adapter**.
- `data_sources.get(target_id) -> DataSource` is the only public entry point for resolving credentials and connection materials.
- **Layer 2 conveniences** consume a `DataSource` and produce a live client; they never resolve secrets themselves.

## Example dialogue

> **Dev:** "How do I query the on-prem sales DB?"
> **Maintainer:** "Add an entry to `data_sources:` in `config.yml` with `kind: mssql` and `auth: windows_trusted`, then `ds = data_sources.get('onprem_sales_db'); engine = db.engine(ds); pd.read_sql(q, engine)`."
> **Dev:** "What if it's a Synapse warehouse with a Key Vault password?"
> **Maintainer:** "Same shape — `kind: synapse`, `auth: azure_keyvault`, plus `key_vault` and `secret` fields. The auth adapter resolves the secret; you still get a `SynapseDataSource` back and call `db.engine(ds)`."
> **Dev:** "And for blob?"
> **Maintainer:** "`kind: blob`, `auth: azure_keyvault`. But no `db.engine` — you take the `BlobDataSource`'s account URL and key and hand them to whichever blob SDK you're using."

## Flagged ambiguities

- "platform" was used ambiguously to mean both the runtime (`databricks`/`aml`/`local`) and the auth path. Resolved: runtime context stays as `platform`; auth path is **auth kind**.
- "credentials" was used for both the secret value and the full connection bundle. Resolved: a **DataSource** is the bundle; the bare secret has no dedicated name and lives only inside an auth adapter.
