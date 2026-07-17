<p align="center">
  <img src="images/ds_toolbox_logo.png" alt="DS Toolbox" width="380"/>
</p>

# DS Toolbox

> **Looking for the top-level project README?** See [../README.md](../README.md) for install steps, extras matrix, quick sanity check, and CI/dev workflow. This page is the deep-dive: per-module samples, config, and RAG / web_reader walkthroughs.

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Azure](<https://img.shields.io/badge/microsoft%20azure-0089D0?style=for-the-badge&logo=microsoft-azure&logoColor=white>)](https://azure.microsoft.com/)
[![Apache Spark](<https://img.shields.io/badge/Apache%20Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black>)](https://spark.apache.org/)

A grab-bag of Python utilities used across day-to-day data science and ML work — moving data in and out of Azure / MSSQL / Synapse / Delta / PI Web API, training and evaluating models, NLP and LLM tagging, RAG pipelines (crawl → convert → chunk → vector store), Spark ETL, statistical process control, and a long tail of pandas / plotting / stats helpers.

The intent is small: bundle the things you'd otherwise copy-paste into every notebook into one importable package, so a new project starts with `import dstoolbox` and goes.

---

## 🧭 What's inside

| Package           | What it covers                                                                                                                                                                                                                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `io_funcs`      | Azure (Synapse, Blob, ADLS, Key Vault), Databricks Delta, PI Web API, MSSQL, Postgres, named SQL templates, Colab/Kaggle bootstrap. All credentials flow through a single seam:`data_sources.get(target_id)` returns a typed `DataSource`; `db.engine(ds)` / `db.pyodbc_connection(ds)` open live clients |
| `ml_funcs`      | Model templates, training (incl. nested CV), scoring, performance plots (ROC, PR, gain/lift, reliability, confusion matrix), tuning, PCA/CCA, SHAP/PDP/feature importance, multilabel, regression assumption checks                                                                                               |
| `utils`         | Lists, text/regex, datetime, paths, logging, SQL parsing, web download, encoding, dataframe ops, stats (incl. Statistical Process Control sigma / I-MR limits), plotly/matplotlib helpers (incl. I-MR and series-overlay charts)                                                                                                                                                                                 |
| `spark_funcs`   | Asof joins, dataframe reshape utils, column finder, incremental ETL pipelines, rolling/tumbling time-series features                                                                                                                                                                                              |
| `nlp_llm_funcs` | Text cleaning + anonymization, fuzzy/embedding similarity, LLM tagging chains                                                                                                                                                                                                                                     |
| `rag_funcs`     | RAG library helpers: docling-based document conversion (`custom_converter`), chunking, vector store — each stage exposed as a `*_Config` / `*_Error` / `*_Processor` triple                                                                                                                              |
| `web_reader`    | Integrated CLI subpackage for the URL → Markdown pipeline: `scraper.py` (fetch URLs), `harvest.py` (intranet listing-page file harvest with Basic/NTLM auth), `convert.py` (HTML/PDF → Markdown), `run_pipeline.py` (one-shot wrapper). All modules share a `PipelineRecord` JSONL audit trail and are exposed as `webreader-scrape` / `-harvest` / `-convert` / `-pipeline` console scripts. |

The repo itself is the importable `dstoolbox` package (see `setup.cfg`).

---

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/mrrnours/ds_toolbox_public.git
cd ds_toolbox_public
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell
python -m pip install --upgrade pip setuptools wheel
```

`dstoolbox` installs entirely via `pip` — **no conda is used**. Python 3.9+ is required (`setup.cfg`: `python_requires = >=3.9`).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Pinned versions cover the core stack: pandas, scipy, scikit-learn, xgboost, plotly, matplotlib, seaborn, SQLAlchemy, pyodbc, azure-* SDKs, langchain, docling, etc.

For Databricks clusters, use the matching pin set:

```bash
pip install -r requirement__dBricks.txt
```

### 4. Install the package

From the repo root:

```bash
pip install -e .
```

Editable mode is convenient while iterating; drop `-e` for a regular install.

### 5. Verify the install

Import the top-level packages and call a pure-Python helper to confirm things resolve:

```python
import dstoolbox
import dstoolbox.io_funcs as io_funcs
import dstoolbox.ml_funcs as ml_funcs
from dstoolbox import utils

assert utils.flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]
print(dstoolbox.__name__, "ok")
```

This only checks that the package and its hard dependencies import — actual platform calls (Azure, MSSQL, Spark, etc.) need credentials and a runtime that can reach those services.

---

## 📦 Package layout

```text
dstoolbox/
├── ml_funcs/        scores · templates · training · performance_plots ·
│                    feature_importance · tuning · pca · assumptions ·
│                    helpers · multilabel
├── io_funcs/        bootstrap · data_sources · db · synapse · delta ·
│                    templates · blob · pi · mssql · colab
├── utils/    lists · text · datetime_utils · paths · logging_utils ·
│                    sql · web · encoding · dataframes · stats · plots
├── spark_funcs/     joins · reshape · columns · col_finder · pipelines · features
├── nlp_llm_funcs/   text_utils · similarity · llm_tagging
├── rag_funcs/       setup_helpers · custom_converter · chunking · vectorstore
├── web_reader/      Integrated CLI subpackage: scraper.py + harvest.py + convert.py + run_pipeline.py (+ tools/)
├── config.yml       Default platform config
├── sql_template.yml Named SQL queries
└── requirements.txt
```

Each package's `__init__.py` re-exports its public names — both `from dstoolbox.io_funcs import blob2pd` and `from dstoolbox.io_funcs.blob import blob2pd` work.

---

## ⚡ Samples

### Query Synapse (Databricks)

```python
import dstoolbox.io_funcs as io_funcs

query = """(SELECT order_id, order_ts, customer_id, total_amount
            FROM analytics.vw_orders WITH (nolock)
            WHERE CAST(order_ts AS date)
                  BETWEEN '2023-01-01' AND '2023-01-15') query"""

df = io_funcs.query_synapse_db(
    query=query,
    target_id='azure_synapse',   # key under `data_sources:` in config.yml
    verbose=True,
)
```

### Run a named SQL template with parameters

```python
df = io_funcs.query_template_run(
    query_temp_name='vw_orders',
    replace_dict={'start___date': '2023-01-01', 'end___date': '2023-01-15'},
    runtime='databricks',
)
```

### Read / write Azure Blob (parquet, xlsx)

```python
df = io_funcs.blob2pd(
    blob_dict={'target_id': 'example_blob',    # key under `data_sources:`
               'container': 'analytics',
               'blob':      'exports/orders_15_new.parquet'},
    parse_dates=['order_ts'],
    usecols=['order_ts', 'customer_id', 'total_amount'],
    verbose=True,
)

io_funcs.xls2blob(
    {'sheet1': df.head(1000), 'sheet2': df.tail(1000)},
    blob_dict={'target_id': 'example_blob',
               'container': 'analytics',
               'blob':      'exports/sample.xlsx'},
    runtime='databricks',
)
```

### Pull tags from PI Web API

```python
tags = ['plant1_unit100-101-FIC-1310-1A.PV', 'plant1_unit100-101-FIC-1410-1A.PV']
df = io_funcs.pi2pd_raw_data(
    tags,
    start_date='2023-07-13', end_date='2023-08-10',
    target_id='webapi',
)
```

### Compare models on a time-series split

[`ml_comparison`](ml_funcs/training.py) runs a list of estimators through the same CV splitter and returns a per-fold + aggregate score table. Feed it a `TimeSeriesSplit` and it becomes a walk-forward evaluator; no extras are needed for regression on the base install:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from dstoolbox.ml_funcs.training import ml_comparison

rng = np.random.default_rng(0)
n = 200
X = pd.DataFrame({"lag1": rng.normal(size=n), "trend": np.arange(n) / n})
y = pd.Series(0.5 * X["lag1"] + 2.0 * X["trend"] + rng.normal(scale=0.3, size=n))

tscv = TimeSeriesSplit(n_splits=5)
models = [Ridge(alpha=1.0, random_state=0),
          GradientBoostingRegressor(random_state=0)]

metrics = ml_comparison(
    ml_models=models, X=X, y=y,
    scores_names=["R2", "mean_absolute_error", "mean_squared_error"],
    sk_fold=tscv,
    map_names={0: "Ridge", 1: "GBR"},
    plot=False, verbose=False, show_capabilities=False,
)
print(metrics[metrics["CV"] == "CV_scores_Mean"].to_string(index=False))
```

Expected output (rounded):

```text
model             CV        R2 mean_absolute_error mean_squared_error
Ridge CV_scores_Mean  0.534978            0.303646           0.145993
  GBR CV_scores_Mean  0.374834            0.330461           0.194721
```

The returned frame has one row per (model, fold) plus per-model aggregate rows (`CV_scores_Mean`, `CV_scores_STD`, `scores_all`) and an `elapsed_time` column. Pass `plot=True` to render a bar chart via [`ml_comparison_plot`](ml_funcs/performance_plots.py). Classification metrics use names like `accuracy`, `f1`, `roc_auc`, `mcc`, `kappa` — the full list lives in [`scores.metric_dict`](ml_funcs/scores.py).

### Spark asof join + rolling features

```python
import dstoolbox.spark_funcs as sp_funcs

joined = sp_funcs.asof_join_spark2(
    sp_left, sp_right,
    on='timestamp', by='equipment_id', direction='backward',
)

rolled = sp_funcs.create_rolling_features(
    df=sensor_df,
    timestamp_column='ts',
    groupby_column='equipment_id',
    window_duration='30 minutes',
    aggregation_type='avg',
)
```

### Statistical Process Control (I-MR chart)

```python
from dstoolbox.utils import i_mr_ctrl_limits, plot_I_MR

limits = i_mr_ctrl_limits(df, cols=['measurement'], grpby_col=['machine'], coef=3)
fig = plot_I_MR(df, limits=limits.set_index('machine').iloc[0], x_col='TimeStamp')
fig.show()
```

### web_reader CLI — URLs to Markdown

The [`web_reader/`](web_reader/) subpackage is a command-line pipeline. Four
modules that share a `PipelineRecord` JSONL audit trail, so any pair composes:

| Module | Console script | Role |
| --- | --- | --- |
| [`scraper.py`](web_reader/scraper.py) | `webreader-scrape` | Fetch each URL to disk. `--fetcher requests` (default) or `--fetcher stealthy` (needs `[webreader-stealth]`). Retry-on-429 with backoff, browser User-Agent, idempotent skip-on-existing. |
| [`harvest.py`](web_reader/harvest.py) | `webreader-harvest` | Treat each URL as a *listing page*. Scrapes `<a href>` links matching `--extensions` and downloads them per-URL with `.meta.json` sidecars. HTTP Basic / NTLM auth via env vars. |
| [`convert.py`](web_reader/convert.py) | `webreader-convert` | HTML/PDF (or anything via `--use-docling`) → Markdown with YAML frontmatter. |
| [`run_pipeline.py`](web_reader/run_pipeline.py) | `webreader-pipeline` | One-shot wrapper that runs the three stages above with checkpointing. |

```bash
pip install -e ".[webreader]"

webreader-pipeline urls.txt              # scrape + convert end-to-end
# or step-by-step:
webreader-scrape  urls.txt --max-workers 4
webreader-convert --input output/crawling_summary.jsonl

# intranet variant: harvest matching files from listing pages
echo "https://intranet/reports/" > listings.txt
webreader-harvest listings.txt --extensions pdf,docx --auth ntlm \
    --output-dir output/files
webreader-convert --input output/harvest_summary.jsonl
```

`urls.txt` and `listings.txt` are one URL per line; lines starting with `#`
are ignored. For `--auth basic` / `--auth ntlm`, set `WEB_READER_USER` and
`WEB_READER_PASSWORD` (see `web_reader/.env.example`). Full flag list and
troubleshooting in [`web_reader/README.md`](web_reader/README.md); per-module
API reference under [`docs/api/dstoolbox/web_reader.html`](../docs/api/dstoolbox/web_reader.html).

### RAG library API (chunking + vector store)

Once you have markdown files (from `web_reader` or anywhere else), the rest of the RAG pipeline is library code under `dstoolbox.rag_funcs`:

```python
from dstoolbox.rag_funcs import (
    ChunkingConfig,    ChunkProcessor,
    VectorstoreConfig, VectorStoreProcessor,
)
```

`ChunkProcessor` and `VectorStoreProcessor` follow the same `*Config` / `*ErrorRecord` / `*Processor` shape used elsewhere — see [`rag_funcs/chunking.py`](rag_funcs/chunking.py) and [`rag_funcs/vectorstore.py`](rag_funcs/vectorstore.py).

---

## 🔧 Configuration

`config.yml` (repo root) holds a single `data_sources:` namespace. Each entry declares its `kind` (what world it talks to — `mssql`, `synapse`, `blob`, `adls`, `pi`, `postgres`) and `auth` (how to resolve the secret — `azure_keyvault`, `windows_trusted`, `inline_password`). Every I/O helper resolves credentials through the same seam: `data_sources.get(target_id)`. See [`CONTEXT.md`](CONTEXT.md) and [`docs/adr/0001-data-sources-seam.md`](docs/adr/0001-data-sources-seam.md) for the rationale.

`sql_template.yml` holds named SQL queries; the template's `db:` field is interpreted as a `target_id`.

```yaml
data_sources:
  analytics_db:
    kind: mssql
    auth: windows_trusted
    db_server: analytics-sql.company.com
    database:  CustomerAnalytics
    trust_server_certificate: true

  analytics_synapse:
    kind: synapse
    auth: azure_keyvault
    key_vault: analytics-kv
    secret:    synapse-password
    hostname:  analytics-synapse.sql.azuresynapse.net
    database:  analytics_dw
    username:  analytics_user
    port:      1433

  analytics_blob:
    kind: blob
    auth: azure_keyvault
    key_vault: analytics-kv
    secret:    analytics-blob-key
    storage_account: analyticsstore
```

Resolving a target:

```python
from dstoolbox.io_funcs import data_sources, db

ds = data_sources.get('analytics_db')   # -> MSSQLDataSource
engine = db.engine(ds)                  # Layer 2: live SQLAlchemy engine

ds = data_sources.get('analytics_blob') # -> BlobDataSource
spark.conf.set(ds.spark_conf_key, ds.account_key)
uri = ds.wasbs_uri('processed-data', 'report.parquet')
```

The runtime (Databricks / AML / local / vm_docker) for Key Vault resolution is selected via the `DSTOOLBOX_RUNTIME` env var, or via the `runtime=` argument on each I/O call (defaults to `databricks`).

## 📝 License

GNU GPL v3 — see [LICENSE](LICENSE).
