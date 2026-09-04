"""dstoolbox — data-science utilities.

A grab-bag of Python helpers for day-to-day data-science and ML work —
I/O against Azure / MSSQL / Synapse / Delta / PI Web API, model training
and scoring, NLP + LLM tagging, RAG pipelines (crawl → convert → chunk →
vector store), Spark ETL, statistical process control, and a long tail of
pandas / plotting / stats helpers.

The intent is small: bundle the things you'd otherwise copy-paste into
every notebook into one importable package, so a new project starts with
`import dstoolbox` and goes.

Package map
-----------

```mermaid
flowchart LR
    root(("dstoolbox"))

    utils["utils<br/><small>lists · text · datetime · paths ·<br/>logging · sql · encoding · dataframes ·<br/>stats · plots</small>"]
    io["io_funcs<br/><small>azure · mssql · synapse · delta ·<br/>pi web api · blob · templates ·<br/>data_sources seam</small>"]
    ml["ml_funcs<br/><small>training · scoring · tuning ·<br/>performance plots · SHAP · PCA ·<br/>time-series EDA + forecasters</small>"]
    spark["spark_funcs<br/><small>asof joins · reshape · features ·<br/>incremental ETL · geo</small>"]
    nlp["nlp_llm_funcs<br/><small>cleaning · anonymization ·<br/>fuzzy + embedding similarity ·<br/>LLM tagging</small>"]
    rag["rag_funcs<br/><small>docling conversion ·<br/>chunking · vector store</small>"]
    web["web_reader<br/><small>URL → Markdown CLI</small>"]

    root --> utils
    root --> io
    root --> ml
    root --> spark
    root --> nlp
    root --> rag
    root --> web

    utils -. shared helpers .-> io
    utils -. shared helpers .-> ml
    utils -. shared helpers .-> spark
    utils -. shared helpers .-> nlp
    nlp  -. embeddings .-> rag
    web  -. feeds .-> rag

    classDef core fill:#FFE0B2,stroke:#E75B12,color:#3E2723,stroke-width:1.5px;
    classDef data fill:#B3E5FC,stroke:#0277BD,color:#0D47A1,stroke-width:1.5px;
    classDef ml   fill:#C8E6C9,stroke:#2E7D32,color:#1B5E20,stroke-width:1.5px;
    classDef big  fill:#FFF9C4,stroke:#F9A825,color:#5D4037,stroke-width:1.5px;
    classDef text fill:#E1BEE7,stroke:#6A1B9A,color:#311B92,stroke-width:1.5px;

    class utils core;
    class io data;
    class ml ml;
    class spark big;
    class nlp,rag,web text;
```

Subpackages
-----------

- `dstoolbox.utils` — general-purpose helpers (lists, text, datetime,
  paths, logging, SQL, encoding, dataframes, stats — including
  Statistical Process Control sigma / I-MR limits — and plots —
  including I-MR and series-overlay charts).
- `dstoolbox.io_funcs` — Azure / MSSQL / Synapse / Delta / PI Web API
  I/O and the `data_sources` credential seam.
- `dstoolbox.ml_funcs` — model training, scoring, plotting, tuning.
- `dstoolbox.spark_funcs` — Spark ETL helpers.
- `dstoolbox.nlp_llm_funcs` — text cleaning, similarity, LLM tagging.
- `dstoolbox.rag_funcs` — document conversion, chunking, vector store.
- `dstoolbox.web_reader` — URL → Markdown pipeline
  (`scraper`, `harvest`, `convert`, `run_pipeline`), also exposed as
  the `webreader-scrape` / `-harvest` / `-convert` / `-pipeline`
  console scripts.

Design notes
------------

- **Small imports, optional heavyweights.** The top-level package does
  lazy `try`/`except` imports so a missing extra (spark, hyperopt,
  docling, ...) never breaks `import dstoolbox`.
- **One credential seam.** Every I/O path funnels through
  `dstoolbox.io_funcs.data_sources.get`. Notebooks never touch raw
  connection strings; tests swap in a dict-backed fake.
- **Numpy-style docstrings everywhere.** This site is generated straight
  from the source docstrings by [pdoc](https://pdoc.dev/) — keep them
  accurate and they will render here.

Install
-------

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[all]"        # everything except dev / test
    pip install -e ".[azure,ml]"   # or pick only what you need
"""

__version__ = "0.4.0"

# Subpackages are imported lazily via try/except so that missing optional
# extras (spark, hyperopt, etc.) don't break `import dstoolbox` — but pdoc
# and IDE tooling can still discover them when the extras are installed.
__all__ = ["__version__"]

for _subpkg in (
    "utils",
    "io_funcs",
    "ml_funcs",
    "spark_funcs",
    "nlp_llm_funcs",
    "rag_funcs",
    "web_reader",
):
    try:
        __import__(f"{__name__}.{_subpkg}")
    except Exception:  # noqa: BLE001,S110 — optional extras may fail to import
        pass
    else:
        # Expose successfully imported subpackages so pdoc lists them as
        # Submodules in the top-level sidebar.
        __all__.append(_subpkg)
del _subpkg
