"""dstoolbox.web_reader — URL → Markdown pipeline.

Four cooperating modules, invoked either as importable modules or as
``python -m dstoolbox.web_reader.<name>`` command-line entry points:

- ``scraper``     — fetch HTML / PDF from a list of URLs.
- ``harvest``     — auth-aware harvester (basic / NTLM) for gated pages.
- ``convert``     — turn crawled HTML / PDF into clean Markdown.
- ``run_pipeline`` — orchestrate the four stages end-to-end with
  checkpointing.

See ``dstoolbox.web_reader.params`` for the shared filesystem layout and
``dstoolbox.web_reader.utils`` for the pipeline record schema.
"""
