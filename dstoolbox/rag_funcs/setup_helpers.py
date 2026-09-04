"""
RAG (Retrieval-Augmented Generation) utility functions for document processing,
web scraping, chunking, and vector store operations.
"""

# Standard library imports
import json
import logging
import os
from typing import Any

# Third-party imports
import requests
import yaml

# Docling imports
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
    SimplePipeline,
    StandardPdfPipeline,
    WordFormatOption,
)

# LangChain imports
from langchain_core.documents import Document

# Local imports


# -------------------------------------------------------------------------
# General utility functions
# -------------------------------------------------------------------------
def setup_http_session(
    credentials: dict[str, str],
    verify_ssl: bool = False,
    auth_type: str = "ntlm",  # 'none', 'basic', 'ntlm'
) -> requests.Session:
    """
    Set up an HTTP session with the appropriate configuration.

    Args:
        credentials: Dictionary containing username and password
        verify_ssl: Whether to verify SSL certificates
        auth_type: Authentication type to use

    Returns:
        Configured requests.Session object
    """
    import urllib3

    session = requests.Session()
    session.verify = verify_ssl

    # Disable SSL warnings if verification is disabled
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Set up authentication
    if auth_type == "ntlm":
        from requests_ntlm import HttpNtlmAuth

        session.auth = HttpNtlmAuth(
            credentials.get("username", ""), credentials.get("password", "")
        )
    elif auth_type == "basic":
        session.auth = (credentials.get("username", ""), credentials.get("password", ""))

    # Set headers to mimic a browser
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    )

    return session


def setup_docling(artifacts_path: str) -> DocumentConverter:
    """
    Configure document converter with advanced processing capabilities.

    Args:
        artifacts_path: Path to store OCR and other processing artifacts

    Returns:
        Configured DocumentConverter instance
    """
    # Create pipeline options with OCR and table structure recognition
    pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["en"]
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # Configure hardware acceleration if available
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.AUTO,  # Auto-select best available device
    )

    # Configure format-specific options
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
            backend=PyPdfiumDocumentBackend,
            pipeline_options=pipeline_options,
        ),
        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
    }

    # Create and return the document converter
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
            InputFormat.XLSX,
        ],
        format_options=format_options,
    )


def load_document(file_path: str, logger: logging.Logger) -> list[dict[str, Any]] | None:
    """
    Load a document based on its file extension.

    Args:
        file_path: Path to the document file
        logger: Logger instance for tracking events

    Returns:
        List of document dictionaries or None if loading fails
    """
    from langchain.document_loaders import UnstructuredMarkdownLoader

    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        elif file_ext == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return [
                    {"page_content": str(item), "metadata": {"source": file_path}} for item in data
                ]
            elif isinstance(data, dict):
                content_field = data.get("content", str(data))
                return [{"page_content": content_field, "metadata": {"source": file_path}}]
            else:
                return [{"page_content": str(data), "metadata": {"source": file_path}}]

        elif file_ext in (".yaml", ".yml"):
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Similar handling as JSON
            if isinstance(data, list):
                return [
                    {"page_content": str(item), "metadata": {"source": file_path}} for item in data
                ]
            elif isinstance(data, dict):
                content_field = data.get("content", str(data))
                return [{"page_content": content_field, "metadata": {"source": file_path}}]
            else:
                return [{"page_content": str(data), "metadata": {"source": file_path}}]

        else:
            logger.warning(f"Unsupported file format: {file_ext} for file {file_path}")
            return None

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        return None


def get_metadata_file(
    doc_path: str, metadata_dir: str, logger: logging.Logger
) -> dict[str, Any] | None:
    """
    Find and load metadata for a document.

    Args:
        doc_path: Path to the document file
        metadata_dir: Directory containing metadata files
        logger: Logger instance

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        if not metadata_dir or not os.path.isdir(metadata_dir):
            return None

        # Get document filename without extension
        doc_name = os.path.splitext(os.path.basename(doc_path))[0]

        # Look for metadata file with matching name
        potential_metadata_files = [
            os.path.join(metadata_dir, f"{doc_name}.meta.json"),
            os.path.join(metadata_dir, f"{doc_name}.meta.yaml"),
            os.path.join(metadata_dir, f"{doc_name}.meta.yml"),
        ]

        for meta_path in potential_metadata_files:
            if os.path.exists(meta_path):
                logger.info("Metadata found")
                ext = os.path.splitext(meta_path)[1].lower()

                try:
                    if ext == ".json":
                        with open(meta_path, encoding="utf-8") as f:
                            return json.load(f)
                    elif ext in (".yaml", ".yml"):
                        with open(meta_path, encoding="utf-8") as f:
                            return yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"Error loading metadata file {meta_path}: {str(e)}")
                    continue

        return {}

    except Exception as e:
        logger.error(f"Error finding metadata for {doc_path}: {str(e)}")
        return None


def save_metadata(
    save_location: str, base_name: str, metadata: dict[str, Any], logger: logging.Logger
) -> str | None:
    """
    Save document metadata to a separate JSON file.

    Args:
        save_location: Base output directory
        base_name: Base name of the file without extension
        metadata: Document metadata dictionary
        logger: Logger instance

    Returns:
        Path to the saved metadata file or None if saving failed
    """
    try:
        # Create metadata directory
        metadata_dir = os.path.join(save_location, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Create full output path
        metadata_path = os.path.join(metadata_dir, f"{base_name}.meta.json")

        # Save metadata as JSON with error handling
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        return metadata_path

    except Exception as e:
        logger.error(f"Error saving metadata for {base_name}: {str(e)}")
        return None


# -------------------------------------------------------------------------
# Utility functions for loading documents
# -------------------------------------------------------------------------
def load_langchain_docs_from_jsonl(file_path: str) -> list[Document]:
    """
    Load LangChain documents from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of LangChain Document objects
    """
    documents = []
    with open(file_path) as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            documents.append(Document(**data))
    return documents
