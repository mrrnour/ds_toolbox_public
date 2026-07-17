"""
RAG (Retrieval-Augmented Generation) utility functions for document processing,
web scraping, chunking, and vector store operations.
"""

# Standard library imports
import json
import logging
import os, sys, shutil
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urldefrag
import datetime as dt
from datetime import datetime
import hashlib
from IPython.display import clear_output

# Third-party imports
import pandas as pd
import requests
import yaml
from html import unescape
from bs4 import BeautifulSoup
from tqdm import tqdm

# Local imports
from .. import utils
from .. import io_funcs as io_funcs

# Docling imports
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import BaseChunk
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode
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
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .setup_helpers import setup_docling, get_metadata_file

@dataclass
class ChunkingConfig:
    """Configuration inputs for :class:`ChunkProcessor`.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory to scan for source documents.
    save_location : pathlib.Path
        Destination directory for chunk JSON and error reports.
    supported_formats : list of str
        File extensions the pipeline is allowed to ingest (e.g. ``['.pdf', '.md']``).
    metadata_dir : str, optional
        Directory holding per-document metadata JSON files. If unset,
        metadata is inferred from the document itself.
    save_individual_chunks : bool, default True
        If True, write a ``<basename>_chunks.json`` alongside each
        processed document.
    overwrite : bool, default False
        If True, re-chunk files that already have a chunks file on disk.
    tokenizer : Any, optional
        Tokenizer forwarded to ``docling.chunking.HybridChunker``.
    artifacts_path : str, optional
        Path forwarded to :func:`setup_docling`.
    """
    input_dir: Path
    save_location: Path
    supported_formats: List[str]
    metadata_dir: Optional[str] = None
    save_individual_chunks: bool = True
    overwrite: bool = False
    tokenizer: Optional[Any] = None
    artifacts_path: Optional[str] = None

@dataclass
class ChunkingErrorRecord:
    """Record describing a single chunking failure.

    Parameters
    ----------
    file_path : str
        Source document that failed.
    error_message : str
        Human-readable error string.
    chunk_index : int, optional
        Index of the chunk being processed when the error occurred
        (unset for whole-document failures).
    chunk_timestamp : datetime.datetime
        Time the error was recorded. Defaults to ``dt.datetime.now()``.
    """
    file_path: str
    error_message: str
    chunk_index: Optional[int] = None
    chunk_timestamp: datetime = dt.datetime.now()

class ChunkProcessor:
    """Chunk documents with docling and export to LangChain / LanceDB formats.

    Wraps a docling :class:`DocumentConverter` and
    :class:`HybridChunker`, tracks per-document/per-chunk errors, and
    persists chunks either as consolidated JSON or one file per
    document.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for progress and error reporting.
    config : ChunkingConfig
        Pipeline configuration (input dir, save location, tokenizer, ...).

    Attributes
    ----------
    logger : logging.Logger
    config : ChunkingConfig
    chunking_errors : list of ChunkingErrorRecord
        Errors accumulated across all documents processed by this instance.
    doc_converter : docling.document_converter.DocumentConverter or None
        Lazily initialised by :meth:`setup_chunker_tools`.
    chunker : docling.chunking.HybridChunker or None
        Lazily initialised by :meth:`setup_chunker_tools`.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: ChunkingConfig
    ):
        """
        Initialize the chunk processor.
        
        Args:
            logger: Logger for recording processing information
            config: Configuration object containing chunking parameters
        """
        self.logger = logger
        self.config = config
        self.chunking_errors: List[ChunkingErrorRecord] = []
        self.doc_converter = None
        self.chunker = None
    
    def setup_chunker_tools(self):
        """
        Set up document converter and chunker using configuration parameters.
        """
        try:
            # Import here to avoid circular imports
            from docling.chunking import HybridChunker
            
            # Set up document converter if not already done
            if self.doc_converter is None:
                self.doc_converter = setup_docling(self.config.artifacts_path)
                
            # Set up chunker if not already done
            if self.chunker is None:
                self.chunker = HybridChunker(tokenizer=self.config.tokenizer)
                
            self.logger.info("Document processing tools set up successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up processing tools: {str(e)}")
            raise

    def chunk_dl2langchain(self, chunk: Any, chunker: BaseChunk, metadata_add: Dict[str, Any]) -> Document:
        """
        Convert chunk from docling format to LangChain format.
        
        Args:
            chunk: The document chunk to convert
            chunker: The chunker object used to serialize the chunk
            metadata_add: Additional metadata to include
            
        Returns:
            LangChain Document object
        """
        return Document(
            page_content=chunker.serialize(chunk=chunk), # type: ignore
            metadata={
                "source": metadata_add['source'],
                "dl_meta": chunk.meta.export_json_dict(),
            }
        )
    
    def chunk_dl2lancedb(self, chunk: Any, chunker: BaseChunk, metadata_add: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert chunk from docling format to LanceDB format.
        
        Args:
            chunk: The document chunk to convert
            chunker: The chunker object used to serialize the chunk
            metadata_add: Additional metadata to include
            
        Returns:
            Dictionary containing chunk metadata for LanceDB
        """
        metadata = {
            "text": chunker.serialize(chunk) if hasattr(chunk, "text") else "", # type: ignore
            "headings": [],
            "page_info": None,
            "content_type": None, 
        }

        # Add additional metadata if provided
        if metadata_add:
            metadata = {**metadata, **metadata_add}
            
        # Extract metadata from chunk
        if hasattr(chunk, 'meta'):
            # Extract headings
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings
            
            # Extract page information and content type
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'label'):
                        metadata["content_type"] = str(item.label)
                    
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                metadata["page_info"] = prov.page_no

        return metadata

    def save_langchain_to_jsonl(self, documents: Iterable[Document], file_path: str) -> None:
        """
        Save LangChain Document objects to a JSONL file.
        
        Args:
            documents: Iterable of LangChain Document objects
            file_path: Path to save the JSONL file
        """
        with open(file_path, 'w') as jsonl_file:
            for doc in documents:
                jsonl_file.write(doc.json() + '\n')
    
    def save_docling_to_json(self, chunks: List[Any], file_path: str) -> None:
        """
        Save docling chunks to a JSON file.
        
        Args:
            chunks: List of docling chunk objects
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as fp:
            json.dump([chunk.export_json_dict() for chunk in chunks], fp, indent=2, ensure_ascii=False)

    def chunk_document(
        self, 
        doc_source: str,
        metadata_add: Dict[str, Any],
        save_individual_chunks: bool = True
    ) -> List[Any]:
        """
        Chunk document into smaller segments and save their metadata.
        
        Args:
            doc_source: Path to the source document
            metadata_add: Additional metadata to include
            save_individual_chunks: Whether to save individual chunk files
            
        Returns:
            List of chunked document segments
        """
        # Get document filename without extension
        doc_filename = os.path.splitext(os.path.basename(doc_source))[0]
        chunks_dl = []
        
        self.logger.info(f"Chunking document: {doc_source}")
        
        # Ensure output directories exist
        os.makedirs(self.config.save_location, exist_ok=True)
        
        try:
            # Ensure tools are set up
            if self.doc_converter is None or self.chunker is None:
                self.setup_chunker_tools()
                
            # Convert and chunk document
            doc = self.doc_converter.convert(source=doc_source).document
            chunks_dl0 = list(self.chunker.chunk(dl_doc=doc))
            
            for chunk in chunks_dl0:
                chunk.meta.origin.uri = metadata_add['source']
                chunks_dl.append(chunk)

            self.logger.info(f"Generated {len(chunks_dl)} chunks for document: {doc_filename}")
            
            if save_individual_chunks:
                # Save consolidated chunks metadata
                chunks_dl_file = os.path.join(
                    self.config.save_location, 
                    f"{doc_filename}_chunks.json"
                )
                
                with open(chunks_dl_file, 'w', encoding='utf-8') as fp:
                    json.dump([chunk.export_json_dict() for chunk in chunks_dl], fp, indent=2, ensure_ascii=False)
                    
                self.logger.info(f"Successfully saved all chunks for {doc_filename}")
                    
        except Exception as e:
            error_msg = f"Error in document chunking pipeline for {doc_source}: {str(e)}"
            self.logger.error(error_msg)
            self.chunking_errors.append(
                ChunkingErrorRecord(doc_source, error_msg)
            )
        
        # Return all successfully created chunks
        return chunks_dl

    def convert_chunks(
        self,
        chunks_dl: List[Any],
        doc_source: str,
        metadata_add: Dict[str, Any],
    ) -> List[Document]:
        """
        Convert document chunks to LangChain format with metadata.
        
        Args:
            chunks_dl: List of document chunks in docling format
            doc_source: Path to the source document
            metadata_add: Additional metadata to include
            
        Returns:
            List of chunks in LangChain format
        """
        doc_filename = os.path.splitext(os.path.basename(doc_source))[0]
        chunks_converted = []
        
        self.logger.info(f"Converting {len(chunks_dl)} chunks for document: {doc_filename}")
        
        for i, chunk in enumerate(chunks_dl):
            try:
                # Convert chunk to LangChain format
                chunk_converted = self.chunk_dl2langchain(chunk, self.chunker, metadata_add)
                chunks_converted.append(chunk_converted)
                
            except Exception as e:
                error_msg = f"Error converting chunk {i} for {doc_filename}: {str(e)}"
                self.logger.error(error_msg)
                self.chunking_errors.append(
                    ChunkingErrorRecord(doc_source, error_msg, i)
                )
        
        return chunks_converted

    def chunk_all_documents(
        self,
    ) -> Tuple[List[Any], List[Document], Optional[pd.DataFrame]]:
        """
        Chunk all documents in the input directory and save their segments.
        Uses a concise progress display to avoid excessive output.
        
        Returns:
            Tuple containing:
            - List of all chunked document metadata in docling format
            - List of all chunked document metadata in converted format
            - DataFrame of chunking errors (or None if no errors)
        """
        try:
            # Create output directory
            os.makedirs(self.config.save_location, exist_ok=True)
            
            # Ensure tools are set up
            if self.doc_converter is None or self.chunker is None:
                self.setup_chunker_tools()
                
            # Get all files to chunk
            input_paths = io_funcs.get_files(
                self.config.input_dir, 
                self.config.supported_formats, 
                self.logger
            )
            
            all_chunks_converted = []
            all_chunks_dl = []
            skipped_documents = []
            total_chunks_count = 0
            
            # Configure tqdm with a more concise format
            progress_bar = tqdm(
                total=len(input_paths),
                desc="Chunking documents",
                bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - Current: {postfix}",
                ncols=100
            )
            
            # Initialize postfix
            progress_bar.set_postfix_str("Starting...")
            
            for i, input_path in enumerate(input_paths):
                try:
                    # Get short filename for display
                    short_filename = os.path.basename(input_path)
                    if len(short_filename) > 40:
                        short_filename = short_filename[:37] + "..."
                    
                    # Update progress bar with current file
                    progress_bar.set_postfix_str(f"File: {short_filename}")
                    
                    # Check if document already has chunks
                    doc_filename = os.path.splitext(os.path.basename(input_path))[0]
                    chunks_dl_file = os.path.join(
                        self.config.save_location, 
                        f"{doc_filename}_chunks.json"
                    )
                    
                    # Check if file has zero size
                    if os.path.getsize(input_path) == 0:
                        error_msg = "File has zero size, skipping chunking"
                        self.logger.warning(f"{error_msg} for {input_path}")
                        self.chunking_errors.append(
                            ChunkingErrorRecord(input_path, error_msg)
                        )
                        progress_bar.update(1)
                        continue
                    
                    # Skip if already chunked and overwrite is False
                    if os.path.exists(chunks_dl_file) and not self.config.overwrite:
                        self.logger.info(f"Skipping already chunked document: {input_path}")
                        skipped_documents.append(input_path)
                        
                        # Increment progress
                        progress_bar.update(1)
                        continue
                    
                    # Get metadata for document
                    metadata_dir = self.config.metadata_dir
                    metadata = get_metadata_file(input_path, metadata_dir, self.logger) or {}
                    metadata_add = {
                        'folder_tags': metadata.get('folder_tags', []),
                        'source': metadata.get('url', input_path)
                    }
                    
                    # Step 1: Chunk the document
                    chunks_dl = self.chunk_document(
                        input_path, 
                        metadata_add, 
                        self.config.save_individual_chunks
                    )
                    
                    if not chunks_dl:
                        error_msg = "No chunks were generated for document"
                        self.logger.warning(f"{error_msg}: {input_path}")
                        self.chunking_errors.append(
                            ChunkingErrorRecord(input_path, error_msg)
                        )
                        progress_bar.update(1)
                        continue

                    # Step 2: Convert chunks to LangChain format
                    chunks_converted = self.convert_chunks(chunks_dl, input_path, metadata_add)
                    chunk_count = len(chunks_dl)
                    all_chunks_dl.extend(chunks_dl)
                    all_chunks_converted.extend(chunks_converted)
                    total_chunks_count += chunk_count
                    
                    # Update progress with chunks count
                    progress_bar.set_postfix_str(f"File: {short_filename} → {chunk_count} chunks")
                    
                    # Log success
                    self.logger.info(
                        f"Document {os.path.basename(input_path)} chunked into {chunk_count} segments"
                    )
                    
                    # Increment progress bar after processing file
                    progress_bar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error chunking document {input_path}: {str(e)}", exc_info=True)
                    self.chunking_errors.append(
                        ChunkingErrorRecord(input_path, str(e))
                    )
                    # Still update progress bar for failed files
                    progress_bar.set_postfix_str(f"Error: {os.path.basename(input_path)}")
                    progress_bar.update(1)
                    continue
                
            # Close progress bar
            progress_bar.close()
            
            # Create error report
            error_report_df = None
            if self.chunking_errors:
                # Convert errors to DataFrame
                error_data = {
                    'file_path': [error.file_path for error in self.chunking_errors],
                    'error_message': [error.error_message for error in self.chunking_errors],
                    'chunk_index': [error.chunk_index for error in self.chunking_errors],
                    'chunk_timestamp': [error.timestamp for error in self.chunking_errors] # type: ignore
                }
                error_report_df = pd.DataFrame(error_data)
                
                # Save errors to CSV
                error_csv_path = os.path.join(self.config.save_location, "chunking_errors.csv")
                error_report_df.to_csv(error_csv_path, index=False)
                
                self.logger.warning(
                    f"{len(self.chunking_errors)} errors occurred during chunking. "
                    f"See {error_csv_path} for details."
                )
            
            # Print summary
            processed_count = len(input_paths) - len(skipped_documents)
            print(f"\nSummary: Processed {processed_count} documents into {total_chunks_count} chunks")
            if len(skipped_documents) > 0:
                print(f"         Skipped {len(skipped_documents)} documents")
            if len(self.chunking_errors) > 0:
                print(f"         Encountered {len(self.chunking_errors)} errors")
            
            # Log summary
            self.logger.info(f"Chunked {processed_count} documents into {total_chunks_count} segments")
            self.logger.info(f"Skipped {len(skipped_documents)} existing documents")
            self.logger.info(f"Encountered {len(self.chunking_errors)} errors")
            
            return all_chunks_dl, all_chunks_converted, error_report_df
            
        except Exception as e:
            self.logger.error(f"Critical error in document chunking: {str(e)}", exc_info=True)
            raise

    def get_error_report(self) -> List[Dict[str, Any]]:
        """
        Get a list of errors that occurred during chunking.
        
        Returns:
            List of error dictionaries
        """
        return [
            {
                'file_path': error.file_path,
                'error_message': error.error_message,
                'chunk_index': error.chunk_index,
                'chunk_timestamp': error.timestamp.isoformat() # type: ignore
            }
            for error in self.chunking_errors
        ]

# -------------------------------------------------------------------------
# Vector store creation
# -------------------------------------------------------------------------
