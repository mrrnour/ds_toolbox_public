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
import src.common_funcs as cfuncs
import src.io_funcs_msql_local as io_funcs

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

# -------------------------------------------------------------------------
# General utility functions
# -------------------------------------------------------------------------
def setup_http_session(
    credentials: Dict[str, str], 
    verify_ssl: bool = False,
    auth_type: str = 'ntlm'  # 'none', 'basic', 'ntlm'
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
    if auth_type == 'ntlm':
        from requests_ntlm import HttpNtlmAuth
        session.auth = HttpNtlmAuth(
            credentials.get('username', ''),
            credentials.get('password', '')
        )
    elif auth_type == 'basic':
        session.auth = (
            credentials.get('username', ''),
            credentials.get('password', '')
        )
    
    # Set headers to mimic a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    
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
        device=AcceleratorDevice.AUTO  # Auto-select best available device
    )
    
    # Configure format-specific options
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline, 
            backend=PyPdfiumDocumentBackend,
            pipeline_options=pipeline_options,
        ),
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=SimplePipeline
        )
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
        format_options=format_options
    )

def load_document(file_path: str, logger: logging.Logger) -> Optional[List[Dict[str, Any]]]:
    """
    Load a document based on its file extension.
    
    Args:
        file_path: Path to the document file
        logger: Logger instance for tracking events
        
    Returns:
        List of document dictionaries or None if loading fails
    """
    from langchain.document_loaders import (
        TextLoader,
        UnstructuredMarkdownLoader,
        JSONLoader
    )
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return [{"page_content": str(item), "metadata": {"source": file_path}} for item in data]
            elif isinstance(data, dict):
                content_field = data.get("content", str(data))
                return [{"page_content": content_field, "metadata": {"source": file_path}}]
            else:
                return [{"page_content": str(data), "metadata": {"source": file_path}}]
                
        elif file_ext in ('.yaml', '.yml'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Similar handling as JSON
            if isinstance(data, list):
                return [{"page_content": str(item), "metadata": {"source": file_path}} for item in data]
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

def get_metadata_file(doc_path: str, metadata_dir: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
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
            os.path.join(metadata_dir, f"{doc_name}.meta.yml")
        ]         
        
        for meta_path in potential_metadata_files:
            if os.path.exists(meta_path):
                logger.info("Metadata found")
                ext = os.path.splitext(meta_path)[1].lower()
                
                try:
                    if ext == '.json':
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    elif ext in ('.yaml', '.yml'):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            return yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"Error loading metadata file {meta_path}: {str(e)}")
                    continue
                    
        return {}
        
    except Exception as e:
        logger.error(f"Error finding metadata for {doc_path}: {str(e)}")
        return None

def save_metadata(save_location: str, base_name: str, metadata: Dict[str, Any], logger: logging.Logger) -> Optional[str]:
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
        metadata_dir = os.path.join(save_location, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create full output path
        metadata_path = os.path.join(metadata_dir, f"{base_name}.meta.json")
        
        # Save metadata as JSON with error handling
        with open(metadata_path, 'w', encoding='utf-8') as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)
            
        return metadata_path
            
    except Exception as e:
        logger.error(f"Error saving metadata for {base_name}: {str(e)}")
        return None

# -------------------------------------------------------------------------
# Utility functions for loading documents
# -------------------------------------------------------------------------
def load_langchain_docs_from_jsonl(file_path: str) -> List[Document]:
    """
    Load LangChain documents from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            documents.append(Document(**data))
    return documents

# -------------------------------------------------------------------------
# Web Crawling
# -------------------------------------------------------------------------
@dataclass
class WebCrawler_Config:
    """Configuration for web crawling operations."""
    save_location:    Path
    input_urls:       List[str]
    credentials:      Dict[str, str]
    max_depth:        int       = 2
    max_workers:      int       = 4
    timeout:          int       = 30
    verify_ssl:       bool      = True
    auth_type:        str       = 'none'  # 'none', 'basic', 'ntlm'
    same_domain_only: bool      = True
    url_patterns:     List[str] = None  # Regex patterns for URLs to follow

@dataclass
class WebCrawler_Error:
    """Stores information about web crawling errors."""
    url: str
    error_message: str
    error_type: str
    depth: int
    crawl_path: list
    crawl_timestamp: datetime = datetime.now()

class WebCrawler_Processor:
    """
    Processes web crawling operations by following links up to a specified depth.
    Tracks the parent URL and crawl path for each URL visited.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: WebCrawler_Config
    ):
        """
        Initialize the web crawl processor.
        
        Args:
            logger: Logger for recording crawling information
            config: Configuration object containing crawling parameters
        """
        self.logger = logger
        self.config = config
        self.session = None
        self.crawling_errors = []
        
        # Tracking data
        self.lock = threading.RLock()
        self.visited = set()
        self.crawled_pages = 0
        self.page_data = []
        self.errors = []
        self.start_time = None
        self.end_time = None
        
        # Set up logging info
        self.auth_type = config.auth_type if config.auth_type != 'none' else 'No Authentication'
        
        # Compile URL patterns if provided
        self.url_patterns = [re.compile(pattern) for pattern in config.url_patterns] if config.url_patterns else None
        
        # Ensure save directory exists
        os.makedirs(config.save_location, exist_ok=True)

    def setup_session(self):
        """Set up HTTP session with appropriate configuration."""
        self.session = setup_http_session(
            credentials=self.config.credentials,
            verify_ssl=self.config.verify_ssl,
            auth_type=self.config.auth_type
        )
        
    def crawl(self) -> Dict[str, Any]:
        """
        Crawl websites by following links up to a specified depth.
        Main entry point for class functionality.
        
        Returns:
            Dictionary containing crawling results and metadata
        """
        # Set up HTTP session if not already done
        if self.session is None:
            self.setup_session()
            
        self.logger.info(f"Starting web crawl | Save location: {self.config.save_location} | Max depth: {self.config.max_depth} | Auth: {self.auth_type}")
        
        # Initialize crawl state
        self.start_time = dt.datetime.now().isoformat()
        self.visited = set()
        self.crawled_pages = 0
        self.page_data = []
        self.errors = []
        
        try:
            # Process each starting URL with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for url in self.config.input_urls:
                    futures.append(
                        executor.submit(
                            self._crawl_from_url,
                            start_url=url,
                            parent_url=None,  # Starting URLs have no parent
                            crawl_path=[]  # Start with empty path
                        )
                    )
                
                # Wait for all futures to complete and handle errors
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error in crawling thread: {str(e)}")
                        with self.lock:
                            self.errors.append(str(e))
            
            # Finalize crawl
            self.end_time = dt.datetime.now().isoformat()
            
            # Prepare crawl summary
            crawl_summary = {
                'crawled_pages': self.crawled_pages,
                'page_data':     self.page_data,
                'errors':        self.errors,
                'start_time':    self.start_time,
                'end_time':      self.end_time
            }
            
            # Save results to files
            self._save_crawl_results(crawl_summary)
            
            return crawl_summary
            
        except Exception as e:
            error_msg = f"Error in crawl operation: {str(e)}"
            self.logger.error(error_msg)
            self.crawling_errors.append(
                WebCrawler_Error("global", error_msg, "Unexpected error", 0, [])
            )
            raise
        
    def _save_crawl_results(self, crawl_summary: Dict[str, Any]) -> None:
        """
        Save web crawling results to files.
        
        Args:
            crawl_summary: Crawling results summary
        """
        try:
            # Ensure save directory exists
            os.makedirs(self.config.save_location, exist_ok=True)
            
            # Save summary report
            summary_path = os.path.join(self.config.save_location, "crawling_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(crawl_summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Crawling summary saved to: {summary_path}")
            
        except Exception as e:
            error_msg = f"Failed to save crawling results: {str(e)}"
            self.logger.error(error_msg)
            self.crawling_errors.append(
                WebCrawler_Error("save_results", error_msg, "File operation error", 0, [])
            )
            
    def _crawl_from_url(
        self,
        start_url: str, 
        parent_url: Optional[str] = None,
        crawl_path: Optional[List[str]] = []
    ) -> None:
        """
        Crawl from a starting URL up to a specified depth.
        
        Args:
            start_url: URL to start crawling from
            parent_url: URL that led to this URL (None for starting URLs)
            crawl_path: List of URLs in the path to this URL
        """
        
        # Create queue for BFS traversal - now includes parent URL and crawl path
        # Format: (url, depth, parent_url, crawl_path)
        queue = deque([(start_url, 0, parent_url, crawl_path)])

        # Process URLs breadth-first
        while queue:
            url, depth, parent, path = queue.popleft()
            
            # Create current path by adding parent to the path
            current_path = path.copy()
            if parent:
                current_path.append(parent)
            
            # Skip if already visited
            with self.lock:
                if url in self.visited:
                    continue
                self.visited.add(url)
            
            try:
                self.logger.info(f"Crawling {url} (depth {depth}, parent: {parent})...")
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                
                # Skip non-HTML content
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type.lower():
                    self.logger.info(f"Skipping non-HTML content: {url} ({content_type})")
                    continue
                
                # Parse the URL to check if it has a file extension
                parsed_url = urlparse(url)
                path = parsed_url.path
                _, extension = os.path.splitext(path)

                # If the URL has an extension (and it's not .html or .htm), skip adding it to page_data
                if extension and extension.lower() not in ['.html', '.htm', '.aspx', '.php', '.jsp']:
                    self.logger.info(f"Skipping URL with non-HTML extension: {url} ({extension})")
                    continue

                # Parse HTML and extract page data
                soup = BeautifulSoup(response.text, 'html.parser')
                page_title = soup.title.string if soup.title else urlparse(url).path
                crawl_path = current_path+[url]

                # Record this page with parent information and crawl path
                with self.lock:
                    self.crawled_pages += 1
                    self.page_data.append({
                        'url'       : url,
                        'title'     : page_title,
                        'depth'     : depth,
                        'crawl_path': crawl_path,
                        'crawling_timestamp': dt.datetime.now().isoformat()
                    })
                
                # If not at max depth, extract and queue links
                if depth < self.config.max_depth:
                    links = self._extract_links(soup, url)
                    
                    for link in links:
                        # Remove URL fragments
                        normalized_url, _ = urldefrag(link)
                        
                        # Skip if already visited or queued
                        with self.lock:
                            if normalized_url in self.visited:
                                continue
                        
                        # Add to queue for next level processing with current URL as parent
                        # and current path as the path to this point
                        queue.append((normalized_url, depth + 1, url, current_path))
                
            except requests.exceptions.RequestException as e:
                self._log_crawl_error(url, e, is_unexpected=False, current_depth=depth, crawl_path=crawl_path)
            except Exception as e:
                self._log_crawl_error(url, e, is_unexpected=True, current_depth=depth, crawl_path=crawl_path)

    def _log_crawl_error(
        self, 
        url: str, 
        error: Exception, 
        is_unexpected: bool = False,
        current_depth: int = 0,
        crawl_path: Optional[str] = None
    ) -> None:
        """
        Log crawler errors in a consistent format.
        
        Args:
            url: URL that caused the error
            error: Exception that occurred
            is_unexpected: Whether the error was unexpected
            current_depth: The depth at which the error occurred
            crawl_path: The path taken to reach this URL
        """
        error_type = "Unexpected error" if is_unexpected else "Request error"
        error_msg = f"{error_type} crawling {url} (depth {current_depth}, parent: {crawl_path}): {str(error)}"
        self.logger.error(error_msg)
        
        # Add to class-level error tracking
        self.crawling_errors.append(
            WebCrawler_Error(url, str(error), error_type, current_depth, crawl_path)
        )
        
        # Add to class-level errors tracking
        with self.lock:
            self.errors.append({
                'url'         : url,
                'error'       : str(error),
                'type'        : error_type,
                'depth'       : current_depth,
                'crawl_path'  : crawl_path,  
                'crawl_timestamp'   : dt.datetime.now().isoformat()
            })

    def _extract_links(
        self,
        soup: BeautifulSoup, 
        base_url: str
    ) -> List[str]:
        """
        Extract filtered links from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object containing parsed HTML
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        base_domain = urlparse(base_url).netloc
        
        # Find all <a> tags with href attributes
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            
            # Skip empty or non-HTTP links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
                    
            # Fix duplicate path segments (e.g., /page/page/)
            parsed_url = urlparse(absolute_url)
            path_parts = [part for part in parsed_url.path.split('/') if part]
            
            # Improved handling of duplicate path segments
            # First, handle the specific case of duplicate 'page' segments
            if 'page' in path_parts:
                # Find all occurrences of 'page'
                page_indices = [i for i, part in enumerate(path_parts) if part == 'page']
                if len(page_indices) > 1:
                    # Keep only the first occurrence of 'page'
                    for idx in sorted(page_indices[1:], reverse=True):
                        path_parts.pop(idx)
            
            # Then handle any consecutive duplicates of any segment
            fixed_parts = []
            prev_part = None
            for part in path_parts:
                # Skip if this part is the same as the previous one
                if part == prev_part:
                    continue
                fixed_parts.append(part)
                prev_part = part
                
            # Reconstruct the URL with fixed path
            fixed_path = '/' + '/'.join(fixed_parts)
            if parsed_url.path.endswith('/') and not fixed_path.endswith('/'):
                fixed_path += '/'
                
            fixed_url = parsed_url._replace(path=fixed_path).geturl()
            absolute_url = fixed_url

            # Apply domain filter if required
            if self.config.same_domain_only and urlparse(absolute_url).netloc != base_domain:
                continue
            
            # Apply pattern filter if provided
            if self.url_patterns and not any(pattern.search(absolute_url) for pattern in self.url_patterns):
                continue
            
            links.append(absolute_url)
        
        return links

# -------------------------------------------------------------------------
# Web Scraping:
# -------------------------------------------------------------------------
@dataclass
class WebScraper_Config:
    """Configuration for download operations."""
    save_location:   Path
    input_urls:      List[str]
    credentials:     Dict[str, str]
    metadata  :      Dict[str, Any] =field(default_factory=dict)
    file_extensions: List[str] = None
    chunk_size:      int       = 8192
    max_workers:     int       = 4
    verify_ssl:      bool      = True
    auth_type:       str       = 'none'  # 'none', 'basic', 'ntlm'

@dataclass
class WebScraper_Error:
    """Stores information about web scraping errors."""
    url: str
    error_message: str
    error_type: str
    file_name: Optional[str] = None
    scraper_timestamp: datetime = datetime.now()
    
class WebScraper_Processor:
    """
    Processes download operations for files and webpage content.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: WebScraper_Config
    ):
        """
        Initialize the download processor.
        
        Args:
            logger: Logger for recording download information
            config: Configuration object containing download parameters
        """
        self.logger = logger
        self.config = config
        self.session = None
        self.download_errors = []
        
        # Set up logging info
        self.auth_type = config.auth_type if config.auth_type != 'none' else 'No Authentication'
        
        # Ensure save directory exists
        os.makedirs(config.save_location, exist_ok=True)
        
    def setup_session(self):
        """Set up HTTP session with appropriate configuration."""
        self.session = setup_http_session(
            credentials=self.config.credentials,
            verify_ssl=self.config.verify_ssl,
            auth_type=self.config.auth_type
        )
        
    def download_files(self) -> None:
        """
        Download files from multiple URLs in parallel.
        Main entry point for file download functionality.
        """
        self.logger.info(f"Starting file download | Save: {self.config.save_location} | Types: {', '.join(self.config.file_extensions)} | Auth: {self.auth_type}")
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for url, metadata in zip(self.config.input_urls, self.config.metadata):
                    futures.append(
                        executor.submit(
                            self._download_single_source,
                            url=url,
                            metadata=metadata, 
                            download_type='files'
                        )
                    )
                for future in futures:
                    future.result()  # Raises exceptions if any occurred
                
        except Exception as e:
            error_msg = f"Error in download_files operation: {str(e)}"
            self.logger.error(error_msg)
            self.download_errors.append(
                WebScraper_Error("global", error_msg, "Unexpected error")
            )
            raise

    def download_content(self) -> None:
        """
        Download webpage content from multiple URLs in parallel.
        Main entry point for content download functionality.
        """
        self.logger.info(f"Starting content download | Save: {self.config.save_location} | Auth: {self.auth_type}")
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for url, metadata in zip(self.config.input_urls, self.config.metadata):
                    futures.append(
                        executor.submit(
                            self._download_single_source,
                            url=url,
                            metadata=metadata,
                            download_type='content'
                        )
                    )
                for future in futures:
                    future.result()
                    
        except Exception as e:
            error_msg = f"Error in download_content operation: {str(e)}"
            self.logger.error(error_msg)
            self.download_errors.append(
                WebScraper_Error("global", error_msg, "Unexpected error")
            )
            raise
            
    def _download_single_source(self, url: str, metadata: Dict[str, Any], download_type: str) -> None:
        """
        Download from a single source URL.
        
        Args:
            url: URL to download from
            metadata: Metadata dictionary for the download
            download_type: Type of download ('files' or 'content')
        """
        try:
            self.logger.info(f"Downloading from {url}")
            save_path = Path(os.path.abspath(os.path.join(self.config.save_location, cfuncs.sanitize_folder_name(Path(url).name))))
            save_path.mkdir(parents=True, exist_ok=True)
            
            if download_type == 'files':
                self._download_intranet_files(url, metadata, str(save_path))
            else:
                self._download_webpage_content(url, metadata, str(save_path))

        except Exception as e:
            error_msg = f"Failed to download from {url}: {str(e)}"
            self.logger.error(error_msg)
            self.download_errors.append(
                WebScraper_Error(url, error_msg, "Download error")
            )
            raise
        finally:
            self.logger.info("-" * 80)    

    def _download_intranet_files(self, url: str, metadata: Dict[str, Any], save_location: str) -> None:
        """
        Download all files from specified intranet URL using authentication.
        
        Args:
            url: URL to download files from
            metadata: Metadata dictionary for the download
            save_location: Directory to save downloaded files
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_location, exist_ok=True)
        
        # Set up session if not already done
        if self.session is None:
            self.setup_session()

        try:
            self.logger.info(f"Accessing {url}...")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            downloaded_files = []
            failed_files = []

            for link in links:
                href = link.get('href')
                if not any(href.lower().endswith(ext) for ext in self.config.file_extensions):
                    continue
                    
                file_url = urljoin(url, href)
                filename = os.path.basename(href)
                save_path = os.path.join(save_location, filename)
                
                try:
                    self.logger.info(f"\nDownloading: {filename}")
                    response = self.session.get(file_url, stream=True)
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(save_path, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for data in response.iter_content(chunk_size=self.config.chunk_size):
                            size = f.write(data)
                            pbar.update(size)
                            
                    downloaded_files.append(filename)
                    self.logger.info(f"Saved: {save_path}")

                    # Update the passed metadata dictionary with file information
                    metadata.update({
                        'url': file_url,
                        'folder_tags': url.split('/')[2:],
                        'save_path': str(save_path),
                        'scrapping_timestamp': dt.datetime.now().isoformat(),
                        'file_size_bytes': os.path.getsize(save_path),
                        'file_extension': os.path.splitext(save_path)[1].lower(),
                    })

                    base_name = os.path.splitext(os.path.basename(save_path))[0]
                    save_metadata(os.path.dirname(save_location), base_name, metadata, self.logger)

                except Exception as e:
                    error_msg = f"Failed to download {filename}: {str(e)}"
                    self.logger.error(error_msg)
                    failed_files.append(filename)
                    
                    self.download_errors.append(
                        WebScraper_Error(file_url, str(e), "File download error", filename)
                    )
                    continue
                    
            # Print summary
            self.logger.info("\nDownload Summary:")
            self.logger.info(f"Successfully downloaded ({len(downloaded_files)}):")
            for file in downloaded_files:
                self.logger.info(f"  ✓ {file}")
                
            if failed_files:
                self.logger.warning(f"\nFailed downloads ({len(failed_files)}):")
                for file in failed_files:
                    self.logger.warning(f"  ✗ {file}")
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Error occurred: {str(e)}"
            self.logger.error(error_msg)
            self.download_errors.append(
                WebScraper_Error(url, str(e), "Request error")
            )
            
    def _download_webpage_content(self, url: str, metadata: Dict[str, Any], save_location: str) -> None:
        """
        Download content of a specified webpage using authentication.
        
        Args:
            url: URL to download content from
            metadata: Metadata dictionary for the download
            save_location: Directory to save downloaded content
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_location, exist_ok=True)

        # Set up session if not already done
        if self.session is None:
            self.setup_session()

        try:
            self.logger.info(f"Accessing {url}...")
            response = self.session.get(url)
            response.raise_for_status()

            # Create a shorter filename by using a hash instead of the full URL basename
            # This helps avoid path length limitations
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            base_name = cfuncs.sanitize_folder_name(os.path.basename(url))
            
            # Limit the base_name length to avoid excessively long paths
            if len(base_name) > 30:
                base_name = base_name[:27] + "..."
                
            filename = f'{url_hash}_{base_name}__webpage_content.html'
            save_path = os.path.join(save_location, filename)
            
            # If the path is still too long, use an even shorter filename
            if len(save_path) > 240:  # Setting a safer limit below most OS limits
                filename = f'{url_hash}__webpage_content.html'
                save_path = os.path.join(save_location, filename)
                
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            self.logger.info(f"Successfully saved webpage content to: {save_path}")

            # Update the passed metadata dictionary with webpage content information
            metadata.update({
                'url': url,
                'folder_tags': url.split('/')[2:],
                'save_path': str(save_path),
                'scrapping_timestamp': dt.datetime.now().isoformat(),
                'file_size_bytes': os.path.getsize(save_path),
                'file_extension': os.path.splitext(save_path)[1].lower(),
                'original_url_basename': os.path.basename(url)
            })

            # Use a shorter base name for metadata file as well
            metadata_base_name = os.path.splitext(os.path.basename(save_path))[0]
            save_metadata(os.path.dirname(save_location), metadata_base_name, metadata, self.logger)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error occurred: {str(e)}"
            self.logger.error(error_msg)
            
            self.download_errors.append(
                WebScraper_Error(url, str(e), "Content download error")
            )
        except OSError as e:
            error_msg = f"File system error: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Problem path length: {len(save_path)} characters")
            
            # Try one more time with an extremely short filename
            try:
                short_filename = f'{url_hash[:8]}.html'
                short_save_path = os.path.join(save_location, short_filename)
                
                with open(short_save_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                    
                self.logger.info(f"Saved with fallback short name to: {short_save_path}")
                
                # Update metadata
                metadata.update({
                    'url': url,
                    'folder_tags': url.split('/')[2:],
                    'save_path': str(short_save_path),
                    'scrapping_timestamp': dt.datetime.now().isoformat(),
                    'file_size_bytes': os.path.getsize(short_save_path),
                    'file_extension': '.html',
                    'original_url_basename': os.path.basename(url),
                    'filename_shortened': True
                })
                
                save_metadata(os.path.dirname(save_location), url_hash[:8], metadata, self.logger)
                
            except OSError as e2:
                error_msg = f"Failed with short filename too: {str(e2)}"
                self.logger.error(error_msg)
                
                self.download_errors.append(
                    WebScraper_Error(url, str(e2), "Path length limitation")
                )
                
    def get_error_report(self) -> List[Dict[str, Any]]:
        """
        Get a list of errors that occurred during downloads.
        
        Returns:
            List of error dictionaries
        """
        return [
            {
                'url': error.url,
                'error_message': error.error_message,
                'error_type': error.error_type,
                'file_name': error.file_name,
                'scraper_timestamp': error.scraper_timestamp.isoformat()
            }
            for error in self.download_errors
        ]

# -------------------------------------------------------------------------
# Document conversion
# -------------------------------------------------------------------------
@dataclass
class CustomConverter_Config:
    """Configuration for HTML to Markdown conversion operations."""
    input_dir: Path
    save_location: Path
    excluded_names: List[str] = None
    special_image_patterns: Dict[str, str] = None
    overwrite: bool = True

@dataclass
class CustomConverter_Error:
    """Stores information about HTML conversion errors."""
    file_path: str
    error_message: str
    conversion_timestamp: datetime = datetime.now()

class CustomConverter_Processor:
    """
    Processes HTML files by extracting embedded JSON data and converting it to Markdown.
    Tracks errors and provides detailed logging for the conversion process.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: CustomConverter_Config
    ):
        """
        Initialize the HTML Markdown converter.
        
        Args:
            logger: Logger for recording conversion information
            config: Configuration object containing conversion parameters
        """
        self.logger = logger
        self.config = config
        self.conversion_errors = []
        
        # Set default special image patterns if none provided
        self.special_image_patterns = config.special_image_patterns or {
            'asterisks1-orange.png': '> **Note:**',
            'asterisks7-blue.png': '**•**'
        }
        
        # Set default excluded names if none provided
        self.excluded_names = config.excluded_names or ['PrivacyNotice', 'TermsOfUse', 'Copyright']
        
        # Ensure save directory exists
        os.makedirs(config.save_location, exist_ok=True)
    
    def convert_documents(self) -> Tuple[List[Dict[str, Any]], Optional[pd.DataFrame]]:
        """
        Process all HTML files in the input directory, extract JSON, and convert to markdown.
        Main entry point for class functionality.
        
        Returns:
            Tuple: (List of processed data objects, error report DataFrame or None)
        """
        self.logger.info(f"Starting batch HTML to Markdown conversion from: {self.config.input_dir}")
        
        try:
            # Get all HTML files to process
            input_paths = io_funcs.get_files(self.config.input_dir, ['.html', '.htm', '.aspx', '.php', '.jsp'], self.logger)
            
            processed_files = []
            skipped_files = []
            
            # Process each HTML file with progress tracking
            for input_file in tqdm(input_paths, desc="Converting HTML files"):
                try:
                    clear_output(wait=True)
                    # Get base name for output files
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    
                    # Check if output files already exist
                    if not self.config.overwrite:
                        markdown_path = os.path.join(self.config.save_location, 'md', f"{base_name}.md")
                        json_path = os.path.join(self.config.save_location, 'flattened_data', f"{base_name}_flattened_data.json")
                        
                        if os.path.exists(markdown_path) or os.path.exists(json_path):
                            tqdm.write(f"Skipping (already exists): {input_file}")
                            self.logger.info(f"Skipping existing file: {input_file}")
                            skipped_files.append(input_file)
                            continue
                    
                    # Display current file being processed
                    tqdm.write(f"Processing: {input_file}")

                    # Get metadata
                    metadata_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_file))), 'metadata')
                    metadata     = get_metadata_file(input_file, metadata_dir, self.logger) or {}
                    
                    # Add processing conversion_timestamp
                    metadata['conversion_timestamp'] = datetime.now().isoformat()
                    
                    # Process individual file
                    result = self._process_single_file(input_file, metadata)
                    
                    if result:
                        flattened_data, markdown_content, processed_metadata = result
                        processed_files.append({
                            'file_path'       : input_file,
                            'base_name'       : base_name,
                            'flattened_data'  : flattened_data,
                            'markdown_content': markdown_content,
                            'metadata'        : processed_metadata
                        })
                        
                        # Save complete metadata to separate file
                        metadata_path = save_metadata(self.config.save_location, base_name, processed_metadata, self.logger)
                        
                        # Log success with metadata
                        if metadata_path:
                            self.logger.info(f"Metadata updated {metadata_path}")
                        
                        self.logger.info(f"Successfully processed: {input_file}")
                    
                except Exception as e:
                    error_msg = f"Error processing file {input_file}: {str(e)}"
                    self.logger.error(error_msg)
                    self.conversion_errors.append(
                        CustomConverter_Error(input_file, error_msg)
                    )
            
            # Create error report
            error_report_df = None
            if self.conversion_errors:
                # Convert errors to DataFrame
                error_data = {
                    'file_path': [error.file_path for error in self.conversion_errors],
                    'error_message': [error.error_message for error in self.conversion_errors],
                    'conversion_timestamp': [error.conversion_timestamp for error in self.conversion_errors]
                }
                error_report_df = pd.DataFrame(error_data)
                
                # Save errors to CSV
                error_csv_path = os.path.join(self.config.save_location, "conversion_errors.csv")
                error_report_df.to_csv(error_csv_path, index=False)
                
                self.logger.warning(
                    f"{len(self.conversion_errors)} files failed to process. "
                    f"See {error_csv_path} for details."
                )
            
            # Log summary
            self.logger.info(f"Processed {len(processed_files)} HTML files successfully")
            self.logger.info(f"Skipped {len(skipped_files)} existing files")
            self.logger.info(f"Encountered {len(self.conversion_errors)} errors")
            
            return processed_files, error_report_df
            
        except Exception as e:
            error_msg = f"Critical error in HTML conversion batch process: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.conversion_errors.append(
                CustomConverter_Error("batch_process", error_msg)
            )
            return [], None
    
    def _process_single_file(self, input_file: str, metadata: Dict[str, Any] = None) -> Optional[Tuple[Any, str, Dict[str, Any]]]:
        """
        Process a single HTML file, extract JSON, flatten data, and convert to markdown.
        
        Args:
            input_file: Path to the HTML file
            metadata: Optional metadata for the file
            
        Returns:
            Tuple: (flattened_data, markdown_content, metadata) or None if processing failed
        """
        self.logger.info(f"Processing HTML file: {input_file}")
        
        if metadata is None:
            metadata = {}
        
        try:
            # Extract JSON from HTML
            json_data = self._extract_json_from_html(input_file)
            if isinstance(json_data, str) and json_data.startswith("Error"):
                error_msg = f"Failed to extract JSON: {json_data}"
                self.logger.error(error_msg)
                self.conversion_errors.append(
                    CustomConverter_Error(input_file, error_msg)
                )
                return None
            
            # Flatten the content data
            flattened_data = self._flatten_content_data(json_data)
            
            doc_filename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Create a copy of metadata
            document_metadata = metadata.copy()

            keys = [
                    'crawl_path',
                    'crawling_timestamp',
                    'scrapping_timestamp',
                    'conversion_timestamp',
                    'depth',
                    'file_extension',
                    'file_size_bytes',
                    'save_path',
                    # 'folder_tags',
                    # 'original_url_basename',
                    # 'title',
                    # 'url'
                    ]

            for field in keys:
                document_metadata.pop(field, None)

            # document_metadata['content_count'] = len(flattened_data)
            # if 'title' not in document_metadata and flattened_data and 'title' in flattened_data[0]:
            #     document_metadata['title'] = flattened_data[0]['title']
            
            # Save flattened data as JSON with metadata
            output_folder = os.path.join(self.config.save_location,'flattened_data')
            os.makedirs(output_folder, exist_ok=True)
            json_path = os.path.join(output_folder, f"{doc_filename}_flattened_data.json")
            
            # Add metadata to flattened data for JSON
            json_data_with_metadata = {
                'data': flattened_data,
                'metadata': document_metadata
            }
            
            self._save_json_to_file(json_data_with_metadata, json_path)
            self.logger.info(f"JSON data saved to: {json_path}")
            
            # Convert to markdown
            markdown_content = self._convert_html_to_markdown_list(flattened_data)
            
            # Add metadata as YAML frontmatter to markdown
            if document_metadata:
                metadata_yaml = yaml.safe_dump(document_metadata)
                markdown_content = f"{markdown_content}---\n\n### Metadata:\n{metadata_yaml}---"
            
            # Save markdown
            output_folder =  os.path.join(self.config.save_location,'md')
            os.makedirs(output_folder, exist_ok=True)
            markdown_path = os.path.join(output_folder, f"{doc_filename}.md")
            self._save_markdown_to_file(markdown_content, markdown_path)
            self.logger.info(f"Markdown content saved to: {markdown_path}")
            
            # Update metadata with processed file locations
            metadata['processed_file_locations'] = {
                'json': json_path,
                'markdown': markdown_path
            }
            
            return flattened_data, markdown_content, metadata
            
        except Exception as e:
            error_msg = f"Conversion error for {input_file}: {str(e)}"
            self.logger.error(error_msg)
            return None

    def _extract_json_from_html(self, input_file: str) -> Any:
        """
        Extract and sanitize JSON data embedded in HTML file.
        
        Args:
            input_file: Path to the HTML file containing embedded JSON
            
        Returns:
            Parsed JSON data or error message
        """
        self.logger.info(f"Extracting JSON from HTML: {input_file}")
        
        try:
            # Read the HTML file
            with open(input_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Extract JSON data from the page-input attribute
            json_match = re.search(r'page-input="(.*?)"', html_content)
            if not json_match:
                return "Error: No JSON data found in the HTML file"
            
            # Unescape HTML entities in the JSON string
            json_str = unescape(json_match.group(1))
            json_str = json_str.replace('&nbsp;', ' ')
            json_str = json_str.replace('\u200b', '')
            
            # Handle URLs with base64 content
            json_str = re.sub(
                r'(https://portal\.apacorp\.net.*?data:image/[^;]+;base64,)[A-Za-z0-9+/=]+', 
                r'\1REMOVED64ENCODING', 
                json_str
            )
            
            # Remove HTML formatting tags
            format_tags = ['strong', 'em', 'span', 'sub', 'sup', 'mark', 'small']
            for tag in format_tags:
                json_str = re.sub(r'<' + tag + r'[^>]*>', '', json_str)
                json_str = re.sub(r'</' + tag + r'>', '', json_str)
            
            # Parse the JSON data
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing JSON: {str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
            
        except Exception as e:
            error_msg = f"Unexpected error extracting JSON: {str(e)}"
            self.logger.error(error_msg)
            return f"Error: {error_msg}"
    
    def _flatten_content_data(self, data_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract nested body, name, title and titleElement data from a nested dictionary
        structure and flatten it into a dictionary. Items with empty body are excluded.
        
        Args:
            data_dict (dict): The nested dictionary containing the data
            
        Returns:
            list: A list of dictionaries with flattened data (excluding empty bodies)
        """
        self.logger.info("Flattening content data...")
        flattened_data = []
        
        try:
            # Process rows -> columns -> panels -> contents
            if 'rows' in data_dict:
                for row in data_dict['rows']:
                    if 'columns' in row:
                        for column in row['columns']:
                            if 'panels' in column:
                                for panel in column['panels']:
                                    if 'contents' in panel:
                                        for content in panel['contents']:
                                            body = content.get('body', '')
                                              
                                            content_data = {
                                                'source': 'row_content',
                                                'name': content.get('name', ''),
                                                'title': content.get('title', ''),
                                                'titleElement': content.get('titleElement', ''),
                                                'body': body,
                                                'sequence': content.get('sequence', ''),
                                                'content_id': panel.get('id', '')
                                            }
                                            flattened_data.append(content_data)
            
            # Process namedContents
            if 'namedContents' in data_dict:
                for content in data_dict['namedContents']:
                    body = content.get('body', '')

                    content_data = {
                        'source': 'named_content',
                        'name': content.get('name', ''),
                        'title': content.get('title', ''),
                        'titleElement': content.get('titleElement', ''),
                        'body': body,
                        'sequence': content.get('sequence', ''),
                        'content_id': content.get('id', '')
                    }
                    flattened_data.append(content_data)
            
            # Filter out items with specific names and empty content
            filtered_data = [item for item in flattened_data 
                            if item['name'] not in self.excluded_names 
                            and not ((item['body']=='') and (item['title']==''))]

            # Simplify the output structure
            simplified_data = [
                {
                    'title': item['title'],
                    'body': item['body']
                }
                for item in filtered_data
            ]
            
            self.logger.info(f"Flattened {len(simplified_data)} content items")
            return simplified_data
            
        except Exception as e:
            error_msg = f"Error flattening content data: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(
                CustomConverter_Error("content_flattening", error_msg)
            )
            return []
    
    def _convert_html_to_markdown_list(self, html_content_list: List[Dict[str, Any]]) -> str:
        """
        Convert a list of HTML content dictionaries to markdown format.
        
        Args:
            html_content_list: List of dictionaries with 'title' and 'body' keys
            
        Returns:
            String containing the markdown conversion
        """
        self.logger.info(f"Converting {len(html_content_list)} HTML items to Markdown...")
        markdown_content = []
        
        for item in html_content_list:
            title = item.get('title', '')
            body = item.get('body', '')
            
            if title:
                markdown_content.append(f"# {title}\n")
            
            if body:
                markdown_content.append(self._convert_html_to_markdown(body))
            
            markdown_content.append("\n---\n")
        
        # Remove the last separator if it exists
        if markdown_content and markdown_content[-1] == "\n---\n":
            markdown_content.pop()
            
        return "".join(markdown_content)
    
    def _convert_html_to_markdown(self, html: str) -> str:
        """
        Convert HTML string to markdown format.
        
        Args:
            html: HTML content as string
            
        Returns:
            Markdown formatted string
        """
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return self._process_tag(soup)
            
        except Exception as e:
            error_msg = f"Error converting HTML to Markdown: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(
                CustomConverter_Error("html_conversion", error_msg)
            )
            return f"*Error converting content: {str(e)}*"
    
    def _process_tag(self, tag: Any) -> str:
        """
        Process a BeautifulSoup tag and its children recursively.
        
        Args:
            tag: BeautifulSoup tag to process
            
        Returns:
            Processed markdown string
        """
        if tag.name is None:
            return tag.string if tag.string else ""
        
        result = []
        
        # Process by tag type
        if tag.name == 'div':
            result.append(self._process_children(tag))
        
        elif tag.name == 'h1':
            result.append(f"# {tag.get_text().strip()}\n\n")
        
        elif tag.name == 'h2':
            result.append(f"## {tag.get_text().strip()}\n\n")
        
        elif tag.name == 'h3':
            result.append(f"### {tag.get_text().strip()}\n\n")
        
        elif tag.name == 'h4':
            result.append(f"#### {tag.get_text().strip()}\n\n")
        
        elif tag.name == 'p':
            result.append(f"{self._process_children(tag)}\n\n")
        
        elif tag.name == 'a':
            href = tag.get('href', '')
            text = tag.get_text().strip()
            result.append(f"[{text}]({href})")
        
        elif tag.name == 'img':
            src = tag.get('src', '')
            alt = tag.get('alt', '')
            
            # Check if this is a special image pattern
            for pattern, replacement in self.special_image_patterns.items():
                if pattern in src:
                    result.append(f"{replacement} ")
                    return "".join(result)
            
            # Regular image
            result.append(f"![{alt}]({src})")
        
        elif tag.name == 'ul':
            items = []
            for li in tag.find_all('li', recursive=False):
                items.append(f"* {self._process_children(li)}")
            result.append("\n".join(items) + "\n\n")
        
        elif tag.name == 'li':
            result.append(self._process_children(tag))
        
        elif tag.name == 'blockquote':
            content = self._process_children(tag)
            # Add > prefix to each line
            quoted_content = "\n> ".join(content.split("\n"))
            result.append(f"> {quoted_content}\n\n")
        
        else:
            # Default processing for other tags
            result.append(self._process_children(tag))
        
        return "".join(result)
    
    def _process_children(self, tag: Any) -> str:
        """
        Process all children of a tag.
        
        Args:
            tag: BeautifulSoup tag whose children should be processed
            
        Returns:
            Processed markdown string
        """
        result = []
        for child in tag.children:
            result.append(self._process_tag(child))
        return "".join(result)
    
    def _save_markdown_to_file(self, markdown_content: str, output_file: str) -> str:
        """
        Save markdown content to a file.
        
        Args:
            markdown_content: String containing markdown content
            output_file: Path to output file
            
        Returns:
            Path to the output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return output_file
            
        except Exception as e:
            error_msg = f"Error saving markdown to {output_file}: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(
                CustomConverter_Error(output_file, error_msg)
            )
            return ""
    
    def _save_json_to_file(self, data: Any, output_file: str) -> str:
        """
        Save JSON data to a file.
        
        Args:
            data: Data to save as JSON
            output_file: Path to output file
            
        Returns:
            Path to the output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            return output_file
            
        except Exception as e:
            error_msg = f"Error saving JSON to {output_file}: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(
                CustomConverter_Error(output_file, error_msg)
            )
            return ""
   
    def get_error_report(self) -> List[Dict[str, Any]]:
        """
        Get a list of errors that occurred during conversion.
        
        Returns:
            List of error dictionaries
        """
        return [
            {
                'file_path': error.file_path,
                'error_message': error.error_message,
                'conversion_timestamp': error.conversion_timestamp.isoformat()
            }
            for error in self.conversion_errors
        ]

# -------------------------------------------------------------------------
# Document processing
# -------------------------------------------------------------------------
@dataclass
class Converter_Config:
    """Configuration for document format conversion operations."""
    input_dir: Path
    save_location: Path
    supported_formats: List[str]
    output_formats: List[str] = field(default_factory=lambda: ['yaml', 'md', 'json'])
    overwrite: bool = True

@dataclass
class Converter_Error:
    """Stores information about document processing errors."""
    file_path: str
    error_message: str
    conversion_timestamp: datetime = dt.datetime.now()
    url: Optional[str] = None  # Add URL field to store document URL
    crawl_path: Optional[str] = None  # Add crawl_path field from metadata

class Converter_Processor:
    """
    Processes documents by converting them to different formats and extracting metadata.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: Converter_Config
    ):
        """
        Initialize the document processor.
        
        Args:
            logger: Logger for recording processing information
            config: Configuration object containing conversion parameters
        """
        self.logger = logger
        self.config = config
        self.processing_errors: List[Converter_Error] = []
        self.doc_converter = None

    def save_document(
        self, 
        save_location: Path, 
        base_name: str, 
        document: Any, 
        metadata: Dict[str, Any], 
        format: str
    ) -> Optional[str]:
        """
        Save document in specified format with optional metadata.
        
        Args:
            save_location: Base output directory
            base_name: Base name of the file without extension
            document: Document object to save
            metadata: Document metadata to include
            format: Format to save in ('md', 'json', or 'yaml')
            
        Returns:
            Path to the saved document or None if saving failed
        """
        try:
            # Create format-specific directory
            format_dir = os.path.join(save_location, format)
            os.makedirs(format_dir, exist_ok=True)
            
            # Create full output path
            output_path = os.path.join(format_dir, base_name)
            full_path = f"{output_path}.{format}"
            
            if format == 'md':
                # For markdown, save content with metadata as YAML frontmatter
                content = document.export_to_markdown()
                
                if metadata:
                    # Add metadata as YAML frontmatter at the end
                    content += "\n\n"
                    content += "### Metadata:\n"
                    content += yaml.safe_dump(metadata)
                    
                with open(full_path, 'w', encoding='utf-8') as fp:
                    fp.write(content)
                    
            elif format == 'json':
                # For JSON, include metadata in output dictionary
                output_dict = document.export_to_dict()
                if metadata:
                    output_dict['metadata'] = metadata

                with open(full_path, 'w', encoding='utf-8') as fp:
                    json.dump(output_dict, fp, indent=2, ensure_ascii=False)
                    
            elif format == 'yaml':
                # For YAML, include metadata in output dictionary
                output_dict = document.export_to_dict()
                if metadata:
                    output_dict['metadata'] = metadata

                with open(full_path, 'w', encoding='utf-8') as fp:
                    yaml.safe_dump(output_dict, fp, allow_unicode=True)
                    
            return full_path

        except Exception as e:
            error_msg = f"Error saving {format} format for {base_name}: {str(e)}"
            self.logger.error(error_msg)
            self.processing_errors.append(
                Converter_Error(str(output_path), f"Failed to save {format} format: {str(e)}")
            )
            return None

    def convert_legacy_formats(self, input_path: str) -> str:
        """
        Convert legacy formats (.doc to .docx, .xls to .xlsx) if needed.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Path to the converted file or original path if no conversion needed
        """
        file_ext = os.path.splitext(input_path)[1].lower().lstrip('.')
        
        # Check if this is a legacy format that needs conversion
        if file_ext in self.config.supported_formats:
            self.logger.info(f"Detected legacy format: {file_ext} for {input_path}")
            
            if file_ext == 'doc':
                return io_funcs.convert_doc_to_html(input_path)
            elif file_ext == 'xls':
                return io_funcs.convert_xls_to_xlsx(input_path)
        
        # Return original path if no conversion needed
        return input_path

    def convert_documents(
        self, 
        artifacts_path: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[pd.DataFrame]]:
        """
        Process all documents in the input directory and save to output formats.
        Files with zero size will be skipped and logged as errors.
        
        Args:
            artifacts_path: Path to store OCR and other processing artifacts

        Returns:
            Tuple containing:
            - List of processed document metadata
            - DataFrame of processing errors (or None if no errors)
        """
        try:
            # Create output directory
            os.makedirs(self.config.save_location, exist_ok=True)
            
            # Get all files to process
            input_paths = io_funcs.get_files(self.config.input_dir, self.config.supported_formats, self.logger)

            converted_documents = []
            # meta_documents = []
            skipped_documents = []
            
            # Set up document converter if not already done
            if self.doc_converter is None:
                doc_converter = setup_docling(artifacts_path)
            else:
                doc_converter = self.doc_converter
        
            # Process each document with progress tracking
            for input_path in tqdm(input_paths, desc="Processing documents"):
                try:
                    clear_output(wait=True)
                    # Check if file has zero size
                    if os.path.getsize(input_path) == 0:
                        error_msg = "File has zero size, skipping conversion"
                        tqdm.write(f"Skipping (zero size): {input_path}")
                        self.logger.warning(f"{error_msg} for {input_path}")
                        
                        # Get metadata to extract URL and crawl_path if available
                        metadata_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_path))), 'metadata')
                        metadata = get_metadata_file(input_path, metadata_dir, self.logger) or {}
                        url = metadata.get('url', None)
                        crawl_path = metadata.get('crawl_path', None)
                        
                        self.processing_errors.append(
                            Converter_Error(input_path, error_msg, url=url, crawl_path=crawl_path)
                        )
                        continue
                    
                    # Get base name for output files
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    
                    # Check if file already exists in any of the output formats
                    if not self.config.overwrite:
                        output_exists = False
                        for format in self.config.output_formats:
                            format_dir = os.path.join(self.config.save_location, format)
                            output_path = os.path.join(format_dir, f"{base_name}.{format}")
                            if os.path.exists(output_path):
                                output_exists = True
                                break
                                
                        # Skip if file already exists and overwrite is False
                        if output_exists:
                            tqdm.write(f"Skipping (already exists): {input_path}")
                            self.logger.info(f"Skipping existing document: {input_path}")
                            skipped_documents.append(input_path)
                            continue
                    
                    # Display current file being processed
                    tqdm.write(f"Processing: {input_path}")
                    
                    # Get metadata early to have URL and crawl_path available in case of error
                    metadata_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_path))), 'metadata')
                    metadata = get_metadata_file(input_path, metadata_dir, self.logger) or {}
                    url = metadata.get('url', None)
                    crawl_path = metadata.get('crawl_path', None)
                    
                    # Convert legacy formats if needed
                    converted_path = self.convert_legacy_formats(input_path)
                    
                    # Track if a temporary conversion was made
                    is_temp_conversion = converted_path != input_path
                    
                    if is_temp_conversion:
                        self.logger.info(f"Converted {input_path} to {converted_path}")
                    
                    # Convert the document
                    res_doc = doc_converter.convert(converted_path).document

                    # Check for empty document
                    if res_doc is None or (res_doc.export_to_markdown() == ''):
                        error_msg = "Document conversion resulted in empty document"
                        self.logger.warning(f"{error_msg} for {input_path}")
                        self.processing_errors.append(
                            Converter_Error(input_path, error_msg, url=url, crawl_path=crawl_path)
                        )
                        continue

                    # Add converted format info if applicable
                    if is_temp_conversion:
                        metadata['converted_from'] = os.path.splitext(input_path)[1].lower()
                        metadata['converted_to'] = os.path.splitext(converted_path)[1].lower()
                    
                    # Create a copy of metadata 
                    document_metadata = metadata.copy()
                    keys = [
                            'crawl_path',
                            'crawling_timestamp',
                            'scrapping_timestamp',
                            'depth',
                            'file_extension',
                            'file_size_bytes',
                            'save_path',
                            # 'folder_tags',
                            # 'original_url_basename',
                            # 'title',
                            # 'url'
                            ]

                    for field in keys:
                        document_metadata.pop(field, None)
                    
                    # Save document in requested formats
                    processed_file_locations = {}
                    for format in self.config.output_formats:
                        format_path = self.save_document(
                            self.config.save_location, 
                            base_name, 
                            res_doc, 
                            document_metadata, 
                            format
                        )
                        if format_path:
                            processed_file_locations[format] = format_path

                    converted_documents.append(res_doc)

                    # Add processed file locations to metadata
                    metadata['processed_file_locations'] = processed_file_locations
                    
                    # Save complete metadata to separate file
                    metadata_path = save_metadata(self.config.save_location, base_name, metadata, self.logger)
                    
                    # Log success
                    self.logger.info(
                        f"Document {os.path.basename(input_path)} converted and saved to {self.config.save_location}"
                    )
                    if metadata_path:
                        self.logger.info(f"Metadata updated {metadata_path}")

                    # meta_documents.append(metadata)

                except Exception as e:
                    # Get metadata to extract URL and crawl_path if available (in case we're inside the exception handler)
                    try:
                        metadata_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(input_path))), 'metadata')
                        metadata = get_metadata_file(input_path, metadata_dir, self.logger) or {}
                        url = metadata.get('url', None)
                        crawl_path = metadata.get('crawl_path', None)
                    except Exception:
                        url = None
                        crawl_path = None
                    
                    self.logger.error(f"Error processing {input_path}: {str(e)}", exc_info=True)
                    self.processing_errors.append(
                        Converter_Error(input_path, str(e), url=url, crawl_path=crawl_path)
                    )
                    continue
            
            # Create error report
            error_report_df = None
            if self.processing_errors:
                # Convert errors to DataFrame
                error_data = {
                    'file_path': [error.file_path for error in self.processing_errors],
                    'error_message': [error.error_message for error in self.processing_errors],
                    'conversion_timestamp': [error.conversion_timestamp for error in self.processing_errors],
                    'url': [error.url for error in self.processing_errors],  # URL column
                    'crawl_path': [error.crawl_path for error in self.processing_errors]  # Add crawl_path column
                }
                error_report_df = pd.DataFrame(error_data)
                
                # Save errors to CSV
                error_csv_path = os.path.join(self.config.save_location, "processing_errors.csv")
                error_report_df.to_csv(error_csv_path, index=False)
                
                self.logger.warning(
                    f"{len(self.processing_errors)} files failed to process. "
                    f"See {error_csv_path} for details."
                )
            
            # Log summary
            self.logger.info(f"Processed {len(converted_documents)} documents successfully")
            self.logger.info(f"Skipped {len(skipped_documents)} existing documents")
            self.logger.info(f"Encountered {len(self.processing_errors)} errors")

            return converted_documents, error_report_df

        except Exception as e:
            self.logger.error(f"Critical error in document processing: {str(e)}", exc_info=True)
            raise

    def get_error_report(self) -> List[Dict[str, Any]]:
        """
        Get a list of errors that occurred during processing.
        
        Returns:
            List of error dictionaries
        """
        return [
            {
                'file_path': error.file_path,
                'error_message': error.error_message,
                'conversion_timestamp': error.conversion_timestamp.isoformat(),
                'url': error.url,  # Include URL in error report
                'crawl_path': error.crawl_path  # Include crawl_path in error report
            }
            for error in self.processing_errors
        ]

# -------------------------------------------------------------------------
# Document chunking
# -------------------------------------------------------------------------
@dataclass
class Chunking_Config:
    """Configuration for document chunking operations."""
    input_dir: Path
    save_location: Path
    supported_formats: List[str]
    metadata_dir: Optional[str] = None
    save_individual_chunks: bool = True
    overwrite: bool = False
    tokenizer: Optional[Any] = None
    artifacts_path: Optional[str] = None

@dataclass
class Chunking_Error:
    """Stores information about document chunking errors."""
    file_path: str
    error_message: str
    chunk_index: Optional[int] = None
    chunk_timestamp: datetime = dt.datetime.now()

class Chunk_Processor:
    """
    Chunks documents by converting them to smaller segments and extracting relevant metadata.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        config: Chunking_Config
    ):
        """
        Initialize the chunk processor.
        
        Args:
            logger: Logger for recording processing information
            config: Configuration object containing chunking parameters
        """
        self.logger = logger
        self.config = config
        self.chunking_errors: List[Chunking_Error] = []
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

    def chunk_dl2langChain(self, chunk: Any, chunker: BaseChunk, metadata_add: Dict[str, Any]) -> Document:
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
                Chunking_Error(doc_source, error_msg)
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
                chunk_converted = self.chunk_dl2langChain(chunk, self.chunker, metadata_add)
                chunks_converted.append(chunk_converted)
                
            except Exception as e:
                error_msg = f"Error converting chunk {i} for {doc_filename}: {str(e)}"
                self.logger.error(error_msg)
                self.chunking_errors.append(
                    Chunking_Error(doc_source, error_msg, i)
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
                            Chunking_Error(input_path, error_msg)
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
                            Chunking_Error(input_path, error_msg)
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
                        Chunking_Error(input_path, str(e))
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
@dataclass
class Vectorstore_Config:
    """Configuration for vector store operations."""
    documents: List[Document] 
    save_location: str
    embed_model_id: str
    store_type: str = "chroma"
    overwrite:bool = True
    show_progress: bool = True

@dataclass
class Vectorstore_Error:
    """Stores information about document embedding errors."""
    document_index: int
    error_message: str
    document_id: Optional[str] = None

class VectorStore_Processor:
    """Process document chunks and store them in vector databases."""
    
    def __init__(
        self,
        logger: logging.Logger,
        config: Vectorstore_Config
    ):
        """
        Initialize the vector store processor.
        
        Args:
            logger: Logger instance
            config: Configuration object containing vector store parameters
        """
        self.logger = logger
        self.config = config
        self.embedding_errors = []
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbeddings(model_name=config.embed_model_id)

    def meta4chroma(self, documents: List[Document]) -> List[Document]:
        """
        Clean document metadata for Chroma vector store compatibility.
        
        Args:
            documents: List of documents to clean
            
        Returns:
            List of documents with cleaned metadata
        """
        cleaned_docs = []
        total_docs = len(documents)
        
        # Initialize progress bar
        with tqdm(total=total_docs, desc="Converting metadata") as pbar:
            for i, doc in enumerate(documents):
                try:
                    # Create new document with simplified metadata
                    metadata = {}
                    if hasattr(doc, 'metadata'):
                        for key, value in doc.metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                metadata[key] = value
                            else:
                                try:
                                    metadata[key] = str(value)
                                except:
                                    pass
                    
                    cleaned_docs.append(Document(
                        page_content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        metadata=metadata
                    ))
                except Exception as e:
                    self.logger.error(f"Error converting metadata for chroma {i}: {str(e)}")
                    self.embedding_errors.append(Vectorstore_Error(i, str(e)))
                
                # Update progress bar
                pbar.update(1)
                
        return cleaned_docs

    def create_vector_store(
        self
    ):
        """
        Create a vector store of the specified type.
        
        Args:
            documents: List of documents to embed and store, defaults to config value
            
        Returns:
            Vector store instance
        """
        # Use parameters from config
        documents = self.config.documents
        store_type = self.config.store_type
        show_progress = self.config.show_progress
        vector_db_uri = self.config.save_location

        self.logger.info(f"Creating {store_type} vector store with {len(documents)} documents")
        
        try:
            # Display progress message
            if show_progress:
                print(f"Embedding documents into {store_type} vector store...")
                
            if store_type == "chroma":
                return Chroma.from_documents(
                    documents=self.meta4chroma(documents),
                    embedding=self.embed_model,
                    persist_directory=vector_db_uri,
                )
            elif store_type == "faiss":
                vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embed_model,
                )
                vectorstore.save_local(vector_db_uri)
                return vectorstore
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
                
        except Exception as e:
            error_msg = f"Error creating {store_type} vector store: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
    def get_error_report(self, save_to_csv: bool = False, output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Generate a report of embedding errors and optionally save to CSV.
        
        Args:
            save_to_csv: Whether to save the error report to a CSV file
            output_path: Path to save the error report. Defaults to save_location/embedding_errors.csv
            
        Returns:
            DataFrame containing error information or None if no errors occurred
        """
        if not self.embedding_errors:
            return None
            
        error_data = {
            'document_index': [e.document_index for e in self.embedding_errors],
            'error_message': [e.error_message for e in self.embedding_errors],
            'document_id': [e.document_id for e in self.embedding_errors]
        }
        error_df = pd.DataFrame(error_data)
        
        if save_to_csv:
            error_csv_path = output_path or os.path.join(self.config.save_location, "embedding_errors.csv")
            error_df.to_csv(error_csv_path, index=False)
            self.logger.warning(f"Encountered {len(self.embedding_errors)} errors during embedding")
            
        return error_df