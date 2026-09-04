"""
RAG (Retrieval-Augmented Generation) utility functions for document processing,
web scraping, chunking, and vector store operations.
"""

# Standard library imports
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any

# Third-party imports
import pandas as pd
import yaml
from bs4 import BeautifulSoup

# Docling imports
from IPython.display import clear_output

# LangChain imports
from tqdm import tqdm

from .. import io_funcs

# Local imports
from .setup_helpers import get_metadata_file, save_metadata


@dataclass
class CustomConverterConfig:
    """Configuration inputs for :class:`CustomConverterProcessor`.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory to scan for input HTML files.
    save_location : pathlib.Path
        Destination directory for the converted Markdown/JSON output.
    excluded_names : list of str, optional
        Substrings; any file whose name contains one of these is
        skipped. Defaults to ``['PrivacyNotice', 'TermsOfUse', 'Copyright']``.
    special_image_patterns : dict of {str: str}, optional
        Filename -> replacement-Markdown map applied when an inline
        ``<img>`` is encountered. Defaults to note/bullet callouts.
    overwrite : bool, default True
        If True, re-convert files that already have output on disk.
    """

    input_dir: Path
    save_location: Path
    excluded_names: list[str] = None
    special_image_patterns: dict[str, str] = None
    overwrite: bool = True


@dataclass
class CustomConverterErrorRecord:
    """Record describing a single HTML->Markdown conversion failure.

    Parameters
    ----------
    file_path : str
        Source HTML file that failed.
    error_message : str
        Human-readable error string.
    conversion_timestamp : datetime.datetime
        Time the error was recorded. Defaults to ``datetime.now()``.
    """

    file_path: str
    error_message: str
    conversion_timestamp: datetime = datetime.now()


class CustomConverterProcessor:
    """Extract embedded JSON from HTML files and convert it to Markdown.

    Iterates over files under :attr:`config.input_dir`, filters out
    excluded names, flattens the JSON content tree, and emits per-file
    Markdown plus a summary metadata frame. Errors are accumulated in
    :attr:`conversion_errors`.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for progress and error reporting.
    config : CustomConverterConfig
        Pipeline configuration.

    Attributes
    ----------
    logger : logging.Logger
    config : CustomConverterConfig
    conversion_errors : list of CustomConverterErrorRecord
    special_image_patterns : dict of {str: str}
    excluded_names : list of str
    """

    def __init__(self, logger: logging.Logger, config: CustomConverterConfig):
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
            "asterisks1-orange.png": "> **Note:**",
            "asterisks7-blue.png": "**•**",
        }

        # Set default excluded names if none provided
        self.excluded_names = config.excluded_names or ["PrivacyNotice", "TermsOfUse", "Copyright"]

        # Ensure save directory exists
        os.makedirs(config.save_location, exist_ok=True)

    def convert_documents(self) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
        """
        Process all HTML files in the input directory, extract JSON, and convert to markdown.
        Main entry point for class functionality.

        Returns:
            Tuple: (List of processed data objects, error report DataFrame or None)
        """
        self.logger.info(
            f"Starting batch HTML to Markdown conversion from: {self.config.input_dir}"
        )

        try:
            # Get all HTML files to process
            input_paths = io_funcs.get_files(
                self.config.input_dir, [".html", ".htm", ".aspx", ".php", ".jsp"], self.logger
            )

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
                        markdown_path = os.path.join(
                            self.config.save_location, "md", f"{base_name}.md"
                        )
                        json_path = os.path.join(
                            self.config.save_location,
                            "flattened_data",
                            f"{base_name}_flattened_data.json",
                        )

                        if os.path.exists(markdown_path) or os.path.exists(json_path):
                            tqdm.write(f"Skipping (already exists): {input_file}")
                            self.logger.info(f"Skipping existing file: {input_file}")
                            skipped_files.append(input_file)
                            continue

                    # Display current file being processed
                    tqdm.write(f"Processing: {input_file}")

                    # Get metadata
                    metadata_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(input_file))), "metadata"
                    )
                    metadata = get_metadata_file(input_file, metadata_dir, self.logger) or {}

                    # Add processing conversion_timestamp
                    metadata["conversion_timestamp"] = datetime.now().isoformat()

                    # Process individual file
                    result = self._process_single_file(input_file, metadata)

                    if result:
                        flattened_data, markdown_content, processed_metadata = result
                        processed_files.append(
                            {
                                "file_path": input_file,
                                "base_name": base_name,
                                "flattened_data": flattened_data,
                                "markdown_content": markdown_content,
                                "metadata": processed_metadata,
                            }
                        )

                        # Save complete metadata to separate file
                        metadata_path = save_metadata(
                            self.config.save_location, base_name, processed_metadata, self.logger
                        )

                        # Log success with metadata
                        if metadata_path:
                            self.logger.info(f"Metadata updated {metadata_path}")

                        self.logger.info(f"Successfully processed: {input_file}")

                except Exception as e:
                    error_msg = f"Error processing file {input_file}: {str(e)}"
                    self.logger.error(error_msg)
                    self.conversion_errors.append(CustomConverterErrorRecord(input_file, error_msg))

            # Create error report
            error_report_df = None
            if self.conversion_errors:
                # Convert errors to DataFrame
                error_data = {
                    "file_path": [error.file_path for error in self.conversion_errors],
                    "error_message": [error.error_message for error in self.conversion_errors],
                    "conversion_timestamp": [
                        error.conversion_timestamp for error in self.conversion_errors
                    ],
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
            self.conversion_errors.append(CustomConverterErrorRecord("batch_process", error_msg))
            return [], None

    def _process_single_file(
        self, input_file: str, metadata: dict[str, Any] = None
    ) -> tuple[Any, str, dict[str, Any]] | None:
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
                self.conversion_errors.append(CustomConverterErrorRecord(input_file, error_msg))
                return None

            # Flatten the content data
            flattened_data = self._flatten_content_data(json_data)

            doc_filename = os.path.splitext(os.path.basename(input_file))[0]

            # Create a copy of metadata
            document_metadata = metadata.copy()

            keys = [
                "crawl_path",
                "crawling_timestamp",
                "scrapping_timestamp",
                "conversion_timestamp",
                "depth",
                "file_extension",
                "file_size_bytes",
                "save_path",
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
            output_folder = os.path.join(self.config.save_location, "flattened_data")
            os.makedirs(output_folder, exist_ok=True)
            json_path = os.path.join(output_folder, f"{doc_filename}_flattened_data.json")

            # Add metadata to flattened data for JSON
            json_data_with_metadata = {"data": flattened_data, "metadata": document_metadata}

            self._save_json_to_file(json_data_with_metadata, json_path)
            self.logger.info(f"JSON data saved to: {json_path}")

            # Convert to markdown
            markdown_content = self._convert_html_to_markdown_list(flattened_data)

            # Add metadata as YAML frontmatter to markdown
            if document_metadata:
                metadata_yaml = yaml.safe_dump(document_metadata)
                markdown_content = f"{markdown_content}---\n\n### Metadata:\n{metadata_yaml}---"

            # Save markdown
            output_folder = os.path.join(self.config.save_location, "md")
            os.makedirs(output_folder, exist_ok=True)
            markdown_path = os.path.join(output_folder, f"{doc_filename}.md")
            self._save_markdown_to_file(markdown_content, markdown_path)
            self.logger.info(f"Markdown content saved to: {markdown_path}")

            # Update metadata with processed file locations
            metadata["processed_file_locations"] = {"json": json_path, "markdown": markdown_path}

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
            with open(input_file, encoding="utf-8") as file:
                html_content = file.read()

            # Extract JSON data from the page-input attribute
            json_match = re.search(r'page-input="(.*?)"', html_content)
            if not json_match:
                return "Error: No JSON data found in the HTML file"

            # Unescape HTML entities in the JSON string
            json_str = unescape(json_match.group(1))
            json_str = json_str.replace("&nbsp;", " ")
            json_str = json_str.replace("\u200b", "")

            # Handle URLs with base64 content
            json_str = re.sub(
                r"(https://portal\.apacorp\.net.*?data:image/[^;]+;base64,)[A-Za-z0-9+/=]+",
                r"\1REMOVED64ENCODING",
                json_str,
            )

            # Remove HTML formatting tags
            format_tags = ["strong", "em", "span", "sub", "sup", "mark", "small"]
            for tag in format_tags:
                json_str = re.sub(r"<" + tag + r"[^>]*>", "", json_str)
                json_str = re.sub(r"</" + tag + r">", "", json_str)

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

    def _flatten_content_data(self, data_dict: dict[str, Any]) -> list[dict[str, Any]]:
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
            if "rows" in data_dict:
                for row in data_dict["rows"]:
                    if "columns" in row:
                        for column in row["columns"]:
                            if "panels" in column:
                                for panel in column["panels"]:
                                    if "contents" in panel:
                                        for content in panel["contents"]:
                                            body = content.get("body", "")

                                            content_data = {
                                                "source": "row_content",
                                                "name": content.get("name", ""),
                                                "title": content.get("title", ""),
                                                "titleElement": content.get("titleElement", ""),
                                                "body": body,
                                                "sequence": content.get("sequence", ""),
                                                "content_id": panel.get("id", ""),
                                            }
                                            flattened_data.append(content_data)

            # Process namedContents
            if "namedContents" in data_dict:
                for content in data_dict["namedContents"]:
                    body = content.get("body", "")

                    content_data = {
                        "source": "named_content",
                        "name": content.get("name", ""),
                        "title": content.get("title", ""),
                        "titleElement": content.get("titleElement", ""),
                        "body": body,
                        "sequence": content.get("sequence", ""),
                        "content_id": content.get("id", ""),
                    }
                    flattened_data.append(content_data)

            # Filter out items with specific names and empty content
            filtered_data = [
                item
                for item in flattened_data
                if item["name"] not in self.excluded_names
                and not ((item["body"] == "") and (item["title"] == ""))
            ]

            # Simplify the output structure
            simplified_data = [
                {"title": item["title"], "body": item["body"]} for item in filtered_data
            ]

            self.logger.info(f"Flattened {len(simplified_data)} content items")
            return simplified_data

        except Exception as e:
            error_msg = f"Error flattening content data: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(
                CustomConverterErrorRecord("content_flattening", error_msg)
            )
            return []

    def _convert_html_to_markdown_list(self, html_content_list: list[dict[str, Any]]) -> str:
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
            title = item.get("title", "")
            body = item.get("body", "")

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
            soup = BeautifulSoup(html, "html.parser")
            return self._process_tag(soup)

        except Exception as e:
            error_msg = f"Error converting HTML to Markdown: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(CustomConverterErrorRecord("html_conversion", error_msg))
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
        if tag.name == "div":
            result.append(self._process_children(tag))

        elif tag.name == "h1":
            result.append(f"# {tag.get_text().strip()}\n\n")

        elif tag.name == "h2":
            result.append(f"## {tag.get_text().strip()}\n\n")

        elif tag.name == "h3":
            result.append(f"### {tag.get_text().strip()}\n\n")

        elif tag.name == "h4":
            result.append(f"#### {tag.get_text().strip()}\n\n")

        elif tag.name == "p":
            result.append(f"{self._process_children(tag)}\n\n")

        elif tag.name == "a":
            href = tag.get("href", "")
            text = tag.get_text().strip()
            result.append(f"[{text}]({href})")

        elif tag.name == "img":
            src = tag.get("src", "")
            alt = tag.get("alt", "")

            # Check if this is a special image pattern
            for pattern, replacement in self.special_image_patterns.items():
                if pattern in src:
                    result.append(f"{replacement} ")
                    return "".join(result)

            # Regular image
            result.append(f"![{alt}]({src})")

        elif tag.name == "ul":
            items = []
            for li in tag.find_all("li", recursive=False):
                items.append(f"* {self._process_children(li)}")
            result.append("\n".join(items) + "\n\n")

        elif tag.name == "li":
            result.append(self._process_children(tag))

        elif tag.name == "blockquote":
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
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            return output_file

        except Exception as e:
            error_msg = f"Error saving markdown to {output_file}: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(CustomConverterErrorRecord(output_file, error_msg))
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
            with open(output_file, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            return output_file

        except Exception as e:
            error_msg = f"Error saving JSON to {output_file}: {str(e)}"
            self.logger.error(error_msg)
            self.conversion_errors.append(CustomConverterErrorRecord(output_file, error_msg))
            return ""

    def get_error_report(self) -> list[dict[str, Any]]:
        """
        Get a list of errors that occurred during conversion.

        Returns:
            List of error dictionaries
        """
        return [
            {
                "file_path": error.file_path,
                "error_message": error.error_message,
                "conversion_timestamp": error.conversion_timestamp.isoformat(),
            }
            for error in self.conversion_errors
        ]


# -------------------------------------------------------------------------
# Document processing
# -------------------------------------------------------------------------
