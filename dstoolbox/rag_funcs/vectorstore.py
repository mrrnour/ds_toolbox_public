"""
RAG (Retrieval-Augmented Generation) utility functions for document processing,
web scraping, chunking, and vector store operations.
"""

# Standard library imports
import logging
import os
from dataclasses import dataclass

# Third-party imports
import pandas as pd

# Docling imports
# LangChain imports
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Local imports


@dataclass
class VectorstoreConfig:
    """Configuration inputs for :class:`VectorStoreProcessor`.

    Parameters
    ----------
    documents : list of langchain_core.documents.Document
        Documents (chunks) to embed.
    save_location : str
        Path where the vector index / collection is persisted.
    embed_model_id : str
        HuggingFace model id used by :class:`HuggingFaceEmbeddings`.
    store_type : str, default ``'chroma'``
        Which backend to build (``'chroma'``, ``'faiss'``, ``'milvus'``).
    overwrite : bool, default True
        If True, wipe any existing store at ``save_location``.
    show_progress : bool, default True
        If True, render tqdm progress bars.
    """

    documents: list[Document]
    save_location: str
    embed_model_id: str
    store_type: str = "chroma"
    overwrite: bool = True
    show_progress: bool = True


@dataclass
class VectorstoreErrorRecord:
    """Record describing a single embedding failure.

    Parameters
    ----------
    document_index : int
        Position of the offending document within the input batch.
    error_message : str
        Human-readable error string.
    document_id : str, optional
        Backend-assigned document id, if any.
    """

    document_index: int
    error_message: str
    document_id: str | None = None


class VectorStoreProcessor:
    """Embed document chunks and build a persistent vector store.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for progress and error reporting.
    config : VectorstoreConfig
        Pipeline configuration (documents, backend, embedding model).

    Attributes
    ----------
    logger : logging.Logger
    config : VectorstoreConfig
    embedding_errors : list of VectorstoreErrorRecord
    embed_model : langchain_huggingface.embeddings.HuggingFaceEmbeddings
    """

    def __init__(self, logger: logging.Logger, config: VectorstoreConfig):
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

    def meta4chroma(self, documents: list[Document]) -> list[Document]:
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
                    if hasattr(doc, "metadata"):
                        for key, value in doc.metadata.items():
                            if isinstance(value, str | int | float | bool):
                                metadata[key] = value
                            else:
                                try:
                                    metadata[key] = str(value)
                                except (TypeError, ValueError, AttributeError):
                                    # Skip metadata entries whose repr is not producible.
                                    continue

                    cleaned_docs.append(
                        Document(
                            page_content=doc.page_content
                            if hasattr(doc, "page_content")
                            else str(doc),
                            metadata=metadata,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error converting metadata for chroma {i}: {str(e)}")
                    self.embedding_errors.append(VectorstoreErrorRecord(i, str(e)))

                # Update progress bar
                pbar.update(1)

        return cleaned_docs

    def create_vector_store(self):
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

    def get_error_report(
        self, save_to_csv: bool = False, output_path: str | None = None
    ) -> pd.DataFrame | None:
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
            "document_index": [e.document_index for e in self.embedding_errors],
            "error_message": [e.error_message for e in self.embedding_errors],
            "document_id": [e.document_id for e in self.embedding_errors],
        }
        error_df = pd.DataFrame(error_data)

        if save_to_csv:
            error_csv_path = output_path or os.path.join(
                self.config.save_location, "embedding_errors.csv"
            )
            error_df.to_csv(error_csv_path, index=False)
            self.logger.warning(f"Encountered {len(self.embedding_errors)} errors during embedding")

        return error_df
