"""Typed exception hierarchy for :mod:`dstoolbox.rag_funcs`."""

from __future__ import annotations


class RAGFuncsError(Exception):
    """Base for :mod:`dstoolbox.rag_funcs` runtime errors."""


class ConversionError(RAGFuncsError):
    """Raised when a document conversion fails."""


class ChunkingError(RAGFuncsError):
    """Raised when a chunking operation cannot produce valid chunks."""


class VectorStoreError(RAGFuncsError):
    """Raised when a vector-store operation fails (embedding, upsert, query)."""


__all__ = [
    "RAGFuncsError",
    "ConversionError",
    "ChunkingError",
    "VectorStoreError",
]
