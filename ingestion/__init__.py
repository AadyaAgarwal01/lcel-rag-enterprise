"""Document ingestion: loading, chunking, and vector indexing."""

from ingestion.indexer import build_vector_store, index_documents
from ingestion.loader import (
    load_mock_document,
    load_text_documents,
    load_uploaded_file,
    load_uploaded_files,
)
from ingestion.splitter import create_text_splitter, split_documents

__all__ = [
    "load_mock_document",
    "load_text_documents",
    "load_uploaded_file",
    "load_uploaded_files",
    "create_text_splitter",
    "split_documents",
    "build_vector_store",
    "index_documents",
]
