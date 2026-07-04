"""
Shared RAG orchestration used by the CLI and web UI.

Keeps ingestion, indexing, and querying in one place so UI and scripts
stay aligned with the modular package layout.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from chains.rag_chain import RAGResponse, build_rag_chain, query_rag
from ingestion.indexer import build_vector_store, index_documents
from ingestion.splitter import split_documents

logger = logging.getLogger(__name__)


def get_vector_store() -> VectorStore:
    """Connect to the persistent Chroma collection (creates dir if missing)."""
    return build_vector_store()


def ingest_documents(raw_documents: list) -> dict[str, Any]:
    """
    Chunk and index documents into the persistent vector store.

    Returns:
        Summary dict with chunk_count and indexed file names.
    """
    if not raw_documents:
        raise ValueError("No documents provided for ingestion.")

    chunks = split_documents(raw_documents)
    vector_store = get_vector_store()
    index_documents(chunks, vector_store=vector_store)

    sources = sorted(
        {
            doc.metadata.get("source", "unknown")
            for doc in raw_documents
        }
    )
    file_count = len(
        {
            doc.metadata.get("doc_id") or doc.metadata.get("source", "unknown")
            for doc in raw_documents
        }
    )

    logger.info("Indexed %d chunks from %d source(s).", len(chunks), len(sources))
    return {
        "chunk_count": len(chunks),
        "document_count": file_count,
        "sources": sources,
    }


def build_chain(vector_store: VectorStore | None = None) -> Runnable[str, RAGResponse]:
    """Build the LCEL RAG chain against the current vector store."""
    store = vector_store or get_vector_store()
    return build_rag_chain(store)


def ask(question: str, *, vector_store: VectorStore | None = None) -> RAGResponse:
    """Run a grounded question against indexed research papers."""
    chain = build_chain(vector_store)
    return query_rag(chain, question)


def collection_stats() -> dict[str, Any]:
    """Return basic stats about the indexed collection."""
    store = get_vector_store()
    collection = store._collection  # noqa: SLF001 — Chroma internal; acceptable for local UI stats
    count = collection.count()
    return {"chunk_count": count}
