"""
Vector store construction and document indexing.

Default backend: Chroma (embedded, persistent). Embeddings use Google Gemini
(`GoogleGenerativeAIEmbeddings`). For Pinecone, swap the store factory while
keeping the same embedding + retriever interfaces.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_google_api_key, get_google_embedding_model


def create_embeddings(
    *,
    model: str | None = None,
    **kwargs: Any,
) -> GoogleGenerativeAIEmbeddings:
    """
    Factory for Gemini embedding models.

    Default: `gemini-embedding-001` — replaces deprecated `text-embedding-004`.
    Re-index documents after changing embedding models.
    """
    return GoogleGenerativeAIEmbeddings(
        model=model or get_google_embedding_model(),
        google_api_key=get_google_api_key(),
        **kwargs,
    )


def build_vector_store(
    *,
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> Chroma:
    """
    Create or reconnect to a persistent Chroma collection.

    Persistence enables warm restarts without re-embedding every document.
    """
    return Chroma(
        collection_name=collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
        embedding_function=embeddings or create_embeddings(),
        persist_directory=persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    )


def index_documents(
    documents: list[Document],
    *,
    vector_store: VectorStore | None = None,
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
) -> VectorStore:
    """
    Embed and index document chunks.

    Returns:
        A VectorStore ready for retriever construction.
    """
    if not documents:
        raise ValueError("Cannot index an empty document list.")

    embedding_model = embeddings or create_embeddings()

    if vector_store is None:
        store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        )
        return store

    vector_store.add_documents(documents)
    return vector_store
