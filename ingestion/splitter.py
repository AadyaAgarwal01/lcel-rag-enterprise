"""
Text splitting for embedding-friendly chunks.

Uses RecursiveCharacterTextSplitter, which recursively splits on paragraph,
sentence, and word boundaries to preserve semantic coherence.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a configured RecursiveCharacterTextSplitter.

    Environment overrides:
        CHUNK_SIZE (default: 1000)
        CHUNK_OVERLAP (default: 200)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200")),
        length_function=len,
        # Prefer natural language boundaries before hard character cuts.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into retrieval-sized chunks while preserving metadata."""
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
