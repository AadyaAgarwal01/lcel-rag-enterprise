"""
Retriever tools wrapped for LCEL pipelines.

Includes:
  - Vector store retriever with configurable top-k
  - Resilient retriever with `.with_fallbacks()` for timeout / outage handling
  - Document formatting helpers for prompt context injection
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


def format_docs(documents: list[Document]) -> str:
    """
    Collapse retrieved chunks into a single context string for the LLM.

    Metadata headers improve citation quality in downstream answers.
    """
    if not documents:
        return "No relevant context was retrieved."

    formatted_blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        header = f"[{index}] source={source}"
        if title:
            header += f" | title={title}"
        formatted_blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_blocks)


def serialize_source_documents(documents: list[Document]) -> list[dict[str, Any]]:
    """
    Convert LangChain Documents into JSON-serializable dicts for API responses.

    Keeps only stable, client-facing fields — avoids leaking internal metadata.
    """
    serialized: list[dict[str, Any]] = []
    for doc in documents:
        serialized.append(
            {
                "content": doc.page_content,
                "metadata": {
                    key: value
                    for key, value in doc.metadata.items()
                    if key in {"source", "title", "doc_id", "page"}
                },
            }
        )
    return serialized


def create_vector_retriever(
    vector_store: VectorStore,
    *,
    search_type: str = "similarity",
    top_k: int | None = None,
) -> BaseRetriever:
    """
    Build a standard vector store retriever.

    `similarity` is the default; switch to `mmr` in high-redundancy corpora
    to diversify retrieved passages at a small latency cost.
    """
    k = top_k or int(os.getenv("RETRIEVER_TOP_K", "4"))
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def _empty_retrieval(_: str) -> list[Document]:
    """
    Graceful degradation path when the primary retriever fails.

    Returning an empty list lets the LLM answer with an explicit "no context"
    signal rather than crashing the entire request.
    """
    logger.warning("Primary retriever failed; returning empty context via fallback.")
    return []


def create_fallback_retriever() -> Runnable[str, list[Document]]:
    """Secondary retriever invoked by LCEL `.with_fallbacks()` on primary failure."""
    return RunnableLambda(_empty_retrieval).with_config({"run_name": "fallback_retriever"})


def create_resilient_retriever(
    vector_store: VectorStore,
    *,
    search_type: str = "similarity",
    top_k: int | None = None,
) -> Runnable[str, list[Document]]:
    """
    Wrap the vector retriever with LCEL fallbacks for production resilience.

    Catches broad Exception types to cover network timeouts, rate limits, and
    transient vector DB outages. Narrow `exceptions_to_handle` in strict environments.
    """
    primary = create_vector_retriever(
        vector_store,
        search_type=search_type,
        top_k=top_k,
    )
    fallback = create_fallback_retriever()

    return primary.with_fallbacks(
        fallbacks=[fallback],
        exceptions_to_handle=(Exception,),
    ).with_config({"run_name": "resilient_vector_retriever"})
