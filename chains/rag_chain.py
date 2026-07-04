"""
Main RAG pipeline composed with LangChain Expression Language (LCEL).

Pipeline shape:
  1. Retrieve documents once (with `.with_fallbacks()` resilience)
  2. Inject formatted context + preserve source documents
  3. Prompt -> LLM -> structured JSON output
"""

from __future__ import annotations

import json
import logging
from operator import itemgetter
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_google_api_key, get_google_model

from prompts.rag_prompt import build_rag_prompt
from tools.retriever import (
    create_resilient_retriever,
    format_docs,
    serialize_source_documents,
)

logger = logging.getLogger(__name__)


class RAGResponse(TypedDict):
    """Structured JSON contract returned to callers."""

    answer: str
    source_documents: list[dict[str, Any]]


def _create_llm(*, model: str | None = None, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Factory for the Gemini chat model.

    temperature=0.0 prioritizes factual consistency in enterprise Q&A workloads.
    """
    return ChatGoogleGenerativeAI(
        model=model or get_google_model(),
        temperature=temperature,
        google_api_key=get_google_api_key(),
    )


def _build_json_response(state: dict[str, Any]) -> RAGResponse:
    """Merge LLM answer with retrieved sources into the public response schema."""
    return {
        "answer": state["answer"],
        "source_documents": serialize_source_documents(state["source_documents"]),
    }


def _normalize_question_input(question: str) -> dict[str, str]:
    """Wrap bare question strings so downstream LCEL branches receive a dict."""
    return {"question": question}


def build_rag_chain(
    vector_store: VectorStore,
    *,
    llm: BaseChatModel | None = None,
    include_history: bool = False,
) -> Runnable[str, RAGResponse]:
    """
    Assemble the full RAG pipeline using LCEL pipe composition.

    Canonical context-injection shape (as specified):
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }

    Production optimization: retrieve once, then derive both `context` and
    `source_documents` from the same document list to avoid duplicate vector
    DB round-trips under load.

    Args:
        vector_store: Indexed vector store backing retrieval.
        llm: Optional chat model override (useful for testing).
        include_history: Forwarded to prompt builder for multi-turn support.

    Returns:
        A Runnable accepting a question string and returning structured JSON.
    """
    retriever = create_resilient_retriever(vector_store)
    prompt = build_rag_prompt(include_history=include_history)
    model = llm or _create_llm()

    # --- Retrieval + context injection (single retriever call) ---
    #
    # Equivalent to the requested pattern, without double retrieval:
    #   {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #
    retrieval_and_context: Runnable[str, dict[str, Any]] = (
        RunnableLambda(_normalize_question_input)
        .assign(source_documents=itemgetter("question") | retriever)
        .assign(context=lambda state: format_docs(state["source_documents"]))
    )

    # --- Generation leg: prompt | llm | parser ---
    generation = prompt | model | StrOutputParser()

    # --- Merge answer + sources into JSON contract ---
    rag_chain: Runnable[str, RAGResponse] = (
        retrieval_and_context.assign(answer=generation)
        | RunnableLambda(_build_json_response)
    )

    return rag_chain.with_config({"run_name": "rag_pipeline"})


def query_rag(
    chain: Runnable[str, RAGResponse],
    question: str,
) -> RAGResponse:
    """
    Execute the RAG chain and return a JSON-serializable dict.

    Wraps `.invoke()` for ergonomic use from scripts and API handlers.
    """
    logger.info("Running RAG query.")
    result = chain.invoke(question)
    return result


def dumps_rag_response(response: RAGResponse, *, indent: int = 2) -> str:
    """Serialize a RAGResponse to a valid JSON string for logging or HTTP bodies."""
    return json.dumps(response, indent=indent, ensure_ascii=False)
