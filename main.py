"""
RAG application entry point.

Workflow:
  1. Load environment configuration (API keys via .env — never hard-code secrets)
  2. Ingest a mock document -> chunk -> embed -> index in Chroma
  3. Build the LCEL RAG chain with retriever fallbacks
  4. Query and emit structured JSON: {"answer", "source_documents"}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is importable when executed as `python main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chains.rag_chain import dumps_rag_response
from ingestion.loader import load_mock_document
from pipeline import ask, ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(question: str) -> dict:
    """
    End-to-end ingestion + query pipeline.

    Args:
        question: Natural-language query passed to the RAG chain.

    Returns:
        Dict with keys `answer` and `source_documents`.
    """
    # --- Ingestion ---
    logger.info("Loading mock document.")
    raw_documents = load_mock_document()

    logger.info("Splitting and indexing documents.")
    ingest_documents(raw_documents)

    logger.info("Query: %s", question)
    response = ask(question)
    return response


def main() -> None:
    """CLI entry: load config, run pipeline, print JSON to stdout."""
    # Load secrets from .env (git-ignored). `.env.example` documents required keys.
    load_dotenv(PROJECT_ROOT / ".env")

    default_question = "What is RAG and how does LCEL help compose pipelines?"
    question = sys.argv[1] if len(sys.argv) > 1 else default_question

    try:
        result = run_pipeline(question)
        print(dumps_rag_response(result))
    except EnvironmentError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("RAG pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
