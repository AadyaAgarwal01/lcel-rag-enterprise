"""
Streamlit UI for uploading research papers and querying the RAG pipeline.

Run from project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_google_embedding_model, get_google_model
from ingestion.loader import SUPPORTED_UPLOAD_SUFFIXES, load_uploaded_files
from pipeline import ask, collection_stats, ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_LABEL = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_ingest" not in st.session_state:
        st.session_state.last_ingest = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Index status")
        try:
            stats = collection_stats()
            st.metric("Indexed chunks", stats["chunk_count"])
        except EnvironmentError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.warning(f"Could not read index: {exc}")

        if st.session_state.last_ingest:
            info = st.session_state.last_ingest
            st.success(
                f"Last upload: **{info['document_count']}** file(s), "
                f"**{info['chunk_count']}** chunks indexed."
            )
            with st.expander("Indexed sources"):
                for source in info["sources"]:
                    st.write(f"- `{source}`")

        st.divider()
        st.subheader("Gemini models")
        st.text(f"Chat: {get_google_model()}")
        st.text(f"Embeddings: {get_google_embedding_model()}")

        st.divider()
        st.caption(
            "Set `GOOGLE_API_KEY` in `.env` (AI Studio key starting with AIza). "
            "Re-index papers after changing the embedding model."
        )


def _render_upload_section() -> None:
    st.subheader("Upload research papers")
    st.caption(f"Supported formats: {SUPPORTED_LABEL}")

    uploaded_files = st.file_uploader(
        "Choose PDF or text files",
        type=[suffix.lstrip(".") for suffix in SUPPORTED_UPLOAD_SUFFIXES],
        accept_multiple_files=True,
        help="Papers are chunked, embedded, and appended to your local Chroma index.",
    )

    if st.button("Index uploaded papers", type="primary", disabled=not uploaded_files):
        _handle_index(uploaded_files)


def _handle_index(uploaded_files) -> None:
    try:
        uploads = [(file.name, file.getvalue()) for file in uploaded_files]
        with st.spinner("Parsing, chunking, and embedding documents..."):
            documents = load_uploaded_files(uploads)
            summary = ingest_documents(documents)

        st.session_state.last_ingest = summary
        st.success(
            f"Indexed {summary['document_count']} file(s) "
            f"into {summary['chunk_count']} searchable chunks."
        )
    except EnvironmentError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Indexing failed.")
        st.error("Indexing failed. Check logs for details.")


def _render_chat_section() -> None:
    st.subheader("Ask your research library")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                _render_sources(message["sources"])

    question = st.chat_input("Ask a question about your uploaded papers...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            _handle_question(question)


def _handle_question(question: str) -> None:
    try:
        with st.spinner("Retrieving context and generating answer..."):
            response = ask(question)

        answer = response["answer"]
        sources = response["source_documents"]

        st.markdown(answer)
        _render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except EnvironmentError as exc:
        st.error(str(exc))
    except Exception:
        logger.exception("Query failed.")
        st.error("Query failed. Ensure papers are indexed and your API key is valid.")


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No source passages retrieved.")
        return

    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            title = metadata.get("title") or metadata.get("source") or f"Source {index}"
            page = metadata.get("page")
            header = f"**{index}. {title}**"
            if page is not None:
                header += f" — page {page + 1 if isinstance(page, int) else page}"
            st.markdown(header)
            st.write(source.get("content", ""))
            st.divider()


def _render_raw_json_toggle() -> None:
    if not st.session_state.messages:
        return

    last_assistant = next(
        (msg for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"),
        None,
    )
    if not last_assistant:
        return

    with st.expander("Raw JSON response"):
        payload = {
            "answer": last_assistant["content"],
            "source_documents": last_assistant.get("sources", []),
        }
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(
        page_title="Research RAG",
        page_icon="📄",
        layout="wide",
    )

    _init_session_state()
    _render_sidebar()

    st.title("Research Paper RAG")
    st.write(
        "Upload PDF or text research papers, index them locally, "
        "and ask grounded questions with cited source passages."
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_upload_section()
    with right:
        _render_chat_section()
        _render_raw_json_toggle()


if __name__ == "__main__":
    main()
