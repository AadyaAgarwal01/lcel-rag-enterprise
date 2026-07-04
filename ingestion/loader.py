"""
Document loading utilities.

Supports mock data, plain text, and research-paper PDFs via PyPDFLoader.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md"}


def load_mock_document() -> list[Document]:
    """
    Return a deterministic mock document for local development and CI.

    Keeps the pipeline runnable without external files while still exercising
    chunking, embedding, and retrieval end-to-end.
    """
    content = """
    Retrieval-Augmented Generation (RAG) combines a retriever with a language model.
    The retriever fetches relevant passages from a vector store based on semantic similarity.
    LangChain Expression Language (LCEL) composes runnables with the pipe operator (|),
    enabling declarative pipelines with built-in streaming, batching, and fallbacks.

    Production RAG systems should enforce least-privilege API key access, persist vectors
    in a managed store (Chroma, Pinecone, Weaviate), and return structured outputs that
    include both the generated answer and the source documents used for grounding.
    """
    return [
        Document(
            page_content=content.strip(),
            metadata={
                "source": "mock://rag-overview",
                "title": "RAG Overview",
                "doc_id": "mock-001",
            },
        )
    ]


def load_text_documents(path: str | Path, *, source_name: str | None = None) -> list[Document]:
    """
    Load plain-text files from disk.

    Args:
        path: File or directory path.
        source_name: Optional override for metadata["source"].

    Raises:
        FileNotFoundError: When path does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    documents: list[Document] = []

    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.txt"))

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name or str(file_path),
                    "title": file_path.stem,
                },
            )
        )

    return documents


def load_pdf(path: str | Path) -> list[Document]:
    """
    Load a research paper PDF with per-page metadata.

    Each page becomes a Document with `page` in metadata for citations.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = PyPDFLoader(str(file_path)).load()
    for page in pages:
        page.metadata.setdefault("source", str(file_path))
        page.metadata.setdefault("title", file_path.stem)
        page.metadata["doc_id"] = file_path.name
    return pages


def load_uploaded_file(filename: str, file_bytes: bytes) -> list[Document]:
    """
    Load an uploaded research paper from in-memory bytes.

    Writes to a secure temp file because PDF loaders expect a filesystem path.
    Temp files are deleted when the context exits.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_UPLOAD_SUFFIXES))}"
        )

    if suffix == ".pdf":
        # delete=False: Windows cannot reopen NamedTemporaryFile while it is open.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            temp_path = Path(tmp.name)

        try:
            pages = load_pdf(temp_path)
            for page in pages:
                page.metadata["source"] = filename
                page.metadata["title"] = Path(filename).stem
                page.metadata["doc_id"] = filename
            return pages
        finally:
            temp_path.unlink(missing_ok=True)

    text = file_bytes.decode("utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "title": Path(filename).stem,
                "doc_id": filename,
            },
        )
    ]


def load_uploaded_files(uploads: list[tuple[str, bytes]]) -> list[Document]:
    """Batch-load multiple uploaded files into a single document list."""
    documents: list[Document] = []
    for filename, file_bytes in uploads:
        documents.extend(load_uploaded_file(filename, file_bytes))
    return documents
