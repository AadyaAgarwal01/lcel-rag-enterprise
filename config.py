"""Shared configuration helpers for Google Gemini credentials."""

from __future__ import annotations

import os


def get_google_api_key() -> str:
    """
    Resolve the Google API key from environment variables.

    Supports both GOOGLE_API_KEY and GEMINI_API_KEY (LangChain fallback).

    Obtain a key at: https://aistudio.google.com/apikey
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return api_key.strip().strip('"').strip("'")


def get_google_model() -> str:
    """Chat model used for answer generation."""
    return os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")


def get_google_embedding_model() -> str:
    """Embedding model used for vector indexing and retrieval."""
    return os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
