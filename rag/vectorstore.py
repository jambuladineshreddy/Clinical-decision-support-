"""
rag/vectorstore.py
------------------
ChromaDB client and collection management with singleton caching.

Manages persistent collections:
    - diabetes_knowledge
    - anemia_knowledge
"""

from __future__ import annotations
import os
import threading
from typing import Any

import chromadb
from chromadb.config import Settings

from core.config import CHROMA_PATH, SYSTEM_CONFIG, DISEASE_CONFIGS, logger

# Single source of truth from config
EMBEDDING_MODEL = SYSTEM_CONFIG.embedding_model
COLLECTION_NAMES = {
    disease: cfg.collection_name for disease, cfg in DISEASE_CONFIGS.items()
}

_cache_lock = threading.Lock()
_vectorstore_cache: dict[str, Any] = {}
_chroma_client_singleton: chromadb.PersistentClient | None = None
_embedding_singleton = None


def get_embedding_function():
    """
    Return thread-safe singleton LangChain HuggingFace embedding function.
    Avoids re-allocating PyTorch transformer weights on each call.
    """
    global _embedding_singleton
    if _embedding_singleton is not None:
        return _embedding_singleton

    with _cache_lock:
        if _embedding_singleton is not None:
            return _embedding_singleton

        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Loading singleton HuggingFace embeddings: {EMBEDDING_MODEL}")
        _embedding_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": SYSTEM_CONFIG.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return _embedding_singleton


def get_chroma_client() -> chromadb.PersistentClient:
    """Return persistent ChromaDB client singleton."""
    global _chroma_client_singleton
    if _chroma_client_singleton is not None:
        return _chroma_client_singleton

    with _cache_lock:
        if _chroma_client_singleton is not None:
            return _chroma_client_singleton

        os.makedirs(str(CHROMA_PATH), exist_ok=True)
        _chroma_client_singleton = chromadb.PersistentClient(path=str(CHROMA_PATH))
        return _chroma_client_singleton


def collection_exists(disease: str) -> bool:
    """Check whether collection exists and contains document chunks."""
    try:
        client = get_chroma_client()
        cname = COLLECTION_NAMES.get(disease, f"{disease}_knowledge")
        col = client.get_collection(name=cname)
        return col.count() > 0
    except Exception:
        return False


def get_langchain_vectorstore(disease: str):
    """
    Return cached LangChain Chroma vectorstore instance for the specified disease.
    """
    if disease in _vectorstore_cache:
        return _vectorstore_cache[disease]

    with _cache_lock:
        if disease in _vectorstore_cache:
            return _vectorstore_cache[disease]

        from langchain_chroma import Chroma
        embeddings = get_embedding_function()
        cname = COLLECTION_NAMES.get(disease, f"{disease}_knowledge")
        vs = Chroma(
            collection_name=cname,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PATH),
        )
        _vectorstore_cache[disease] = vs
        return vs
