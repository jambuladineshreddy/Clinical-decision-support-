"""
rag/ingest.py
-------------
Clinical knowledge base ingestion pipeline.

Run once before starting the application:
    python rag/ingest.py          # first-time ingestion
    python rag/ingest.py --reset  # full rebuild (drop existing collections)

Pipeline:
    TextLoader -> RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> ChromaDB

Each chunk is stored with structured metadata for filtering and provenance tracking.
"""

import os
import sys
import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from core.config import BASE_DIR, CHROMA_PATH, DOCS_DIR, SYSTEM_CONFIG, logger
from rag.vectorstore import (
    get_embedding_function,
    get_chroma_client,
    COLLECTION_NAMES,
)


# ─────────────────────────────────────────────
# Document Registry
# Add entries here when expanding the knowledge base.
# ─────────────────────────────────────────────
DOCUMENTS = [
    {
        "path": os.path.join(str(DOCS_DIR), "diabetes_sample.txt"),
        "disease": "diabetes",
        "source_type": "sample",
        "source_name": "Development Sample Clinical Document",
        "document_name": "diabetes_sample.txt",
    },
    {
        "path": os.path.join(str(DOCS_DIR), "anemia_sample.txt"),
        "disease": "anemia",
        "source_type": "sample",
        "source_name": "Development Sample Clinical Document",
        "document_name": "anemia_sample.txt",
    },
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=SYSTEM_CONFIG.chunk_size,
    chunk_overlap=SYSTEM_CONFIG.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_and_split(doc_entry: dict) -> list[Document]:
    """Load a document and split it into chunks with provenance metadata."""
    path = doc_entry["path"]
    if not os.path.exists(path):
        logger.warning(f"Document not found, skipping: {path}")
        return []

    loader = TextLoader(path, encoding="utf-8")
    raw_docs = loader.load()
    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "disease": doc_entry["disease"],
            "source_type": doc_entry["source_type"],
            "source_name": doc_entry["source_name"],
            "document_name": doc_entry["document_name"],
            "chunk_id": i,
        })

    logger.info(f"Loaded '{doc_entry['document_name']}': {len(chunks)} chunks produced.")
    return chunks


def ingest_all(reset: bool = False):
    """
    Ingest all registered documents into ChromaDB.

    Parameters
    ----------
    reset : bool
        Delete existing collections before ingesting for a clean rebuild.
    """
    print("=" * 65)
    print("  CLINICAL KNOWLEDGE BASE INGESTION")
    print("=" * 65)

    embeddings = get_embedding_function()
    client = get_chroma_client()

    # Group chunks by disease collection
    by_disease: dict[str, list] = {d: [] for d in COLLECTION_NAMES}
    for doc_entry in DOCUMENTS:
        disease = doc_entry["disease"]
        chunks = load_and_split(doc_entry)
        if chunks:
            by_disease[disease].extend(chunks)

    # Ingest per disease collection
    for disease, chunks in by_disease.items():
        cname = COLLECTION_NAMES[disease]

        if reset:
            try:
                client.delete_collection(name=cname)
                logger.info(f"Dropped existing collection: {cname}")
            except Exception:
                pass

        if not chunks:
            logger.warning(f"No documents registered for disease '{disease}' — skipping.")
            continue

        from langchain_chroma import Chroma
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=cname,
            persist_directory=str(CHROMA_PATH),
        )
        logger.info(f"Ingested {len(chunks)} chunks into collection '{cname}'.")

    print("\n  Ingestion complete.")
    print(f"  ChromaDB stored at: {CHROMA_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical documents into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collections before ingesting (full rebuild).",
    )
    args = parser.parse_args()
    ingest_all(reset=args.reset)
