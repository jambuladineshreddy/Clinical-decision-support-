"""
rag/retriever.py
----------------
Disease-specific clinical evidence retriever with calibrated scoring.

Retrieves top-k relevant document chunks from the disease collection.
Maps raw vector distance to normalized relevance confidence scores.

Function:
    retrieve_evidence(disease, query, k=4) -> list of dicts
"""

from __future__ import annotations
import math
from typing import Any

from core.config import SYSTEM_CONFIG, logger
from core.schemas import EvidenceChunk
from rag.vectorstore import get_langchain_vectorstore, collection_exists


def _distance_to_similarity(distance: float) -> float:
    """
    Convert normalized embedding L2 distance into a bounded [0, 1] similarity score.
    With normalized vectors: d^2 = 2 - 2*cos(theta).
    cos_sim = 1 - (d^2)/2.
    """
    if distance is None:
        return 0.0
    try:
        cos_sim = 1.0 - (float(distance) ** 2) / 2.0
        return max(0.0, min(1.0, cos_sim))
    except Exception:
        return max(0.0, 1.0 / (1.0 + float(distance)))


def retrieve_evidence(disease: str, query: str, k: int = 4) -> list[dict[str, Any]]:
    """
    Retrieve top-k relevant document chunks with calibrated relevance scoring.
    Returns empty list immediately when k=0 (for pipeline testing without RAG).
    """
    if k <= 0:
        return []

    if not collection_exists(disease):
        raise RuntimeError(
            f"ChromaDB collection for '{disease}' is empty or missing. "
            "Execute 'python rag/ingest.py' to index clinical knowledge."
        )

    vectorstore = get_langchain_vectorstore(disease)
    results = vectorstore.similarity_search_with_score(query, k=k)

    evidence_list: list[dict[str, Any]] = []
    min_thresh = SYSTEM_CONFIG.min_similarity_threshold

    for doc, raw_score in results:
        sim_score = _distance_to_similarity(raw_score)
        is_relevant = sim_score >= min_thresh

        chunk = EvidenceChunk(
            content=doc.page_content.strip(),
            source_name=doc.metadata.get("source_name", "Clinical Guideline Sample"),
            document_name=doc.metadata.get("document_name", "Unknown Document"),
            source_type=doc.metadata.get("source_type", "sample"),
            disease=doc.metadata.get("disease", disease),
            chunk_id=int(doc.metadata.get("chunk_id", -1)),
            score=round(float(raw_score), 4) if raw_score is not None else None,
            similarity_score=round(sim_score, 4),
            is_relevance_passed=is_relevant,
        )
        evidence_list.append(chunk.to_dict())

    logger.info(
        f"Retrieved {len(evidence_list)} chunks for '{disease}' "
        f"(top similarity: {evidence_list[0]['similarity_score'] if evidence_list else 0:.3f})"
    )
    return evidence_list
