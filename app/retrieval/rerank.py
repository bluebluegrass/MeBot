from __future__ import annotations

from app.retrieval.hybrid import RetrievalResult


def rerank(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """No-op reranker for the scaffold."""
    return results
