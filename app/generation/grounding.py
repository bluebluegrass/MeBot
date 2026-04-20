from __future__ import annotations


def should_abstain(retrieved_chunks: int, distinct_documents: int, top_score: float) -> bool:
    if retrieved_chunks < 2:
        return True
    if distinct_documents < 1:
        return True
    if top_score < 0.2:
        return True
    return False
