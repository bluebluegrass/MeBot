from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import SQLITE_PATH
from app.db import connect, log_run
from app.generation.draft_writer import generate_draft
from app.generation.grounding import should_abstain
from app.retrieval.hybrid import RetrievalConfig, build_query_profile, retrieve, retrieve_documents
from app.services.source_formatter import format_source_block

# Lenient config for the passage-review step — the user sees and curates results,
# so we prefer recall over precision here.
_REVIEW_CONFIG = RetrievalConfig(
    semantic_limit=20,
    lexical_limit=20,
    final_limit=12,
    min_score_threshold=0.05,
    relative_score_floor=0.15,
    document_relative_score_floor=0.35,
    max_documents_for_prompt=10,
    support_phrase_match_floor=0.05,
    support_term_coverage_floor=0.05,
)

_INTENT_WITH_MATERIAL = re.compile(
    r"^我[想要]用[^，,写]{1,40}写[^，,。.！!]{0,20}[，,]\s*",
    re.UNICODE,
)
_INTENT_SIMPLE = re.compile(
    r"^我[想要](?:写|起草|输出|生成|创作)(?:一[篇个段条则])?(?:关于|有关|讲|谈|说)?\s*",
    re.UNICODE,
)
_SECONDARY_MARKER = re.compile(r"^主要[是写讲说]?[，,]?\s*", re.UNICODE)
_INTRO_MARKER = re.compile(r"^(?:关于|讲|表达|描述)\s*", re.UNICODE)


def _extract_retrieval_query(topic: str) -> str:
    """Strip writing-intent boilerplate so retrieval focuses on the actual subject."""
    text = topic.strip()

    # "我想用[material]写[format]，[actual topic]"
    match = _INTENT_WITH_MATERIAL.match(text)
    if match:
        remainder = text[match.end():]
        remainder = _SECONDARY_MARKER.sub("", remainder)
        remainder = _INTRO_MARKER.sub("", remainder).strip()
        if len(remainder) >= 4:
            return remainder

    # "我想写一篇关于[actual topic]"
    match = _INTENT_SIMPLE.match(text)
    if match:
        remainder = text[match.end():].strip()
        if len(remainder) >= 4:
            return remainder

    return topic


@dataclass
class SourcePreview:
    document_id: str
    title: str
    source_type: str
    date_published: str | None
    canonical_url: str | None
    excerpts: list[str]
    score: float


@dataclass
class DraftResponse:
    draft_text: str
    source_count: int
    abstained: bool
    sources: list[SourcePreview]
    retrieval_summary: str
    prompt_context_preview: str
    adjacent_topics: list[str]


def retrieve_for_review(topic: str) -> list[SourcePreview]:
    """Step 1: retrieve passages for user to review and select."""
    retrieval_query = _extract_retrieval_query(topic)
    document_results = retrieve_documents(retrieval_query, max_chunks_per_document=3, config=_REVIEW_CONFIG)
    if not document_results:
        return []
    return [
        SourcePreview(
            document_id=document.document_id,
            title=document.title,
            source_type=document.source_type,
            date_published=document.date_published,
            canonical_url=document.canonical_url,
            excerpts=[build_excerpt(chunk.text, limit=500) for chunk in document.chunks],
            score=document.score,
        )
        for document in document_results
    ]


def draft_from_selected(
    topic: str, sources: list[SourcePreview], requested_format: str = "free-form"
) -> DraftResponse:
    """Step 2: generate draft from user-selected sources."""
    if not sources:
        return DraftResponse(
            draft_text="没有选择任何来源，无法起草。",
            source_count=0,
            abstained=True,
            sources=[],
            retrieval_summary="No sources selected.",
            prompt_context_preview="",
            adjacent_topics=[],
        )

    prompt_context = build_prompt_context(sources)
    draft = generate_draft(topic, prompt_context, requested_format)
    return DraftResponse(
        draft_text=draft,
        source_count=len(sources),
        abstained=False,
        sources=sources,
        retrieval_summary=f"distinct_documents={len(sources)} (user-selected)",
        prompt_context_preview=prompt_context,
        adjacent_topics=[],
    )


def topic_to_draft(topic: str, requested_format: str = "free-form") -> DraftResponse:
    retrieval_results = retrieve(topic)
    document_results = retrieve_documents(topic, max_chunks_per_document=2)
    query_profile = build_query_profile(topic)
    if not retrieval_results or not document_results:
        return DraftResponse(
            draft_text="这个话题之前没写过，或者现有材料不足以支持可靠起草。",
            source_count=0,
            abstained=True,
            sources=[],
            retrieval_summary="No retrieval matches found.",
            prompt_context_preview="",
            adjacent_topics=[],
        )

    distinct_documents = len(document_results)
    top_score = retrieval_results[0].score
    top_document_score = document_results[0].score
    sources = [
        SourcePreview(
            document_id=document.document_id,
            title=document.title,
            source_type=document.source_type,
            date_published=document.date_published,
            canonical_url=document.canonical_url,
            excerpts=[build_excerpt(chunk.text) for chunk in document.chunks],
            score=document.score,
        )
        for document in document_results
    ]

    prompt_context = build_prompt_context(sources)
    adjacent_topics = build_adjacent_topics(sources, topic)

    if should_abstain(len(retrieval_results), distinct_documents, top_score) or should_abstain_for_query_profile(
        query_profile,
        distinct_documents,
        top_document_score,
    ):
        response = DraftResponse(
            draft_text=build_abstention_text(adjacent_topics),
            source_count=distinct_documents,
            abstained=True,
            sources=sources,
            retrieval_summary=build_retrieval_summary(len(retrieval_results), distinct_documents, top_score),
            prompt_context_preview=prompt_context,
            adjacent_topics=adjacent_topics,
        )
        _persist_run(topic, requested_format, retrieval_results, document_results, response)
        return response

    draft = generate_draft(topic, prompt_context, requested_format)
    response = DraftResponse(
        draft_text=draft,
        source_count=distinct_documents,
        abstained=False,
        sources=sources,
        retrieval_summary=build_retrieval_summary(len(retrieval_results), distinct_documents, top_score),
        prompt_context_preview=prompt_context,
        adjacent_topics=adjacent_topics,
    )
    _persist_run(topic, requested_format, retrieval_results, document_results, response)
    return response


def build_excerpt(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def build_retrieval_summary(retrieved_chunks: int, distinct_documents: int, top_score: float) -> str:
    return (
        f"retrieved_chunks={retrieved_chunks} "
        f"distinct_documents={distinct_documents} "
        f"top_score={top_score:.3f}"
    )


def build_prompt_context(sources: list[SourcePreview]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            format_source_block(
                index=index,
                title=source.title,
                source_type=source.source_type,
                date_published=source.date_published,
                canonical_url=source.canonical_url,
                excerpt="\n\n".join(source.excerpts),
            )
        )
    return "\n\n".join(blocks)


def build_adjacent_topics(sources: list[SourcePreview], topic: str, limit: int = 3) -> list[str]:
    normalized_topic = topic.strip().lower()
    suggestions: list[str] = []
    for source in sources:
        candidate = normalize_title_for_suggestion(source.title)
        if not candidate:
            continue
        if normalized_topic and candidate.lower() == normalized_topic:
            continue
        if candidate not in suggestions:
            suggestions.append(candidate)
        if len(suggestions) >= limit:
            break
    return suggestions


def normalize_title_for_suggestion(title: str) -> str:
    candidate = " ".join(title.split()).strip()
    candidate = candidate.replace("🆓 ", "").replace("🆓", "").strip()
    return candidate


def build_abstention_text(adjacent_topics: list[str]) -> str:
    base = "这个话题之前没写过，或者现有材料不足以支持可靠起草。"
    if not adjacent_topics:
        return base
    return f"{base}\n\n你也许可以试试这些相近题目：\n- " + "\n- ".join(adjacent_topics)


def _persist_run(
    topic: str,
    requested_format: str,
    retrieval_results: list,
    document_results: list,
    response: DraftResponse,
) -> None:
    connection = connect(SQLITE_PATH)
    log_run(
        connection=connection,
        run_id=uuid.uuid4().hex,
        query=topic,
        query_language="zh" if any("\u4e00" <= c <= "\u9fff" for c in topic) else "en",
        requested_format=requested_format,
        retrieved_chunk_ids=[r.chunk_id for r in retrieval_results],
        retrieved_document_ids=[d.document_id for d in document_results],
        retrieval_score_summary=response.retrieval_summary,
        draft_text=response.draft_text,
        status="abstained" if response.abstained else "generated",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    connection.close()


def should_abstain_for_query_profile(query_profile, distinct_documents: int, top_score: float) -> bool:
    if query_profile.is_specific:
        return False
    if distinct_documents <= 1 and top_score < 0.65:
        return True
    return False
