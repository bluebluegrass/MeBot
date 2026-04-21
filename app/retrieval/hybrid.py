from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from pathlib import Path

from app.config import SQLITE_PATH, Settings
from app.db import connect, fetch_chunks_for_documents, fetch_chunks_with_documents, fetch_lexical_matches
from app.ingestion.embedder import embed_text
from app.retrieval.scoring import normalize_scores


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")
QUERY_SEGMENT_PATTERN = re.compile(r"[\s,，。!！?？:：;；/|]+")


@dataclass
class RetrievalConfig:
    semantic_limit: int = 12
    lexical_limit: int = 12
    final_limit: int = 6
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    min_score_threshold: float = 0.22
    relative_score_floor: float = 0.45
    keyword_weight: float = 0.08
    title_match_weight: float = 0.12
    document_hit_weight: float = 0.08
    title_coverage_weight: float = 0.18
    doc_title_match_weight: float = 0.06
    doc_title_coverage_weight: float = 0.09
    doc_keyword_weight: float = 0.04
    max_chunks_per_document: int = 2
    chunk_diversity_threshold: float = 0.72
    lead_chunk_bonus: float = 0.18
    markdown_noise_penalty: float = 0.12
    document_relative_score_floor: float = 0.62
    max_documents_for_prompt: int = 3
    support_phrase_match_floor: float = 0.34
    support_term_coverage_floor: float = 0.28


@dataclass
class QueryProfile:
    is_specific: bool
    term_count: int
    phrase_count: int
    latin_term_count: int
    long_term_count: int


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    title: str
    source_type: str
    date_published: str | None
    canonical_url: str | None
    score: float
    semantic_score: float
    lexical_score: float


@dataclass
class RetrievedDocument:
    document_id: str
    title: str
    source_type: str
    date_published: str | None
    canonical_url: str | None
    score: float
    chunks: list[RetrievalResult]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def build_fts_query(query: str) -> str:
    tokens = TOKEN_PATTERN.findall(query.lower())
    return " OR ".join(f'"{token}"' for token in tokens[:8])


def extract_query_terms(query: str) -> list[str]:
    raw_segments = [segment.strip().lower() for segment in QUERY_SEGMENT_PATTERN.split(query) if segment.strip()]
    terms: list[str] = []

    for segment in raw_segments:
        if segment not in terms:
            terms.append(segment)
        if contains_cjk(segment) and len(segment) >= 4:
            for size in (4, 3, 2):
                for index in range(0, len(segment) - size + 1):
                    gram = segment[index : index + size]
                    if gram not in terms:
                        terms.append(gram)

    compact = "".join(raw_segments)
    if compact and compact not in terms:
        terms.append(compact)
    return terms[:24]


def extract_query_phrases(query: str) -> list[str]:
    raw_segments = [segment.strip().lower() for segment in QUERY_SEGMENT_PATTERN.split(query) if segment.strip()]
    phrases: list[str] = []

    for segment in raw_segments:
        if len(segment) >= 2 and segment not in phrases:
            phrases.append(segment)

    for index in range(len(raw_segments) - 1):
        phrase = " ".join(raw_segments[index : index + 2]).strip()
        if len(phrase) >= 4 and phrase not in phrases:
            phrases.append(phrase)

    normalized = " ".join(raw_segments).strip()
    if len(normalized) >= 4 and normalized not in phrases:
        phrases.append(normalized)
    return phrases[:12]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def build_query_profile(query: str) -> QueryProfile:
    raw_segments = [segment.strip() for segment in QUERY_SEGMENT_PATTERN.split(query) if segment.strip()]
    terms = extract_query_terms(query)
    phrases = extract_query_phrases(query)
    latin_term_count = sum(1 for segment in raw_segments if re.search(r"[A-Za-z]", segment))
    long_term_count = sum(1 for segment in raw_segments if len(segment) >= 4)
    is_specific = latin_term_count >= 1 or (len(raw_segments) <= 2 and long_term_count >= 1)
    return QueryProfile(
        is_specific=is_specific,
        term_count=len(terms),
        phrase_count=len(phrases),
        latin_term_count=latin_term_count,
        long_term_count=long_term_count,
    )


def keyword_overlap_score(query: str, title: str, text: str) -> float:
    terms = extract_query_terms(query)
    if not terms:
        return 0.0

    haystacks = ((title.lower(), 2.0), (text.lower(), 1.0))
    score = 0.0
    max_score = 0.0

    for term in terms:
        base = 1.0 if len(term) < 4 else 1.5
        for haystack, weight in haystacks:
            max_score += base * weight
            if term in haystack:
                score += base * weight

    if max_score == 0:
        return 0.0
    return score / max_score


def phrase_match_score(query: str, title: str, text: str) -> float:
    phrases = extract_query_phrases(query)
    if not phrases:
        return 0.0

    title_lower = title.lower()
    text_lower = text.lower()
    score = 0.0

    for phrase in phrases:
        if phrase in title_lower:
            score += 1.0
        elif phrase in text_lower:
            score += 0.4

    return min(score / len(phrases), 1.0)


def title_term_coverage_score(query: str, title: str) -> float:
    terms = extract_query_terms(query)
    if not terms:
        return 0.0

    title_lower = title.lower()
    matched = 0.0
    total = 0.0
    for term in terms:
        weight = 1.5 if len(term) >= 4 else 1.0
        total += weight
        if term in title_lower:
            matched += weight

    if total == 0:
        return 0.0
    return matched / total


def term_coverage_score(query: str, text: str) -> float:
    terms = extract_query_terms(query)
    if not terms:
        return 0.0

    text_lower = text.lower()
    matched = 0.0
    total = 0.0
    for term in terms:
        weight = 1.5 if len(term) >= 4 else 1.0
        total += weight
        if term in text_lower:
            matched += weight

    if total == 0:
        return 0.0
    return matched / total


def chunk_position_bonus(chunk_index: int) -> float:
    if chunk_index <= 0:
        return 1.0
    if chunk_index == 1:
        return 0.75
    if chunk_index == 2:
        return 0.45
    if chunk_index == 3:
        return 0.2
    return 0.0


def markdown_noise_score(text: str) -> float:
    patterns = (
        "![](",
        "[Club Only",
        "__GHOST_URL__",
        "[欢迎订阅]",
        "mailto:",
        "http://https://",
        "Podcast link ->",
    )
    hits = sum(1 for pattern in patterns if pattern in text)
    heading_hits = text.count("### ") + text.count("#### ")
    return min((hits + heading_hits * 0.2) / 4, 1.0)


def semantic_search(db_path: Path, query: str, settings: Settings, limit: int) -> list[RetrievalResult]:
    query_vector = embed_text(query, settings)
    connection = connect(db_path)
    rows = fetch_chunks_with_documents(connection)
    connection.close()

    results: list[RetrievalResult] = []
    for row in rows:
        results.append(build_retrieval_result_from_row(row, query_vector))

    return sorted(results, key=lambda item: item.semantic_score, reverse=True)[:limit]


def lexical_search(db_path: Path, query: str, limit: int) -> list[RetrievalResult]:
    match_query = build_fts_query(query)
    if not match_query:
        return []

    connection = connect(db_path)
    rows = fetch_lexical_matches(connection, match_query, limit)
    connection.close()

    results: list[RetrievalResult] = []
    for row in rows:
        lexical_score = float(-row["lexical_score"])
        results.append(
            RetrievalResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                title=row["title"],
                source_type=row["source_type"],
                date_published=row["date_published"],
                canonical_url=row["canonical_url"],
                score=lexical_score,
                semantic_score=0.0,
                lexical_score=lexical_score,
            )
        )
    return results


def build_retrieval_result_from_row(row, query_vector: list[float]) -> RetrievalResult:
    semantic_score = cosine_similarity(query_vector, json.loads(row["embedding_vector"]))
    return RetrievalResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        title=row["title"],
        source_type=row["source_type"],
        date_published=row["date_published"],
        canonical_url=row["canonical_url"],
        score=semantic_score,
        semantic_score=semantic_score,
        lexical_score=0.0,
    )


def merge_results(
    semantic_results: list[RetrievalResult],
    lexical_results: list[RetrievalResult],
    config: RetrievalConfig,
    query: str,
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}

    for result, normalized in zip(
        semantic_results,
        normalize_scores([result.semantic_score for result in semantic_results]),
    ):
        merged[result.chunk_id] = RetrievalResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            chunk_index=result.chunk_index,
            text=result.text,
            title=result.title,
            source_type=result.source_type,
            date_published=result.date_published,
            canonical_url=result.canonical_url,
            score=config.semantic_weight * normalized,
            semantic_score=result.semantic_score,
            lexical_score=0.0,
        )

    for result, normalized in zip(
        lexical_results,
        normalize_scores([result.lexical_score for result in lexical_results]),
    ):
        component = config.lexical_weight * normalized
        existing = merged.get(result.chunk_id)
        if existing is None:
            merged[result.chunk_id] = RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                text=result.text,
                title=result.title,
                source_type=result.source_type,
                date_published=result.date_published,
                canonical_url=result.canonical_url,
                score=component,
                semantic_score=0.0,
                lexical_score=result.lexical_score,
            )
        else:
            existing.score += component
            existing.lexical_score = result.lexical_score

    for result in merged.values():
        overlap = keyword_overlap_score(query, result.title, result.text)
        phrase_score = phrase_match_score(query, result.title, result.text)
        title_coverage = title_term_coverage_score(query, result.title)
        lead_bonus = chunk_position_bonus(result.chunk_index)
        noise_penalty = markdown_noise_score(result.text)
        result.score += config.keyword_weight * overlap
        result.score += config.title_match_weight * phrase_score
        result.score += config.title_coverage_weight * title_coverage
        result.score += config.lead_chunk_bonus * lead_bonus
        result.score -= config.markdown_noise_penalty * noise_penalty

    apply_document_level_bonus(list(merged.values()), query, config)

    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    ranked = prune_low_confidence_results(ranked, config)
    return diversify_by_document(ranked, config.final_limit)


def apply_document_level_bonus(results: list[RetrievalResult], query: str, config: RetrievalConfig) -> None:
    grouped: dict[str, list[RetrievalResult]] = {}
    for result in results:
        grouped.setdefault(result.document_id, []).append(result)

    for document_results in grouped.values():
        title = document_results[0].title
        title_bonus = phrase_match_score(query, title, "")
        coverage_bonus = title_term_coverage_score(query, title)
        keyword_bonus = max(keyword_overlap_score(query, item.title, item.text) for item in document_results)
        hit_bonus = min(len(document_results), 2) / 2
        total_bonus = (
            config.doc_title_match_weight * title_bonus
            + config.doc_title_coverage_weight * coverage_bonus
            + config.doc_keyword_weight * keyword_bonus
            + config.document_hit_weight * hit_bonus
        )
        for result in document_results:
            result.score += total_bonus


def prune_low_confidence_results(results: list[RetrievalResult], config: RetrievalConfig) -> list[RetrievalResult]:
    if not results:
        return []

    top_score = results[0].score
    floor = max(config.min_score_threshold, top_score * config.relative_score_floor)
    pruned = [result for result in results if result.score >= floor]

    if pruned:
        return pruned
    return results[:1]


def diversify_by_document(results: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    per_document_counts: dict[str, int] = {}

    for result in results:
        current = per_document_counts.get(result.document_id, 0)
        if current >= 2:
            continue
        selected.append(result)
        per_document_counts[result.document_id] = current + 1
        if len(selected) >= limit:
            break

    return selected


def group_results_by_document(results: list[RetrievalResult], max_chunks_per_document: int = 2) -> list[RetrievedDocument]:
    grouped: dict[str, list[RetrievalResult]] = {}
    order: list[str] = []

    for result in results:
        if result.document_id not in grouped:
            grouped[result.document_id] = []
            order.append(result.document_id)
        if len(grouped[result.document_id]) < max_chunks_per_document:
            grouped[result.document_id].append(result)

    documents: list[RetrievedDocument] = []
    for document_id in order:
        chunks = select_representative_chunks(grouped[document_id], max_chunks=max_chunks_per_document)
        lead = chunks[0]
        documents.append(
            RetrievedDocument(
                document_id=document_id,
                title=lead.title,
                source_type=lead.source_type,
                date_published=lead.date_published,
                canonical_url=lead.canonical_url,
                score=max(chunk.score for chunk in chunks),
                chunks=chunks,
            )
        )

    return documents


def select_representative_chunks(
    chunks: list[RetrievalResult],
    max_chunks: int = 2,
    diversity_threshold: float = 0.72,
) -> list[RetrievalResult]:
    ranked = sorted(chunks, key=lambda item: item.score, reverse=True)
    selected: list[RetrievalResult] = []

    for chunk in ranked:
        if not selected:
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break
            continue

        if all(text_similarity(chunk.text, existing.text) < diversity_threshold for existing in selected):
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break

    if not selected and ranked:
        return [ranked[0]]

    return selected or ranked[:1]


def text_similarity(left: str, right: str) -> float:
    left_tokens = set(TOKEN_PATTERN.findall(left.lower()))
    right_tokens = set(TOKEN_PATTERN.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    total = len(left_tokens | right_tokens)
    if total == 0:
        return 0.0
    return overlap / total


def retrieve(
    query: str,
    db_path: Path = SQLITE_PATH,
    settings: Settings | None = None,
    config: RetrievalConfig | None = None,
) -> list[RetrievalResult]:
    cleaned_query = query.strip()
    if not cleaned_query or not db_path.exists():
        return []

    active_settings = settings or Settings()
    query_profile = build_query_profile(cleaned_query)
    active_config = adapt_config_for_query(config or RetrievalConfig(), query_profile)
    semantic_results = semantic_search(db_path, cleaned_query, active_settings, active_config.semantic_limit)
    lexical_results = lexical_search(db_path, cleaned_query, active_config.lexical_limit)
    if not semantic_results and not lexical_results:
        return []
    return merge_results(semantic_results, lexical_results, active_config, cleaned_query)


def retrieve_documents(
    query: str,
    db_path: Path = SQLITE_PATH,
    settings: Settings | None = None,
    config: RetrievalConfig | None = None,
    max_chunks_per_document: int = 2,
) -> list[RetrievedDocument]:
    query_profile = build_query_profile(query)
    active_config = adapt_config_for_query(config or RetrievalConfig(), query_profile)
    results = retrieve(query, db_path=db_path, settings=settings, config=active_config)
    chunk_limit = max_chunks_per_document if max_chunks_per_document is not None else active_config.max_chunks_per_document
    grouped = group_results_by_document(results, max_chunks_per_document=chunk_limit)
    rescored = rescore_documents_with_all_chunks(
        grouped,
        query=query,
        db_path=db_path,
        settings=settings or Settings(),
        config=active_config,
        max_chunks_per_document=chunk_limit,
    )
    return prune_documents_for_prompt(rescored, active_config, query=query, query_profile=query_profile)


def rescore_documents_with_all_chunks(
    documents: list[RetrievedDocument],
    query: str,
    db_path: Path,
    settings: Settings,
    config: RetrievalConfig,
    max_chunks_per_document: int,
) -> list[RetrievedDocument]:
    if not documents:
        return []

    connection = connect(db_path)
    rows = fetch_chunks_for_documents(connection, [document.document_id for document in documents])
    connection.close()
    query_vector = embed_text(query, settings)

    by_document: dict[str, list[RetrievalResult]] = {}
    for row in rows:
        result = build_retrieval_result_from_row(row, query_vector)
        apply_chunk_scoring(result, query, config)
        by_document.setdefault(result.document_id, []).append(result)

    rescored_documents: list[RetrievedDocument] = []
    for document in documents:
        candidates = by_document.get(document.document_id, document.chunks)
        selected_chunks = select_representative_chunks(
            candidates,
            max_chunks=max_chunks_per_document,
            diversity_threshold=config.chunk_diversity_threshold,
        )
        lead = selected_chunks[0]
        rescored_documents.append(
            RetrievedDocument(
                document_id=document.document_id,
                title=document.title,
                source_type=document.source_type,
                date_published=document.date_published,
                canonical_url=document.canonical_url,
                score=max(chunk.score for chunk in selected_chunks),
                chunks=selected_chunks,
            )
        )

    return rescored_documents


def apply_chunk_scoring(result: RetrievalResult, query: str, config: RetrievalConfig) -> None:
    overlap = keyword_overlap_score(query, result.title, result.text)
    phrase_score = phrase_match_score(query, result.title, result.text)
    title_coverage = title_term_coverage_score(query, result.title)
    lead_bonus = chunk_position_bonus(result.chunk_index)
    noise_penalty = markdown_noise_score(result.text)
    result.score += config.keyword_weight * overlap
    result.score += config.title_match_weight * phrase_score
    result.score += config.title_coverage_weight * title_coverage
    result.score += config.lead_chunk_bonus * lead_bonus
    result.score -= config.markdown_noise_penalty * noise_penalty


def prune_documents_for_prompt(
    documents: list[RetrievedDocument],
    config: RetrievalConfig,
    query: str | None = None,
    query_profile: QueryProfile | None = None,
) -> list[RetrievedDocument]:
    if not documents:
        return []

    ranked = sorted(documents, key=lambda document: document.score, reverse=True)
    top_score = ranked[0].score
    floor = top_score * effective_document_score_floor(config, query_profile)

    pruned = [document for document in ranked if document.score >= floor]
    if not pruned:
        pruned = ranked[:1]

    if query:
        pruned = filter_complementary_supporting_documents(pruned, query, config, query_profile=query_profile)

    return pruned[: effective_max_documents(config, query_profile)]


def effective_document_score_floor(config: RetrievalConfig, query_profile: QueryProfile | None) -> float:
    if not query_profile:
        return config.document_relative_score_floor
    if query_profile.is_specific:
        return config.document_relative_score_floor
    return min(config.document_relative_score_floor, 0.48)


def effective_max_documents(config: RetrievalConfig, query_profile: QueryProfile | None) -> int:
    if not query_profile:
        return config.max_documents_for_prompt
    if query_profile.is_specific:
        return config.max_documents_for_prompt
    return max(config.max_documents_for_prompt, 4)


def filter_complementary_supporting_documents(
    documents: list[RetrievedDocument],
    query: str,
    config: RetrievalConfig,
    query_profile: QueryProfile | None = None,
) -> list[RetrievedDocument]:
    if not documents:
        return []

    kept = [documents[0]]
    phrase_floor = effective_support_phrase_floor(config, query_profile)
    coverage_floor = effective_support_term_floor(config, query_profile)
    for document in documents[1:]:
        combined_text = " ".join(chunk.text for chunk in document.chunks)
        phrase_score = phrase_match_score(query, document.title, combined_text)
        coverage_score = max(
            title_term_coverage_score(query, document.title),
            term_coverage_score(query, combined_text),
        )
        if phrase_score >= phrase_floor or coverage_score >= coverage_floor:
            kept.append(document)
    return kept


def effective_support_phrase_floor(config: RetrievalConfig, query_profile: QueryProfile | None) -> float:
    if not query_profile:
        return config.support_phrase_match_floor
    if query_profile.is_specific:
        return config.support_phrase_match_floor
    return min(config.support_phrase_match_floor, 0.18)


def effective_support_term_floor(config: RetrievalConfig, query_profile: QueryProfile | None) -> float:
    if not query_profile:
        return config.support_term_coverage_floor
    if query_profile.is_specific:
        return config.support_term_coverage_floor
    return min(config.support_term_coverage_floor, 0.16)


def adapt_config_for_query(config: RetrievalConfig, query_profile: QueryProfile) -> RetrievalConfig:
    if query_profile.is_specific:
        return config

    return replace(
        config,
        semantic_limit=max(config.semantic_limit, 20),
        lexical_limit=max(config.lexical_limit, 20),
        final_limit=max(config.final_limit, 10),
        relative_score_floor=min(config.relative_score_floor, 0.28),
        document_relative_score_floor=min(config.document_relative_score_floor, 0.48),
        max_documents_for_prompt=max(config.max_documents_for_prompt, 4),
        support_phrase_match_floor=min(config.support_phrase_match_floor, 0.18),
        support_term_coverage_floor=min(config.support_term_coverage_floor, 0.16),
    )
