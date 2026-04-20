from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import RAW_DIR, SQLITE_PATH, Settings, ensure_directories
from app.db import connect, init_db, replace_chunks, upsert_document
from app.ingestion.chunker import chunk_text, normalize_text
from app.ingestion.embedder import embed_text
from app.ingestion.loader import discover_files, load_document
from app.models import Chunk, SourceDocument


@dataclass
class IngestionResult:
    documents_seen: int = 0
    documents_ingested: int = 0
    chunks_created: int = 0


def infer_source_type(path: Path, metadata: dict[str, str | None]) -> str:
    if metadata.get("source_type"):
        return metadata["source_type"] or "essay"

    lowered = path.as_posix().lower()
    file_name = path.name.lower()
    if "ywdp" in file_name or "穿堂风" in path.name:
        return "newsletter"
    if "newsletter" in lowered:
        return "newsletter"
    if "xhs" in lowered:
        return "xhs"
    if "podcast" in lowered or "transcript" in lowered:
        return "podcast_transcript"
    return "essay"


def infer_title(path: Path, metadata: dict[str, str | None]) -> str:
    return metadata.get("title") or path.stem.replace("_", " ").replace("-", " ").strip()


def infer_language(text: str, metadata: dict[str, str | None]) -> str:
    if metadata.get("language"):
        return metadata["language"] or "unknown"
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    if chinese_chars > max(20, len(text) * 0.1):
        return "zh"
    return "en"


def build_document(path: Path, raw_text: str, metadata: dict[str, str | None]) -> SourceDocument:
    checksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    document_id = checksum[:16]
    return SourceDocument(
        document_id=document_id,
        title=infer_title(path, metadata),
        source_type=infer_source_type(path, metadata),
        date_published=(
            metadata.get("date")
            or metadata.get("date_published")
            or metadata.get("publishedAt")
            or metadata.get("createdAt")
        ),
        language=infer_language(raw_text, metadata),
        canonical_url=metadata.get("canonical_url") or metadata.get("url"),
        local_path=str(path),
        raw_text=raw_text,
        era_tag=metadata.get("era_tag"),
        checksum=checksum,
    )


def build_chunks(document: SourceDocument, settings: Settings) -> list[Chunk]:
    texts = chunk_text(document.raw_text, settings)
    chunks: list[Chunk] = []

    for index, text in enumerate(texts):
        chunk_id = f"{document.document_id}-{index:03d}-{uuid.uuid4().hex[:8]}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document.document_id or "",
                chunk_index=index,
                text=text,
                token_count=len(text.split()),
                char_count=len(text),
                embedding_model=settings.embedding_model_name,
                embedding_vector=embed_text(text, settings),
            )
        )

    return chunks


def ingest_corpus(raw_dir: Path = RAW_DIR, db_path: Path = SQLITE_PATH, settings: Settings | None = None) -> IngestionResult:
    ensure_directories()
    active_settings = settings or Settings()
    connection = connect(db_path)
    init_db(connection)

    result = IngestionResult()
    for path in discover_files(raw_dir, active_settings):
        result.documents_seen += 1
        loaded = load_document(path)
        normalized_text = normalize_text(loaded.raw_text)
        if not normalized_text:
            continue

        document = build_document(path, normalized_text, loaded.metadata)
        chunks = build_chunks(document, active_settings)
        upsert_document(connection, document)
        replace_chunks(connection, document.document_id or "", chunks)

        result.documents_ingested += 1
        result.chunks_created += len(chunks)

    connection.close()
    return result
