from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SourceDocument:
    title: str
    source_type: str
    language: str
    local_path: str
    raw_text: str
    checksum: str
    document_id: str | None = None
    date_published: str | None = None
    canonical_url: str | None = None
    era_tag: str | None = None
    ingested_at: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    char_count: int
    embedding_model: str
    embedding_vector: list[float]
    chunk_id: str | None = None
    section_label: str | None = None


@dataclass
class LoadedDocument:
    path: Path
    metadata: dict[str, str | None]
    raw_text: str
