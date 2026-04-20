from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.models import Chunk, SourceDocument


DRAFT_RUNS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS draft_generation_records (
        run_id TEXT PRIMARY KEY,
        query TEXT NOT NULL,
        query_language TEXT NOT NULL,
        requested_format TEXT NOT NULL,
        retrieved_chunk_ids TEXT NOT NULL,
        retrieved_document_ids TEXT NOT NULL,
        retrieval_score_summary TEXT NOT NULL,
        draft_text TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
"""

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        source_type TEXT NOT NULL,
        date_published TEXT,
        language TEXT NOT NULL,
        canonical_url TEXT,
        local_path TEXT NOT NULL UNIQUE,
        raw_text TEXT NOT NULL,
        era_tag TEXT,
        ingested_at TEXT NOT NULL,
        checksum TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        char_count INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        embedding_vector TEXT NOT NULL,
        section_label TEXT,
        UNIQUE(document_id, chunk_index),
        FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        chunk_id UNINDEXED,
        document_id UNINDEXED,
        text,
        content=''
    )
    """,
)


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute(DRAFT_RUNS_SCHEMA)
    connection.commit()


def log_run(
    connection: sqlite3.Connection,
    run_id: str,
    query: str,
    query_language: str,
    requested_format: str,
    retrieved_chunk_ids: list[str],
    retrieved_document_ids: list[str],
    retrieval_score_summary: str,
    draft_text: str,
    status: str,
    created_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO draft_generation_records (
            run_id, query, query_language, requested_format,
            retrieved_chunk_ids, retrieved_document_ids,
            retrieval_score_summary, draft_text, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            query,
            query_language,
            requested_format,
            json.dumps(retrieved_chunk_ids),
            json.dumps(retrieved_document_ids),
            retrieval_score_summary,
            draft_text,
            status,
            created_at,
        ),
    )
    connection.commit()


def upsert_document(connection: sqlite3.Connection, document: SourceDocument) -> None:
    connection.execute("DELETE FROM documents WHERE local_path = ? AND document_id != ?", (document.local_path, document.document_id))
    connection.execute(
        """
        INSERT INTO documents (
            document_id, title, source_type, date_published, language,
            canonical_url, local_path, raw_text, era_tag, ingested_at, checksum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            title=excluded.title,
            source_type=excluded.source_type,
            date_published=excluded.date_published,
            language=excluded.language,
            canonical_url=excluded.canonical_url,
            local_path=excluded.local_path,
            raw_text=excluded.raw_text,
            era_tag=excluded.era_tag,
            ingested_at=excluded.ingested_at,
            checksum=excluded.checksum
        """,
        (
            document.document_id,
            document.title,
            document.source_type,
            document.date_published,
            document.language,
            document.canonical_url,
            document.local_path,
            document.raw_text,
            document.era_tag,
            document.ingested_at,
            document.checksum,
        ),
    )


def replace_chunks(connection: sqlite3.Connection, document_id: str, chunks: list[Chunk]) -> None:
    connection.execute("DELETE FROM chunk_fts WHERE document_id = ?", (document_id,))
    connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    for chunk in chunks:
        connection.execute(
            """
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text, token_count, char_count,
                embedding_model, embedding_vector, section_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.char_count,
                chunk.embedding_model,
                json.dumps(chunk.embedding_vector),
                chunk.section_label,
            ),
        )
        connection.execute(
            "INSERT INTO chunk_fts (chunk_id, document_id, text) VALUES (?, ?, ?)",
            (chunk.chunk_id, chunk.document_id, chunk.text),
        )

    connection.commit()


def fetch_chunks_with_documents(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            c.embedding_model,
            c.embedding_vector,
            d.title,
            d.source_type,
            d.date_published,
            d.canonical_url
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        ORDER BY c.chunk_index ASC
        """
    ).fetchall()


def fetch_lexical_matches(connection: sqlite3.Connection, match_query: str, limit: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            c.embedding_model,
            c.embedding_vector,
            d.title,
            d.source_type,
            d.date_published,
            d.canonical_url,
            bm25(chunk_fts) AS lexical_score
        FROM chunk_fts
        JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
        JOIN documents d ON d.document_id = c.document_id
        WHERE chunk_fts MATCH ?
        ORDER BY lexical_score
        LIMIT ?
        """,
        (match_query, limit),
    ).fetchall()


def fetch_chunks_for_documents(connection: sqlite3.Connection, document_ids: list[str]) -> list[sqlite3.Row]:
    if not document_ids:
        return []
    placeholders = ",".join("?" for _ in document_ids)
    return connection.execute(
        f"""
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            c.embedding_model,
            c.embedding_vector,
            d.title,
            d.source_type,
            d.date_published,
            d.canonical_url
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE c.document_id IN ({placeholders})
        ORDER BY c.document_id, c.chunk_index ASC
        """,
        document_ids,
    ).fetchall()
