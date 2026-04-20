from __future__ import annotations

import re

from app.config import Settings


MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
WHITESPACE_LINE_PATTERN = re.compile(r"[ \t]+\n")


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = WHITESPACE_LINE_PATTERN.sub("\n", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text


def chunk_text(text: str, settings: Settings) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= settings.chunk_target_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = paragraph

        while len(current) > settings.chunk_target_chars + settings.min_chunk_chars:
            split_at = settings.chunk_target_chars
            chunks.append(current[:split_at].strip())
            overlap = current[max(0, split_at - settings.chunk_overlap_chars) :]
            current = overlap.strip()

    if current:
        chunks.append(current)

    return merge_small_tail(chunks, settings)


def merge_small_tail(chunks: list[str], settings: Settings) -> list[str]:
    if len(chunks) < 2:
        return chunks

    merged = chunks[:]
    if len(merged[-1]) < settings.min_chunk_chars:
        merged[-2] = f"{merged[-2]}\n\n{merged[-1]}".strip()
        merged.pop()
    return merged
