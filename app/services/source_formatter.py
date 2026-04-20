from __future__ import annotations


def format_source(title: str, date_published: str | None, canonical_url: str | None) -> str:
    parts = [title]
    if date_published:
        parts.append(date_published)
    if canonical_url:
        parts.append(canonical_url)
    return " | ".join(parts)


def format_source_block(
    index: int,
    title: str,
    source_type: str,
    date_published: str | None,
    canonical_url: str | None,
    excerpt: str,
) -> str:
    header = (
        f"[Source {index}] {title}\n"
        f"type: {source_type}\n"
        f"date: {date_published or 'undated'}\n"
        f"url: {canonical_url or 'n/a'}"
    )
    return f"{header}\nexcerpt:\n{excerpt}"
