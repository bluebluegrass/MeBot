from __future__ import annotations

import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from yaml import YAMLError

from app.config import Settings
from app.models import LoadedDocument


FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
FIRST_HEADING_PATTERN = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
TAG_LINE_PATTERN = re.compile(r"^\s*(#[^\s#]+(?:\s+#[^\s#]+)*)\s*$")


def discover_files(root: Path, settings: Settings) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in settings.supported_suffixes
    )


def load_document(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, str | None] = {}

    if path.suffix.lower() in {".html", ".htm"}:
        text = html_to_text(text)

    match = FRONTMATTER_PATTERN.match(text)
    if match:
        raw_metadata = parse_frontmatter(match.group(1))
        metadata = {str(key): stringify(value) for key, value in raw_metadata.items()}
        text = text[match.end() :]

    if path.suffix.lower() in {".md", ".txt"}:
        text, derived_metadata = preprocess_markdown_text(text)
        for key, value in derived_metadata.items():
            metadata.setdefault(key, value)

    return LoadedDocument(path=path, metadata=metadata, raw_text=text.strip())


def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return soup.get_text("\n")


def stringify(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_frontmatter(raw_text: str) -> dict[str, object]:
    fallback: dict[str, object] = {}
    for line in raw_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            fallback[key] = value

    try:
        loaded = yaml.safe_load(raw_text) or {}
        if isinstance(loaded, dict):
            merged = {str(key): value for key, value in loaded.items()}
            merged.update(fallback)
            return merged
    except YAMLError:
        pass

    return fallback


def preprocess_markdown_text(text: str) -> tuple[str, dict[str, str | None]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    metadata: dict[str, str | None] = {}

    heading = extract_first_heading(text)
    if heading:
        metadata["title"] = heading

    cleaned_lines: list[str] = []
    skipped_first_heading = False
    skipped_duplicate_title = False
    effective_title = heading.strip() if heading else None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if effective_title and not skipped_first_heading and stripped == f"# {effective_title}":
            skipped_first_heading = True
            continue

        if effective_title and not skipped_duplicate_title and stripped == effective_title:
            skipped_duplicate_title = True
            continue

        if TAG_LINE_PATTERN.match(stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip(), metadata


def extract_first_heading(text: str) -> str | None:
    match = FIRST_HEADING_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip()
