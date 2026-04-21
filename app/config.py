from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
EXPORTS_DIR = DATA_DIR / "exports"
SQLITE_PATH = DB_DIR / "mebot.sqlite3"


@dataclass(frozen=True)
class Settings:
    chunk_target_chars: int = 700
    chunk_overlap_chars: int = 120
    min_chunk_chars: int = 180
    supported_suffixes: tuple[str, ...] = (".md", ".txt", ".html", ".htm")
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    enable_llm: bool = os.getenv("MEBOT_ENABLE_LLM", "0") == "1"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("MEBOT_OPENAI_MODEL", "gpt-5.4-mini")
    openai_timeout_seconds: int = int(os.getenv("MEBOT_OPENAI_TIMEOUT", "120"))


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, DB_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
