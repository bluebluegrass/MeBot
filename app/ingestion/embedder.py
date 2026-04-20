from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import Settings


def embed_text(text: str, settings: Settings) -> list[float]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")

    payload = {
        "model": settings.embedding_model_name,
        "input": text,
        "dimensions": settings.embedding_dimensions,
    }
    request = Request(
        url=f"{settings.openai_base_url.rstrip('/')}/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.openai_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Embedding API connection failed: {exc.reason}") from exc

    return body["data"][0]["embedding"]
