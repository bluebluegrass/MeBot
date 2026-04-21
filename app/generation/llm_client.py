from __future__ import annotations

import json
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import Settings


class LLMConfigurationError(RuntimeError):
    """Raised when the model client is not configured."""


class LLMRequestError(RuntimeError):
    """Raised when the model request fails."""


def generate_with_openai(instructions: str, prompt: str, settings: Settings) -> str:
    if not settings.enable_llm:
        raise LLMConfigurationError("MEBOT_ENABLE_LLM is not set to 1.")
    if not settings.openai_api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not set.")

    payload = {
        "model": settings.openai_model,
        "instructions": instructions,
        "input": prompt,
    }
    request = Request(
        url=f"{settings.openai_base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.openai_timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except (TimeoutError, SocketTimeout) as exc:
        raise LLMRequestError(
            f"OpenAI API request timed out after {settings.openai_timeout_seconds}s"
        ) from exc
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMRequestError(f"OpenAI API request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise LLMRequestError(f"OpenAI API connection failed: {exc.reason}") from exc

    parsed = json.loads(body)
    text = extract_output_text(parsed)
    if not text:
        raise LLMRequestError("OpenAI API returned no output_text content.")
    return text


def extract_output_text(response_json: dict[str, object]) -> str:
    outputs = response_json.get("output")
    if not isinstance(outputs, list):
        return ""

    parts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(part for part in parts if part).strip()
