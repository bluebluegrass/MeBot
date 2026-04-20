from __future__ import annotations

from app.generation.prompts import build_prompt
from app.generation.llm_client import LLMConfigurationError, LLMRequestError, generate_with_openai
from app.generation.prompts import SYSTEM_INSTRUCTIONS
from app.config import Settings


def generate_draft(query: str, retrieved_context: str, requested_format: str = "free-form") -> str:
    prompt = build_prompt(query, retrieved_context, requested_format)
    settings = Settings()

    try:
        return generate_with_openai(SYSTEM_INSTRUCTIONS, prompt, settings)
    except LLMConfigurationError:
        return (
            "Draft generation is not wired to an LLM yet because OPENAI_API_KEY is not set.\n\n"
            "Prepared prompt preview:\n"
            f"{prompt}"
        )
    except LLMRequestError as exc:
        return (
            "Draft generation failed while calling the LLM.\n\n"
            f"Error: {exc}\n\n"
            "Prepared prompt preview:\n"
            f"{prompt}"
        )
