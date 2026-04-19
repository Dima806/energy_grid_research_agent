from __future__ import annotations

import json
from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.prompts import get_prompt_registry


def _call_llm(prompt: str) -> str:
    import ollama

    settings = get_settings()
    response = ollama.chat(
        model=settings.ollama.llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    return str(response.message.content)


def run_validation(report: dict[str, Any], chunks: list[dict[str, Any]]) -> bool:
    registry = get_prompt_registry()
    prompt = registry.get("validation_agent")
    report_text = json.dumps(report, indent=2)[:2000]
    chunks_text = json.dumps([c.get("content", "")[:200] for c in chunks[:6]], indent=2)
    message = f"{prompt.system}\n\n" + registry.format_user(
        "validation_agent", report=report_text, chunks=chunks_text
    )

    text = _call_llm(message)
    try:
        parsed = json.loads(text)
        passed = bool(parsed.get("validation_passed", False))
    except (json.JSONDecodeError, TypeError, ValueError):
        passed = "true" in text.lower() and "false" not in text.lower()

    logger.debug(f"Validation passed={passed}")
    return passed
