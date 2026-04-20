from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.prompts import get_prompt_registry


def _call_llm(prompt: str) -> str:
    import ollama

    settings = get_settings()
    model = settings.ollama.llm_model
    logger.debug("[validation] llm start | model={} | chars={}", model, len(prompt))
    t0 = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    logger.debug("[validation] llm done | elapsed={:.2f}s", time.perf_counter() - t0)
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

    logger.debug("[validation] passed={}", passed)
    return passed
