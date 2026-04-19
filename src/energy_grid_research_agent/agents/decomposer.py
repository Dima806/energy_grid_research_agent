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
        options={"temperature": 0.1},
    )
    return str(response.message.content)


def _parse_subtasks(content: Any, query: str) -> list[str]:
    if isinstance(content, list):
        return [str(s) for s in content if s]

    text = str(content)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(s) for s in parsed if s]
        if isinstance(parsed, dict) and "subtasks" in parsed:
            return [str(s) for s in parsed["subtasks"] if s]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    lines = [ln.strip().lstrip("-•*0123456789. )") for ln in text.splitlines()]
    subtasks = [ln for ln in lines if len(ln) > 8]
    return subtasks if subtasks else [query]


def run_decomposition(query: str) -> list[str]:
    registry = get_prompt_registry()
    prompt = registry.get("decomposer")
    message = f"{prompt.system}\n\n{registry.format_user('decomposer', query=query)}"
    content = _call_llm(message)
    subtasks = _parse_subtasks(content, query)
    logger.debug(f"Decomposed into {len(subtasks)} subtasks")
    return subtasks
