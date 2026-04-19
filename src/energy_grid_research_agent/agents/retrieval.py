from __future__ import annotations

from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.tools.search import search_grid_documents


def run_retrieval(subtasks: list[str]) -> list[dict[str, Any]]:
    settings = get_settings()
    top_k = settings.retrieval.top_k
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()

    for subtask in subtasks:
        results = search_grid_documents(subtask, top_k=top_k)
        for chunk in results:
            key = chunk.get("content", "")[:80]
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

    logger.debug(f"Retrieved {len(chunks)} unique chunks for {len(subtasks)} subtasks")
    return chunks
