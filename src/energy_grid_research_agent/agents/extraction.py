from __future__ import annotations

import contextlib
import json
import time
from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.prompts import get_prompt_registry
from energy_grid_research_agent.report import GridFinding


def _call_llm(prompt: str) -> str:
    import ollama

    settings = get_settings()
    model = settings.ollama.llm_model
    logger.debug("[extraction] llm start | model={} | chars={}", model, len(prompt))
    t0 = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    logger.debug("[extraction] llm done | elapsed={:.2f}s", time.perf_counter() - t0)
    return str(response.message.content)


def _parse_findings(content: Any) -> list[GridFinding]:
    text = str(content)
    parsed: Any = None
    with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
        parsed = json.loads(text)

    items: list[Any] = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "findings" in parsed:
        items = parsed["findings"]

    findings: list[GridFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            findings.append(GridFinding(**item))
        except Exception:
            findings.append(
                GridFinding(
                    category="unknown",
                    description=str(item),
                    source_document="unknown",
                    source_section="unknown",
                    confidence=0.3,
                    requires_human_review=True,
                )
            )

    if not findings:
        findings.append(
            GridFinding(
                category="general",
                description=text[:300].strip(),
                source_document="unknown",
                source_section="unknown",
                confidence=0.3,
                requires_human_review=True,
            )
        )

    return findings


def run_extraction(query: str, chunks: list[dict[str, Any]]) -> tuple[list[GridFinding], bool]:
    settings = get_settings()
    registry = get_prompt_registry()
    prompt = registry.get("extraction_agent")
    context = "\n\n".join(c.get("content", "") for c in chunks[:8])
    system = prompt.system.format(confidence_threshold=settings.confidence.hitl_threshold)
    user = registry.format_user("extraction_agent", query=query, context=context)
    message = f"{system}\n\n{user}"

    content = _call_llm(message)
    findings = _parse_findings(content)

    requires_review = any(
        f.confidence < settings.confidence.hitl_threshold or f.requires_human_review
        for f in findings
    )
    logger.debug(f"Extracted {len(findings)} findings, requires_review={requires_review}")
    return findings, requires_review
