from __future__ import annotations

import contextlib
import json
from typing import Any

from loguru import logger

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.prompts import get_prompt_registry
from energy_grid_research_agent.report import ComplianceArtefact, GridFinding, ResearchReport


def _call_llm(prompt: str) -> str:
    import ollama

    settings = get_settings()
    response = ollama.chat(
        model=settings.ollama.llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return str(response.message.content)


def _build_report(
    query: str,
    findings_raw: list[dict[str, Any]],
    content: Any,
    prompt_version: str,
) -> ResearchReport:
    findings = [GridFinding(**f) for f in findings_raw if isinstance(f, dict)]

    text = str(content)
    parsed: dict[str, Any] = {}
    with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
        parsed = json.loads(text)

    summary = parsed.get("summary", text[:400].strip())

    artefacts: list[ComplianceArtefact] = []
    for raw in parsed.get("compliance_artefacts", []):
        if isinstance(raw, dict):
            with contextlib.suppress(Exception):
                artefacts.append(ComplianceArtefact(**raw))

    if not artefacts:
        for f in findings:
            artefacts.append(
                ComplianceArtefact(
                    claim=f.description,
                    evidence=[f.source_document],
                    confidence=f.confidence,
                    prompt_version=prompt_version,
                )
            )

    validation_passed = bool(parsed.get("validation_passed", False))
    requires_review = any(f.requires_human_review for f in findings)

    return ResearchReport(
        query=query,
        summary=str(summary),
        findings=findings,
        compliance_artefacts=artefacts,
        validation_passed=validation_passed,
        requires_human_review=requires_review,
        prompt_version=prompt_version,
    )


def run_synthesis(query: str, findings_raw: list[dict[str, Any]]) -> ResearchReport:
    registry = get_prompt_registry()
    prompt_entry = registry.get("synthesis_agent")
    findings_text = json.dumps(findings_raw, indent=2)
    system = prompt_entry.system
    user = registry.format_user("synthesis_agent", query=query, findings=findings_text)
    message = f"{system}\n\n{user}"

    content = _call_llm(message)
    report = _build_report(query, findings_raw, content, prompt_entry.version)
    logger.debug(f"Synthesised report: {len(report.findings)} findings")
    return report
