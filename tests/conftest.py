from __future__ import annotations

import pytest

from energy_grid_research_agent.config import get_settings
from energy_grid_research_agent.prompts import get_prompt_registry
from energy_grid_research_agent.report import ComplianceArtefact, GridFinding, ResearchReport


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    get_settings.cache_clear()
    get_prompt_registry.cache_clear()


@pytest.fixture
def sample_finding() -> GridFinding:
    return GridFinding(
        category="fault",
        description="Three-phase fault on Line-123 cleared in 75 ms per IEC 60909.",
        source_document="fault_report_0001.txt",
        source_section="Executive Summary",
        page_number=1,
        confidence=0.85,
        requires_human_review=False,
    )


@pytest.fixture
def low_confidence_finding() -> GridFinding:
    return GridFinding(
        category="measurement",
        description="Voltage deviation possibly exceeds ±5 % threshold.",
        source_document="grid_standard_01.txt",
        source_section="3.3 Measurement Criteria",
        confidence=0.4,
        requires_human_review=True,
    )


@pytest.fixture
def sample_report(sample_finding: GridFinding) -> ResearchReport:
    artefact = ComplianceArtefact(
        claim=sample_finding.description,
        evidence=[sample_finding.source_document],
        confidence=sample_finding.confidence,
        validated=True,
        prompt_version="v1.0",
    )
    return ResearchReport(
        query="What are fault clearance time requirements?",
        summary="Fault clearance within IEC 60909 zone 1 limits.",
        findings=[sample_finding],
        compliance_artefacts=[artefact],
        validation_passed=True,
        requires_human_review=False,
        human_approved=False,
        prompt_version="v1.0",
        agent_calls=5,
        latency_seconds=3.2,
    )


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "content": "Zone 1 protection shall operate within 80 ms per IEC 60909.",
            "source": "grid_standard_01.txt",
            "score": 0.9,
        },
        {
            "content": "GOOSE messages maximum transfer time 4 ms under IEC 61850.",
            "source": "grid_standard_02.txt",
            "score": 0.87,
        },
        {
            "content": "Power factor maintained above 0.95 lagging during peak load.",
            "source": "grid_standard_03.txt",
            "score": 0.82,
        },
    ]


@pytest.fixture
def sample_state(sample_chunks: list[dict]) -> dict:
    return {
        "query": "What are IEC 61850 GOOSE requirements?",
        "subtasks": ["GOOSE timing", "IEC 61850 scope"],
        "retrieved_chunks": sample_chunks,
        "extracted_findings": [],
        "report": None,
        "requires_human_review": False,
        "human_approved": False,
        "agent_calls": 2,
        "validation_attempts": 0,
    }
