from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class GridFinding(BaseModel):
    category: str
    description: str
    source_document: str
    source_section: str
    page_number: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool = False


class ComplianceArtefact(BaseModel):
    claim: str
    evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    validated: bool = False
    prompt_version: str = "v0.0"
    model_version: str = "qwen2.5:1.5b"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResearchReport(BaseModel):
    query: str
    summary: str
    findings: list[GridFinding] = Field(default_factory=list)
    compliance_artefacts: list[ComplianceArtefact] = Field(default_factory=list)
    validation_passed: bool = False
    requires_human_review: bool = False
    human_approved: bool = False
    prompt_version: str = "v0.0"
    agent_calls: int = 0
    latency_seconds: float = 0.0
