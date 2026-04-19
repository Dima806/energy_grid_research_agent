from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from energy_grid_research_agent.agents.decomposer import _parse_subtasks, run_decomposition
from energy_grid_research_agent.agents.extraction import _parse_findings, run_extraction
from energy_grid_research_agent.agents.retrieval import run_retrieval
from energy_grid_research_agent.agents.synthesis import _build_report, run_synthesis
from energy_grid_research_agent.agents.validation import run_validation

# ── decomposer ────────────────────────────────────────────────────────────────


def test_parse_subtasks_from_json_array() -> None:
    result = _parse_subtasks('["subtask A", "subtask B"]', "query")
    assert result == ["subtask A", "subtask B"]


def test_parse_subtasks_from_json_dict() -> None:
    result = _parse_subtasks('{"subtasks": ["a", "b", "c"]}', "query")
    assert result == ["a", "b", "c"]


def test_parse_subtasks_from_list_object() -> None:
    result = _parse_subtasks(["task one", "task two"], "query")
    assert result == ["task one", "task two"]


def test_parse_subtasks_fallback_newlines() -> None:
    text = "- What is zone 1 protection?\n- How does IEC 61850 work?"
    result = _parse_subtasks(text, "original query")
    assert len(result) >= 1
    assert all(isinstance(s, str) for s in result)


def test_parse_subtasks_empty_fallback_to_query() -> None:
    result = _parse_subtasks("", "my original query")
    assert result == ["my original query"]


def test_parse_subtasks_short_lines_filtered() -> None:
    result = _parse_subtasks("a\nb\nc", "fallback query")
    assert result == ["fallback query"]


def test_run_decomposition_returns_list(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.decomposer._call_llm",
        return_value='["GOOSE timing", "IEC 61850 scope"]',
    ):
        result = run_decomposition("IEC 61850 GOOSE requirements")
    assert isinstance(result, list)
    assert len(result) == 2
    assert "GOOSE timing" in result


def test_run_decomposition_llm_called_once() -> None:
    mock_llm = MagicMock(return_value='["task1"]')
    with patch("energy_grid_research_agent.agents.decomposer._call_llm", mock_llm):
        run_decomposition("test query")
    mock_llm.assert_called_once()


def test_run_decomposition_fallback_on_bad_response() -> None:
    with patch(
        "energy_grid_research_agent.agents.decomposer._call_llm",
        return_value="not json at all",
    ):
        result = run_decomposition("my query")
    assert isinstance(result, list)
    assert len(result) >= 1


# ── retrieval agent ───────────────────────────────────────────────────────────


def test_run_retrieval_deduplicates_chunks() -> None:
    chunk = {"content": "Zone 1 protection shall operate within 80 ms.", "source": "doc.txt"}
    with patch(
        "energy_grid_research_agent.agents.retrieval.search_grid_documents",
        return_value=[chunk],
    ):
        result = run_retrieval(["subtask1", "subtask2"])
    assert len(result) == 1


def test_run_retrieval_aggregates_unique_chunks() -> None:
    chunks_a = [{"content": "Chunk A content here abc.", "source": "a.txt"}]
    chunks_b = [{"content": "Chunk B content here xyz.", "source": "b.txt"}]
    side_effects = [chunks_a, chunks_b]
    with patch(
        "energy_grid_research_agent.agents.retrieval.search_grid_documents",
        side_effect=side_effects,
    ):
        result = run_retrieval(["task1", "task2"])
    assert len(result) == 2


def test_run_retrieval_empty_subtasks() -> None:
    result = run_retrieval([])
    assert result == []


def test_run_retrieval_calls_search_per_subtask() -> None:
    mock_search = MagicMock(return_value=[])
    with patch("energy_grid_research_agent.agents.retrieval.search_grid_documents", mock_search):
        run_retrieval(["a", "b", "c"])
    assert mock_search.call_count == 3


# ── extraction agent ──────────────────────────────────────────────────────────


_FINDING_JSON = (
    '[{"category": "fault", "description": "desc", "source_document": "doc.txt",'
    ' "source_section": "s1", "confidence": 0.8, "requires_human_review": false}]'
)

_FINDINGS_DICT_JSON = (
    '{"findings": [{"category": "standard", "description": "d", "source_document": "f.txt",'
    ' "source_section": "3.1", "confidence": 0.9, "requires_human_review": false}]}'
)


def test_parse_findings_from_json_list() -> None:
    findings = _parse_findings(_FINDING_JSON)
    assert len(findings) == 1
    assert findings[0].category == "fault"


def test_parse_findings_from_dict_with_findings_key() -> None:
    findings = _parse_findings(_FINDINGS_DICT_JSON)
    assert len(findings) == 1


def test_parse_findings_fallback_on_invalid_json() -> None:
    findings = _parse_findings("some plain text response")
    assert len(findings) == 1
    assert findings[0].requires_human_review is True
    assert findings[0].confidence == pytest.approx(0.3)


def test_parse_findings_bad_item_gets_default() -> None:
    content = '[{"not_a_finding": true}]'
    findings = _parse_findings(content)
    assert len(findings) == 1
    assert findings[0].category == "unknown"


_EX_HIGH = (
    '[{"category": "fault", "description": "fault found", "source_document": "doc.txt",'
    ' "source_section": "s", "confidence": 0.9, "requires_human_review": false}]'
)
_EX_LOW = (
    '[{"category": "fault", "description": "d", "source_document": "doc.txt",'
    ' "source_section": "s", "confidence": 0.3, "requires_human_review": true}]'
)
_EX_VERY_HIGH = (
    '[{"category": "standard", "description": "d", "source_document": "doc.txt",'
    ' "source_section": "s", "confidence": 0.95, "requires_human_review": false}]'
)


def test_run_extraction_returns_tuple(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.extraction._call_llm",
        return_value=_EX_HIGH,
    ):
        findings, needs_review = run_extraction("fault analysis", sample_chunks)
    assert isinstance(findings, list)
    assert isinstance(needs_review, bool)


def test_run_extraction_low_confidence_triggers_review(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.extraction._call_llm",
        return_value=_EX_LOW,
    ):
        _, needs_review = run_extraction("query", sample_chunks)
    assert needs_review is True


def test_run_extraction_high_confidence_no_review(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.extraction._call_llm",
        return_value=_EX_VERY_HIGH,
    ):
        _, needs_review = run_extraction("query", sample_chunks)
    assert needs_review is False


# ── synthesis agent ───────────────────────────────────────────────────────────


def test_build_report_from_parsed_content() -> None:
    findings_raw = [
        {
            "category": "fault",
            "description": "d",
            "source_document": "doc.txt",
            "source_section": "s",
            "confidence": 0.8,
            "requires_human_review": False,
        }
    ]
    content = '{"summary": "Summary text.", "compliance_artefacts": [], "validation_passed": true}'
    report = _build_report("test query", findings_raw, content, "v1.1")
    assert report.summary == "Summary text."
    assert report.validation_passed is True
    assert report.prompt_version == "v1.1"


def test_build_report_fallback_summary() -> None:
    report = _build_report("query", [], "plain text response", "v1.0")
    assert "plain text" in report.summary.lower()


def test_build_report_creates_artefacts_from_findings() -> None:
    findings_raw = [
        {
            "category": "fault",
            "description": "claim text",
            "source_document": "doc.txt",
            "source_section": "s",
            "confidence": 0.75,
            "requires_human_review": False,
        }
    ]
    report = _build_report("query", findings_raw, "{}", "v1.0")
    assert len(report.compliance_artefacts) == 1
    assert report.compliance_artefacts[0].claim == "claim text"


def test_run_synthesis_returns_report(sample_chunks) -> None:
    _synth = (
        '{"summary": "Grid compliant.", "compliance_artefacts": [], "validation_passed": false}'
    )
    findings_raw = [
        {
            "category": "standard",
            "description": "IEC 61850 zone 1 met.",
            "source_document": "doc.txt",
            "source_section": "3.2",
            "confidence": 0.88,
            "requires_human_review": False,
        }
    ]
    with patch("energy_grid_research_agent.agents.synthesis._call_llm", return_value=_synth):
        report = run_synthesis("IEC 61850 query", findings_raw)
    assert report.query == "IEC 61850 query"
    assert report.summary == "Grid compliant."


# ── validation agent ──────────────────────────────────────────────────────────


def test_run_validation_returns_true_on_passed(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.validation._call_llm",
        return_value='{"validation_passed": true, "issues": []}',
    ):
        result = run_validation({"summary": "test"}, sample_chunks)
    assert result is True


def test_run_validation_returns_false_on_failed(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.validation._call_llm",
        return_value='{"validation_passed": false, "issues": ["claim not grounded"]}',
    ):
        result = run_validation({"summary": "test"}, sample_chunks)
    assert result is False


def test_run_validation_fallback_on_bad_json(sample_chunks) -> None:
    with patch(
        "energy_grid_research_agent.agents.validation._call_llm",
        return_value="validation looks true overall",
    ):
        result = run_validation({}, sample_chunks)
    assert isinstance(result, bool)


def test_run_validation_empty_chunks() -> None:
    with patch(
        "energy_grid_research_agent.agents.validation._call_llm",
        return_value='{"validation_passed": true}',
    ):
        result = run_validation({"summary": "test"}, [])
    assert result is True
