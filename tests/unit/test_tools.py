from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from energy_grid_research_agent.tools.calculate import calculate_metric
from energy_grid_research_agent.tools.extract import extract_structured_data
from energy_grid_research_agent.tools.registry import ALL_TOOLS
from energy_grid_research_agent.tools.search import search_grid_documents

# ── calculate_metric ───────────────────────────────────────────────────────────


def test_calculate_mean() -> None:
    result = calculate_metric("mean", [2.0, 4.0, 6.0])
    assert result["result"] == pytest.approx(4.0)


def test_calculate_max() -> None:
    result = calculate_metric("max", [1.0, 5.0, 3.0])
    assert result["result"] == pytest.approx(5.0)


def test_calculate_min() -> None:
    result = calculate_metric("min", [1.0, 5.0, 3.0])
    assert result["result"] == pytest.approx(1.0)


def test_calculate_sum() -> None:
    result = calculate_metric("sum", [10.0, 20.0, 30.0])
    assert result["result"] == pytest.approx(60.0)


def test_calculate_rms() -> None:
    result = calculate_metric("rms", [3.0, 4.0])
    expected = math.sqrt((9 + 16) / 2)
    assert result["result"] == pytest.approx(expected)


def test_calculate_impedance_two_values() -> None:
    result = calculate_metric("impedance", [3.0, 4.0])
    assert result["result"] == pytest.approx(5.0)


def test_calculate_impedance_one_value() -> None:
    result = calculate_metric("impedance", [5.0])
    assert result["result"] == pytest.approx(5.0)


def test_calculate_power_factor() -> None:
    result = calculate_metric("power_factor", [1.0, 0.0])
    assert 0.0 <= result["result"] <= 1.0


def test_calculate_empty_values() -> None:
    result = calculate_metric("mean", [])
    assert result["result"] == 0.0
    assert result["n"] == 0.0


def test_calculate_unknown_metric_falls_back_to_mean() -> None:
    result = calculate_metric("unknown_metric", [2.0, 4.0])
    assert result["result"] == pytest.approx(3.0)


def test_calculate_returns_n() -> None:
    result = calculate_metric("mean", [1.0, 2.0, 3.0])
    assert result["n"] == 3.0


def test_calculate_single_value() -> None:
    result = calculate_metric("mean", [42.0])
    assert result["result"] == pytest.approx(42.0)


# ── extract_structured_data ────────────────────────────────────────────────────


def test_extract_valid_json() -> None:
    text = '{"category": "fault", "confidence": 0.9}'
    result = extract_structured_data(text)
    assert result["category"] == "fault"
    assert result["confidence"] == 0.9


def test_extract_json_array_returns_first_parse() -> None:
    text = '[{"a": 1}]'
    result = extract_structured_data(text)
    assert isinstance(result, list)


def test_extract_json_embedded_in_text() -> None:
    text = 'Some prefix {"key": "value"} some suffix'
    result = extract_structured_data(text)
    assert result.get("key") == "value"


def test_extract_plain_text_fallback() -> None:
    text = "No JSON here, just plain analysis text about IEC standards."
    result = extract_structured_data(text)
    assert "description" in result
    assert result["confidence"] == pytest.approx(0.3)
    assert result["requires_human_review"] is True


def test_extract_empty_string_fallback() -> None:
    result = extract_structured_data("")
    assert "description" in result


def test_extract_with_schema_hint() -> None:
    result = extract_structured_data("plain text", schema_hint="GridFinding")
    assert result["schema_hint"] == "GridFinding"


def test_extract_nested_json() -> None:
    text = '{"findings": [{"category": "standard"}]}'
    result = extract_structured_data(text)
    assert "findings" in result


def test_extract_invalid_json_with_brace() -> None:
    text = "value is {not valid json}"
    result = extract_structured_data(text)
    assert "description" in result


# ── search_grid_documents ─────────────────────────────────────────────────────


def test_search_returns_list_of_dicts(sample_chunks) -> None:
    mock_result = {
        "source_documents": [
            MagicMock(
                page_content=c["content"],
                metadata={"source": c["source"], "score": c["score"]},
            )
            for c in sample_chunks
        ]
    }
    with patch("energy_grid_research_agent.tools.search.query_chain", return_value=mock_result):
        results = search_grid_documents("GOOSE timing")
    assert isinstance(results, list)
    assert all("content" in r and "source" in r for r in results)


def test_search_respects_top_k(sample_chunks) -> None:
    mock_result = {
        "source_documents": [
            MagicMock(
                page_content=c["content"],
                metadata={"source": c["source"], "score": c["score"]},
            )
            for c in sample_chunks
        ]
    }
    with patch("energy_grid_research_agent.tools.search.query_chain", return_value=mock_result):
        results = search_grid_documents("query", top_k=2)
    assert len(results) <= 2


def test_search_handles_empty_source_documents() -> None:
    with patch(
        "energy_grid_research_agent.tools.search.query_chain",
        return_value={"source_documents": []},
    ):
        results = search_grid_documents("query")
    assert results == []


def test_search_handles_missing_source_key() -> None:
    with patch("energy_grid_research_agent.tools.search.query_chain", return_value={}):
        results = search_grid_documents("query")
    assert results == []


def test_search_missing_metadata_uses_defaults() -> None:
    doc = MagicMock(page_content="content", metadata={})
    with patch(
        "energy_grid_research_agent.tools.search.query_chain",
        return_value={"source_documents": [doc]},
    ):
        results = search_grid_documents("query")
    assert results[0]["source"] == "unknown"
    assert results[0]["score"] == 1.0


# ── registry ──────────────────────────────────────────────────────────────────


def test_all_tools_is_list() -> None:
    assert isinstance(ALL_TOOLS, list)
    assert len(ALL_TOOLS) == 3


def test_all_tools_are_callable() -> None:
    for t in ALL_TOOLS:
        assert callable(t)
