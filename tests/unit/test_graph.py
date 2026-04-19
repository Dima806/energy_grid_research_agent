from __future__ import annotations

from unittest.mock import MagicMock, patch

from energy_grid_research_agent.graph import (
    _route_hitl,
    _route_validation,
    decompose_node,
    extraction_node,
    hitl_node,
    make_initial_state,
    retrieve_node,
    synthesis_node,
    validation_node,
)
from energy_grid_research_agent.report import GridFinding, ResearchReport

# ── make_initial_state ─────────────────────────────────────────────────────────


def test_make_initial_state_sets_query() -> None:
    state = make_initial_state("grid fault query")
    assert state["query"] == "grid fault query"


def test_make_initial_state_zeros() -> None:
    state = make_initial_state("q")
    assert state["agent_calls"] == 0
    assert state["validation_attempts"] == 0
    assert state["retrieved_chunks"] == []
    assert state["extracted_findings"] == []
    assert state["report"] is None
    assert state["requires_human_review"] is False
    assert state["human_approved"] is False


# ── decompose_node ─────────────────────────────────────────────────────────────


def test_decompose_node_returns_subtasks(sample_state) -> None:
    with patch(
        "energy_grid_research_agent.graph.run_decomposition",
        return_value=["subtask A", "subtask B"],
    ):
        result = decompose_node(sample_state)
    assert result["subtasks"] == ["subtask A", "subtask B"]


def test_decompose_node_increments_agent_calls(sample_state) -> None:
    with patch("energy_grid_research_agent.graph.run_decomposition", return_value=["t"]):
        result = decompose_node(sample_state)
    assert result["agent_calls"] == sample_state["agent_calls"] + 1


def test_decompose_node_calls_run_decomposition_with_query(sample_state) -> None:
    mock = MagicMock(return_value=["task"])
    with patch("energy_grid_research_agent.graph.run_decomposition", mock):
        decompose_node(sample_state)
    mock.assert_called_once_with(sample_state["query"])


# ── retrieve_node ──────────────────────────────────────────────────────────────


def test_retrieve_node_returns_chunks(sample_state, sample_chunks) -> None:
    with patch("energy_grid_research_agent.graph.run_retrieval", return_value=sample_chunks):
        result = retrieve_node(sample_state)
    assert result["retrieved_chunks"] == sample_chunks


def test_retrieve_node_increments_agent_calls(sample_state, sample_chunks) -> None:
    with patch("energy_grid_research_agent.graph.run_retrieval", return_value=sample_chunks):
        result = retrieve_node(sample_state)
    assert result["agent_calls"] == sample_state["agent_calls"] + 1


def test_retrieve_node_passes_subtasks(sample_state) -> None:
    mock = MagicMock(return_value=[])
    with patch("energy_grid_research_agent.graph.run_retrieval", mock):
        retrieve_node(sample_state)
    mock.assert_called_once_with(sample_state["subtasks"])


# ── extraction_node ────────────────────────────────────────────────────────────


def _make_finding(**kwargs) -> GridFinding:
    defaults = {
        "category": "fault",
        "description": "test finding",
        "source_document": "doc.txt",
        "source_section": "3.1",
        "confidence": 0.85,
        "requires_human_review": False,
    }
    defaults.update(kwargs)
    return GridFinding(**defaults)


def test_extraction_node_stores_findings(sample_state) -> None:
    finding = _make_finding()
    with patch(
        "energy_grid_research_agent.graph.run_extraction",
        return_value=([finding], False),
    ):
        result = extraction_node(sample_state)
    assert len(result["extracted_findings"]) == 1
    assert result["extracted_findings"][0]["category"] == "fault"


def test_extraction_node_sets_requires_human_review(sample_state) -> None:
    finding = _make_finding(confidence=0.3, requires_human_review=True)
    with patch(
        "energy_grid_research_agent.graph.run_extraction",
        return_value=([finding], True),
    ):
        result = extraction_node(sample_state)
    assert result["requires_human_review"] is True


def test_extraction_node_false_requires_review(sample_state) -> None:
    finding = _make_finding()
    with patch(
        "energy_grid_research_agent.graph.run_extraction",
        return_value=([finding], False),
    ):
        result = extraction_node(sample_state)
    assert result["requires_human_review"] is False


def test_extraction_node_increments_agent_calls(sample_state) -> None:
    with patch(
        "energy_grid_research_agent.graph.run_extraction",
        return_value=([_make_finding()], False),
    ):
        result = extraction_node(sample_state)
    assert result["agent_calls"] == sample_state["agent_calls"] + 1


# ── hitl_node ─────────────────────────────────────────────────────────────────


def test_hitl_node_auto_approves_when_no_review_needed(sample_state) -> None:
    sample_state["requires_human_review"] = False
    result = hitl_node(sample_state)
    assert result["human_approved"] is True


def test_hitl_node_prompts_when_review_needed(sample_state) -> None:
    sample_state["requires_human_review"] = True
    with patch("builtins.input", return_value="y"):
        result = hitl_node(sample_state)
    assert result["human_approved"] is True


def test_hitl_node_rejected_by_human(sample_state) -> None:
    sample_state["requires_human_review"] = True
    with patch("builtins.input", return_value="n"):
        result = hitl_node(sample_state)
    assert result["human_approved"] is False


def test_hitl_node_case_insensitive_yes(sample_state) -> None:
    sample_state["requires_human_review"] = True
    with patch("builtins.input", return_value="Y"):
        result = hitl_node(sample_state)
    assert result["human_approved"] is True


def test_hitl_node_whitespace_yes(sample_state) -> None:
    sample_state["requires_human_review"] = True
    with patch("builtins.input", return_value="  y  "):
        result = hitl_node(sample_state)
    assert result["human_approved"] is True


def test_hitl_node_unknown_input_means_rejected(sample_state) -> None:
    sample_state["requires_human_review"] = True
    with patch("builtins.input", return_value="maybe"):
        result = hitl_node(sample_state)
    assert result["human_approved"] is False


# ── synthesis_node ─────────────────────────────────────────────────────────────


def test_synthesis_node_stores_report(sample_state) -> None:
    report = ResearchReport(
        query="q",
        summary="Grid is compliant.",
        validation_passed=False,
    )
    with patch("energy_grid_research_agent.graph.run_synthesis", return_value=report):
        result = synthesis_node(sample_state)
    assert result["report"]["summary"] == "Grid is compliant."


def test_synthesis_node_increments_agent_calls(sample_state) -> None:
    report = ResearchReport(query="q", summary="s")
    with patch("energy_grid_research_agent.graph.run_synthesis", return_value=report):
        result = synthesis_node(sample_state)
    assert result["agent_calls"] == sample_state["agent_calls"] + 1


def test_synthesis_node_passes_query_and_findings(sample_state) -> None:
    report = ResearchReport(query="q", summary="s")
    mock = MagicMock(return_value=report)
    with patch("energy_grid_research_agent.graph.run_synthesis", mock):
        synthesis_node(sample_state)
    mock.assert_called_once_with(sample_state["query"], sample_state["extracted_findings"])


# ── validation_node ────────────────────────────────────────────────────────────


def test_validation_node_sets_validation_passed(sample_state) -> None:
    sample_state["report"] = {"summary": "report", "validation_passed": False}
    with patch("energy_grid_research_agent.graph.run_validation", return_value=True):
        result = validation_node(sample_state)
    assert result["report"]["validation_passed"] is True


def test_validation_node_sets_failed(sample_state) -> None:
    sample_state["report"] = {"summary": "report"}
    with patch("energy_grid_research_agent.graph.run_validation", return_value=False):
        result = validation_node(sample_state)
    assert result["report"]["validation_passed"] is False


def test_validation_node_handles_none_report(sample_state) -> None:
    sample_state["report"] = None
    with patch("energy_grid_research_agent.graph.run_validation", return_value=True):
        result = validation_node(sample_state)
    assert result["report"]["validation_passed"] is True


def test_validation_node_increments_agent_calls(sample_state) -> None:
    sample_state["report"] = {"summary": "s"}
    with patch("energy_grid_research_agent.graph.run_validation", return_value=True):
        result = validation_node(sample_state)
    assert result["agent_calls"] == sample_state["agent_calls"] + 1


# ── routing functions ─────────────────────────────────────────────────────────


def test_route_hitl_approved(sample_state) -> None:
    sample_state["human_approved"] = True
    assert _route_hitl(sample_state) == "synthesise"


def test_route_hitl_rejected(sample_state) -> None:
    sample_state["human_approved"] = False
    from langgraph.graph import END

    assert _route_hitl(sample_state) == END


def test_route_validation_passed(sample_state) -> None:
    from langgraph.graph import END

    sample_state["report"] = {"validation_passed": True}
    assert _route_validation(sample_state) == END


def test_route_validation_failed(sample_state) -> None:
    sample_state["report"] = {"validation_passed": False}
    sample_state["validation_attempts"] = 1
    assert _route_validation(sample_state) == "synthesise"


def test_route_validation_none_report(sample_state) -> None:
    sample_state["report"] = None
    sample_state["validation_attempts"] = 0
    assert _route_validation(sample_state) == "synthesise"


def test_route_validation_missing_key(sample_state) -> None:
    sample_state["report"] = {}
    sample_state["validation_attempts"] = 0
    assert _route_validation(sample_state) == "synthesise"


def test_route_validation_cap_at_3_attempts(sample_state) -> None:
    from langgraph.graph import END

    sample_state["report"] = {"validation_passed": False}
    sample_state["validation_attempts"] = 3
    assert _route_validation(sample_state) == END


def test_route_validation_cap_not_triggered_before_3(sample_state) -> None:
    sample_state["report"] = {"validation_passed": False}
    sample_state["validation_attempts"] = 2
    assert _route_validation(sample_state) == "synthesise"


def test_validation_node_increments_validation_attempts(sample_state) -> None:
    sample_state["report"] = {"summary": "s"}
    sample_state["validation_attempts"] = 1
    with patch("energy_grid_research_agent.graph.run_validation", return_value=False):
        result = validation_node(sample_state)
    assert result["validation_attempts"] == 2
