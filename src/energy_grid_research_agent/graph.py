from __future__ import annotations

import asyncio
import operator
import time
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from energy_grid_research_agent.agents.decomposer import run_decomposition
from energy_grid_research_agent.agents.extraction import run_extraction
from energy_grid_research_agent.agents.retrieval import run_retrieval
from energy_grid_research_agent.agents.synthesis import run_synthesis
from energy_grid_research_agent.agents.validation import run_validation
from energy_grid_research_agent.report import ResearchReport


class AgentState(TypedDict):
    query: str
    subtasks: list[str]
    retrieved_chunks: Annotated[list[dict[str, Any]], operator.add]
    extracted_findings: Annotated[list[dict[str, Any]], operator.add]
    report: dict[str, Any] | None
    requires_human_review: bool
    human_approved: bool
    agent_calls: int
    validation_attempts: int


def decompose_node(state: AgentState) -> dict[str, Any]:
    logger.info("[decompose] start | query={!r}", state["query"][:80])
    t0 = time.perf_counter()
    subtasks = run_decomposition(state["query"])
    elapsed = time.perf_counter() - t0
    logger.info("[decompose] done | subtasks={} | elapsed={:.2f}s", len(subtasks), elapsed)
    return {"subtasks": subtasks, "agent_calls": state["agent_calls"] + 1}


def retrieve_node(state: AgentState) -> dict[str, Any]:
    logger.info("[retrieve] start | subtasks={}", len(state["subtasks"]))
    t0 = time.perf_counter()
    chunks = run_retrieval(state["subtasks"])
    elapsed = time.perf_counter() - t0
    logger.info("[retrieve] done | chunks={} | elapsed={:.2f}s", len(chunks), elapsed)
    return {"retrieved_chunks": chunks, "agent_calls": state["agent_calls"] + 1}


def extraction_node(state: AgentState) -> dict[str, Any]:
    logger.info("[extract] start | chunks={}", len(state["retrieved_chunks"]))
    t0 = time.perf_counter()
    findings, needs_review = run_extraction(state["query"], state["retrieved_chunks"])
    elapsed = time.perf_counter() - t0
    logger.info(
        "[extract] done | findings={} | needs_review={} | elapsed={:.2f}s",
        len(findings),
        needs_review,
        elapsed,
    )
    return {
        "extracted_findings": [f.model_dump() for f in findings],
        "requires_human_review": needs_review,
        "agent_calls": state["agent_calls"] + 1,
    }


def hitl_node(state: AgentState) -> dict[str, Any]:
    if not state["requires_human_review"]:
        logger.info("[hitl] auto-approved (confidence ok)")
        return {"human_approved": True}
    logger.info("[hitl] prompting human — low-confidence findings detected")
    response = input(
        f"\nLow-confidence findings require review.\nQuery: {state['query']}\nApprove? [y/n]: "
    )
    approved = response.strip().lower() == "y"
    logger.info("[hitl] human decision: {}", "approved" if approved else "rejected")
    return {"human_approved": approved}


async def hitl_node_async(state: AgentState) -> dict[str, Any]:
    if not state["requires_human_review"]:
        logger.info("[hitl] auto-approved (confidence ok)")
        return {"human_approved": True}
    logger.info("[hitl] prompting human — low-confidence findings detected")
    response = await asyncio.to_thread(
        input,
        f"\nLow-confidence findings require review.\nQuery: {state['query']}\nApprove? [y/n]: ",
    )
    approved = response.strip().lower() == "y"
    logger.info("[hitl] human decision: {}", "approved" if approved else "rejected")
    return {"human_approved": approved}


def synthesis_node(state: AgentState) -> dict[str, Any]:
    logger.info("[synthesise] start | findings={}", len(state["extracted_findings"]))
    t0 = time.perf_counter()
    report: ResearchReport = run_synthesis(state["query"], state["extracted_findings"])
    elapsed = time.perf_counter() - t0
    n = len(report.compliance_artefacts)
    logger.info("[synthesise] done | artefacts={} | elapsed={:.2f}s", n, elapsed)
    return {
        "report": report.model_dump(),
        "agent_calls": state["agent_calls"] + 1,
    }


def validation_node(state: AgentState) -> dict[str, Any]:
    attempt = state["validation_attempts"] + 1
    logger.info("[validate] start | attempt={}/{}", attempt, _MAX_VALIDATION_ATTEMPTS)
    t0 = time.perf_counter()
    passed = run_validation(state.get("report") or {}, state["retrieved_chunks"])
    elapsed = time.perf_counter() - t0
    logger.info("[validate] done | passed={} | elapsed={:.2f}s", passed, elapsed)
    report = dict(state.get("report") or {})
    report["validation_passed"] = passed
    return {
        "report": report,
        "agent_calls": state["agent_calls"] + 1,
        "validation_attempts": attempt,
    }


def _route_hitl(state: AgentState) -> str:
    return "synthesise" if state["human_approved"] else END  # type: ignore[return-value]


_MAX_VALIDATION_ATTEMPTS = 3


def _route_validation(state: AgentState) -> str:
    report = state.get("report") or {}
    exhausted = state["validation_attempts"] >= _MAX_VALIDATION_ATTEMPTS
    if report.get("validation_passed", False) or exhausted:
        return END  # type: ignore[return-value]
    return "synthesise"  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_app() -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(AgentState)  # type: ignore

    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract", extraction_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("synthesise", synthesis_node)
    graph.add_node("validate", validation_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "hitl")
    graph.add_conditional_edges("hitl", _route_hitl)
    graph.add_edge("synthesise", "validate")
    graph.add_conditional_edges("validate", _route_validation)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def make_initial_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        subtasks=[],
        retrieved_chunks=[],
        extracted_findings=[],
        report=None,
        requires_human_review=False,
        human_approved=False,
        agent_calls=0,
        validation_attempts=0,
    )
