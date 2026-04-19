from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from energy_grid_research_agent.graph import get_app, make_initial_state
from energy_grid_research_agent.report import ResearchReport

app = FastAPI(title="Energy Grid Research Agent", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str
    session_id: str = ""


def _doc_count() -> int:
    from pathlib import Path

    corpus_dir = Path("data/corpus")
    return len(list(corpus_dir.glob("*.txt"))) if corpus_dir.exists() else 0


async def _event_stream(request: ResearchRequest) -> AsyncIterator[str]:
    graph_app = get_app()
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial = make_initial_state(request.query)

    async for state_chunk in graph_app.astream(initial, config=config):
        node_name = next(iter(state_chunk))
        payload = {"node": node_name, "state": state_chunk[node_name]}
        yield f"data: {json.dumps(payload, default=str)}\n\n"
        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


@app.post("/research")
async def research(request: ResearchRequest) -> StreamingResponse:
    return StreamingResponse(_event_stream(request), media_type="text/event-stream")


@app.post("/research/sync")
async def research_sync(request: ResearchRequest) -> ResearchReport:
    graph_app = get_app()
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial = make_initial_state(request.query)

    t0 = time.perf_counter()
    result: dict[str, Any] = await graph_app.ainvoke(initial, config=config)
    latency = time.perf_counter() - t0

    report_dict: dict[str, Any] = result.get("report") or {}
    report_dict.setdefault("query", request.query)
    report_dict.setdefault("summary", "No summary generated.")
    report_dict["latency_seconds"] = round(latency, 3)
    report_dict["agent_calls"] = result.get("agent_calls", 0)
    report_dict["human_approved"] = result.get("human_approved", False)
    report_dict["requires_human_review"] = result.get("requires_human_review", False)

    logger.info(f"Sync research completed in {latency:.2f}s")
    return ResearchReport(**report_dict)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "qwen2.5:1.5b",
        "embed_model": "nomic-embed-text",
        "docs_indexed": _doc_count(),
    }
