from __future__ import annotations

from typing import Any

from energy_grid_research_agent.retrieval.chain import query_chain


def search_grid_documents(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Search the power grid document corpus for relevant passages."""
    result = query_chain(query)
    docs = result.get("source_documents", [])[:top_k]
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": doc.metadata.get("score", 1.0),
        }
        for doc in docs
    ]
