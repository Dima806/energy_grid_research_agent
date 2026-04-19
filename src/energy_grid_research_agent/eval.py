from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "IEC 61850 GOOSE message transfer time requirements",
        "expected_keywords": ["GOOSE", "4 ms", "61850"],
    },
    {
        "query": "fault clearance time for zone 1 protection",
        "expected_keywords": ["80 ms", "zone 1", "protection"],
    },
    {
        "query": "power factor requirements at point of common coupling",
        "expected_keywords": ["0.95", "power factor", "common coupling"],
    },
]


@dataclass
class EvalResult:
    query: str
    retrieval_precision: float
    chunks_retrieved: int
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def mean_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.retrieval_precision for r in self.results) / len(self.results)


def _precision(chunks: list[dict[str, Any]], keywords: list[str]) -> tuple[float, list[str]]:
    matched: list[str] = []
    for kw in keywords:
        for chunk in chunks:
            if kw.lower() in chunk.get("content", "").lower():
                matched.append(kw)
                break
    return len(matched) / max(len(keywords), 1), matched


def run_eval() -> EvalReport:
    from energy_grid_research_agent.agents.retrieval import run_retrieval

    report = EvalReport()
    for test in _TEST_QUERIES:
        query: str = test["query"]
        keywords: list[str] = test["expected_keywords"]
        chunks = run_retrieval([query])
        precision, matched = _precision(chunks, keywords)
        result = EvalResult(
            query=query,
            retrieval_precision=precision,
            chunks_retrieved=len(chunks),
            matched_keywords=matched,
        )
        report.results.append(result)
        logger.info(f"Query: '{query[:50]}' | precision={precision:.2f} | chunks={len(chunks)}")

    logger.info(f"Mean retrieval precision: {report.mean_precision:.2f}")
    return report


if __name__ == "__main__":
    run_eval()
