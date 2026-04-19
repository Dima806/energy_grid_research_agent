from __future__ import annotations

from collections.abc import Callable
from typing import Any

from energy_grid_research_agent.tools.calculate import calculate_metric
from energy_grid_research_agent.tools.extract import extract_structured_data
from energy_grid_research_agent.tools.search import search_grid_documents

ALL_TOOLS: list[Callable[..., Any]] = [
    search_grid_documents,
    extract_structured_data,
    calculate_metric,
]
