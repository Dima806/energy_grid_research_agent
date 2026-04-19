from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mock-hitl",
        action="store_true",
        default=False,
        help="Auto-approve HITL prompts (no live Ollama required for smoke tests)",
    )


@pytest.fixture
def mock_hitl(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--mock-hitl", default=False))
