from __future__ import annotations

import urllib.request

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mock-hitl",
        action="store_true",
        default=False,
        help="Auto-approve HITL prompts (no live Ollama required for smoke tests)",
    )


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2)  # noqa: S310
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_ollama(request: pytest.FixtureRequest) -> None:
    """Skip all live-Ollama tests if server is unreachable and --mock-hitl not set."""
    if not request.config.getoption("--mock-hitl", default=False) and not _ollama_running():
        pytest.skip("Ollama not running — start with `ollama serve` or run `make setup`")


@pytest.fixture
def mock_hitl(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--mock-hitl", default=False))
