Run lint and unit tests, report summary of failures only.

```bash
uv run ruff format src tests && uv run ruff check --fix src tests && uv run ty check src
uv run pytest tests/unit/ -q --cov=src/energy_grid_research_agent --cov-fail-under=80 2>&1 | tail -20
```
