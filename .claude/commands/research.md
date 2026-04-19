Run a live research query through the full LangGraph pipeline. Requires Ollama running and corpus ingested.

If $ARGUMENTS is provided, use it as the query. Otherwise use the default demo query.

```bash
if [ -n "$ARGUMENTS" ]; then
  uv run python -m energy_grid_research_agent "$ARGUMENTS"
else
  uv run python -m energy_grid_research_agent "What are IEC 61850 GOOSE timing requirements?"
fi
```

Pre-requisites: `make setup`, `make generate`, `make ingest` must have been run.
To check Ollama: use `/ollama-status`.
