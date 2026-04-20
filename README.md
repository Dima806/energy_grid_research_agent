# energy_grid_research_agent

Multi-framework agentic RAG system for power grid technical document analysis. Analyses IEC standards and fault reports to produce structured compliance artefacts with Human-in-the-Loop review.

## Architecture

| Layer | Framework | Role |
|---|---|---|
| Agent logic | **ollama** (native `ollama.chat()`) | LLM calls — decompose, extract, synthesise, validate |
| Retrieval | **LangChain** | Chroma vector store, `RetrievalQA`, doc chunking |
| Orchestration | **LangGraph** | 6-node `StateGraph`, HITL conditional edge, validation retry loop |
| API | **FastAPI** | SSE streaming + sync endpoints |

All LLM inference is **localhost-only** via `qwen2.5:1.5b` on Ollama. No GPU required. No external API calls.

## Graph Flow

```
decompose → retrieve → extract → [HITL gate] → synthesise → validate
                                                     ↑____________| (max 3 retries)
```

- **HITL gate**: findings with confidence < 0.6 display each flagged finding (category, description, source, confidence score) then prompt `[y/n]` before synthesis.
- **Validation loop**: if `validation_passed=False`, re-synthesises up to 3 times then exits.

## Quick Start

```bash
# 1. One-time setup (installs uv, ollama, pulls models)
make setup

# 2. Generate synthetic corpus + ingest into Chroma
make generate
make ingest

# 3. Run a research query
make research
```

## Commands

| Command | When to run |
|---|---|
| `make setup` | Once, after clone |
| `make generate` | After `make clean` or first time |
| `make ingest` | After `make generate` |
| `make research` | Query the pipeline interactively (auto-starts Ollama) |
| `make serve` | Start FastAPI server on :8000 (auto-starts Ollama) |
| `make smoke` | Fast CI check — mocked LLM, no Ollama needed |
| `make test-unit` | Unit tests with coverage report |
| `make test-integ` | Full pipeline test (auto-starts Ollama) |
| `make lint` | ruff + ty type check |
| `make clean` | Remove corpus and vector store |

## Output

Every query produces a `ResearchReport`:

```json
{
  "query": "What are IEC 61850 GOOSE timing requirements?",
  "summary": "GOOSE messages must complete transfer within 4 ms...",
  "findings": [...],
  "compliance_artefacts": [...],
  "validation_passed": true,
  "requires_human_review": false,
  "prompt_version": "v1.0"
}
```

## API

```bash
# Streaming SSE
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "IEC 61850 GOOSE requirements"}'

# Sync
curl -X POST http://localhost:8000/research/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "fault clearance time zone 1"}'

# Health
curl http://localhost:8000/health
```

## Models

- **LLM**: `qwen2.5:1.5b` — ~935 MB RAM
- **Embeddings**: `nomic-embed-text` — ~274 MB RAM
- Total: ~1.6 GB RAM, ~1.5 GB disk

## Testing

```bash
make test-unit    # 112 unit tests, ~90% coverage, no Ollama needed
make smoke        # 2 integration tests, mocked LLM
make test-integ   # full pipeline against live Ollama (auto-started if needed)
```

If Ollama is unreachable and `--mock-hitl` is not passed, integration tests skip with a clear message rather than failing.

## Logging

Every graph node emits `INFO` logs with elapsed time:

```
2026-04-20 12:00:01 | INFO | [decompose] start | query='What are IEC 61850...'
2026-04-20 12:00:03 | INFO | [decompose] done | subtasks=3 | elapsed=2.33s
2026-04-20 12:00:03 | INFO | [retrieve] start | subtasks=3
2026-04-20 12:00:04 | INFO | [retrieve] done | chunks=12 | elapsed=0.43s
2026-04-20 12:00:04 | INFO | [extract] start | chunks=12
2026-04-20 12:00:10 | INFO | [extract] done | findings=4 | needs_review=False | elapsed=5.81s
2026-04-20 12:00:10 | INFO | [hitl] auto-approved (confidence ok)
2026-04-20 12:00:10 | INFO | [synthesise] start | findings=4
2026-04-20 12:00:14 | INFO | [synthesise] done | artefacts=4 | elapsed=3.92s
2026-04-20 12:00:14 | INFO | [validate] start | attempt=1/3
2026-04-20 12:00:17 | INFO | [validate] done | passed=True | elapsed=3.11s
```

Set `LOGURU_LEVEL=DEBUG` to also see per-LLM-call timings (model, prompt size, duration).

## Development

```bash
make lint         # ruff format + ruff check --fix + ty check
uv add <pkg>      # add dependency (updates uv.lock)
uv sync --frozen --extra dev  # sync after pulling
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (auto-installed by `make setup`)
- [Ollama](https://ollama.com/) (auto-installed by `make setup`)
- ~2 GB free RAM, ~2 GB disk
