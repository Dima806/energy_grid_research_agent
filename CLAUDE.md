# CLAUDE.md — energy_grid_research_agent

Multi-framework agentic RAG system for power grid technical document analysis.

---

## Framework Roles (strict separation)

| Layer | Framework | Responsibility |
|---|---|---|
| Agent logic | **ollama** (native client) | LLM calls via `ollama.chat()` in each agent module |
| Retrieval/tools | **LangChain** | Doc loading, chunking, Chroma vector store, `RetrievalQA` |
| Orchestration | **LangGraph** | `StateGraph`, state transitions, HITL conditional edge |

Do not use `agno` — it chains OpenAI imports even for Ollama models. All agents call `ollama.chat()` directly via `_call_llm()` helpers. LangGraph does not know about agent internals.

---

## Key Commands

```bash
make setup        # install uv + ollama, sync deps, pull models (one-time)
make generate     # generate synthetic grid corpus
make ingest       # embed + store into Chroma via LangChain
make research     # single query via CLI (Rich output); auto-starts Ollama
make serve        # async FastAPI on localhost:8000; auto-starts Ollama
make stream       # demo SSE streaming endpoint
make hitl         # interactive HITL demo; auto-starts Ollama
make eval         # retrieval precision + claim grounding eval; auto-starts Ollama
make smoke        # fast e2e smoke test (mocked LLM, no live Ollama needed)
make test-unit    # pytest tests/unit/ with ≥80% coverage
make test-integ   # pytest tests/integration/; auto-starts Ollama if needed
make lint         # ruff format + ruff check + ty check
make clean        # remove data/corpus/, data/chroma/, mlruns/
```

---

## Models (Ollama, localhost only)

- **LLM**: `qwen2.5:1.5b` (~935 MB RAM)
- **Embeddings**: `nomic-embed-text` (~300 MB RAM)
- Total budget: ~1.6 GB RAM, ~1.5 GB disk
- `NetworkGuard` (`src/.../network_guard.py`) enforces localhost-only LLM calls
- No GPU: `CUDA_VISIBLE_DEVICES=` and `OLLAMA_NUM_GPU=0` exported in Makefile

---

## Project Structure

```
src/energy_grid_research_agent/
├── config.py           # Pydantic Settings (loaded from config/settings.yaml)
├── network_guard.py    # localhost-only URL enforcement
├── prompts.py          # prompt registry loader (reads config/prompts.yaml)
├── corpus/generator.py # synthetic IEC standard + fault report generator
├── retrieval/
│   ├── embedder.py     # OllamaEmbeddings + Chroma (langchain_ollama)
│   └── chain.py        # RetrievalQA (langchain_classic)
├── tools/
│   ├── registry.py
│   ├── search.py       # search_grid_documents tool
│   ├── extract.py      # extract_structured_data tool
│   └── calculate.py    # calculate_metric tool
├── agents/
│   ├── decomposer.py   # query decomposition via ollama.chat()
│   ├── retrieval.py    # document retrieval (wraps search tool)
│   ├── extraction.py   # structured finding extraction via ollama.chat()
│   ├── synthesis.py    # report synthesis via ollama.chat()
│   └── validation.py   # claim grounding validation via ollama.chat()
├── graph.py            # LangGraph StateGraph + HITL node
├── report.py           # ResearchReport + ComplianceArtefact Pydantic models
└── serve.py            # Async FastAPI: /research (SSE) + /research/sync
config/
├── settings.yaml
└── prompts.yaml        # versioned prompt registry (version logged per response)
tests/
├── unit/               # 112 tests, ~90% coverage, no live Ollama needed
└── integration/        # corpus + full pipeline tests (requires live Ollama)
data/
├── corpus/             # synthetic grid documents (git-ignored)
└── chroma/             # Chroma persist directory (git-ignored)
```

---

## Prompt Registry

Prompts live in `config/prompts.yaml`, loaded by `src/.../prompts.py`.
Changing a prompt = config change, not code change.
Every API response includes `prompt_version` field for A/B auditability.

---

## LangGraph State

```python
class AgentState(TypedDict):
    query: str
    subtasks: list[str]
    retrieved_chunks: Annotated[list[dict], operator.add]
    extracted_findings: Annotated[list[dict], operator.add]
    report: dict | None
    requires_human_review: bool   # set by extraction agent
    human_approved: bool          # set by HITL node
    agent_calls: int
    validation_attempts: int      # capped at 3; prevents infinite synthesis loop
```

Graph flow: `decompose → retrieve → extract → hitl → synthesise → validate`

- **HITL gate**: if `requires_human_review` (confidence < 0.6) → display each flagged finding (category, description, source, confidence) then prompt `[y/n]`; else auto-approve.
- **Validation loop**: if `validation_passed` is False → re-enter `synthesise`, max **3 attempts** then exit.

---

## Output Schemas

- `GridFinding`: category, description, source_document, source_section, page_number, confidence, requires_human_review
- `ComplianceArtefact`: claim, evidence, confidence, validated, prompt_version, model_version, timestamp
- `ResearchReport`: aggregates all findings + metadata

---

## Engineering Conventions

- **Package manager**: `uv` with frozen `uv.lock` — never `pip install` directly
- **Typing**: fully typed Python, checked with `ty` (Astral); `unresolved-import = "warn"` for langchain stubs
- **Testing**: 112 unit tests, ~90% coverage; HITL and LLM mocked via `patch("...._call_llm")`; integration tests use `--mock-hitl` flag or live Ollama; `require_ollama` session fixture skips live tests if server unreachable
- **CI**: GitHub Actions — lint + unit tests on every push
- **Ollama auto-start**: `_ensure-ollama` Makefile target polls `http://127.0.0.1:11434/` and starts `ollama serve` if needed; used by `research`, `serve`, `hitl`, `eval`, `test-integ`
- **Logging**: `logger.info` at each graph node entry/exit with elapsed time; `logger.debug` logs each `_call_llm` start/end with model name, prompt char count, and duration
- **HITL async**: `asyncio.to_thread` wraps blocking `input()` in async context
- **Streaming**: node-level SSE (fast), not token-level
- **Chroma over FAISS**: chosen for LangChain integration fit, not raw speed

---

## Key Dependencies

```toml
ollama>=0.3
langchain>=0.2
langchain-classic>=1.0      # RetrievalQA
langchain-community>=0.2
langchain-ollama>=0.1       # OllamaEmbeddings, OllamaLLM
langchain-chroma>=0.1
langchain-text-splitters>=0.1
langgraph>=0.2
chromadb>=0.5
pdfplumber>=0.11
pydantic>=2.6
pydantic-settings>=2.3
fastapi>=0.111
uvicorn>=0.30
httpx>=0.27
rich>=13.7
loguru>=0.7
numpy>=1.26
# dev
pytest>=8.0, pytest-asyncio>=0.23, pytest-cov>=5.0, ruff>=0.4, ty>=0.0.1a6
```

---

## FastAPI Endpoints

- `POST /research` — streaming SSE, yields node states as events
- `POST /research/sync` — returns complete `ResearchReport`
- `GET /health` — model status + indexed doc count
