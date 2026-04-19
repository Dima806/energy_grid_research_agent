# CLAUDE.md — energy_grid_research_agent

Multi-framework agentic RAG system for power grid technical document analysis.
Extends `financial_research_agent` with Agno + LangChain + LangGraph + HITL.

---

## Framework Roles (strict separation)

| Layer | Framework | Responsibility |
|---|---|---|
| Agent definition | **Agno** | Typed agents, tool registries, instruction prompts, `response_model` |
| Retrieval/tools | **LangChain** | Doc loading, chunking, Chroma vector store, `RetrievalQA` |
| Orchestration | **LangGraph** | `StateGraph`, state transitions, HITL conditional edge |

Do not blur these roles. Agno agents do not know about graph routing. LangGraph does not know about agent internals.

---

## Key Commands

```bash
make setup        # pull Ollama models + uv sync --frozen (one-time)
make generate     # generate synthetic grid corpus
make ingest       # embed + store into Chroma via LangChain
make research     # single query via CLI (Rich output)
make serve        # async FastAPI on localhost:8000
make stream       # demo SSE streaming endpoint
make hitl         # interactive HITL demo
make eval         # retrieval precision + claim grounding eval
make smoke        # fast e2e smoke test (1 doc, 20 chunks, mock HITL)
make test-unit    # pytest tests/unit/
make test-integ   # pytest tests/integration/ (requires live Ollama)
make lint         # ruff format + ruff check + ty check
make clean        # remove data/corpus/, data/chroma/, mlruns/
```

---

## Models (Ollama, localhost only)

- **LLM**: `qwen2.5:1.5b` (~935 MB RAM)
- **Embeddings**: `nomic-embed-text` (~300 MB RAM)
- Total budget: ~1.6 GB RAM, ~1.5 GB disk
- `NetworkGuard` (`src/.../network_guard.py`) enforces localhost-only LLM calls

---

## Project Structure

```
src/energy_grid_research_agent/
├── config.py           # Pydantic Settings
├── network_guard.py    # localhost-only enforcement
├── prompts.py          # prompt registry loader (reads config/prompts.yaml)
├── corpus/generator.py # synthetic IEC standard + fault report generator
├── retrieval/
│   ├── embedder.py     # OllamaEmbeddings + Chroma
│   └── chain.py        # RetrievalQA with return_source_documents=True
├── tools/
│   ├── registry.py
│   ├── search.py       # search_grid_documents Agno tool
│   ├── extract.py      # extract_structured_data Agno tool
│   └── calculate.py    # calculate_metric Agno tool
├── agents/
│   ├── decomposer.py   # Agno: query decomposition
│   ├── retrieval.py    # Agno: document retrieval
│   ├── extraction.py   # Agno: structured finding extraction
│   ├── synthesis.py    # Agno: report synthesis
│   └── validation.py   # Agno: claim grounding validation
├── graph.py            # LangGraph StateGraph + HITL node
├── report.py           # ResearchReport + ComplianceArtefact Pydantic models
└── serve.py            # Async FastAPI: /research (SSE) + /research/sync
config/
├── settings.yaml
└── prompts.yaml        # versioned prompt registry (version logged per response)
data/
├── corpus/             # synthetic grid documents
└── chroma/             # Chroma persist directory
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
```

Graph flow: `decompose → retrieve → extract → hitl → synthesise → validate`
HITL gate: if `requires_human_review` (confidence < 0.6) → prompt human; else pass through.
Validation loop: if `validation_passed` is False → re-enter `synthesise`.

---

## Output Schemas

- `GridFinding`: category, description, source_document, source_section, page_number, confidence, requires_human_review
- `ComplianceArtefact`: claim, evidence, confidence, validated, prompt_version, model_version, timestamp
- `ResearchReport`: aggregates all findings + metadata

---

## Engineering Conventions

- **Package manager**: `uv` with frozen `uv.lock` — never `pip install` directly
- **Typing**: fully typed Python, checked with `ty` (Astral)
- **Testing**: HITL mocked in unit tests; integration tests require live Ollama
- **CI**: GitHub Actions — lint + unit tests on every push
- **HITL async**: `asyncio.to_thread` wraps blocking `input()` in async context
- **Streaming**: node-level SSE (fast), not token-level (avoids slow LLM streaming)
- **Chroma over FAISS**: chosen for LangChain integration fit, not raw speed

---

## Key Dependencies

```toml
agno>=1.0
langchain>=0.2
langchain-community>=0.2
langchain-chroma>=0.1
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
pytest>=8.0, pytest-asyncio>=0.23, ruff>=0.4, ty>=0.0.1a6
```

---

## FastAPI Endpoints

- `POST /research` — streaming SSE, yields node states as events
- `POST /research/sync` — returns complete `ResearchReport`
- `GET /health` — model status + indexed doc count
