.PHONY: setup generate ingest research serve stream hitl eval smoke \
        test-unit test-integ lint clean _ensure-ollama

# ── Variables ─────────────────────────────────────────────────────────────────

# Force CPU-only — hide all GPU devices from CUDA, ROCm, and PyTorch
export CUDA_VISIBLE_DEVICES  :=
export HIP_VISIBLE_DEVICES   :=
export ROCR_VISIBLE_DEVICES  :=
export OLLAMA_CUDA_DEVICE    := -1
export OLLAMA_NUM_GPU        := 0

PYTHON      := uv run python
PYTEST      := uv run pytest
RUFF        := uv run ruff
TY          := uv run ty
OLLAMA      := ollama
APP_MODULE  := energy_grid_research_agent
SRC         := src/$(APP_MODULE)
PORT        := 8000
QUERY       ?= "What are the fault tolerance requirements for IEC 61850 substations?"
SESSION_ID  ?= demo-session-1

# =============================================================================
# 1. FIRST TIME SETUP — run once after cloning the repo
#    Installs uv, Ollama, Python deps, pulls LLM + embedding models.
#    Re-run after adding new dependencies to pyproject.toml.
# =============================================================================

setup:
	@if ! command -v uv &>/dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@if ! command -v ollama &>/dev/null; then \
		echo "Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
	fi
	@if [ -f uv.lock ]; then \
		uv sync --frozen --extra dev; \
	else \
		uv sync --extra dev; \
	fi
	@if ! ollama list &>/dev/null; then \
		echo "Starting ollama serve..."; \
		ollama serve &>/tmp/ollama.log & \
		echo "Waiting for Ollama API..."; \
		until curl -sf http://127.0.0.1:11434/ &>/dev/null; do sleep 1; done; \
	fi
	$(OLLAMA) pull qwen2.5:1.5b
	$(OLLAMA) pull nomic-embed-text
	mkdir -p data/corpus data/chroma

# =============================================================================
# 2. DATA PIPELINE — run in order after setup, and after any corpus changes
#    generate → creates synthetic IEC standard + fault report documents
#    ingest   → chunks, embeds, and stores documents in Chroma vector store
# =============================================================================

generate:
	$(PYTHON) -m $(APP_MODULE).corpus.generator

ingest:
	$(PYTHON) -m $(APP_MODULE).retrieval.embedder

# =============================================================================
# 3. RUN — requires setup + data pipeline to be complete
#    research → one-shot CLI query with Rich output (good for quick testing)
#    serve    → start FastAPI server; needed before running `stream`
#    stream   → SSE demo against running server (run `make serve` first)
#    hitl     → interactive query that prompts for human approval if confidence < 0.6
# =============================================================================

research: _ensure-ollama
	$(PYTHON) -m $(APP_MODULE) --query $(QUERY)

serve: _ensure-ollama
	uv run uvicorn $(APP_MODULE).serve:app \
		--host 127.0.0.1 --port $(PORT) --reload

stream:
	curl -sN -X POST http://127.0.0.1:$(PORT)/research \
		-H "Content-Type: application/json" \
		-d '{"query": $(QUERY), "session_id": "$(SESSION_ID)"}' \
		--no-buffer

hitl: _ensure-ollama
	$(PYTHON) -m $(APP_MODULE) --query $(QUERY) --hitl

# =============================================================================
# 4. EVALUATION — requires setup + data pipeline to be complete
#    Measures retrieval precision and claim grounding across test queries.
#    Run after ingesting new corpus or changing prompts/models.
# =============================================================================

eval: _ensure-ollama
	$(PYTHON) -m $(APP_MODULE).eval

# =============================================================================
# 5. TESTS — no live Ollama needed for unit tests; integration tests require it
#    test-unit  → fast, fully mocked, run on every commit (CI)
#    test-integ → requires live Ollama + ingested Chroma; run before releasing
#    smoke      → minimal end-to-end check (1 doc, mock HITL); run after ingest
# =============================================================================

test-unit:
	$(PYTEST) tests/unit/ -v \
		--cov=$(SRC) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80

test-integ: _ensure-ollama
	$(PYTEST) tests/integration/ -v

_ensure-ollama:
	@if ! curl -sf http://127.0.0.1:11434/ &>/dev/null; then \
		echo "Starting Ollama..."; \
		ollama serve &>/tmp/ollama.log & \
		until curl -sf http://127.0.0.1:11434/ &>/dev/null; do sleep 1; done; \
		echo "Ollama ready."; \
	fi

smoke:
	$(PYTEST) tests/integration/test_pipeline.py \
		-k smoke \
		--mock-hitl \
		-v

# =============================================================================
# 6. QUALITY — run before every commit; also enforced by CI
#    Runs ruff formatter, ruff linter (auto-fix), and ty type checker.
# =============================================================================

lint:
	$(RUFF) format $(SRC) tests
	$(RUFF) check --fix $(SRC) tests
	$(TY) check $(SRC)

# =============================================================================
# 7. CLEANUP — removes generated data and caches; does not touch source code
#    Run to reset the data pipeline (corpus + vector store) or free disk space.
# =============================================================================

clean:
	rm -rf data/corpus data/chroma mlruns htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
