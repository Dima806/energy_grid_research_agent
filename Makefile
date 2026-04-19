.PHONY: setup generate ingest research serve stream hitl eval smoke \
        test-unit test-integ lint clean

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

# ── Setup ─────────────────────────────────────────────────────────────────────

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
		uv sync --frozen; \
	else \
		uv sync; \
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

# ── Corpus ────────────────────────────────────────────────────────────────────

generate:
	$(PYTHON) -m $(APP_MODULE).corpus.generator

ingest:
	$(PYTHON) -m $(APP_MODULE).retrieval.embedder

# ── Run ───────────────────────────────────────────────────────────────────────

research:
	$(PYTHON) -m $(APP_MODULE) --query $(QUERY)

serve:
	uv run uvicorn $(APP_MODULE).serve:app \
		--host 127.0.0.1 --port $(PORT) --reload

stream:
	curl -sN -X POST http://127.0.0.1:$(PORT)/research \
		-H "Content-Type: application/json" \
		-d '{"query": $(QUERY), "session_id": "$(SESSION_ID)"}' \
		--no-buffer

hitl:
	$(PYTHON) -m $(APP_MODULE) --query $(QUERY) --hitl

# ── Evaluation ────────────────────────────────────────────────────────────────

eval:
	$(PYTHON) -m $(APP_MODULE).eval

# ── Tests ─────────────────────────────────────────────────────────────────────

smoke:
	$(PYTEST) tests/integration/test_pipeline.py \
		-k smoke \
		--num-docs 1 --num-chunks 20 --mock-hitl \
		-v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integ:
	$(PYTEST) tests/integration/ -v

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(RUFF) format $(SRC) tests
	$(RUFF) check --fix $(SRC) tests --fix
	$(TY) check $(SRC)

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf data/corpus data/chroma mlruns
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
