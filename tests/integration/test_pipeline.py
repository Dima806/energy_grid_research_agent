from __future__ import annotations

from unittest.mock import patch


def test_smoke_corpus_generation(tmp_path) -> None:
    from energy_grid_research_agent.corpus.generator import generate_corpus

    out = generate_corpus(output_dir=tmp_path)
    txt_files = list(out.glob("*.txt"))
    assert len(txt_files) > 0


def test_smoke_full_pipeline(mock_hitl: bool) -> None:
    from energy_grid_research_agent.graph import get_app, make_initial_state

    get_app.cache_clear()
    app = get_app()
    initial = make_initial_state("What are IEC 61850 GOOSE requirements?")
    config = {"configurable": {"thread_id": "smoke-test"}}

    if mock_hitl:
        stub_chunks = [
            {
                "content": "GOOSE messages maximum transfer time 4 ms under IEC 61850.",
                "source": "grid_standard_01.txt",
                "score": 0.95,
            }
        ]
        with (
            patch(
                "energy_grid_research_agent.agents.decomposer._call_llm",
                return_value='["GOOSE timing requirements", "IEC 61850 scope"]',
            ),
            patch(
                "energy_grid_research_agent.agents.retrieval.search_grid_documents",
                return_value=stub_chunks,
            ),
            patch(
                "energy_grid_research_agent.agents.extraction._call_llm",
                return_value=(
                    '[{"category":"standard","description":"GOOSE max transfer time 4 ms",'
                    '"source_document":"grid_standard_01.txt","source_section":"4.1",'
                    '"confidence":0.9,"requires_human_review":false}]'
                ),
            ),
            patch(
                "energy_grid_research_agent.agents.synthesis._call_llm",
                return_value=(
                    '{"summary":"GOOSE max transfer time is 4 ms per IEC 61850.",'
                    '"compliance_artefacts":[],"validation_passed":true}'
                ),
            ),
            patch(
                "energy_grid_research_agent.agents.validation._call_llm",
                return_value='{"validation_passed":true}',
            ),
        ):
            result = app.invoke(initial, config=config)
    else:
        # Live Ollama run — requires `make ingest` to have been run first
        with patch("builtins.input", return_value="y"):
            result = app.invoke(initial, config=config)

    assert result.get("report") is not None
    assert result["report"].get("summary")
