from __future__ import annotations

import pytest

from energy_grid_research_agent.config import Settings, get_settings
from energy_grid_research_agent.network_guard import NetworkGuardError, assert_localhost
from energy_grid_research_agent.prompts import PromptEntry, PromptRegistry, get_prompt_registry

# ── PromptEntry ────────────────────────────────────────────────────────────────


def test_prompt_entry_defaults() -> None:
    entry = PromptEntry(version="v1.0", system="You are helpful.")
    assert entry.user_template == ""


def test_prompt_entry_fields() -> None:
    entry = PromptEntry(version="v2.1", system="Expert.", user_template="Query: {query}")
    assert entry.version == "v2.1"
    assert "Expert" in entry.system


# ── PromptRegistry ─────────────────────────────────────────────────────────────


def test_registry_loads_known_prompt() -> None:
    data = {
        "prompts": {
            "my_agent": {
                "version": "v1.3",
                "system": "You are a power grid expert.",
                "user_template": "Analyse: {query}",
            }
        }
    }
    registry = PromptRegistry(data)
    entry = registry.get("my_agent")
    assert entry.version == "v1.3"
    assert "power grid" in entry.system


def test_registry_returns_default_for_missing_key() -> None:
    registry = PromptRegistry({})
    entry = registry.get("nonexistent_agent")
    assert entry.version == "v0.0"
    assert "helpful" in entry.system


def test_registry_format_user() -> None:
    data = {
        "prompts": {
            "decomposer": {
                "version": "v1.0",
                "system": "You decompose.",
                "user_template": "Decompose: {query}",
            }
        }
    }
    registry = PromptRegistry(data)
    result = registry.format_user("decomposer", query="grid faults")
    assert result == "Decompose: grid faults"


def test_registry_format_user_missing_key_returns_empty_template() -> None:
    registry = PromptRegistry({})
    result = registry.format_user("missing", query="x")
    assert isinstance(result, str)


def test_registry_list_names() -> None:
    data = {
        "prompts": {"a": {"version": "v1", "system": "s"}, "b": {"version": "v2", "system": "t"}}
    }
    registry = PromptRegistry(data)
    names = registry.list_names()
    assert set(names) == {"a", "b"}


def test_registry_empty_prompts_section() -> None:
    registry = PromptRegistry({"prompts": {}})
    assert registry.list_names() == []


def test_registry_none_prompts_section() -> None:
    registry = PromptRegistry({"prompts": None})
    assert registry.list_names() == []


def test_registry_non_dict_entry_ignored() -> None:
    data = {"prompts": {"bad": "not-a-dict"}}
    registry = PromptRegistry(data)
    entry = registry.get("bad")
    assert entry.version == "v1.0"


# ── get_prompt_registry (lru_cache, uses config/prompts.yaml) ─────────────────


def test_get_prompt_registry_returns_registry_instance() -> None:
    registry = get_prompt_registry()
    assert isinstance(registry, PromptRegistry)


def test_get_prompt_registry_cached() -> None:
    r1 = get_prompt_registry()
    r2 = get_prompt_registry()
    assert r1 is r2


# ── Settings / Config ──────────────────────────────────────────────────────────


def test_settings_defaults() -> None:
    s = Settings()
    assert s.ollama.base_url == "http://127.0.0.1:11434"
    assert s.ollama.llm_model == "qwen2.5:1.5b"
    assert s.retrieval.top_k == 4
    assert s.confidence.hitl_threshold == 0.6
    assert s.chroma.collection_name == "grid_docs"
    assert s.corpus.num_standards == 5


def test_settings_from_yaml_missing_file(tmp_path) -> None:
    s = Settings.from_yaml(tmp_path / "nonexistent.yaml")
    assert s.ollama.llm_model == "qwen2.5:1.5b"


def test_settings_from_yaml_partial(tmp_path) -> None:
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text("ollama:\n  llm_model: 'custom-model'\n")
    s = Settings.from_yaml(yaml_file)
    assert s.ollama.llm_model == "custom-model"
    assert s.retrieval.top_k == 4


def test_get_settings_returns_settings_instance() -> None:
    s = get_settings()
    assert isinstance(s, Settings)


def test_get_settings_cached() -> None:
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ── NetworkGuard ───────────────────────────────────────────────────────────────


def test_assert_localhost_passes_for_127() -> None:
    assert_localhost("http://127.0.0.1:11434")


def test_assert_localhost_passes_for_localhost() -> None:
    assert_localhost("http://localhost:8000/api")


def test_assert_localhost_blocks_external() -> None:
    with pytest.raises(NetworkGuardError, match="blocked"):
        assert_localhost("http://api.openai.com/v1")


def test_assert_localhost_blocks_external_ip() -> None:
    with pytest.raises(NetworkGuardError):
        assert_localhost("http://8.8.8.8:11434")


def test_assert_localhost_empty_host() -> None:
    with pytest.raises(NetworkGuardError):
        assert_localhost("not-a-url")


def test_network_guard_error_is_exception() -> None:
    err = NetworkGuardError("blocked")
    assert isinstance(err, Exception)
    assert "blocked" in str(err)
