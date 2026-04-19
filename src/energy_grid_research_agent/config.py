from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:1.5b"
    embed_model: str = "nomic-embed-text"


class RetrievalConfig(BaseModel):
    top_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64


class ConfidenceConfig(BaseModel):
    hitl_threshold: float = 0.6


class ChromaConfig(BaseModel):
    persist_dir: str = "data/chroma"
    collection_name: str = "grid_docs"


class CorpusConfig(BaseModel):
    output_dir: str = "data/corpus"
    num_standards: int = 5
    num_fault_reports: int = 5


class Settings(BaseModel):
    ollama: OllamaConfig = OllamaConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    chroma: ChromaConfig = ChromaConfig()
    corpus: CorpusConfig = CorpusConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            return cls(**raw)
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_yaml(Path("config/settings.yaml"))
