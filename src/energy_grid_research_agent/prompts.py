from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PromptEntry:
    version: str
    system: str
    user_template: str = field(default="")


class PromptRegistry:
    def __init__(self, data: dict[str, Any]) -> None:
        self._prompts: dict[str, PromptEntry] = {}
        prompts_dict: dict[str, Any] = data.get("prompts") or {}
        for name, raw in prompts_dict.items():
            entry: dict[str, Any] = raw if isinstance(raw, dict) else {}
            self._prompts[name] = PromptEntry(
                version=str(entry.get("version", "v1.0")),
                system=str(entry.get("system", "")),
                user_template=str(entry.get("user_template", "")),
            )

    def get(self, name: str) -> PromptEntry:
        return self._prompts.get(
            name,
            PromptEntry(version="v0.0", system="You are a helpful assistant."),
        )

    def format_user(self, name: str, **kwargs: object) -> str:
        return self.get(name).user_template.format(**kwargs)

    def list_names(self) -> list[str]:
        return list(self._prompts.keys())


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    path = Path("config/prompts.yaml")
    data: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}
    return PromptRegistry(data or {})
