from __future__ import annotations

import json
import re
from typing import Any


def extract_structured_data(text: str, schema_hint: str = "") -> dict[str, Any]:
    """Extract structured JSON data from unstructured text."""
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "description": text[:500].strip(),
        "schema_hint": schema_hint,
        "confidence": 0.3,
        "requires_human_review": True,
    }
