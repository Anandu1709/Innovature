import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse first JSON object from model output (handles optional markdown fences)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:500]!r}")
    return json.loads(t[start : end + 1])
