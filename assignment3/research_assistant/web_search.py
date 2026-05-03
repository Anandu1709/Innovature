"""Lightweight web context via DuckDuckGo.

Works with either of the two upstream package names:
- `ddgs` (current, since duckduckgo-search 8.x was renamed)
- `duckduckgo_search` (legacy <= 7.x)
"""

from typing import List


def _import_ddgs():
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore

        return DDGS


def duckduckgo_snippets(query: str, max_results: int = 5) -> str:
    DDGS = _import_ddgs()

    snippets: List[str] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            title = item.get("title") or ""
            body = item.get("body") or ""
            if title or body:
                snippets.append(f"- {title}: {body}")
    if not snippets:
        return "No web results returned."
    return "\n".join(snippets)
