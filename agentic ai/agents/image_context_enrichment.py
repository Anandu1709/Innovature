"""
Image Context Enrichment — Rule-based query enrichment for RAG retrieval.

Converts the Vision Agent's structured analysis into a text-enriched query
that the existing Hybrid Retrieval Agent can search with.

This is NOT an LLM agent. No Gemini calls. Pure Python string concatenation.

The enriched query overwrites state["query"] so the existing retrieval
agents pick it up automatically (hybrid_retrieval reads state["refined_query"]
or state["query"]).
"""

import logging

from agents.state import AgentState

log = logging.getLogger(__name__)


def image_context_enrichment(state: AgentState) -> dict:
    """
    LangGraph node function: Build enriched retrieval query from image analysis.

    Reads:  state["query"], state["image_summary"], state["components"],
            state["observations"], state["possible_issues"], state["intent"]
    Writes: state["query"], state["intent"]
    """
    query = state.get("query", "")
    summary = state.get("image_summary", "")
    components = state.get("components", [])
    observations = state.get("observations", [])
    issues = state.get("possible_issues", [])

    parts = []

    if query:
        parts.append(f"User Question: {query}")
    if summary:
        parts.append(f"Image Summary: {summary}")
    if components:
        parts.append(f"Components: {', '.join(components)}")
    if observations:
        parts.append(f"Observations: {', '.join(observations)}")
    if issues:
        parts.append(f"Possible Issues: {', '.join(issues)}")

    enriched = "\n".join(parts)

    log.info(
        f"[ENRICHMENT] Enriched query built — "
        f"{len(parts)} sections, {len(enriched)} chars"
    )

    # Use the vision agent's intent, default to "both" for electronics images
    intent = state.get("intent", "both")

    return {
        "query": enriched,
        "intent": intent,
    }
