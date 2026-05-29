"""
Input Router — Lightweight entry-point node for the LangGraph StateGraph.

Routes based purely on whether an uploaded image is present:
  - Image present  → vision_agent
  - No image       → context_router (existing text-only flow)

No LLM calls. No Gemini calls. Pure Python logic.

Usage:
  This module provides both the pass-through node function and the
  routing function used by add_conditional_edges().
"""

import logging

from agents.state import AgentState

# --- Config ------------------------------------------------------------------
log = logging.getLogger(__name__)


def input_router_node(state: AgentState) -> dict:
    """
    LangGraph node function: Pass-through node that logs routing decision.

    This node does NOT mutate state. It exists solely as the graph's
    entry point so that add_conditional_edges() can branch from it.
    """
    has_image = state.get("image_path") is not None
    log.info(
        f"[INPUT_ROUTER] Image present: {has_image} | "
        f"Query: '{state.get('query', '')[:80]}'"
    )
    return {"query": state.get("query")}


def route_from_input(state: AgentState) -> str:
    """
    Conditional edge function: Decide next node based on image presence.

    Returns:
        'vision_agent'   — if image_path is set
        'context_router' — otherwise (existing text-only flow)
    """
    if state.get("image_path") is not None:
        return "vision_agent"
    return "context_router"
