"""
Image Clarification Agent — Generates a structured clarification response
when a user uploads an image without any text query.

This is a pure Python function — no LLM call, no Gemini call.
Uses the Vision Agent's structured output to compose the clarification.

Triggered when: image present AND query is empty.
Terminal node → graph routes to END.
"""

import logging

from agents.state import AgentState

log = logging.getLogger(__name__)


def image_clarification_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Generate a clarification response for image-only uploads.

    Reads:  state["image_summary"], state["components"]
    Writes: state["intent"], state["clarification_question"]
    """
    summary = state.get("image_summary", "an uploaded image")
    components = state.get("components", [])

    log.info(f"[IMG_CLARIFY] Image-only upload — generating clarification")

    comp_str = ", ".join(components) if components else "some components"

    question = (
        f"I can see **{summary}** with {comp_str}.\n\n"
        "What would you like me to help with?\n\n"
        "• **Explain the circuit** — How does this setup work?\n"
        "• **Identify components** — What parts are used here?\n"
        "• **Troubleshoot a problem** — Why isn't it working?\n"
        "• **Check wiring** — Are the connections correct?"
    )

    log.info(f"[IMG_CLARIFY] Clarification generated ({len(question)} chars)")

    return {
        "intent": "clarify",
        "clarification_question": question,
    }
