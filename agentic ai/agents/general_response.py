"""
General Response Agent — Direct Gemini answer for non-electronics images.

When the Vision Agent classifies an image as domain="general" (e.g. VS Code
screenshot, browser error, application UI), this agent bypasses the entire
RAG pipeline (no Milvus, no BM25) and answers directly using the image
analysis context + Gemini.

Uses the existing LangChain LLM provider for consistency with other agents.
"""

import os
import logging

from agents.state import AgentState
from utils.llm_provider import invoke_with_rate_limit

log = logging.getLogger(__name__)

GENERAL_RESPONSE_MODEL = os.getenv("GENERAL_RESPONSE_MODEL", "gemini-2.5-flash-lite")


def general_response_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Answer non-electronics image queries directly.

    Reads:  state["query"], state["image_analysis"], state["image_summary"],
            state["observations"]
    Writes: state["intent"], state["final_answer"],
            state["verification_passed"]
    """
    query = state.get("query", "")
    analysis = state.get("image_analysis", {})
    summary = analysis.get("summary", state.get("image_summary", ""))
    observations = analysis.get("observations", state.get("observations", []))

    log.info(f"[GENERAL] Generating direct response for general image")

    # Build context from vision analysis
    context_parts = []
    if summary:
        context_parts.append(f"Image Summary: {summary}")
    if observations:
        context_parts.append(f"Observations: {', '.join(observations)}")
    context_str = "\n".join(context_parts) if context_parts else "No image details available."

    prompt = (
        f"The user uploaded an image (not electronics-related).\n\n"
        f"{context_str}\n\n"
        f"User question: {query}\n\n"
        f"Provide a helpful, direct answer based on what was observed in the image."
    )

    try:
        answer = invoke_with_rate_limit(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely based on the image analysis provided."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            use_cache=False,
            model_name=GENERAL_RESPONSE_MODEL,
        )
    except Exception as e:
        log.exception(f"[GENERAL] Answer generation failed: {e}")
        answer = "I'm sorry, I encountered an error analyzing this image. Please try again."

    log.info(f"[GENERAL] Response generated ({len(answer)} chars)")

    return {
        "intent": "text",
        "final_answer": answer,
        "verification_passed": True,  # Skip loopback for general images
    }
