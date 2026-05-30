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
    LangGraph node function: Answer general/out-of-scope text queries and non-electronics image queries.

    Reads:  state["query"], state["image_analysis"], state["image_summary"],
            state["observations"], state["global_search_requested"]
    Writes: state["intent"], state["final_answer"],
            state["verification_passed"], state["coverage_found"],
            state["offer_global_search"], state["global_search_requested"]
    """
    query = state.get("query", "")
    global_search_requested = state.get("global_search_requested", False)

    # --- Case 1: Out-of-Context Text Web Fallback Search requested by User -----
    if global_search_requested:
        log.info(f"[GENERAL] Generating general web fallback response for query: '{query[:80]}'")
        system_prompt = (
            "You are a helpful assistant. Provide a concise answer using web search "
            "when the knowledge base cannot answer.\n\n"
            "Rules:\n"
            "1. Maximum 5 lines.\n"
            "2. Answer directly.\n"
            "3. No long explanations.\n"
            "4. No markdown headings.\n"
            "5. No unnecessary background information.\n"
            "6. Prioritize the most useful facts."
        )
        prompt = f"Question:\n{query}"

        try:
            answer = invoke_with_rate_limit(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                use_cache=False,
                model_name=GENERAL_RESPONSE_MODEL,
            )
        except Exception as e:
            log.exception(f"[GENERAL] Fallback answer generation failed: {e}")
            answer = "I'm sorry, I could not retrieve information for this query. Please try again."

        log.info(f"[GENERAL] Fallback response generated ({len(answer)} chars)")

        return {
            "intent": "text",
            "final_answer": answer,
            "verification_passed": True,
            "coverage_found": True,
            "offer_global_search": False,
            "global_search_requested": True,
        }

    # --- Case 2: General Uploaded Image (VS Code screenshots, etc) -----------
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
        "coverage_found": True,
        "offer_global_search": False,
        "global_search_requested": False,
    }
