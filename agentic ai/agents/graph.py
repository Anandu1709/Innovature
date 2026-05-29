"""
Graph Orchestrator — Connects all agents into a compiled LangGraph StateGraph.

Topology (upgraded with image input support):
  START → input_router
       → context_router (text-only flow — unchanged)
            → clarification → END                         (intent = clarify)
            → hybrid_retrieval → synthesis → END           (intent = text)
            → visual_retrieval → synthesis → END           (intent = visual)
            → hybrid_retrieval → visual_retrieval → synthesis → END  (intent = both)
       → vision_agent (image present)
            → image_clarification → END                    (image-only, no text)
            → general_response → END                       (domain = general)
            → image_context_enrichment → hybrid_retrieval  (domain = electronics)

  Loopback: synthesis → context_router (if verification fails, up to MAX_LOOPBACKS)

Usage:
  python agents/graph.py                    # Run interactive test
  python agents/graph.py --all              # Run all 5 test queries
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, END

from agents.state import AgentState, MAX_LOOPBACKS, create_initial_state
from agents.context_router import context_router
from agents.clarification import clarification_agent
from agents.hybrid_retrieval import hybrid_retrieval_agent
from agents.visual_retrieval import visual_retrieval_agent
from agents.synthesis import synthesis_agent

# New agents for image input support
from agents.input_router import input_router_node, route_from_input
from agents.vision_agent import vision_agent
from agents.image_clarification import image_clarification_agent
from agents.general_response import general_response_agent
from agents.image_context_enrichment import image_context_enrichment

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  Conditional Edge Functions
# =============================================================================

def route_by_intent(state: AgentState) -> str:
    """
    Route from context_router based on classified intent.

    Returns the name of the next node to execute.
    """
    intent = state.get("intent", "text")
    log.info(f"[GRAPH] Routing by intent: '{intent}'")

    if intent == "clarify":
        return "clarification"
    elif intent == "visual":
        return "visual_retrieval"
    elif intent == "both":
        return "hybrid_retrieval"  # → then visual_retrieval → synthesis
    else:  # "text" or fallback
        return "hybrid_retrieval"


def route_after_hybrid(state: AgentState) -> str:
    """
    Route from hybrid_retrieval:
      - If intent="both", proceed to visual_retrieval next.
      - Otherwise, go directly to synthesis.
    """
    intent = state.get("intent", "text")

    if intent == "both":
        log.info("[GRAPH] Intent is 'both' → routing to visual_retrieval")
        return "visual_retrieval"
    else:
        log.info("[GRAPH] Intent is 'text' → routing to synthesis")
        return "synthesis"


def route_after_synthesis(state: AgentState) -> str:
    """
    Route from synthesis based on verification result:
      - If passed (or max loopbacks reached): END
      - If failed and loopbacks remain: loop back to context_router
    """
    passed = state.get("verification_passed", True)
    loopback_count = state.get("loopback_count", 0)

    if passed:
        log.info("[GRAPH] Verification passed → END")
        return END
    elif loopback_count >= MAX_LOOPBACKS:
        log.warning(
            f"[GRAPH] Max loopbacks ({MAX_LOOPBACKS}) reached → END"
        )
        return END
    else:
        log.info(
            f"[GRAPH] Verification failed (loopback {loopback_count}/{MAX_LOOPBACKS}) "
            f"→ routing back to context_router"
        )
        return "context_router"


def route_after_vision(state: AgentState) -> str:
    """
    Route from vision_agent based on query presence and domain.

    Three possible destinations:
      - Image-only (no text query) → image_clarification
      - General domain → general_response (bypass RAG)
      - Electronics domain → image_context_enrichment (enter RAG)
    """
    query = state.get("query", "").strip()
    domain = state.get("domain", "electronics")

    if not query:
        log.info("[GRAPH] Image-only upload → image_clarification")
        return "image_clarification"
    if domain == "general":
        log.info("[GRAPH] General image → general_response (bypass RAG)")
        return "general_response"

    log.info("[GRAPH] Electronics image → image_context_enrichment → RAG")
    return "image_context_enrichment"


# =============================================================================
#  Graph Builder
# =============================================================================

def build_graph() -> StateGraph:
    """
    Construct and compile the full agent StateGraph.

    Node map:
      input_router              → routes by image presence (NEW)
      vision_agent              → Gemini vision analysis (NEW)
      image_clarification       → image-only clarification (NEW)
      general_response          → direct answer for non-electronics (NEW)
      image_context_enrichment  → rule-based query enrichment (NEW)
      context_router            → classifies intent (existing)
      clarification             → generates follow-up question (existing)
      hybrid_retrieval          → BM25 + dense + RRF text search (existing)
      visual_retrieval          → CLIP cross-modal image search (existing)
      synthesis                 → answer generation + self-verification (existing)
    """
    graph = StateGraph(AgentState)

    # --- Add nodes -----------------------------------------------------------
    # New image input nodes
    graph.add_node("input_router", input_router_node)
    graph.add_node("vision_agent", vision_agent)
    graph.add_node("image_clarification", image_clarification_agent)
    graph.add_node("general_response", general_response_agent)
    graph.add_node("image_context_enrichment", image_context_enrichment)

    # Existing nodes (unchanged)
    graph.add_node("context_router", context_router)
    graph.add_node("clarification", clarification_agent)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_agent)
    graph.add_node("visual_retrieval", visual_retrieval_agent)
    graph.add_node("synthesis", synthesis_agent)

    # --- Set entry point (changed from context_router to input_router) -------
    graph.set_entry_point("input_router")

    # --- New conditional edges for image routing -----------------------------

    # From input_router: route by image presence
    graph.add_conditional_edges(
        "input_router",
        route_from_input,
        {
            "vision_agent": "vision_agent",
            "context_router": "context_router",
        },
    )

    # From vision_agent: route by query presence and domain
    graph.add_conditional_edges(
        "vision_agent",
        route_after_vision,
        {
            "image_clarification": "image_clarification",
            "general_response": "general_response",
            "image_context_enrichment": "image_context_enrichment",
        },
    )

    # Terminal nodes for image flows
    graph.add_edge("image_clarification", END)
    graph.add_edge("general_response", END)

    # From enrichment: enter the existing retrieval pipeline
    graph.add_edge("image_context_enrichment", "hybrid_retrieval")

    # --- Existing conditional edges (unchanged) ------------------------------

    # From context_router: route by intent
    graph.add_conditional_edges(
        "context_router",
        route_by_intent,
        {
            "clarification": "clarification",
            "hybrid_retrieval": "hybrid_retrieval",
            "visual_retrieval": "visual_retrieval",
        },
    )

    # From clarification: terminal → END
    graph.add_edge("clarification", END)

    # From hybrid_retrieval: go to visual (if "both") or synthesis
    graph.add_conditional_edges(
        "hybrid_retrieval",
        route_after_hybrid,
        {
            "visual_retrieval": "visual_retrieval",
            "synthesis": "synthesis",
        },
    )

    # From visual_retrieval: always go to synthesis
    graph.add_edge("visual_retrieval", "synthesis")

    # From synthesis: check verification → END or loopback
    graph.add_conditional_edges(
        "synthesis",
        route_after_synthesis,
        {
            END: END,
            "context_router": "context_router",
        },
    )

    # --- Compile -------------------------------------------------------------
    compiled = graph.compile()
    log.info("[GRAPH] StateGraph compiled successfully")

    return compiled


# =============================================================================
#  Execution Helper
# =============================================================================

def run_query(
    query: str,
    session_id: str = None,
    session_history: list = None,
) -> dict:
    """
    Run a single query through the full agent graph.

    Args:
        query:           User's question.
        session_id:      Optional session identifier.
        session_history: Optional prior conversation turns.

    Returns:
        Final AgentState dict with all fields populated.
    """
    graph = build_graph()
    initial_state = create_initial_state(
        query=query,
        session_id=session_id,
        session_history=session_history,
    )

    log.info(f"[GRAPH] Starting query: '{query}'")
    final_state = graph.invoke(initial_state)
    log.info(f"[GRAPH] Query complete. Intent: {final_state.get('intent')}")

    return final_state


# =============================================================================
#  Pretty Printer
# =============================================================================

def print_result(state: dict) -> None:
    """Print the final state in a human-readable format."""
    encoding = sys.stdout.encoding or "utf-8"

    def safe(text):
        if isinstance(text, str):
            return text.encode(encoding, errors="replace").decode(encoding)
        return str(text)

    print("\n" + "=" * 70)
    print(f"  QUERY:    {safe(state.get('query', ''))}")
    print(f"  INTENT:   {safe(state.get('intent', ''))}")
    print(f"  LOOPBACKS: {state.get('loopback_count', 0)}")
    print("=" * 70)

    intent = state.get("intent", "")

    if intent == "clarify":
        print(f"\n  [CLARIFICATION QUESTION]")
        print(f"  {safe(state.get('clarification_question', 'N/A'))}")
    else:
        print(f"\n  [ANSWER]")
        print(f"  {safe(state.get('final_answer', 'No answer generated'))}")

        sources = state.get("sources", [])
        if sources:
            print(f"\n  [SOURCES]")
            for src in sources:
                print(f"    - {safe(src)}")

        images = state.get("image_results", [])
        if images:
            print(f"\n  [IMAGES RETRIEVED: {len(images)}]")
            for img in images:
                print(f"    - {safe(img.get('image_path', ''))} (score: {img.get('score', 0):.4f})")

        text_count = len(state.get("text_results", []))
        print(f"\n  [TEXT CHUNKS USED: {text_count}]")

        verification = state.get("verification_passed", False)
        print(f"  [VERIFICATION: {'PASSED' if verification else 'FAILED'}]")

    print("=" * 70 + "\n")


# ─── CLI Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Multimodal Electronics Assistant")
    parser.add_argument("--all", action="store_true", help="Run all 5 test queries")
    parser.add_argument("--query", type=str, help="Run a single custom query")
    args = parser.parse_args()

    if args.all:
        # Task 8: Full end-to-end test with 5 query types
        test_queries = [
            # 1. Text query
            "What are the key features of the ATmega328P microcontroller?",
            # 2. Visual query
            "Show me the Arduino Uno pinout diagram",
            # 3. Both (text + visual)
            "Explain SPI communication and show the wiring diagram",
            # 4. Clarification trigger
            "It doesn't work, help me fix it",
            # 5. Loopback test (deliberately narrow query)
            "What is the exact voltage regulator part number on Arduino Nano?",
        ]

        print("\n" + "#" * 70)
        print("  END-TO-END GRAPH TEST — 5 Query Types")
        print("#" * 70)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'-' * 70}")
            print(f"  TEST {i}/5")
            print(f"{'-' * 70}")

            state = run_query(query)
            print_result(state)

        # Print LLM provider stats
        from utils.llm_provider import get_stats
        stats = get_stats()
        print("\n" + "#" * 70)
        print("  LLM PROVIDER STATS")
        print("#" * 70)
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
        print("#" * 70 + "\n")

    elif args.query:
        state = run_query(args.query)
        print_result(state)

    else:
        # Interactive mode
        print("\n" + "=" * 70)
        print("  MULTIMODAL ELECTRONICS ASSISTANT — Interactive Mode")
        print("  Type 'quit' to exit")
        print("=" * 70)

        session_history = []
        session_id = None

        while True:
            try:
                query = input("\n  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if query.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break

            if not query:
                continue

            state = run_query(
                query=query,
                session_id=session_id,
                session_history=session_history,
            )

            # Update session tracking
            if session_id is None:
                session_id = state.get("session_id")

            print_result(state)

            # Append to session history for conversational continuity
            session_history.append({"role": "user", "content": query})
            if state.get("intent") == "clarify":
                session_history.append({
                    "role": "assistant",
                    "content": state.get("clarification_question", ""),
                })
            else:
                session_history.append({
                    "role": "assistant",
                    "content": state.get("final_answer", ""),
                })
