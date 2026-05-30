"""
Context Router Agent — Classifies user intent using Gemini.

Reads the user query (or refined_query on loopback) and session history,
then classifies intent as one of: "text", "visual", "both", "clarify".

This is the entry-point node of the StateGraph.

Usage (standalone test):
  python agents/context_router.py
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState
from utils.llm_provider import invoke_with_rate_limit

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONTEXT_ROUTER_MODEL = os.getenv("CONTEXT_ROUTER_MODEL", "gemini-2.5-flash-lite")

# --- Classification Prompt ---------------------------------------------------
ROUTER_SYSTEM_PROMPT = """\
You are an intent classifier for a Multimodal Electronics Assistant \
that helps users with Arduino and Raspberry Pi questions.

Your job is to classify the user's query into EXACTLY ONE of these intents:

1. "text"    — The user wants a factual, conceptual, or procedural answer.
               Examples: "What is the clock speed of ATmega328P?",
                         "How do I set up SPI communication?",
                         "Explain PWM on Arduino Nano."

2. "visual"  — The user explicitly wants a diagram, schematic, pinout image,
               layout, or wiring illustration, and does NOT need a long text explanation.
               Examples: "Show me the Arduino Uno pinout",
                         "Display the Raspberry Pi GPIO layout",
                         "I need the wiring diagram for an LED."

3. "both"    — The user wants BOTH a detailed text explanation AND a visual/diagram.
               Examples: "Explain SPI and show me the pin connections",
                         "How do I wire a servo motor? Include the diagram.",
                         "What are the GPIO pins? Show the pinout."

4. "clarify" — The query is too vague, ambiguous, or lacks enough context to
               retrieve meaningful results. The user needs to be asked a follow-up.
               Examples: "It doesn't work", "How do I fix it?",
                         "What should I use?", "Help me with the project."

RULES:
- Respond with ONLY the intent string: "text", "visual", "both", or "clarify".
- Do NOT include any explanation, punctuation, or extra words.
- If the user mentions "show", "pinout", "diagram", "schematic", "layout",
  "wiring", or "image" alongside a conceptual question, classify as "both".
- If the query references prior conversation context (e.g. "that LED",
  "the board I mentioned"), check session history. If session history
  provides enough context, classify normally. If not, classify as "clarify".
- When in doubt between "text" and "both", prefer "text".
- When in doubt between any intent and "clarify", prefer the non-clarify intent
  (only use "clarify" for truly meaningless queries).
"""


def _build_user_message(state: AgentState) -> str:
    """
    Build the user message for the classification prompt,
    incorporating session history and loopback context.
    """
    parts = []

    # Include session history for follow-up resolution
    history = state.get("session_history", [])
    if history:
        parts.append("=== CONVERSATION HISTORY ===")
        for turn in history[-5:]:  # Last 5 turns max to stay within context
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("=== END HISTORY ===\n")

    # Include loopback feedback if this is a re-routing cycle
    feedback = state.get("verification_feedback")
    if feedback:
        parts.append(f"[LOOPBACK CONTEXT] Previous answer was insufficient: {feedback}\n")

    # Use refined_query if available (from loopback), otherwise original query
    active_query = state.get("refined_query") or state["query"]
    parts.append(f"USER QUERY: {active_query}")

    return "\n".join(parts)


def context_router(state: AgentState) -> dict:
    """
    LangGraph node function: Classify user intent.

    Reads:  state["query"], state["refined_query"], state["session_history"],
            state["verification_feedback"]
    Writes: state["intent"]
    """
    active_query = state.get("refined_query") or state["query"]
    log.info(f"[ROUTER] Classifying intent for: '{active_query}'")

    user_message = _build_user_message(state)

    cleaned = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # Deterministic classification
        use_cache=True,   # Same query → same intent
        model_name=CONTEXT_ROUTER_MODEL,
    ).lower()

    valid_intents = {"text", "visual", "both", "clarify"}

    # Bulletproof keyword extraction: find any valid intent within the response text
    raw_intent = None
    for intent in valid_intents:
        if intent in cleaned:
            raw_intent = intent
            break

    if not raw_intent:
        log.warning(
            f"[ROUTER] Unexpected intent response '{cleaned}' from LLM. "
            f"Defaulting to 'text'."
        )
        raw_intent = "text"

    log.info(f"[ROUTER] Intent classified: '{raw_intent}'")

    return {"intent": raw_intent}


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    test_queries = [
        "What is the clock speed of ATmega328P?",
        "Show me the Arduino Uno pinout diagram",
        "How do I set up SPI and show the wiring?",
        "It doesn't work",
        "Explain PWM on Arduino Nano",
    ]

    expected = ["text", "visual", "both", "clarify", "text"]

    print("\n" + "=" * 60)
    print("  CONTEXT ROUTER — Standalone Test")
    print("=" * 60)

    for query, exp in zip(test_queries, expected):
        state = create_initial_state(query)
        result = context_router(state)
        intent = result["intent"]
        match = "✓" if intent == exp else "✗"
        print(f"  {match}  [{intent:7s}] (expected: {exp:7s}) — \"{query}\"")

    print("=" * 60 + "\n")
