"""
Query Rewriter Agent — Resolves contextual follow-up queries into
self-contained search strings using session history.

Placed between context_router and retrieval agents in the graph.
Only fires when the query contains contextual language (pronouns,
references) AND session history is present. Otherwise passes through
with zero LLM calls.

Model: gemini-2.5-flash-lite (fast, cheap — this is a simple rewrite task).

Usage (standalone test):
  python agents/query_rewriter.py
"""

import os
import re
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

REWRITER_MODEL = os.getenv("QUERY_REWRITER_MODEL", "gemini-2.5-flash-lite")

# --- Contextual Trigger Detection -------------------------------------------

# Single-word triggers (matched as whole words, case-insensitive)
_TRIGGER_WORDS = {
    "it", "they", "them", "that", "this", "these", "those",
    "its", "their", "above", "previous", "earlier", "same",
    "mentioned", "one",
}

# Multi-word triggers (matched as substrings, case-insensitive)
_TRIGGER_PHRASES = [
    "the board", "the pin", "the pins", "the diagram",
    "the image", "the sensor", "the module", "the chip",
    "the controller", "the component", "the circuit",
    "show me more", "tell me more", "explain more",
]

# Compile a regex that matches any trigger word as a whole word
_TRIGGER_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _TRIGGER_WORDS) + r")\b",
    re.IGNORECASE,
)


def _needs_rewrite(query: str, session_history: list) -> bool:
    """
    Determine if a query needs contextual rewriting.

    Returns True only when BOTH conditions are met:
      1. There is at least one prior turn in session_history.
      2. The query contains a contextual trigger word or phrase.

    This is a pure Python check — zero LLM calls.
    """
    if not session_history:
        return False

    query_lower = query.lower()

    # Check multi-word phrases first
    for phrase in _TRIGGER_PHRASES:
        if phrase in query_lower:
            return True

    # Check single-word triggers
    if _TRIGGER_WORD_PATTERN.search(query):
        return True

    return False


# --- Rewrite Prompt ----------------------------------------------------------

REWRITER_SYSTEM_PROMPT = """\
You are a query rewriter for a Multimodal Electronics Assistant \
that helps users with Arduino and Raspberry Pi questions.

Your ONLY job is to rewrite a follow-up query into a self-contained, \
searchable query by resolving pronouns and contextual references \
using the conversation history.

RULES:
- Output ONLY the rewritten query. No explanations, no quotes, no prefix.
- Preserve the user's original intent exactly.
- Replace pronouns (it, they, them, that, this, those, etc.) with the \
  specific entities from conversation history.
- Include the board/component name explicitly in the rewritten query.
- Keep the rewritten query concise — it will be used as a search string.
- If the query is already self-contained, return it unchanged.
- Do NOT answer the question. Only rewrite it.

EXAMPLES:

History: User asked about SPI pins on Arduino Uno.
Follow-up: "What are they used for?"
Rewritten: "What are the SPI communication pins used for on Arduino Uno?"

History: User asked about GPIO pinout for Raspberry Pi.
Follow-up: "Which of those support I2C?"
Rewritten: "Which Raspberry Pi GPIO pins support I2C?"

History: User asked about wiring an LED to Arduino pin 13.
Follow-up: "What resistor value should I use for that?"
Rewritten: "What resistor value should I use for an LED connected to Arduino pin 13?"

History: User asked about the Arduino Nano board layout.
Follow-up: "Show me the diagram"
Rewritten: "Show me the Arduino Nano board layout diagram"
"""


def _build_rewrite_prompt(query: str, session_history: list) -> str:
    """Build the user message for the rewrite LLM call."""
    parts = []

    parts.append("=== CONVERSATION HISTORY ===")
    for turn in session_history[-5:]:  # Last 5 turns max
        role = turn.get("role", "user").upper()
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("=== END HISTORY ===\n")

    parts.append(f"FOLLOW-UP QUERY: {query}")
    parts.append("\nRewrite the follow-up query into a self-contained search query:")

    return "\n".join(parts)


# --- LangGraph Node ----------------------------------------------------------

def query_rewriter(state: AgentState) -> dict:
    """
    LangGraph node function: Conditionally rewrite follow-up queries.

    If the query contains contextual language and session history exists,
    calls gemini-2.5-flash-lite to resolve pronouns and references.
    Otherwise passes through with zero LLM calls.

    Reads:  state["query"], state["session_history"]
    Writes: state["refined_query"] (only when rewrite occurs)
    """
    query = state["query"]
    history = state.get("session_history", [])

    # Skip if already refined (e.g. from loopback, if re-enabled in future)
    if state.get("refined_query"):
        log.info(f"[REWRITER] Skipping — refined_query already set")
        return {"refined_query": state.get("refined_query")}

    # Pure Python check — no LLM call
    if not _needs_rewrite(query, history):
        log.info(f"[REWRITER] No rewrite needed for: '{query[:80]}'")
        return {"refined_query": state.get("refined_query")}

    log.info(f"[REWRITER] Rewriting follow-up: '{query}'")

    user_message = _build_rewrite_prompt(query, history)

    rewritten = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # Deterministic rewriting
        use_cache=False,   # Each rewrite depends on unique session history
        model_name=REWRITER_MODEL,
    ).strip().strip('"').strip("'")  # Clean any wrapping quotes

    log.info(f"[REWRITER] '{query}' → '{rewritten}'")

    return {"refined_query": rewritten}


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    print("\n" + "=" * 60)
    print("  QUERY REWRITER — Standalone Test")
    print("=" * 60)

    # Test 1: Follow-up with pronoun
    print("\n--- Test 1: Follow-up with pronoun ---")
    state = create_initial_state(
        query="What are they used for?",
        session_history=[
            {"role": "user", "content": "What are the SPI communication pins for Arduino Uno?"},
            {"role": "assistant", "content": "The SPI pins on Arduino Uno are: MOSI (pin 11), MISO (pin 12), SCK (pin 13), and SS (pin 10)."},
        ],
    )
    result = query_rewriter(state)
    print(f"  Original:  '{state['query']}'")
    print(f"  Rewritten: '{result.get('refined_query', '(no rewrite)')}'")

    # Test 2: Follow-up with "the board"
    print("\n--- Test 2: Follow-up with 'the board' ---")
    state = create_initial_state(
        query="Which pins support I2C on the board?",
        session_history=[
            {"role": "user", "content": "Show me the GPIO pinout for Raspberry Pi"},
            {"role": "assistant", "content": "The Raspberry Pi has a 40-pin GPIO header..."},
        ],
    )
    result = query_rewriter(state)
    print(f"  Original:  '{state['query']}'")
    print(f"  Rewritten: '{result.get('refined_query', '(no rewrite)')}'")

    # Test 3: Standalone query (should NOT rewrite)
    print("\n--- Test 3: Standalone query (no rewrite) ---")
    state = create_initial_state(
        query="What is SPI communication?",
        session_history=[
            {"role": "user", "content": "What are the SPI pins for Arduino?"},
            {"role": "assistant", "content": "MOSI, MISO, SCK, SS..."},
        ],
    )
    result = query_rewriter(state)
    print(f"  Original:  '{state['query']}'")
    print(f"  Rewritten: '{result.get('refined_query', '(no rewrite — pass through)')}'")

    # Test 4: No session history (should NOT rewrite)
    print("\n--- Test 4: No session history (no rewrite) ---")
    state = create_initial_state(query="What are they used for?")
    result = query_rewriter(state)
    print(f"  Original:  '{state['query']}'")
    print(f"  Rewritten: '{result.get('refined_query', '(no rewrite — no history)')}'")

    print("\n" + "=" * 60 + "\n")
