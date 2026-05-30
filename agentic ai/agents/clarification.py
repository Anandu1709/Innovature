"""
Clarification Agent — Generates a targeted follow-up question when the
user's query is too vague or ambiguous to retrieve meaningful results.

Triggered when the Context Router classifies intent as "clarify".
This is a terminal node — it writes the clarification question and
the graph routes to END so the frontend can display it.

Usage (standalone test):
  python agents/clarification.py
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

CLARIFICATION_MODEL = os.getenv("CLARIFICATION_MODEL", "gemini-2.5-flash-lite")

# --- Clarification Prompt ----------------------------------------------------
CLARIFICATION_SYSTEM_PROMPT = """\
You are a helpful Multimodal Electronics Assistant specializing in \
Arduino and Raspberry Pi projects.

The user has asked a vague or ambiguous question that lacks enough context \
to provide an accurate answer. Your job is to generate a single, polite, \
specific follow-up question that will help clarify exactly what the user needs.

RULES:
- Ask ONLY ONE follow-up question, not multiple.
- Be specific — offer concrete choices when possible \
  (e.g., board names, protocols, components).
- Keep the tone friendly, helpful, and concise.
- Do NOT attempt to answer the original query.
- Do NOT say "I'm sorry" or apologize — just ask the clarifying question directly.
- If session history provides partial context, acknowledge what you \
  already know and ask for the missing piece.

GOOD EXAMPLES:
- "Which board are you working with — Arduino Uno, Nano, or Raspberry Pi?"
- "Are you trying to read data from the sensor (input) or control a device (output)?"
- "Could you specify which communication protocol you need help with — SPI, I2C, or UART?"

BAD EXAMPLES:
- "Can you please provide more details?" (too generic)
- "What do you mean?" (unhelpful)
- "I'm sorry, I didn't understand. Could you rephrase?" (apologetic, vague)
"""


def _build_user_message(state: AgentState) -> str:
    """Build the user message with session context for clarification."""
    parts = []

    # Include session history for partial context
    history = state.get("session_history", [])
    if history:
        parts.append("=== CONVERSATION HISTORY ===")
        for turn in history[-5:]:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("=== END HISTORY ===\n")

    query = state["query"]
    parts.append(f"VAGUE USER QUERY: {query}")
    parts.append("\nGenerate a single, specific follow-up question:")

    return "\n".join(parts)


def clarification_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Generate a clarifying follow-up question.

    Reads:  state["query"], state["session_history"]
    Writes: state["clarification_question"]
    """
    log.info(f"[CLARIFY] Generating follow-up for: '{state['query']}'")

    user_message = _build_user_message(state)

    question = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # Slight creativity for natural-sounding questions
        use_cache=False,  # Clarifications should be fresh each time
        model_name=CLARIFICATION_MODEL,
    )

    log.info(f"[CLARIFY] Generated: '{question}'")

    return {"clarification_question": question}


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    test_queries = [
        "It doesn't work",
        "How do I fix it?",
        "What should I use?",
        "Help me with the project",
        "Can you show me the thing?",
    ]

    print("\n" + "=" * 60)
    print("  CLARIFICATION AGENT — Standalone Test")
    print("=" * 60)

    for query in test_queries:
        state = create_initial_state(query)
        result = clarification_agent(state)
        print(f"\n  Query:    \"{query}\"")
        print(f"  Follow-up: {result['clarification_question']}")

    # Test with session history context
    print("\n" + "-" * 60)
    print("  Test with session history:")
    state = create_initial_state(
        query="What resistor should I use for that?",
        session_history=[
            {"role": "user", "content": "How do I wire an LED to Arduino Uno?"},
            {"role": "assistant", "content": "Connect the LED anode to digital pin 13 through a current-limiting resistor, and the cathode to GND."},
        ],
    )
    result = clarification_agent(state)
    print(f"\n  Query:    \"{state['query']}\"")
    print(f"  History:   2 prior turns about LED wiring")
    print(f"  Follow-up: {result['clarification_question']}")

    print("\n" + "=" * 60 + "\n")
