"""
Synthesis & Verification Agent — Generates the final answer and self-verifies.

Two-phase pipeline:
  Phase 1 (Synthesis):  Combines retrieved text chunks and images into a
                        comprehensive, cited answer using Gemini.
  Phase 2 (Verification): A second Gemini call acts as a critic, evaluating
                          completeness, grounding, and hallucination risk.
                          If verification fails, triggers a loopback signal.

Usage (standalone test):
  python agents/synthesis.py
"""

import sys
import json
import re
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import AgentState, MAX_LOOPBACKS
from utils.llm_provider import invoke_with_rate_limit

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  Prompts
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """\
You are a Multimodal Electronics Assistant specializing in Arduino and \
Raspberry Pi. Your job is to synthesize a comprehensive, accurate answer \
using ONLY the retrieved context provided below.

RULES:
- Answer the user's question thoroughly using the provided text chunks.
- Cite source documents inline using [Source: filename] format.
- If images are provided, reference them naturally in your answer \
  (e.g., "As shown in the diagram below...").
- Structure your answer with clear headings, bullet points, or numbered \
  steps where appropriate.
- If the retrieved context does not contain enough information to fully \
  answer the question, clearly state what is missing rather than guessing.
- Do NOT hallucinate information that is not in the provided context.
- Do NOT include disclaimers like "based on the provided context" — \
  just answer directly and confidently.
- Keep the answer focused and practical for electronics engineers/makers.
"""

VERIFICATION_SYSTEM_PROMPT = """\
You are a quality verification agent. Your job is to evaluate whether \
the generated answer adequately addresses the user's original question.

Evaluate the answer on these criteria:
1. COMPLETENESS: Does the answer address all aspects of the user's query?
2. GROUNDING: Is the answer grounded in the provided source documents? \
   Are citations present?
3. ACCURACY: Are there any obvious hallucinations or fabricated details \
   not supported by the retrieved context?
4. ACTIONABILITY: For how-to questions, does the answer provide clear, \
   followable steps?

Respond with a JSON object (and NOTHING else) in this exact format:
{
  "passed": true/false,
  "reason": "Brief explanation of why it passed or failed",
  "missing_info": "What specific information is missing (empty string if passed)",
  "refined_query": "A better search query to find the missing info (empty string if passed)"
}
"""



# =============================================================================
#  Phase 1: Answer Synthesis
# =============================================================================

def _build_synthesis_prompt(state: AgentState) -> str:
    """Build the user message for synthesis, including all retrieved context."""
    parts = []

    query = state.get("refined_query") or state["query"]
    parts.append(f"USER QUESTION: {query}\n")

    # Include session history for conversational context
    history = state.get("session_history", [])
    if history:
        parts.append("=== CONVERSATION HISTORY ===")
        for turn in history[-3:]:  # Last 3 turns for synthesis context
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("=== END HISTORY ===\n")

    # Include verification feedback if this is a loopback cycle
    feedback = state.get("verification_feedback")
    if feedback:
        parts.append(
            f"[LOOPBACK INSTRUCTION] The previous answer was insufficient. "
            f"Reason: {feedback}. Please address this gap in your new answer.\n"
        )

    # Include retrieved text chunks
    text_results = state.get("text_results", [])
    if text_results:
        parts.append("=== RETRIEVED TEXT CHUNKS ===")
        for i, chunk in enumerate(text_results, 1):
            source = chunk.get("source_document", "unknown")
            section = chunk.get("section_name", "")
            text = chunk.get("text", "")
            score = chunk.get("fused_score", chunk.get("score", 0))
            parts.append(
                f"\n--- Chunk {i} [Source: {source}] "
                f"[Section: {section}] [Score: {score:.4f}] ---\n{text}"
            )
        parts.append("\n=== END TEXT CHUNKS ===\n")
    else:
        parts.append("[NO TEXT CHUNKS RETRIEVED]\n")

    # Include retrieved images
    image_results = state.get("image_results", [])
    if image_results:
        parts.append("=== RETRIEVED IMAGES ===")
        for i, img in enumerate(image_results, 1):
            parts.append(
                f"\nImage {i}:"
                f"\n  Path: {img.get('image_path', '')}"
                f"\n  Caption: {img.get('caption', 'No caption')}"
                f"\n  Source: {img.get('source_document', 'unknown')}"
                f"\n  Score: {img.get('score', 0):.4f}"
            )
        parts.append("\n=== END IMAGES ===\n")

    parts.append("Please provide a comprehensive answer to the user's question.")

    return "\n".join(parts)


def _synthesize_answer(state: AgentState) -> str:
    """Generate the answer using Gemini with all retrieved context."""
    prompt = _build_synthesis_prompt(state)

    return invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,   # Slightly creative for natural answers
        use_cache=False,   # Answers should always be freshly generated
    )


# =============================================================================
#  Phase 2: Self-Verification
# =============================================================================

def _build_verification_prompt(state: AgentState, answer: str) -> str:
    """Build the verification prompt with the original query and generated answer."""
    query = state.get("refined_query") or state["query"]

    # Summarise what context was available
    text_count = len(state.get("text_results", []))
    image_count = len(state.get("image_results", []))
    sources = state.get("sources", [])

    return (
        f"ORIGINAL USER QUESTION: {query}\n\n"
        f"CONTEXT AVAILABLE: {text_count} text chunks, {image_count} images\n"
        f"SOURCES USED: {', '.join(sources) if sources else 'none'}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"Evaluate this answer and respond with the JSON object."
    )


def _verify_answer(state: AgentState, answer: str) -> dict:
    """
    Run self-verification on the generated answer.

    Returns dict with keys: passed, reason, missing_info, refined_query
    """
    prompt = _build_verification_prompt(state, answer)

    raw = invoke_with_rate_limit(
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,   # Deterministic verification
        use_cache=False,   # Verification must be fresh
    )

    # Bulletproof JSON extraction from anywhere in the LLM response
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object block found in response")
        
        json_str = match.group(0)
        result = json.loads(json_str)
        
        return {
            "passed": bool(result.get("passed", True)),
            "reason": str(result.get("reason", "")),
            "missing_info": str(result.get("missing_info", "")),
            "refined_query": str(result.get("refined_query", "")),
        }
    except Exception as e:
        log.warning(f"[SYNTH] Verification JSON parse failed: {e}. Raw response: {raw}")
        return {
            "passed": True,
            "reason": f"Verification parse failed ({type(e).__name__}) — defaulting to pass",
            "missing_info": "",
            "refined_query": "",
        }


# =============================================================================
#  LangGraph Node
# =============================================================================

def synthesis_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Synthesize answer and verify quality.

    Reads:  state["query"], state["refined_query"], state["text_results"],
            state["image_results"], state["session_history"],
            state["verification_feedback"], state["loopback_count"],
            state["sources"]
    Writes: state["final_answer"], state["verification_passed"],
            state["verification_feedback"], state["refined_query"],
            state["loopback_count"], state["sources"]
    """
    active_query = state.get("refined_query") or state["query"]
    current_loopback = state.get("loopback_count", 0)

    log.info(
        f"[SYNTH] Synthesizing answer for: '{active_query}' "
        f"(loopback {current_loopback}/{MAX_LOOPBACKS})"
    )

    # Phase 1: Generate answer
    try:
        answer = _synthesize_answer(state)
    except Exception as e:
        log.exception(f"[SYNTH] Answer generation failed: {e}")
        return {
            "final_answer": "I'm sorry, I encountered an error generating the answer. Please try again.",
            "verification_passed": True,  # Don't loopback on LLM errors
        }

    log.info(f"[SYNTH] Answer generated ({len(answer)} chars)")

    # Extract source documents from text_results for citation tracking
    result_sources = set()
    for r in state.get("text_results", []):
        src = r.get("source_document", "")
        if src:
            result_sources.add(src)
    for r in state.get("image_results", []):
        src = r.get("source_document", "")
        if src:
            result_sources.add(src)
    all_sources = list(result_sources)

    # Phase 2: Self-verification
    try:
        verification = _verify_answer(state, answer)
    except Exception as e:
        log.exception(f"[SYNTH] Verification failed: {e}")
        verification = {"passed": True, "reason": "Verification error", "missing_info": "", "refined_query": ""}

    if verification["passed"]:
        log.info(f"[SYNTH] Verification PASSED: {verification['reason']}")
        return {
            "final_answer": answer,
            "sources": all_sources,
            "verification_passed": True,
            "verification_feedback": None,
        }
    else:
        log.warning(f"[SYNTH] Verification FAILED: {verification['reason']}")
        log.warning(f"[SYNTH] Missing: {verification['missing_info']}")

        new_loopback = current_loopback + 1

        if new_loopback >= MAX_LOOPBACKS:
            log.warning(
                f"[SYNTH] Max loopbacks ({MAX_LOOPBACKS}) reached. "
                f"Returning best available answer."
            )
            return {
                "final_answer": answer,
                "sources": all_sources,
                "verification_passed": True,  # Force exit
                "verification_feedback": None,
                "loopback_count": new_loopback,
            }

        return {
            "final_answer": answer,
            "sources": all_sources,
            "verification_passed": False,
            "verification_feedback": verification["reason"],
            "refined_query": verification["refined_query"] or active_query,
            "loopback_count": new_loopback,
        }


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    # Simulate a state with pre-populated retrieval results
    state = create_initial_state("What are the key features of ATmega328P?")
    state["text_results"] = [
        {
            "chunk_id": "a000066_datasheet_general_c003",
            "text": "Features ATMega328P Processor Memory AVR CPU at up to 16 MHz "
                    "32 kB Flash 2 kB SRAM 1 kB EEPROM Security UPDI, debugWIRE",
            "source_document": "A000066-datasheet.pdf",
            "section_name": "General",
            "fused_score": 0.032258,
        },
        {
            "chunk_id": "a000005_datasheet_general_c003",
            "text": "Features ATmega328 Microcontroller High-performance low-power "
                    "8-bit processor Achieve up to 16 MIPS throughput at 16 MHz",
            "source_document": "A000005-datasheet.pdf",
            "section_name": "General",
            "fused_score": 0.016393,
        },
    ]
    state["image_results"] = [
        {
            "image_id": "a000066_datasheet_p007_img000",
            "image_path": "data/images/arduino/a000066_datasheet_p007_img000.png",
            "caption": "Block Diagram of Arduino UNO R3",
            "source_document": "A000066-datasheet.pdf",
            "score": 0.72,
        },
    ]
    state["sources"] = ["A000066-datasheet.pdf", "A000005-datasheet.pdf"]

    print("\n" + "=" * 60)
    print("  SYNTHESIS AGENT — Standalone Test")
    print("=" * 60)

    result = synthesis_agent(state)

    print(f"\n  Verification: {'PASSED' if result['verification_passed'] else 'FAILED'}")
    if result.get("verification_feedback"):
        print(f"  Feedback: {result['verification_feedback']}")
    print(f"  Sources: {result.get('sources', [])}")
    print(f"\n  --- ANSWER ---")
    print(f"  {result['final_answer']}")
    print(f"  --- END ---")
    print("=" * 60 + "\n")
