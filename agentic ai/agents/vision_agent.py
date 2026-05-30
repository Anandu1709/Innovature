"""
Vision Agent — Gemini-powered image analysis for uploaded images.

Uses the native Google Generative AI SDK (not LangChain) to send
images directly to Gemini 2.5 Flash for structured analysis.

Pipeline:
  1. Load image from state["image_path"] using PIL
  2. Send image + structured prompt to Gemini
  3. Parse JSON response → populate state fields
  4. Classify domain (electronics vs general) and intent

Output schema:
  {
    "domain": "electronics" | "general",
    "intent": "text" | "visual" | "both" | "clarify",
    "summary": "one-line description",
    "components": ["list", "of", "components"],
    "observations": ["what you see"],
    "possible_issues": ["potential problems"]
  }

Usage (standalone test):
  python agents/vision_agent.py <image_path>
"""

import os
import sys
import json
import re
import logging
from pathlib import Path

from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from agents.state import AgentState

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Use the same model configured in .env
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")

# --- Singleton Gemini client -------------------------------------------------
_genai_model = None


def _get_genai_model():
    """Lazy-load the native Gemini GenerativeModel for vision tasks."""
    global _genai_model
    if _genai_model is None:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")

        genai.configure(api_key=api_key)
        _genai_model = genai.GenerativeModel(VISION_MODEL)
        log.info(f"[VISION] Gemini model loaded: {VISION_MODEL}")

    return _genai_model


# --- Prompt ------------------------------------------------------------------
VISION_SYSTEM_PROMPT = """\
Analyze this image carefully. Return ONLY valid JSON with this exact schema:

{
  "domain": "electronics" or "general",
  "intent": "text" or "visual" or "both" or "clarify",
  "summary": "one-line description of what the image shows",
  "components": ["list", "of", "identified", "components"],
  "observations": ["key visual observations"],
  "possible_issues": ["potential problems or concerns"]
}

RULES:
- If the image shows electronic circuits, Arduino, Raspberry Pi, breadboards, \
  PCBs, wiring, components, schematics, or datasheets → domain = "electronics"
- If the image shows code editors, browser windows, screenshots, text documents, \
  photos of non-electronic items → domain = "general"
- For electronics images: identify specific components (e.g. "Arduino Uno", \
  "220 ohm resistor", "red LED"), note wiring connections, flag potential issues.
- For general images: provide a brief summary, leave components empty, \
  list key observations.
- intent should reflect what kind of answer would best help the user:
  "text" = factual explanation needed
  "visual" = diagram or pinout would help
  "both" = text explanation + visual reference needed
  "clarify" = image is unclear or ambiguous
- Do NOT wrap the JSON in markdown code fences.
- Do NOT include any text outside the JSON object.
"""


# --- Core Analysis -----------------------------------------------------------

def _analyze_image(image_path: str, query: str = "") -> dict:
    """
    Send image to Gemini for structured analysis.

    Args:
        image_path: Path to the image file on disk.
        query: Optional user question about the image.

    Returns:
        Parsed dict matching the vision output schema.
    """
    model = _get_genai_model()

    img = Image.open(image_path)
    log.info(f"[VISION] Image loaded: {image_path} ({img.size[0]}x{img.size[1]})")

    # Build content: image + prompt + optional user query
    content = [img, VISION_SYSTEM_PROMPT]
    if query:
        content.append(f"\nUser question about this image: {query}")

    response = model.generate_content(content)
    raw_text = response.text.strip()

    log.info(f"[VISION] Raw response length: {len(raw_text)} chars")

    return _parse_vision_response(raw_text)


def _parse_vision_response(raw_text: str) -> dict:
    """
    Parse the JSON response from Gemini, with fallback extraction.

    Handles:
    - Clean JSON response
    - JSON wrapped in markdown code fences
    - JSON embedded in free text
    """
    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object from response
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                log.warning(f"[VISION] JSON parse failed. Raw: {raw_text[:300]}")
                result = {}
        else:
            log.warning(f"[VISION] No JSON found in response. Raw: {raw_text[:300]}")
            result = {}

    # Validate and fill defaults
    defaults = {
        "domain": "general",
        "intent": "both",
        "summary": "Image uploaded",
        "components": [],
        "observations": [],
        "possible_issues": [],
    }

    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    # Validate domain
    if result["domain"] not in ("electronics", "general"):
        result["domain"] = "general"

    # Validate intent
    if result["intent"] not in ("text", "visual", "both", "clarify"):
        result["intent"] = "both"

    return result


# --- LangGraph Node ----------------------------------------------------------

def vision_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Analyze uploaded image with Gemini Vision.

    Reads:  state["image_path"], state["query"]
    Writes: state["image_analysis"], state["image_summary"],
            state["components"], state["observations"],
            state["possible_issues"], state["domain"], state["intent"]
    """
    image_path = state.get("image_path")
    query = state.get("query", "")

    if not image_path:
        log.warning("[VISION] No image_path in state — skipping vision analysis")
        return {}

    log.info(f"[VISION] Analyzing image: {image_path}")

    try:
        analysis = _analyze_image(image_path, query)
    except Exception as e:
        log.exception(f"[VISION] Image analysis failed: {e}")
        # Return safe defaults so the graph can continue
        analysis = {
            "domain": "general",
            "intent": "both",
            "summary": "Image analysis failed",
            "components": [],
            "observations": [],
            "possible_issues": [],
        }

    log.info(
        f"[VISION] Analysis complete — "
        f"domain={analysis['domain']}, intent={analysis['intent']}, "
        f"components={len(analysis['components'])}, "
        f"summary='{analysis['summary'][:60]}'"
    )

    return {
        "image_analysis": analysis,
        "image_summary": analysis["summary"],
        "components": analysis["components"],
        "observations": analysis["observations"],
        "possible_issues": analysis["possible_issues"],
        "domain": analysis["domain"],
        "intent": analysis["intent"],
    }


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Vision Agent")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--query", default="", help="Optional question about the image")
    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: File not found: {args.image_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  VISION AGENT — Standalone Test")
    print("=" * 60)

    result = _analyze_image(args.image_path, args.query)
    print(f"\n  Domain:          {result['domain']}")
    print(f"  Intent:          {result['intent']}")
    print(f"  Summary:         {result['summary']}")
    print(f"  Components:      {result['components']}")
    print(f"  Observations:    {result['observations']}")
    print(f"  Possible Issues: {result['possible_issues']}")
    print("\n" + "=" * 60 + "\n")
