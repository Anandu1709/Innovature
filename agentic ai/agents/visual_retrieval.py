"""
Visual Retrieval Agent — CLIP cross-modal search on image_collection.

Encodes the user's text query with CLIP (clip-ViT-B-32) and performs
vector similarity search against image embeddings in Milvus Lite.

Pipeline:
  CLIP retrieval → metadata reranking → threshold filtering →
  fallback recovery → graph-safe output

Score convention:
  Milvus COSINE returns distance (lower=better). We convert to
  similarity = 1.0 - distance so that HIGHER = BETTER.

Fixes applied (v2):
  [FIX-1] _rerank_with_metadata now uses regex tokenization for both
          query and caption/section terms — eliminates punctuation-
          attachment mismatches that caused wrong overlap counts.
  [FIX-2] Metadata bonus is now capped so that caption/section overlap
          cannot inflate a score beyond the CLIP similarity ceiling.
          This prevents irrelevant but keyword-rich images from outranking
          genuinely relevant ones.
  [FIX-3] Fallback threshold raised from 0.25 → 0.35.  0.25 is noise-
          level for CLIP on technical content; returning a wrong pinout
          image is worse than returning nothing.

Usage (standalone test):
  python agents/visual_retrieval.py
"""

import re
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from agents.state import AgentState

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MILVUS_DB_PATH = str(PROJECT_ROOT / "data" / "milvus_lite.db")
IMAGE_COLLECTION = "image_collection"
CLIP_MODEL_NAME = "clip-ViT-B-32"

# Retrieval parameters
IMAGE_TOP_K = 5                # Max candidates to retrieve from Milvus
SIMILARITY_THRESHOLD = 0.55    # Minimum cosine similarity to keep (after inversion)

# [FIX-2] Maximum bonus that metadata overlap can contribute.
# Prevents keyword-stuffed captions from dominating genuinely relevant images.
MAX_METADATA_BONUS = 0.10

# [FIX-3] Raised from 0.25 — 0.25 is noise-level for CLIP on technical content.
FALLBACK_THRESHOLD = 0.35


# =============================================================================
#  Shared Tokenizer — consistent with hybrid_retrieval.py
# =============================================================================

def _tokenize(text: str) -> set[str]:
    """
    Regex-based word tokenizer returning a set of clean tokens.

    [FIX-1] Replaces .split() throughout — eliminates punctuation-attachment
    mismatches (e.g. "specifications." ≠ "specifications").
    """
    return set(re.findall(r"\b\w+\b", text.lower()))


# =============================================================================
#  Singleton Resource Manager
# =============================================================================

_clip_model = None
_milvus_client = None


def _get_clip_model() -> SentenceTransformer:
    """Lazy-load the CLIP model for text → image cross-modal search."""
    global _clip_model
    if _clip_model is None:
        log.info("[VISUAL] Loading CLIP model...")
        _clip_model = SentenceTransformer(CLIP_MODEL_NAME)
        log.info(f"[VISUAL] CLIP model loaded — device: {_clip_model.device}")
    return _clip_model


def _get_milvus_client() -> MilvusClient:
    """Lazy-load the Milvus Lite client and load image_collection."""
    global _milvus_client
    if _milvus_client is None:
        log.info("[VISUAL] Connecting to Milvus Lite...")
        _milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
        _milvus_client.load_collection(IMAGE_COLLECTION)
        log.info("[VISUAL] Milvus connected and image_collection loaded")
    return _milvus_client


# =============================================================================
#  Image Search
# =============================================================================

def _image_search(query: str, top_k: int = IMAGE_TOP_K) -> list[dict]:
    """
    Encode text query with CLIP and search image_collection in Milvus.

    CLIP's shared embedding space allows cross-modal retrieval:
    a text query embedding can be compared directly against image embeddings.

    Returns results with cosine similarity scores (higher = better).
    """
    model = _get_clip_model()
    client = _get_milvus_client()

    # CLIP encodes text into the same 512-dim space as images
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    results = client.search(
        collection_name=IMAGE_COLLECTION,
        data=[query_embedding],
        limit=top_k,
        output_fields=["image_id", "image_path", "caption", "source_document", "section_name"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        },
    )

    formatted = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        distance = float(hit.get("distance", 1.0))
        similarity = 1.0 - distance  # Convert distance → similarity (higher=better)

        formatted.append({
            "image_id":        entity.get("image_id", ""),
            "image_path":      entity.get("image_path", ""),
            "caption":         entity.get("caption", ""),
            "source_document": entity.get("source_document", ""),
            "section_name":    entity.get("section_name", ""),
            "score":           similarity,
        })

    # Score diagnostics
    if formatted:
        scores = [r["score"] for r in formatted]
        log.info(
            f"[VISUAL] Similarity range: "
            f"{min(scores):.4f} → {max(scores):.4f}"
        )

    return formatted


# =============================================================================
#  Metadata-Assisted Reranking
# =============================================================================

def _rerank_with_metadata(
    query: str,
    results: list[dict],
) -> list[dict]:
    """
    Improve CLIP retrieval using metadata overlap signals.

    Technical diagrams often retrieve poorly using CLIP alone.
    We boost images whose captions/sections overlap with query terms.

    [FIX-1] Uses regex tokenizer — clean token matching, no punctuation artifacts.
    [FIX-2] Bonus is capped at MAX_METADATA_BONUS so keyword-rich captions
            cannot arbitrarily outrank high-similarity images.
    """
    query_terms = _tokenize(query)

    for result in results:
        bonus = 0.0

        caption_terms = _tokenize(result.get("caption", ""))     # [FIX-1]
        section_terms = _tokenize(result.get("section_name", ""))  # [FIX-1]

        # Caption overlap boost
        overlap_caption = len(query_terms & caption_terms)
        bonus += overlap_caption * 0.05

        # Section overlap boost
        overlap_section = len(query_terms & section_terms)
        bonus += overlap_section * 0.03

        # [FIX-2] Hard cap — metadata can nudge, not decide
        bonus = min(bonus, MAX_METADATA_BONUS)
        result["score"] += bonus

    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# =============================================================================
#  Threshold Filtering with Fallback
# =============================================================================

def _filter_by_threshold(
    results: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
    fallback_threshold: float = FALLBACK_THRESHOLD,   # [FIX-3] was 0.25
) -> list[dict]:
    """
    Filter image results by similarity threshold.

    If all results are removed but the top image has at least moderate
    relevance (>= fallback_threshold), keep it as a single best-effort result.

    [FIX-3] fallback_threshold raised from 0.25 to 0.35.  At 0.25 CLIP
    scores are essentially random for electronics technical queries —
    returning a wrong image destroys answer quality more than returning none.
    """
    filtered = [r for r in results if r["score"] >= threshold]

    # Normal success path
    if filtered:
        log.info(
            f"[VISUAL] Threshold filter kept "
            f"{len(filtered)} images"
        )
        return filtered

    # Fallback recovery
    if results:
        best = results[0]

        if best["score"] >= fallback_threshold:
            log.warning(
                f"[VISUAL] No images passed threshold "
                f"{threshold:.2f}. Using fallback image "
                f"(score={best['score']:.4f})"
            )
            return [best]

    log.warning("[VISUAL] No sufficiently relevant images found")

    return []


# =============================================================================
#  LangGraph Node
# =============================================================================

def visual_retrieval_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Perform CLIP cross-modal image retrieval.

    Pipeline: CLIP search → metadata reranking → threshold filter → fallback

    Reads:  state["query"], state["refined_query"]
    Writes: state["image_results"], state["sources"]
    """
    active_query = state.get("refined_query") or state["query"]
    log.info(f"[VISUAL] Searching images for: '{active_query}'")

    # 1. CLIP cross-modal search (with exception handling for graph robustness)
    try:
        raw_results = _image_search(active_query, top_k=IMAGE_TOP_K)
    except Exception as e:
        log.exception(f"[VISUAL] Image retrieval failed: {e}")
        return {
            "image_results": [],
            "sources": state.get("sources", []),
        }

    log.info(f"[VISUAL] Raw search returned {len(raw_results)} images")

    # 2. Metadata-assisted reranking — [FIX-1] clean tokenizer, [FIX-2] capped bonus
    reranked_results = _rerank_with_metadata(active_query, raw_results)

    # 3. Threshold filtering with fallback recovery — [FIX-3] raised fallback threshold
    filtered_results = _filter_by_threshold(reranked_results, SIMILARITY_THRESHOLD)
    log.info(f"[VISUAL] After threshold: {len(filtered_results)} images")

    # Collect source documents for citation tracking
    new_sources = list({
        r.get("source_document", "")
        for r in filtered_results
        if r.get("source_document")
    })

    # Merge with existing sources (from hybrid retrieval if intent="both")
    existing_sources = state.get("sources", [])
    merged_sources = list(set(existing_sources + new_sources))

    return {
        "image_results": filtered_results,
        "sources": merged_sources,
    }


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    test_queries = [
        "Arduino Uno pinout diagram",
        "Raspberry Pi GPIO layout",
        "SPI wiring schematic",
        "LED circuit diagram",
        "ATmega328P block diagram",
    ]

    print("\n" + "=" * 60)
    print("  VISUAL RETRIEVAL AGENT — Standalone Test (v2)")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Fallback threshold:   {FALLBACK_THRESHOLD}")
    print("=" * 60)

    for query in test_queries:
        state = create_initial_state(query)
        result = visual_retrieval_agent(state)

        print(f"\n  QUERY: \"{query}\"")
        print(f"  Images found: {len(result['image_results'])}")
        for i, r in enumerate(result["image_results"], 1):
            print(
                f"    {i}. (sim: {r['score']:.4f}) "
                f"[{r['image_id']}] "
                f"caption: \"{r['caption'][:60]}...\""
            )
            print(f"       path: {r['image_path']}")

        if not result["image_results"]:
            print("    No images above similarity threshold.")

    print("\n" + "=" * 60 + "\n")
