"""
Hybrid Retrieval Agent — BM25 + Dense Vector Search + Reciprocal Rank Fusion.

Combines keyword-based retrieval (BM25) with semantic dense search (Milvus)
and fuses results using RRF for high-quality chunk retrieval.

Reuses the Day 2 infrastructure:
  - Milvus Lite text_collection (384-dim, all-MiniLM-L6-v2)
  - BM25 index (data/bm25_index.pkl)

Score convention:
  All scores are normalised so that HIGHER = BETTER.
  - Dense: Milvus COSINE returns distance (lower=better), we convert to
    similarity = 1.0 - distance.
  - BM25: Native scores are already higher=better.
  - RRF: Fused rank scores are higher=better.

Fixes applied (v2):
  [FIX-1] Unified regex tokenizer in _bm25_search matches the BM25 index
          builder tokenizer — eliminates the punctuation-attachment mismatch
          that caused near-zero BM25 recall on every query.
  [FIX-2] BM25 corpus metadata (source_document, section_name) is now
          loaded from the pickle and attached to BM25 results so the RRF
          merger never produces metadata-empty entries.
  [FIX-3] RRF chunk_data merge now always prefers the entry with the most
          complete metadata rather than silently keeping the first-seen one.
  [NEW-1] Phase 2: Exact part-number / technical term boosting applied
          after RRF fusion.  Technical codes (e.g. ATmega328P, RP2040,
          LM35) are extracted from the query and used to boost chunks that
          contain an exact case-insensitive match.
  [NEW-2] Phase 3: Metadata-assisted section reranking mirrors the
          strategy in visual_retrieval.py.  Chunks whose section_name
          overlaps high-value section keywords receive a score bonus.

Usage (standalone test):
  python agents/hybrid_retrieval.py
"""

import re
import sys
import pickle
import logging
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from agents.state import AgentState

# --- Config ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MILVUS_DB_PATH = str(PROJECT_ROOT / "data" / "milvus_lite.db")
BM25_INDEX_FILE = PROJECT_ROOT / "data" / "bm25_index.pkl"
TEXT_COLLECTION = "text_collection"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval parameters
DENSE_TOP_K = 10      # Candidates from dense search for RRF input
BM25_TOP_K = 10       # Candidates from BM25 for RRF input
FINAL_TOP_K = 5       # Final fused results returned to state
RRF_K = 60            # RRF constant

# Boosting parameters
PART_NUMBER_BOOST = 0.05          # [NEW-1] RRF score bonus per exact part-number match
SECTION_KEYWORD_BOOST = 0.015     # [NEW-2] RRF score bonus per high-value section keyword match

# High-value section keywords for electronics datasheets
HIGH_VALUE_SECTIONS = {
    "pinout", "specifications", "electrical", "characteristics",
    "absolute", "ratings", "register", "timing", "diagram",
    "schematic", "interface", "protocol", "memory", "features",
}

# Regex for technical alphanumeric codes (e.g. ATmega328P, RP2040, LM35, GPIO17)
TECH_TERM_PATTERN = re.compile(r"\b[a-zA-Z]+\d+[a-zA-Z\d]*\b")

# [FIX-1] Unified tokenizer — used at both index-build time and query time.
# Must match the tokenizer used in embeddings/bm25_builder.py exactly.
def _tokenize(text: str) -> list[str]:
    """
    Regex-based word tokenizer.

    Strips punctuation attachment (e.g. "flash," → "flash") and preserves
    alphanumeric technical terms like "ATmega328P", "GPIO17", "I2C".
    This is the single canonical tokenizer for all BM25 operations.
    """
    return re.findall(r"\b\w+\b", text.lower())


# =============================================================================
#  Singleton Resource Manager — avoids reloading models on every graph call
# =============================================================================

_text_model = None
_bm25_data = None


def _get_text_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer text embedding model."""
    global _text_model
    if _text_model is None:
        log.info("[HYBRID] Loading text embedding model...")
        _text_model = SentenceTransformer(TEXT_MODEL_NAME)
        log.info(f"[HYBRID] Text model loaded — device: {_text_model.device}")
    return _text_model


def _get_bm25() -> tuple[BM25Okapi, list[str], list[str], list[str], list[str]]:
    """
    Lazy-load the pickled BM25 index.

    [FIX-2] Now also loads source_document and section_name arrays so that
    BM25 results carry full metadata into the RRF merger.  The pickle is
    expected to contain these keys; if absent (old index), empty strings
    are substituted so the code degrades gracefully.
    """
    global _bm25_data
    if _bm25_data is None:
        log.info("[HYBRID] Loading BM25 index...")
        if not BM25_INDEX_FILE.exists():
            raise FileNotFoundError(f"BM25 index not found: {BM25_INDEX_FILE}")
        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)

        n = data["bm25"].corpus_size
        source_docs   = data.get("source_documents", [""] * n)
        section_names = data.get("section_names",    [""] * n)

        _bm25_data = (
            data["bm25"],
            data["chunk_ids"],
            data["texts"],
            source_docs,
            section_names,
        )
        log.info(f"[HYBRID] BM25 index loaded — {n} documents")
    return _bm25_data




# =============================================================================
#  Search Functions
# =============================================================================

def _dense_search(query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
    """Embed query with MiniLM and search text_collection in Milvus.

    Milvus COSINE metric returns distance = 1 - cosine_similarity
    (lower distance = more similar). We convert to similarity so that
    higher score = better match, consistent with BM25 and RRF.
    """
    from utils.milvus_manager import get_client
    model = _get_text_model()
    client = get_client()

    query_embedding = model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    results = client.search(
        collection_name=TEXT_COLLECTION,
        data=[query_embedding],
        limit=top_k,
        output_fields=["chunk_id", "text", "source_document", "section_name"],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    formatted = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        distance = float(hit.get("distance", 1.0))
        similarity = 1.0 - distance  # Convert distance → similarity (higher=better)
        formatted.append({
            "chunk_id": entity.get("chunk_id", ""),
            "text": entity.get("text", ""),
            "source_document": entity.get("source_document", ""),
            "section_name": entity.get("section_name", ""),
            "score": similarity,
            "search_method": "dense",
        })

    return formatted


def _bm25_search(query: str, top_k: int = BM25_TOP_K) -> list[dict]:
    """
    Perform BM25 keyword search on the tokenized corpus.

    [FIX-1] Uses the unified _tokenize() function — same regex used at
    index-build time — so query tokens always match index tokens.
    [FIX-2] Attaches source_document and section_name to each result.
    """
    bm25, chunk_ids, texts, source_docs, section_names = _get_bm25()

    # [FIX-1] Unified tokenizer: "flash," → "flash", "ATmega328P" preserved
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk_id":       chunk_ids[idx],
                "text":           texts[idx],
                "source_document": source_docs[idx],    # [FIX-2]
                "section_name":   section_names[idx],   # [FIX-2]
                "score":          float(scores[idx]),
                "search_method":  "bm25",
            })

    return results


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
) -> list[dict]:
    """
    Combine dense and BM25 results using Reciprocal Rank Fusion.

    For each result appearing in either list:
      fused_score = sum( 1 / (k + rank) ) across all lists where it appears.
    rank is 1-indexed.

    [FIX-3] Chunk metadata merge: when a chunk appears in both lists,
    the dense result's metadata (source_document, section_name) takes
    precedence since Milvus always returns these fields reliably.
    """
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score from dense results (metadata always complete from Milvus)
    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank)
        chunk_data[cid] = result  # Dense result preferred for metadata [FIX-3]

    # Score from BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank)
        if cid not in chunk_data:
            # BM25-only hit: use BM25 metadata (now populated — [FIX-2])
            chunk_data[cid] = result
        else:
            # Already in chunk_data from dense — keep dense metadata,
            # but merge in BM25 metadata fields if dense left them empty.
            existing = chunk_data[cid]
            if not existing.get("source_document") and result.get("source_document"):
                existing["source_document"] = result["source_document"]
            if not existing.get("section_name") and result.get("section_name"):
                existing["section_name"] = result["section_name"]

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid].copy()
        entry["fused_score"] = fused_scores[cid]
        entry["search_method"] = "hybrid_rrf"
        results.append(entry)

    return results


# =============================================================================
#  Phase 2: Exact Part-Number & Technical Keyword Boosting  [NEW-1]
# =============================================================================

def _extract_tech_terms(query: str) -> list[str]:
    """
    Extract technical alphanumeric codes from the query.

    Matches part numbers and chip identifiers like:
      ATmega328P, RP2040, LM35, GPIO17, I2C1, SPI0, USB2.0
    """
    return [m.lower() for m in TECH_TERM_PATTERN.findall(query)]


def _apply_part_number_boost(
    results: list[dict],
    tech_terms: list[str],
    boost: float = PART_NUMBER_BOOST,
) -> list[dict]:
    """
    Boost RRF fused_score for chunks containing exact query tech terms.

    A chunk receives +boost for each tech term it contains (case-insensitive).
    This directly counteracts the dilution that dense embeddings apply to
    exact technical strings like part numbers.
    """
    if not tech_terms:
        return results

    for result in results:
        text_lower = result.get("text", "").lower()
        matches = sum(1 for term in tech_terms if term in text_lower)
        if matches:
            result["fused_score"] += matches * boost
            log.debug(
                f"[HYBRID] Part-number boost +{matches * boost:.4f} "
                f"on chunk {result['chunk_id']} ({matches} term(s) matched)"
            )

    results.sort(key=lambda x: x["fused_score"], reverse=True)
    return results


# =============================================================================
#  Phase 3: Metadata-Assisted Section Reranking  [NEW-2]
# =============================================================================

def _apply_section_boost(
    results: list[dict],
    query: str,
    boost: float = SECTION_KEYWORD_BOOST,
) -> list[dict]:
    """
    Boost chunks whose section_name overlaps with high-value keywords.

    High-value sections (pinout, specifications, electrical characteristics,
    register map, timing diagram, etc.) are the most precise retrieval
    targets for electronics datasheets.  Chunks in these sections get a
    small but deterministic RRF score bonus.

    Additionally, if the query itself mentions a section keyword
    (e.g. "pinout diagram"), only chunks in matching sections are boosted,
    which acts as a soft filter.
    """
    query_tokens = set(_tokenize(query))
    query_section_hints = query_tokens & HIGH_VALUE_SECTIONS

    for result in results:
        section_tokens = set(_tokenize(result.get("section_name", "")))
        section_overlap = section_tokens & HIGH_VALUE_SECTIONS

        if not section_overlap:
            continue

        if query_section_hints:
            # Only boost if the chunk's section matches what the query hints at
            matched = section_overlap & query_section_hints
            result["fused_score"] += len(matched) * boost
        else:
            # No section hint in query — boost any high-value section
            result["fused_score"] += len(section_overlap) * boost

    results.sort(key=lambda x: x["fused_score"], reverse=True)
    return results


# =============================================================================
#  LangGraph Node
# =============================================================================

def hybrid_retrieval_agent(state: AgentState) -> dict:
    """
    LangGraph node function: Perform hybrid BM25 + dense + RRF retrieval.

    Pipeline:
      Dense search → BM25 search (unified tokenizer) → RRF fusion
      → Part-number boosting → Section reranking → Top-K output

    Reads:  state["query"], state["refined_query"]
    Writes: state["text_results"], state["sources"]
    """
    active_query = state.get("refined_query") or state["query"]
    log.info(f"[HYBRID] Searching for: '{active_query}'")

    # 1. Dense vector search
    dense_results = _dense_search(active_query, top_k=DENSE_TOP_K)
    log.info(f"[HYBRID] Dense search returned {len(dense_results)} results")

    # 2. BM25 keyword search — [FIX-1] unified tokenizer now active
    bm25_results = _bm25_search(active_query, top_k=BM25_TOP_K)
    log.info(f"[HYBRID] BM25 search returned {len(bm25_results)} results")

    # 3. RRF fusion — [FIX-3] metadata merge now correct
    fused_results = _reciprocal_rank_fusion(
        dense_results, bm25_results, k=RRF_K, top_k=FINAL_TOP_K
    )
    log.info(f"[HYBRID] RRF fused to {len(fused_results)} candidates")

    # 4. [NEW-1] Phase 2: Exact part-number / technical term boosting
    tech_terms = _extract_tech_terms(active_query)
    if tech_terms:
        log.info(f"[HYBRID] Tech terms detected: {tech_terms}")
    fused_results = _apply_part_number_boost(fused_results, tech_terms)

    # 5. [NEW-2] Phase 3: Section metadata reranking
    fused_results = _apply_section_boost(fused_results, active_query)

    log.info(f"[HYBRID] Final top-{FINAL_TOP_K} results ready")

    # Collect unique source documents for citation tracking
    sources = list({
        r.get("source_document", "")
        for r in fused_results
        if r.get("source_document")
    })

    return {
        "text_results": fused_results,
        "sources": sources,
    }


# ─── Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.state import create_initial_state

    test_queries = [
        "ATmega328P microcontroller specifications",
        "How to set up SPI communication on Arduino",
        "Raspberry Pi Pico W wireless connectivity",
        "LM35 temperature sensor electrical characteristics",
        "RP2040 pinout",
    ]

    print("\n" + "=" * 60)
    print("  HYBRID RETRIEVAL AGENT — Standalone Test (v2)")
    print("=" * 60)

    for query in test_queries:
        state = create_initial_state(query)
        result = hybrid_retrieval_agent(state)

        print(f"\n  QUERY: \"{query}\"")
        print(f"  Sources: {result['sources']}")
        for i, r in enumerate(result["text_results"], 1):
            text_preview = r["text"][:80].replace("\n", " ")
            print(
                f"    {i}. (fused: {r['fused_score']:.6f}) "
                f"[{r['chunk_id']}] "
                f"section: '{r.get('section_name', '')[:30]}' "
                f"\"{text_preview}...\""
            )

    print("\n" + "=" * 60 + "\n")
