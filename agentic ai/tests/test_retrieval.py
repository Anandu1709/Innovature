"""
Hybrid Retrieval Test Script — Runs 5 sample queries through dense search,
BM25 search, and Reciprocal Rank Fusion (RRF). Also retrieves top image result.

This is the Day 2 retrieval evaluation harness that validates:
  - Semantic retrieval (Milvus dense search)
  - Exact keyword retrieval (BM25)
  - Fused ranking improvement (RRF)
  - Image retrieval (CLIP cross-modal search)

Usage:
  python tests/test_retrieval.py
"""

import sys
import time
import pickle
import logging
import numpy as np
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- Config ------------------------------------------------------------------
MILVUS_DB_PATH = str(PROJECT_ROOT / "data" / "milvus_lite.db")
BM25_INDEX_FILE = PROJECT_ROOT / "data" / "bm25_index.pkl"

TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "clip-ViT-B-32"

TOP_K = 3       # Number of results to display per method
RRF_K = 60      # RRF constant

# --- 5 Test Queries ----------------------------------------------------------
QUERIES = [
    "ATmega328P microcontroller specifications and features",
    "How to set up SPI communication on Arduino",
    "Raspberry Pi Pico W wireless connectivity",
    "PWM pins and analog output on Arduino Nano",
    "GPIO pin configuration for Raspberry Pi",
]


# =============================================================================
#  Console Helpers
# =============================================================================

def safe_print(*args, **kwargs):
    """Print safely on Windows consoles with encoding issues."""
    encoding = sys.stdout.encoding or "utf-8"
    processed = []
    for arg in args:
        if isinstance(arg, str):
            processed.append(arg.encode(encoding, errors="replace").decode(encoding))
        else:
            processed.append(arg)
    print(*processed, **kwargs)


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text for display, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# =============================================================================
#  BM25 Search
# =============================================================================

def load_bm25_index() -> tuple[BM25Okapi, list[str], list[str]]:
    """Load the pickled BM25 index from disk."""
    if not BM25_INDEX_FILE.exists():
        raise FileNotFoundError(f"BM25 index not found: {BM25_INDEX_FILE}")

    with open(BM25_INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["chunk_ids"], data["texts"]


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunk_ids: list[str],
    texts: list[str],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Perform BM25 keyword search.

    Returns list of dicts: [{"chunk_id": ..., "text": ..., "score": ...}, ...]
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices by score (descending)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk_id": chunk_ids[idx],
                "text": texts[idx],
                "score": float(scores[idx]),
            })

    return results


# =============================================================================
#  Dense Vector Search (Milvus)
# =============================================================================

def dense_search(
    query: str,
    client: MilvusClient,
    text_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the query with MiniLM and search text_collection in Milvus.

    Returns list of dicts: [{"chunk_id": ..., "text": ..., "score": ...}, ...]
    """
    query_embedding = text_model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    results = client.search(
        collection_name=TEXT_COLLECTION,
        data=[query_embedding],
        limit=top_k,
        output_fields=["chunk_id", "text", "source_document", "section_name"],
    )

    formatted = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        formatted.append({
            "chunk_id": entity.get("chunk_id", ""),
            "text": entity.get("text", ""),
            "source_document": entity.get("source_document", ""),
            "section_name": entity.get("section_name", ""),
            "score": float(hit.get("distance", 0)),
        })

    return formatted


# =============================================================================
#  Image Search (CLIP cross-modal)
# =============================================================================

def image_search(
    query: str,
    client: MilvusClient,
    image_model: SentenceTransformer,
    top_k: int = 1,
) -> list[dict]:
    """
    Embed the text query with CLIP and search image_collection in Milvus.
    CLIP supports cross-modal search: text query → image embeddings.

    Returns list of dicts with image metadata.
    """
    query_embedding = image_model.encode(
        [query], normalize_embeddings=True
    )[0].tolist()

    results = client.search(
        collection_name=IMAGE_COLLECTION,
        data=[query_embedding],
        limit=top_k,
        output_fields=["image_id", "image_path", "caption", "source_document", "section_name"],
    )

    formatted = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        formatted.append({
            "image_id": entity.get("image_id", ""),
            "image_path": entity.get("image_path", ""),
            "caption": entity.get("caption", ""),
            "source_document": entity.get("source_document", ""),
            "section_name": entity.get("section_name", ""),
            "score": float(hit.get("distance", 0)),
        })

    return formatted


# =============================================================================
#  Reciprocal Rank Fusion (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Combine dense and BM25 results using Reciprocal Rank Fusion.

    For each result appearing in either list:
      fused_score = sum( 1 / (k + rank) ) across all lists where it appears

    rank is 1-indexed.
    """
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score from dense results
    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank)
        chunk_data[cid] = result

    # Score from BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0) + 1.0 / (k + rank)
        if cid not in chunk_data:
            chunk_data[cid] = result

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        entry = chunk_data[cid].copy()
        entry["fused_score"] = fused_scores[cid]
        results.append(entry)

    return results


# =============================================================================
#  Main Test Runner
# =============================================================================

def run_tests():
    """Run all 5 queries through dense, BM25, RRF, and image search."""
    total_start = time.time()

    safe_print("\n" + "=" * 70)
    safe_print("  HYBRID RETRIEVAL TEST — Day 2 Validation")
    safe_print("=" * 70)

    # --- Load models and indexes -----------------------------------------
    safe_print("\n  Loading models and indexes...")

    safe_print("    Loading text model (MiniLM)...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    safe_print(f"    Text model loaded — device: {text_model.device}")

    safe_print("    Loading image model (CLIP)...")
    image_model = SentenceTransformer(IMAGE_MODEL_NAME)
    safe_print(f"    Image model loaded — device: {image_model.device}")

    safe_print("    Loading BM25 index...")
    bm25, bm25_chunk_ids, bm25_texts = load_bm25_index()
    safe_print(f"    BM25 index loaded — {bm25.corpus_size} documents")

    safe_print("    Connecting to Milvus Lite...")
    client = MilvusClient(uri=MILVUS_DB_PATH)
    safe_print("    Milvus connected")

    safe_print("    Loading collections into memory...")
    client.load_collection(TEXT_COLLECTION)
    client.load_collection(IMAGE_COLLECTION)
    safe_print("    Collections loaded successfully")

    safe_print("\n  All resources loaded. Running queries...\n")

    # --- Run each query --------------------------------------------------
    for q_idx, query in enumerate(QUERIES, start=1):
        safe_print("=" * 60)
        safe_print(f"  QUERY {q_idx}: {query}")
        safe_print("=" * 60)

        # --- Dense search ------------------------------------------------
        dense_results = dense_search(query, client, text_model, top_k=TOP_K)

        safe_print(f"\n  [DENSE SEARCH — Milvus {TEXT_COLLECTION}]")
        if dense_results:
            for i, r in enumerate(dense_results, 1):
                safe_print(
                    f"    {i}. (score: {r['score']:.4f}) "
                    f"[{r['chunk_id']}] "
                    f"\"{truncate_text(r['text'])}\""
                )
        else:
            safe_print("    No results found.")

        # --- BM25 search -------------------------------------------------
        bm25_results = bm25_search(query, bm25, bm25_chunk_ids, bm25_texts, top_k=TOP_K)

        safe_print(f"\n  [BM25 SEARCH]")
        if bm25_results:
            for i, r in enumerate(bm25_results, 1):
                safe_print(
                    f"    {i}. (score: {r['score']:.4f}) "
                    f"[{r['chunk_id']}] "
                    f"\"{truncate_text(r['text'])}\""
                )
        else:
            safe_print("    No results found.")

        # --- RRF fusion --------------------------------------------------
        # For RRF, we want more candidates from each source
        dense_for_rrf = dense_search(query, client, text_model, top_k=10)
        bm25_for_rrf = bm25_search(query, bm25, bm25_chunk_ids, bm25_texts, top_k=10)
        rrf_results = reciprocal_rank_fusion(dense_for_rrf, bm25_for_rrf, k=RRF_K, top_k=TOP_K)

        safe_print(f"\n  [HYBRID RRF RESULTS]")
        if rrf_results:
            for i, r in enumerate(rrf_results, 1):
                safe_print(
                    f"    {i}. (fused: {r['fused_score']:.6f}) "
                    f"[{r['chunk_id']}] "
                    f"\"{truncate_text(r['text'])}\""
                )
        else:
            safe_print("    No results found.")

        # --- Image search ------------------------------------------------
        img_results = image_search(query, client, image_model, top_k=1)

        safe_print(f"\n  [TOP IMAGE]")
        if img_results:
            img = img_results[0]
            safe_print(f"    image:   {img['image_path']}")
            safe_print(f"    score:   {img['score']:.4f}")
            safe_print(f"    caption: {img['caption']}")
        else:
            safe_print("    No image results found.")

        safe_print("")  # blank line between queries

    # --- Summary ---------------------------------------------------------
    elapsed = time.time() - total_start
    safe_print("=" * 60)
    safe_print("  TEST RUN COMPLETE")
    safe_print("=" * 60)
    safe_print(f"  Queries run:   {len(QUERIES)}")
    safe_print(f"  Total time:    {elapsed:.1f}s")
    safe_print(f"  Methods:       Dense (MiniLM) + BM25 + RRF + CLIP Image")
    safe_print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
