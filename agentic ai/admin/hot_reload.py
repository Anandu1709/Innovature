"""
Hot Reload — Rebuilds BM25 index and invalidates retrieval singletons
after incremental document changes (insert or delete).

This module is the bridge between the admin ingestion service and the
live retrieval agents. It ensures that newly ingested or deleted documents
are immediately searchable without restarting the server.

Usage:
    from admin.hot_reload import reload_after_change
    reload_after_change()  # Call after any Milvus insert/delete
"""

import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"


def reload_bm25() -> None:
    """
    Rebuild the BM25 index from all_chunks.jsonl and invalidate the
    in-memory singleton in hybrid_retrieval.

    BM25Okapi has no incremental .add() method — IDF statistics and
    average document length are computed at construction time. A full
    rebuild from ~2000 chunks takes < 1 second, so this is not a
    bottleneck.
    """
    import json
    from rank_bm25 import BM25Okapi
    import agents.hybrid_retrieval as hr

    start = time.time()

    # Load all chunks from JSONL
    chunks = []
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    if not chunks:
        log.warning("[HOT_RELOAD] No chunks found in JSONL — BM25 index will be empty")
        hr._bm25_data = None
        return

    # Build BM25 index using the same tokenizer as hybrid_retrieval
    import re
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    chunk_ids = []
    texts = []
    source_documents = []
    section_names = []
    tokenized_corpus = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        chunk_ids.append(chunk["chunk_id"])
        texts.append(text)
        source_documents.append(chunk.get("source_document", ""))
        section_names.append(chunk.get("section_name", chunk.get("section", "")))
        tokenized_corpus.append(_tokenize(text))

    bm25 = BM25Okapi(tokenized_corpus)

    # Save to disk
    import pickle
    bm25_path = PROJECT_ROOT / "data" / "bm25_index.pkl"
    index_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "texts": texts,
        "source_documents": source_documents,
        "section_names": section_names,
    }
    with open(bm25_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Invalidate the singleton so next query reloads from disk
    hr._bm25_data = None

    elapsed = time.time() - start
    log.info(
        f"[HOT_RELOAD] BM25 rebuilt — {len(chunk_ids)} docs, "
        f"{elapsed:.2f}s, avgdl={bm25.avgdl:.1f}"
    )


def reload_milvus() -> None:
    """
    Force Milvus to re-load the text_collection after insert/delete.
    Uses the centralized milvus_manager to avoid file lock conflicts.
    """
    from utils.milvus_manager import load_collection
    try:
        load_collection("text_collection")
    except Exception as e:
        log.error(f"[HOT_RELOAD] Failed to reload Milvus collection: {e}")


def reload_after_change() -> None:
    """
    Full reload sequence after a document insert or delete.

    Call this after:
      1. Inserting new chunks into Milvus
      2. Appending/rewriting all_chunks.jsonl
    """
    log.info("[HOT_RELOAD] Starting post-change reload sequence...")
    reload_bm25()
    reload_milvus()
    log.info("[HOT_RELOAD] Reload complete — new data is now searchable")
