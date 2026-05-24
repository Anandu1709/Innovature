"""
BM25 Index Builder — Builds a BM25Okapi index on raw text from all chunks
and serializes it to disk with pickle.

Input:  data/chunks/all_chunks.jsonl
Output: data/bm25_index.pkl

Usage:
  python embeddings/bm25_builder.py
"""

import json
import time
import pickle
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
BM25_INDEX_FILE = PROJECT_ROOT / "data" / "bm25_index.pkl"


def load_chunks() -> list[dict]:
    """Load all chunks from the JSONL file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE.name}")
    return chunks


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenization.

    Consistent with the tokenization used at query time in the test script
    to ensure BM25 scoring is accurate.
    """
    return text.lower().split()


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[str], list[str]]:
    """
    Build a BM25Okapi index over all chunk texts.

    Returns:
        bm25:      fitted BM25Okapi model
        chunk_ids: aligned list of chunk_id strings (BM25 returns indices,
                   this maps them back to IDs)
        texts:     aligned list of raw text strings (for result display)
    """
    chunk_ids = []
    texts = []
    tokenized_corpus = []

    skipped = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            skipped += 1
            continue

        chunk_ids.append(chunk["chunk_id"])
        texts.append(text)
        tokenized_corpus.append(tokenize(text))

    if skipped > 0:
        log.warning(f"Skipped {skipped} chunks with empty text")

    log.info(f"Tokenized {len(tokenized_corpus)} documents for BM25")

    start = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    elapsed = time.time() - start

    log.info(f"BM25 index built in {elapsed:.2f}s")
    log.info(f"  Corpus size: {bm25.corpus_size}")
    log.info(f"  Avg doc length: {bm25.avgdl:.1f} tokens")

    return bm25, chunk_ids, texts


def save_index(bm25: BM25Okapi, chunk_ids: list[str], texts: list[str]) -> None:
    """Serialize BM25 index + aligned metadata to disk with pickle."""
    index_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "texts": texts,
    }

    BM25_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = BM25_INDEX_FILE.stat().st_size / 1024 / 1024
    log.info(f"Saved BM25 index: {BM25_INDEX_FILE.name} ({file_size:.2f} MB)")


def load_index() -> tuple[BM25Okapi, list[str], list[str]]:
    """
    Load a previously saved BM25 index from disk.

    Returns:
        bm25, chunk_ids, texts  (same structure as build_bm25_index output)
    """
    if not BM25_INDEX_FILE.exists():
        raise FileNotFoundError(f"BM25 index not found: {BM25_INDEX_FILE}")

    with open(BM25_INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    log.info(
        f"Loaded BM25 index: {data['bm25'].corpus_size} docs, "
        f"{len(data['chunk_ids'])} chunk_ids"
    )
    return data["bm25"], data["chunk_ids"], data["texts"]


def main():
    """Full BM25 index pipeline: load chunks → tokenize → build → save."""
    print("\n" + "=" * 70)
    print("  BM25 INDEX BUILDER — rank_bm25 (BM25Okapi)")
    print("=" * 70)

    # 1. Load chunks
    chunks = load_chunks()

    # 2. Build index
    bm25, chunk_ids, texts = build_bm25_index(chunks)

    # 3. Save to disk
    save_index(bm25, chunk_ids, texts)

    # 4. Verify by loading back
    bm25_loaded, ids_loaded, _ = load_index()

    # 5. Summary
    file_size = BM25_INDEX_FILE.stat().st_size / 1024 / 1024
    print("\n" + "=" * 70)
    print("  BM25 INDEX COMPLETE")
    print("=" * 70)
    print(f"  Corpus size:       {bm25.corpus_size} documents")
    print(f"  Avg doc length:    {bm25.avgdl:.1f} tokens")
    print(f"  Index file:        {BM25_INDEX_FILE}")
    print(f"  Index file size:   {file_size:.2f} MB")
    print(f"  Reload verified:   {len(ids_loaded)} chunk_ids loaded back OK")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
