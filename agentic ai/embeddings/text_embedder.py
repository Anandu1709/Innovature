"""
Text Embedder — Generates 384-dim dense embeddings for all text chunks.

Model: sentence-transformers/all-MiniLM-L6-v2
Input: data/chunks/all_chunks.jsonl
Output: data/embeddings/text_embeddings.npy + data/embeddings/text_chunk_ids.json

Usage:
  python embeddings/text_embedder.py
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "text_embeddings.npy"
TEXT_CHUNK_IDS_FILE = EMBEDDINGS_DIR / "text_chunk_ids.json"

# --- Model -------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIM = 384
BATCH_SIZE = 64


def load_text_model() -> SentenceTransformer:
    """Load the sentence-transformer model for text embedding."""
    log.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Verify output dimension
    test_emb = model.encode(["dimension check"])
    actual_dim = test_emb.shape[1]
    if actual_dim != EXPECTED_DIM:
        raise ValueError(
            f"Model dimension mismatch: expected {EXPECTED_DIM}, got {actual_dim}"
        )
    log.info(f"Model loaded — device: {model.device}, dim: {actual_dim}")
    return model


def load_chunks() -> list[dict]:
    """Load all chunks from the JSONL file produced by the ingestion pipeline."""
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


def embed_all_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate embeddings for all text chunks.

    Returns:
        embeddings: np.ndarray of shape (N, 384)
        chunk_ids:  list of chunk_id strings, aligned with embeddings
    """
    # Filter out chunks with empty text
    valid_chunks = []
    skipped = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if text:
            valid_chunks.append(chunk)
        else:
            skipped += 1

    if skipped > 0:
        log.warning(f"Skipped {skipped} chunks with empty text")

    if not valid_chunks:
        raise ValueError("No valid chunks to embed — all texts are empty")

    texts = [c["text"] for c in valid_chunks]
    chunk_ids = [c["chunk_id"] for c in valid_chunks]

    log.info(
        f"Embedding {len(texts)} chunks in batches of {batch_size}..."
    )
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    elapsed = time.time() - start
    log.info(
        f"Embedding complete — shape: {embeddings.shape}, "
        f"time: {elapsed:.1f}s ({len(texts) / elapsed:.0f} chunks/sec)"
    )

    return np.array(embeddings, dtype=np.float32), chunk_ids


def save_embeddings(embeddings: np.ndarray, chunk_ids: list[str]) -> None:
    """Persist embeddings and aligned chunk IDs to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(TEXT_EMBEDDINGS_FILE, embeddings)
    log.info(
        f"Saved embeddings: {TEXT_EMBEDDINGS_FILE.name} "
        f"({TEXT_EMBEDDINGS_FILE.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    with open(TEXT_CHUNK_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    log.info(f"Saved chunk IDs: {TEXT_CHUNK_IDS_FILE.name} ({len(chunk_ids)} entries)")


def main():
    """Full text embedding pipeline: load → embed → save."""
    print("\n" + "=" * 70)
    print("  TEXT EMBEDDER — all-MiniLM-L6-v2 (384-dim)")
    print("=" * 70)

    # 1. Load chunks from disk
    chunks = load_chunks()

    # 2. Load model
    model = load_text_model()

    # 3. Generate embeddings
    embeddings, chunk_ids = embed_all_chunks(chunks, model)

    # 4. Save to disk
    save_embeddings(embeddings, chunk_ids)

    # 5. Summary
    print("\n" + "=" * 70)
    print("  TEXT EMBEDDING COMPLETE")
    print("=" * 70)
    print(f"  Chunks embedded:  {len(chunk_ids)}")
    print(f"  Embedding shape:  {embeddings.shape}")
    print(f"  Output files:")
    print(f"    {TEXT_EMBEDDINGS_FILE}")
    print(f"    {TEXT_CHUNK_IDS_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
