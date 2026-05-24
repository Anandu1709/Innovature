"""
Milvus Loader — Initialises Milvus Lite, creates text_collection and
image_collection with defined schemas, and inserts all embedded data.

Input:
  data/embeddings/text_embeddings.npy   + text_chunk_ids.json
  data/embeddings/image_embeddings.npy  + image_ids.json
  data/chunks/all_chunks.jsonl
  data/chunks/all_images.jsonl

Output:
  data/milvus_lite.db  (populated vector database)

Usage:
  python embeddings/milvus_loader.py
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
)

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Embedding files (produced by text_embedder.py and image_embedder.py)
TEXT_EMBEDDINGS_FILE = DATA_DIR / "embeddings" / "text_embeddings.npy"
TEXT_CHUNK_IDS_FILE = DATA_DIR / "embeddings" / "text_chunk_ids.json"
IMAGE_EMBEDDINGS_FILE = DATA_DIR / "embeddings" / "image_embeddings.npy"
IMAGE_IDS_FILE = DATA_DIR / "embeddings" / "image_ids.json"

# Original data files (produced by ingestion pipeline)
CHUNKS_FILE = DATA_DIR / "chunks" / "all_chunks.jsonl"
IMAGES_METADATA_FILE = DATA_DIR / "chunks" / "all_images.jsonl"

# Milvus Lite database
MILVUS_DB_PATH = str(DATA_DIR / "milvus_lite.db")

# --- Constants ---------------------------------------------------------------
TEXT_COLLECTION = "text_collection"
IMAGE_COLLECTION = "image_collection"
TEXT_VECTOR_DIM = 384
IMAGE_VECTOR_DIM = 512
INSERT_BATCH_SIZE = 500


# =============================================================================
#  Schema Definitions (Section 7.2)
# =============================================================================

def _create_text_schema() -> CollectionSchema:
    """
    text_collection schema:
      chunk_id              VARCHAR(64)   — primary key
      text                  VARCHAR(2000) — raw text content
      source_document       VARCHAR(200)  — source PDF filename or URL
      page_number           INT64         — page number in source
      section_name          VARCHAR(200)  — section heading
      associated_image_path VARCHAR(500)  — image on same page, if any
      embedding             FLOAT_VECTOR(384)
    """
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="section_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="associated_image_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=TEXT_VECTOR_DIM),
    ]
    return CollectionSchema(fields=fields, description="Text chunks with sentence-transformer embeddings")


def _create_image_schema() -> CollectionSchema:
    """
    image_collection schema:
      image_id        VARCHAR(64)   — primary key
      image_path      VARCHAR(500)  — relative file path to PNG
      caption         VARCHAR(500)  — extracted caption or surrounding text
      source_document VARCHAR(200)  — source document filename
      section_name    VARCHAR(200)  — section where image appeared
      embedding       FLOAT_VECTOR(512)
    """
    fields = [
        FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="section_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=IMAGE_VECTOR_DIM),
    ]
    return CollectionSchema(fields=fields, description="Image embeddings from CLIP ViT-B-32")


# =============================================================================
#  Data Loading Helpers
# =============================================================================

def _load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(filepath: Path) -> list:
    """Load a JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _truncate(value: str, max_len: int) -> str:
    """Truncate a string to fit a VARCHAR field."""
    if len(value) > max_len:
        return value[:max_len]
    return value


# =============================================================================
#  Build Insertion Data
# =============================================================================

def _build_text_rows(
    chunks: list[dict],
    embeddings: np.ndarray,
    chunk_ids: list[str],
) -> list[dict]:
    """
    Align chunk metadata with embeddings and build insertion rows
    for text_collection.

    Maps Day 1 fields to Milvus schema:
      chunk["section"]                   → section_name
      chunk["images"][0]["image_path"]   → associated_image_path
    """
    # Build a lookup: chunk_id → chunk metadata
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    rows = []
    skipped = 0

    for idx, cid in enumerate(chunk_ids):
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            skipped += 1
            continue

        # Derive associated_image_path from the first image in the chunk's images list
        images_list = chunk.get("images", [])
        assoc_image = images_list[0]["image_path"] if images_list else ""

        rows.append({
            "chunk_id": _truncate(cid, 64),
            "text": _truncate(chunk.get("text", ""), 2000),
            "source_document": _truncate(chunk.get("source_document", ""), 200),
            "page_number": int(chunk.get("page_number", 0)),
            "section_name": _truncate(chunk.get("section", ""), 200),
            "associated_image_path": _truncate(assoc_image, 500),
            "embedding": embeddings[idx].tolist(),
        })

    if skipped > 0:
        log.warning(f"Text rows: skipped {skipped} chunk_ids not found in JSONL")

    return rows


def _build_image_rows(
    image_metas: list[dict],
    chunks: list[dict],
    embeddings: np.ndarray,
    image_ids: list[str],
) -> list[dict]:
    """
    Align image metadata with embeddings and build insertion rows
    for image_collection.

    Resolves section_name by cross-referencing image_id against chunk data
    (each chunk has an "images" array containing image_id fields).
    """
    # Build a lookup: image_id → image metadata
    meta_lookup = {m["image_id"]: m for m in image_metas}

    # Build a cross-reference: image_id → section from chunks
    image_section_lookup = {}
    for chunk in chunks:
        section = chunk.get("section", "")
        for img in chunk.get("images", []):
            img_id = img.get("image_id", "")
            if img_id and img_id not in image_section_lookup:
                image_section_lookup[img_id] = section

    rows = []
    skipped = 0

    for idx, img_id in enumerate(image_ids):
        meta = meta_lookup.get(img_id)
        if meta is None:
            skipped += 1
            continue

        section = image_section_lookup.get(img_id, "")

        rows.append({
            "image_id": _truncate(img_id, 64),
            "image_path": _truncate(meta.get("image_path", ""), 500),
            "caption": _truncate(meta.get("caption", ""), 500),
            "source_document": _truncate(meta.get("source_document", ""), 200),
            "section_name": _truncate(section, 200),
            "embedding": embeddings[idx].tolist(),
        })

    if skipped > 0:
        log.warning(f"Image rows: skipped {skipped} image_ids not found in metadata")

    return rows


# =============================================================================
#  Main Pipeline
# =============================================================================

def init_milvus() -> MilvusClient:
    """Connect to Milvus Lite and return the client."""
    db_path = Path(MILVUS_DB_PATH)
    if db_path.exists():
        import shutil
        log.info(f"Removing existing database at {db_path} to avoid Windows drop_collection bug...")
        try:
            shutil.rmtree(db_path, ignore_errors=True)
            # Give the OS a brief moment to release directory locks
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"Could not remove database directory: {e}")

    log.info(f"Connecting to Milvus Lite: {MILVUS_DB_PATH}")
    client = MilvusClient(uri=MILVUS_DB_PATH)
    log.info("Connected to Milvus Lite")
    return client



def create_collections(client: MilvusClient) -> None:
    """Drop existing collections (if any) and create fresh ones with schemas."""

    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        if client.has_collection(name):
            client.drop_collection(name)
            log.info(f"Dropped existing collection: {name}")

    # Create text_collection
    text_schema = _create_text_schema()
    client.create_collection(
        collection_name=TEXT_COLLECTION,
        schema=text_schema,
    )
    log.info(f"Created collection: {TEXT_COLLECTION}")

    # Create image_collection
    image_schema = _create_image_schema()
    client.create_collection(
        collection_name=IMAGE_COLLECTION,
        schema=image_schema,
    )
    log.info(f"Created collection: {IMAGE_COLLECTION}")


def create_indexes(client: MilvusClient) -> None:
    """Create vector indexes on embedding fields for similarity search."""

    # Index for text_collection
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )
    client.create_index(
        collection_name=TEXT_COLLECTION,
        index_params=index_params,
    )
    log.info(f"Created FLAT/COSINE index on {TEXT_COLLECTION}.embedding")

    # Index for image_collection
    index_params_img = client.prepare_index_params()
    index_params_img.add_index(
        field_name="embedding",
        index_type="FLAT",
        metric_type="COSINE",
    )
    client.create_index(
        collection_name=IMAGE_COLLECTION,
        index_params=index_params_img,
    )
    log.info(f"Created FLAT/COSINE index on {IMAGE_COLLECTION}.embedding")


def insert_data(client: MilvusClient, collection_name: str, rows: list[dict]) -> int:
    """Insert rows in batches into a Milvus collection."""
    total_inserted = 0

    for i in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[i : i + INSERT_BATCH_SIZE]
        result = client.insert(collection_name=collection_name, data=batch)
        batch_count = result.get("insert_count", len(batch))
        total_inserted += batch_count
        log.info(
            f"  Inserted batch {i // INSERT_BATCH_SIZE + 1}: "
            f"{batch_count} rows into {collection_name} "
            f"({total_inserted}/{len(rows)} total)"
        )

    return total_inserted


def main():
    """Full Milvus loading pipeline: connect → create → insert → verify."""
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  MILVUS LOADER — Populating Vector Database")
    print("=" * 70)

    # --- Verify all input files exist ------------------------------------
    required_files = [
        TEXT_EMBEDDINGS_FILE, TEXT_CHUNK_IDS_FILE,
        IMAGE_EMBEDDINGS_FILE, IMAGE_IDS_FILE,
        CHUNKS_FILE, IMAGES_METADATA_FILE,
    ]
    for f in required_files:
        if not f.exists():
            print(f"\n  [ERROR] Required file missing: {f}")
            print("  Run text_embedder.py and image_embedder.py first.")
            return
    print("  All input files verified.")

    # --- Load embeddings and metadata ------------------------------------
    print("\n  Loading embeddings and metadata...")

    text_embeddings = np.load(TEXT_EMBEDDINGS_FILE)
    text_chunk_ids = _load_json(TEXT_CHUNK_IDS_FILE)
    chunks = _load_jsonl(CHUNKS_FILE)

    image_embeddings = np.load(IMAGE_EMBEDDINGS_FILE)
    image_ids = _load_json(IMAGE_IDS_FILE)
    image_metas = _load_jsonl(IMAGES_METADATA_FILE)

    # Alignment checks
    assert text_embeddings.shape[0] == len(text_chunk_ids), (
        f"Text embedding count ({text_embeddings.shape[0]}) != "
        f"chunk_id count ({len(text_chunk_ids)})"
    )
    assert text_embeddings.shape[1] == TEXT_VECTOR_DIM, (
        f"Text embedding dim ({text_embeddings.shape[1]}) != expected ({TEXT_VECTOR_DIM})"
    )
    assert image_embeddings.shape[0] == len(image_ids), (
        f"Image embedding count ({image_embeddings.shape[0]}) != "
        f"image_id count ({len(image_ids)})"
    )
    assert image_embeddings.shape[1] == IMAGE_VECTOR_DIM, (
        f"Image embedding dim ({image_embeddings.shape[1]}) != expected ({IMAGE_VECTOR_DIM})"
    )

    print(f"  Text:  {text_embeddings.shape[0]} embeddings × {text_embeddings.shape[1]}d")
    print(f"  Image: {image_embeddings.shape[0]} embeddings × {image_embeddings.shape[1]}d")
    print(f"  Chunks JSONL: {len(chunks)} records")
    print(f"  Images JSONL: {len(image_metas)} records")

    # --- Build insertion rows --------------------------------------------
    print("\n  Building insertion rows...")

    text_rows = _build_text_rows(chunks, text_embeddings, text_chunk_ids)
    print(f"  Text rows prepared: {len(text_rows)}")

    image_rows = _build_image_rows(image_metas, chunks, image_embeddings, image_ids)
    print(f"  Image rows prepared: {len(image_rows)}")

    # --- Initialize Milvus and insert ------------------------------------
    client = init_milvus()
    create_collections(client)
    create_indexes(client)

    print(f"\n  Inserting into {TEXT_COLLECTION}...")
    text_inserted = insert_data(client, TEXT_COLLECTION, text_rows)

    print(f"\n  Inserting into {IMAGE_COLLECTION}...")
    image_inserted = insert_data(client, IMAGE_COLLECTION, image_rows)

    # --- Summary ---------------------------------------------------------
    elapsed = time.time() - start_time
    db_path = Path(MILVUS_DB_PATH)
    db_size = db_path.stat().st_size / 1024 / 1024 if db_path.exists() else 0

    print("\n" + "=" * 70)
    print("  MILVUS LOADING COMPLETE")
    print("=" * 70)
    print(f"  Database:           {MILVUS_DB_PATH}")
    print(f"  Database size:      {db_size:.1f} MB")
    print(f"  {TEXT_COLLECTION}:  {text_inserted} rows ({TEXT_VECTOR_DIM}-dim)")
    print(f"  {IMAGE_COLLECTION}: {image_inserted} rows ({IMAGE_VECTOR_DIM}-dim)")
    print(f"  Index type:         FLAT / COSINE")
    print(f"  Duration:           {elapsed:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
