"""
Image Embedder — Generates 512-dim CLIP embeddings for all PNG images.

Model: clip-ViT-B-32 (via sentence-transformers)
Input: data/chunks/all_images.jsonl (metadata) + data/images/**/*.png (files)
Output: data/embeddings/image_embeddings.npy + data/embeddings/image_ids.json

Usage:
  python embeddings/image_embedder.py
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from PIL import Image
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
IMAGES_METADATA_FILE = PROJECT_ROOT / "data" / "chunks" / "all_images.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "image_embeddings.npy"
IMAGE_IDS_FILE = EMBEDDINGS_DIR / "image_ids.json"

# --- Model -------------------------------------------------------------------
MODEL_NAME = "clip-ViT-B-32"
EXPECTED_DIM = 512
BATCH_SIZE = 16

# --- Filters -----------------------------------------------------------------
MIN_IMAGE_SIZE = 10  # Skip images smaller than 10x10 pixels (likely artifacts)


def load_image_model() -> SentenceTransformer:
    """Load the CLIP model for image embedding."""
    log.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Verify output dimension with a tiny test image
    test_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    test_emb = model.encode([test_img])
    actual_dim = test_emb.shape[1]
    if actual_dim != EXPECTED_DIM:
        raise ValueError(
            f"Model dimension mismatch: expected {EXPECTED_DIM}, got {actual_dim}"
        )
    log.info(f"Model loaded — device: {model.device}, dim: {actual_dim}")
    return model


def load_image_metadata() -> list[dict]:
    """Load image metadata from the JSONL file produced by the ingestion pipeline."""
    if not IMAGES_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Image metadata file not found: {IMAGES_METADATA_FILE}"
        )

    metadata = []
    with open(IMAGES_METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    log.info(f"Loaded {len(metadata)} image metadata entries from {IMAGES_METADATA_FILE.name}")
    return metadata


def load_and_validate_images(
    image_metas: list[dict],
) -> tuple[list[Image.Image], list[dict]]:
    """
    Open and validate all images referenced in metadata.

    Returns:
        pil_images:   list of PIL Image objects (RGB)
        valid_metas:  corresponding metadata dicts (aligned with pil_images)
    """
    pil_images = []
    valid_metas = []
    skipped_missing = 0
    skipped_corrupt = 0
    skipped_tiny = 0

    for meta in image_metas:
        raw_path = meta.get("image_path", "")
        image_path = PROJECT_ROOT / raw_path

        # Check file exists
        if not image_path.exists():
            skipped_missing += 1
            log.debug(f"Missing file: {raw_path}")
            continue

        # Try opening the image
        try:
            img = Image.open(image_path)

            # Skip tiny images (likely extraction artifacts)
            if img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE:
                skipped_tiny += 1
                log.debug(f"Tiny image skipped ({img.width}x{img.height}): {raw_path}")
                continue

            # Convert to RGB (handles RGBA, palette, grayscale)
            img = img.convert("RGB")

            pil_images.append(img)
            valid_metas.append(meta)

        except Exception as e:
            skipped_corrupt += 1
            log.warning(f"Corrupt image skipped: {raw_path} — {e}")

    log.info(
        f"Image validation: {len(pil_images)} valid, "
        f"{skipped_missing} missing, {skipped_corrupt} corrupt, "
        f"{skipped_tiny} too small"
    )
    return pil_images, valid_metas


def embed_all_images(
    pil_images: list[Image.Image],
    valid_metas: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate CLIP embeddings for all validated images.

    Returns:
        embeddings: np.ndarray of shape (M, 512)
        image_ids:  list of image_id strings, aligned with embeddings
    """
    if not pil_images:
        raise ValueError("No valid images to embed")

    image_ids = [m["image_id"] for m in valid_metas]

    log.info(f"Embedding {len(pil_images)} images in batches of {batch_size}...")
    start = time.time()

    embeddings = model.encode(
        pil_images,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    elapsed = time.time() - start
    log.info(
        f"Embedding complete — shape: {embeddings.shape}, "
        f"time: {elapsed:.1f}s ({len(pil_images) / elapsed:.1f} images/sec)"
    )

    return np.array(embeddings, dtype=np.float32), image_ids


def save_embeddings(embeddings: np.ndarray, image_ids: list[str]) -> None:
    """Persist image embeddings and aligned image IDs to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(IMAGE_EMBEDDINGS_FILE, embeddings)
    log.info(
        f"Saved embeddings: {IMAGE_EMBEDDINGS_FILE.name} "
        f"({IMAGE_EMBEDDINGS_FILE.stat().st_size / 1024 / 1024:.2f} MB)"
    )

    with open(IMAGE_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(image_ids, f)
    log.info(f"Saved image IDs: {IMAGE_IDS_FILE.name} ({len(image_ids)} entries)")


def main():
    """Full image embedding pipeline: load metadata → validate images → embed → save."""
    print("\n" + "=" * 70)
    print("  IMAGE EMBEDDER — clip-ViT-B-32 (512-dim)")
    print("=" * 70)

    # 1. Load image metadata from disk
    image_metas = load_image_metadata()

    # 2. Validate and open all images
    pil_images, valid_metas = load_and_validate_images(image_metas)

    if not pil_images:
        print("\n  [ERROR] No valid images found. Exiting.")
        return

    # 3. Load model
    model = load_image_model()

    # 4. Generate embeddings
    embeddings, image_ids = embed_all_images(pil_images, valid_metas, model)

    # 5. Save to disk
    save_embeddings(embeddings, image_ids)

    # 6. Summary
    print("\n" + "=" * 70)
    print("  IMAGE EMBEDDING COMPLETE")
    print("=" * 70)
    print(f"  Images embedded:   {len(image_ids)}")
    print(f"  Embedding shape:   {embeddings.shape}")
    print(f"  Output files:")
    print(f"    {IMAGE_EMBEDDINGS_FILE}")
    print(f"    {IMAGE_IDS_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
