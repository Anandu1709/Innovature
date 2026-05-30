"""
Ingestion Service — Orchestrates incremental ingestion of PDFs, URLs, and Markdown,
updating SQLite document registry, Milvus vector database, and the BM25 index.

This service reuses existing document parsers and chunkers to avoid code duplication
while ensuring newly added documents are indexed incrementally without rebuilding
the entire database from scratch.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from admin.registry import update_status, get_document, delete_document as registry_delete
from admin.hot_reload import reload_after_change

log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks" / "all_chunks.jsonl"
TEXT_COLLECTION = "text_collection"

# =============================================================================
#  Helper Functions
# =============================================================================

def _get_embedding_model():
    """Lazily load the text embedding model using the live retrieval singleton."""
    from agents.hybrid_retrieval import _get_text_model
    return _get_text_model()



def _append_chunks_to_jsonl(chunks: list[dict]) -> None:
    """Append new chunks to the canonical all_chunks.jsonl file."""
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info(f"[INGEST] Appended {len(chunks)} chunks to {CHUNKS_FILE.name}")


# =============================================================================
#  Core Ingestion Flow
# =============================================================================

def _process_and_insert(
    parse_result: dict,
    document_id: str,
    doc_name: str,
) -> dict:
    """
    Common downstream pipeline:
    Chunk → Generate Embeddings → Insert to Milvus → Append to JSONL → Rebuild/Reload
    """
    from ingestion.chunker import process_document

    try:
        # 1. Chunk document blocks using existing semantic chunker
        log.info(f"[INGEST] Semantic chunking for document: {doc_name}")
        chunks = process_document(parse_result)
        chunk_count = len(chunks)
        log.info(f"[INGEST] Generated {chunk_count} chunks")

        if chunk_count == 0:
            log.warning(f"[INGEST] Document {doc_name} yielded 0 chunks.")
            update_status(document_id, "ready", chunk_count=0, image_count=0)
            return {"document_id": document_id, "chunk_count": 0, "status": "ready"}

        # 2. Embed new chunks using warmed sentence-transformer model
        log.info("[INGEST] Embedding new chunks...")
        model = _get_embedding_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        log.info(f"[INGEST] Embeddings generated successfully (shape: {embeddings.shape})")

        # 3. Construct rows matching text_collection Milvus schema
        rows = []
        for i, chunk in enumerate(chunks):
            # Resolve image if any
            assoc_img = ""
            if chunk.get("images"):
                assoc_img = chunk["images"][0].get("image_path", "")

            rows.append({
                "chunk_id": chunk["chunk_id"][:64],
                "text": chunk["text"][:2000],
                "source_document": chunk.get("source_document", doc_name)[:200],
                "page_number": int(chunk.get("page_number", 1)),
                "section_name": chunk.get("section", "General")[:200],
                "associated_image_path": assoc_img[:500],
                "embedding": embeddings[i].tolist(),
            })

        # 4. Insert directly into Milvus text_collection (Incremental Ingestion)
        log.info(f"[INGEST] Inserting {len(rows)} records into Milvus...")
        from utils.milvus_manager import safe_insert
        insert_result = safe_insert(TEXT_COLLECTION, rows)
        inserted_count = insert_result.get("insert_count", len(rows))
        log.info(f"[INGEST] Inserted {inserted_count} records into Milvus")

        # 5. Append new chunks to all_chunks.jsonl
        _append_chunks_to_jsonl(chunks)

        # 6. Trigger Hot Reload (BM25 Index rebuild + Client load update)
        reload_after_change()

        # 7. Update status to 'ready' in SQLite document registry
        image_count = len(parse_result.get("images", []))
        update_status(document_id, "ready", chunk_count=chunk_count, image_count=image_count)

        log.info(f"[INGEST] Successfully indexed document: {doc_name}")
        return {
            "document_id": document_id,
            "chunk_count": chunk_count,
            "image_count": image_count,
            "status": "ready",
        }

    except Exception as e:
        log.exception(f"[INGEST] Ingestion pipeline failed for document: {doc_name}")
        update_status(document_id, "failed")
        raise e


# =============================================================================
#  Public Ingestion Ingresses
# =============================================================================

def ingest_pdf(
    pdf_path: str,
    document_id: str,
) -> dict:
    """
    Parse, chunk, embed, and index a PDF file.
    Reuses pdf_parser.py to keep formatting consistent.
    """
    from ingestion.pdf_parser import parse_single_pdf
    pdf_file = Path(pdf_path)
    log.info(f"[INGEST] Starting PDF ingestion from: {pdf_file.name}")

    try:
        parse_result = parse_single_pdf(str(pdf_file))
        return _process_and_insert(parse_result, document_id, pdf_file.name)
    except Exception as e:
        update_status(document_id, "failed")
        raise e


def ingest_url(
    url: str,
    document_id: str,
) -> dict:
    """
    Scrape, chunk, embed, and index a web URL page.
    Reuses web_scraper.py to extract main text.
    """
    from ingestion.web_scraper import scrape_single_url
    log.info(f"[INGEST] Starting URL ingestion from: {url}")

    # Deduce category based on keywords
    category = "arduino"
    url_lower = url.lower()
    if any(kw in url_lower for kw in ["raspberry", "rpi", "raspi", "pico"]):
        category = "raspberry_pi"

    url_config = {
        "url": url,
        "name": url,
        "category": category,
    }

    try:
        parse_result = scrape_single_url(url_config)
        if not parse_result:
            raise ValueError(f"Failed to scrape content from URL: {url}")

        return _process_and_insert(parse_result, document_id, url)
    except Exception as e:
        update_status(document_id, "failed")
        raise e


def ingest_markdown(
    title: str,
    markdown_text: str,
    document_id: str,
) -> dict:
    """
    Parse, chunk, embed, and index a raw pasted Markdown block.
    Constructs a parser mock object directly to feed the chunker.
    """
    log.info(f"[INGEST] Starting Markdown block ingestion: {title}")

    # Build basic blocks for chunker consumption
    text_blocks = [
        {
            "text": markdown_text,
            "page_number": 1,
            "source_document": title,
            "section": "Introduction",
            "subsection": "",
            "is_heading": False,
        }
    ]

    parse_result = {
        "source_document": title,
        "category": "arduino",  # default categorisation
        "text_blocks": text_blocks,
        "images": [],
    }

    try:
        return _process_and_insert(parse_result, document_id, title)
    except Exception as e:
        update_status(document_id, "failed")
        raise e


# =============================================================================
#  Deletion Flow
# =============================================================================

def delete_document(
    document_id: str,
) -> dict:
    """
    Remove a document completely from Milvus, all_chunks.jsonl, and the registry.
    """
    doc = get_document(document_id)
    if not doc:
        raise ValueError(f"Document ID {document_id} not found in SQLite registry")

    doc_name = doc["name"]
    log.info(f"[INGEST] Starting deletion of document: {doc_name} ({document_id})")

    try:
        update_status(document_id, "deleting")

        # 1. Delete matching vectors from Milvus text_collection
        log.info(f"[INGEST] Deleting chunks from Milvus for source_document: '{doc_name}'")
        from utils.milvus_manager import safe_delete
        safe_delete(
            TEXT_COLLECTION,
            f'source_document == "{doc_name}"',
        )
        log.info("[INGEST] Milvus vectors deleted")

        # 2. Rewrite all_chunks.jsonl excluding chunks matching source_document
        if CHUNKS_FILE.exists():
            log.info("[INGEST] Rewriting all_chunks.jsonl...")
            kept_chunks = []
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunk = json.loads(line)
                        if chunk.get("source_document") != doc_name:
                            kept_chunks.append(chunk)

            # Overwrite JSONL
            with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
                for chunk in kept_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            log.info(f"[INGEST] Rewrote all_chunks.jsonl, keeping {len(kept_chunks)} chunks")

        # 3. Rebuild BM25 index and refresh singletons
        reload_after_change()

        # 4. Remove from SQLite registry
        registry_delete(document_id)

        log.info(f"[INGEST] Successfully deleted document: {doc_name}")
        return {"document_id": document_id, "status": "deleted"}

    except Exception as e:
        log.exception(f"[INGEST] Deletion failed for document: {doc_name}")
        update_status(document_id, "failed")
        raise e
