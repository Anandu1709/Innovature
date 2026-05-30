"""
Admin API Routes — FastAPI controllers for the Knowledge Management System.

Provides asynchronous REST endpoints for:
  - Listing registered documents
  - Getting database statistics
  - Ingesting PDFs (with background tasks)
  - Ingesting URL documentation (with background tasks)
  - Ingesting pasted Markdown (with background tasks)
  - Deleting indexed documents (with background tasks)
"""

import os
import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl

from admin.registry import (
    register_document,
    update_status,
    list_documents,
    get_statistics,
    get_document,
)
from admin.ingestion_service import (
    ingest_pdf,
    ingest_url,
    ingest_markdown,
    delete_document,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PDFS_DIR = PROJECT_ROOT / "data" / "raw_pdfs"


# =============================================================================
#  Pydantic Schemas
# =============================================================================

class URLIngestRequest(BaseModel):
    url: str


class MarkdownIngestRequest(BaseModel):
    title: str
    markdown: str


# =============================================================================
#  Workers (Background Task Handlers)
# =============================================================================

def bg_pdf_worker(temp_path: Path, document_id: str):
    """Background thread handler for PDF parsing and indexing."""
    try:
        ingest_pdf(str(temp_path), document_id)
    except Exception as e:
        log.error(f"[API_ADMIN] PDF bg ingestion failed for {document_id}: {e}")
        update_status(document_id, "failed")
    finally:
        # Keep temp file cleaned
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def bg_url_worker(url: str, document_id: str):
    """Background thread handler for web scraping and indexing."""
    try:
        ingest_url(url, document_id)
    except Exception as e:
        log.error(f"[API_ADMIN] URL bg ingestion failed for {document_id}: {e}")
        update_status(document_id, "failed")


def bg_markdown_worker(title: str, markdown: str, document_id: str):
    """Background thread handler for markdown block chunking and indexing."""
    try:
        ingest_markdown(title, markdown, document_id)
    except Exception as e:
        log.error(f"[API_ADMIN] Markdown bg ingestion failed for {document_id}: {e}")
        update_status(document_id, "failed")


def bg_deletion_worker(document_id: str):
    """Background thread handler for deep deletion from registries and vector collections."""
    try:
        delete_document(document_id)
    except Exception as e:
        log.error(f"[API_ADMIN] Deletion bg worker failed for {document_id}: {e}")
        update_status(document_id, "failed")


# =============================================================================
#  HTTP Endpoint Handlers
# =============================================================================

@router.get("/documents")
async def get_all_documents():
    """List all indexed documentation packages in the system registry."""
    try:
        return list_documents()
    except Exception as e:
        log.error(f"[API_ADMIN] Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_db_statistics():
    """Retrieve aggregate counts across all documents, chunk indices, and image maps."""
    try:
        return get_statistics()
    except Exception as e:
        log.error(f"[API_ADMIN] Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/pdf")
async def upload_pdf_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a datasheet PDF. Stream it to raw_pdfs, record details inside
    document registry, and kick off chunking + embedding in a background thread.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted")

    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = RAW_PDFS_DIR / f"upload_{file.filename}"

    # Write stream locally
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        log.error(f"[API_ADMIN] Streaming uploaded file failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file stream")

    # 1. Add record into SQLite Registry
    document_id = register_document(
        name=file.filename,
        doc_type="pdf",
        source=str(temp_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    )

    # 2. Fire background task thread
    background_tasks.add_task(bg_pdf_worker, temp_path, document_id)

    return {
        "message": f"Successfully uploaded '{file.filename}'. Ingestion started in background.",
        "document_id": document_id,
        "status": "processing",
    }


@router.post("/ingest/url")
async def ingest_url_page(
    request: URLIngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Register and scrape a web page. Resolves content, chunks elements,
    and inserts vectors incrementally inside a background task thread.
    """
    url_str = request.url.strip()
    if not url_str.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL format provided")

    # 1. Register processing block
    document_id = register_document(
        name=url_str,
        doc_type="url",
        source=url_str,
    )

    # 2. Fire scraping worker thread
    background_tasks.add_task(bg_url_worker, url_str, document_id)

    return {
        "message": f"Registered URL '{url_str}'. Scraping started in background.",
        "document_id": document_id,
        "status": "processing",
    }


@router.post("/ingest/markdown")
async def ingest_markdown_block(
    request: MarkdownIngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Register and ingest a raw pasted Markdown block.
    """
    title = request.title.strip()
    markdown = request.markdown.strip()

    if not title or not markdown:
        raise HTTPException(status_code=400, detail="Title and Markdown content are required")

    # 1. Register manual block
    document_id = register_document(
        name=title,
        doc_type="markdown",
        source="manual_paste",
    )

    # 2. Fire background task
    background_tasks.add_task(bg_markdown_worker, title, markdown, document_id)

    return {
        "message": f"Registered Markdown block '{title}'. Ingestion started in background.",
        "document_id": document_id,
        "status": "processing",
    }


@router.delete("/documents/{document_id}")
async def delete_indexed_document(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Cleanly purge a document package from the SQLite registry,
    all_chunks.jsonl records, and Milvus vector spaces.
    """
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document registry not found")

    if doc["status"] == "deleting":
        return {"message": "Document is already being purged.", "document_id": document_id}

    # Set status to deleting
    update_status(document_id, "deleting")

    # Launch background purger
    background_tasks.add_task(bg_deletion_worker, document_id)

    return {
        "message": f"Purging '{doc['name']}' from all vector collections and files.",
        "document_id": document_id,
        "status": "deleting",
    }
