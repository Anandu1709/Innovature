"""
Document Registry — SQLite-based metadata store for admin document management.

This database tracks which documents have been ingested, their status,
and chunk/image counts. It is NEVER consulted during retrieval — Milvus
and BM25 remain the sole retrieval sources.

Database location: data/admin_registry.db

Usage:
    from admin.registry import init_db, register_document, list_documents
    init_db()
    doc_id = register_document("ATmega328P.pdf", "pdf", "/path/to/file.pdf")
"""

import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "admin_registry.db"

log = logging.getLogger(__name__)


# =============================================================================
#  Database Initialization
# =============================================================================

def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return conn


def init_db() -> None:
    """Create the documents table if it does not exist."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id   TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                type          TEXT NOT NULL,
                source        TEXT NOT NULL,
                chunk_count   INTEGER DEFAULT 0,
                image_count   INTEGER DEFAULT 0,
                uploaded_at   TEXT NOT NULL,
                status        TEXT DEFAULT 'processing'
            )
        """)
        conn.commit()
        log.info("[REGISTRY] SQLite database initialized")
    finally:
        conn.close()


# =============================================================================
#  CRUD Operations
# =============================================================================

def register_document(
    name: str,
    doc_type: str,
    source: str,
) -> str:
    """
    Register a new document in the registry.

    Args:
        name:     Display name (e.g. "ATmega328P.pdf" or "Arduino SPI Docs")
        doc_type: One of "pdf", "url", "markdown"
        source:   File path, URL, or "manual_paste"

    Returns:
        document_id (UUID string)
    """
    document_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO documents (document_id, name, type, source, uploaded_at, status)
            VALUES (?, ?, ?, ?, ?, 'processing')
            """,
            (document_id, name, doc_type, source, uploaded_at),
        )
        conn.commit()
        log.info(f"[REGISTRY] Registered document: {name} ({doc_type}) → {document_id}")
    finally:
        conn.close()

    return document_id


def update_status(
    document_id: str,
    status: str,
    chunk_count: Optional[int] = None,
    image_count: Optional[int] = None,
) -> None:
    """Update document status and optionally chunk/image counts."""
    conn = _get_conn()
    try:
        if chunk_count is not None and image_count is not None:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, image_count = ?
                WHERE document_id = ?
                """,
                (status, chunk_count, image_count, document_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ? WHERE document_id = ?",
                (status, document_id),
            )
        conn.commit()
        log.info(f"[REGISTRY] Updated {document_id}: status={status}")
    finally:
        conn.close()


def list_documents() -> list[dict]:
    """Return all documents as a list of dicts, ordered by upload time (newest first)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_document(document_id: str) -> Optional[dict]:
    """Return a single document by ID, or None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_document(document_id: str) -> bool:
    """Delete a document from the registry. Returns True if a row was deleted."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM documents WHERE document_id = ?",
            (document_id,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log.info(f"[REGISTRY] Deleted document: {document_id}")
        return deleted
    finally:
        conn.close()


def get_statistics() -> dict:
    """Return aggregated statistics across all documents."""
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*)                                    AS total_documents,
                COALESCE(SUM(chunk_count), 0)               AS total_chunks,
                COALESCE(SUM(image_count), 0)               AS total_images,
                COUNT(DISTINCT type)                         AS source_types,
                SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END)      AS ready_count,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)     AS failed_count
            FROM documents
        """).fetchone()
        return dict(row)
    finally:
        conn.close()
