"""
Milvus Connection Manager — Single, application-wide MilvusClient for Milvus Lite.

Problem:
  Milvus Lite stores collections as local files. On Windows, multiple
  MilvusClient instances pointing at the same database directory hold
  competing file handles, causing WinError 183 when one instance tries
  to write (insert/delete) while another holds the collection loaded.

Solution:
  One shared MilvusClient across the entire application, with a
  threading.Lock to serialize write operations (insert, delete, load).
  Read operations (search) proceed freely since MilvusClient reads
  are thread-safe on a single instance.

Usage:
    from utils.milvus_manager import get_client, safe_insert, safe_delete

    # For reads (search) — no lock needed
    client = get_client()
    results = client.search(...)

    # For writes — thread-safe
    safe_insert("text_collection", rows)
    safe_delete("text_collection", filter_expr)
"""

import threading
import logging
from pathlib import Path
from pymilvus import MilvusClient

log = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MILVUS_DB_PATH = str(PROJECT_ROOT / "data" / "milvus_lite.db")

# --- Singleton state ---------------------------------------------------------
_client: MilvusClient | None = None
_lock = threading.Lock()


def get_client() -> MilvusClient:
    """
    Return the application-wide MilvusClient singleton.

    The first call creates the client and connects to the local
    Milvus Lite database. Subsequent calls return the same instance.
    """
    global _client
    if _client is None:
        with _lock:
            # Double-check inside the lock to prevent race conditions
            if _client is None:
                log.info(f"[MILVUS_MGR] Connecting to Milvus Lite: {MILVUS_DB_PATH}")
                _client = MilvusClient(uri=MILVUS_DB_PATH)
                log.info("[MILVUS_MGR] Connected successfully")
    return _client


def load_collection(collection_name: str) -> None:
    """Load a collection into memory (thread-safe)."""
    with _lock:
        client = get_client()
        client.load_collection(collection_name)
        log.info(f"[MILVUS_MGR] Collection '{collection_name}' loaded")


def safe_insert(collection_name: str, data: list[dict]) -> dict:
    """
    Insert rows into a collection with thread-safe locking.

    The lock ensures that no concurrent insert/delete or manifest
    file write can collide on Windows.
    """
    with _lock:
        client = get_client()
        result = client.insert(collection_name=collection_name, data=data)
        client.load_collection(collection_name)
        log.info(
            f"[MILVUS_MGR] Inserted {len(data)} rows into '{collection_name}'"
        )
        return result


def safe_delete(collection_name: str, filter_expr: str) -> None:
    """
    Delete rows matching a filter expression with thread-safe locking.
    """
    with _lock:
        client = get_client()
        client.delete(collection_name=collection_name, filter=filter_expr)
        client.load_collection(collection_name)
        log.info(
            f"[MILVUS_MGR] Deleted from '{collection_name}' where {filter_expr}"
        )
