import os
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()


def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "Set DATABASE_URL (e.g. postgresql://user:pass@localhost:5432/dbname)"
        )
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    return conn


@contextmanager
def connection() -> Generator[Any, None, None]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retrieve_top_k(
    query_embedding: List[float],
    k: int,
) -> List[dict]:
    """Cosine distance via pgvector <=> on normalized vectors."""
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_name, chunk_index, text,
                       (embedding <=> %s::vector) AS dist
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )
            rows = cur.fetchall()
    out: List[dict] = []
    for doc_name, chunk_index, text, dist in rows:
        out.append(
            {
                "doc_name": doc_name,
                "chunk_index": int(chunk_index),
                "text": text,
                "distance": float(dist),
            }
        )
    return out


def list_distinct_doc_names(limit: int = 30) -> List[str]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT doc_name FROM chunks ORDER BY doc_name LIMIT %s",
                (limit,),
            )
            return [r[0] for r in cur.fetchall()]


def delete_docs_for_reingest(doc_names: Optional[List[str]] = None) -> None:
    """If doc_names is None, truncate all chunks."""
    with connection() as conn:
        with conn.cursor() as cur:
            if doc_names:
                cur.execute(
                    "DELETE FROM chunks WHERE doc_name IN %s",
                    (tuple(doc_names),),
                )
            else:
                cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
