"""PDF text extraction and DB insert."""

from pathlib import Path
from typing import List

from pypdf import PdfReader

from . import chunking, db, embeddings


def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts).strip()


def ingest_pdf_file(
    pdf_path: Path,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> int:
    text = read_pdf_text(pdf_path)
    if not text:
        return 0
    doc_name = pdf_path.name
    pieces = chunking.split_text_recursive(
        text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not pieces:
        return 0
    vecs = embeddings.encode_texts(pieces)
    with db.connection() as conn:
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(pieces, vecs)):
                cur.execute(
                    """
                    INSERT INTO chunks (doc_name, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_name, chunk_index)
                    DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (doc_name, i, chunk, emb),
                )
    return len(pieces)
