"""Ingest all PDFs from data/pdfs into Postgres/pgvector."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from research_assistant.ingest_pdf import ingest_pdf_file  # noqa: E402


def main() -> None:
    pdf_dir = ROOT / "data" / "pdfs"
    if not pdf_dir.is_dir():
        raise SystemExit(f"Missing directory {pdf_dir}. Run scripts/fetch_sample_pdfs.py")
    files = sorted(pdf_dir.glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs under {pdf_dir}")
    total_chunks = 0
    for f in files:
        n = ingest_pdf_file(f)
        print(f"{f.name}: {n} chunks")
        total_chunks += n
    print("Total chunks inserted/updated:", total_chunks)


if __name__ == "__main__":
    main()
