"""Manage the PDF corpus and the matching rows in Postgres/pgvector.

Subcommands:
  list                       Show ingested PDFs and their chunk counts.
  remove <name> [<name>...]  Delete all chunks for the given doc_name(s).
  remove --all               Delete every chunk (does not delete files on disk).
  sync                       Re-align Postgres with data/pdfs/:
                               - delete DB rows for PDFs no longer on disk
                               - ingest any PDFs on disk that are not in DB
                               (existing PDFs are left untouched unless --refresh)
  sync --refresh             Same as sync, but also re-ingests existing PDFs
                               (handles edits / shorter replacements safely).

Examples:
  python scripts/docs.py list
  python scripts/docs.py remove 1810.04805_bert.pdf
  python scripts/docs.py remove --all
  python scripts/docs.py sync
  python scripts/docs.py sync --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from research_assistant import db  # noqa: E402
from research_assistant.ingest_pdf import ingest_pdf_file  # noqa: E402


PDF_DIR = ROOT / "data" / "pdfs"


def _doc_counts() -> List[tuple]:
    """Return [(doc_name, chunk_count)] ordered by name."""
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_name, COUNT(*) FROM chunks GROUP BY doc_name ORDER BY doc_name"
            )
            return cur.fetchall()


def _delete_docs(doc_names: Iterable[str]) -> int:
    names = [n for n in doc_names if n]
    if not names:
        return 0
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE doc_name IN %s",
                (tuple(names),),
            )
            return cur.rowcount or 0


def _truncate_all() -> int:
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            n = cur.fetchone()[0]
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
            return int(n)


def _disk_pdfs() -> List[Path]:
    if not PDF_DIR.is_dir():
        return []
    return sorted(PDF_DIR.glob("*.pdf"))


def cmd_list() -> int:
    rows = _doc_counts()
    if not rows:
        print("(no chunks ingested yet)")
        return 0
    total = 0
    print(f"{'doc_name':60} chunks")
    print("-" * 72)
    for name, count in rows:
        print(f"{name:60} {count}")
        total += int(count)
    print("-" * 72)
    print(f"{'TOTAL':60} {total}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    if args.all:
        deleted = _truncate_all()
        print(f"Cleared all chunks ({deleted} rows).")
        return 0
    if not args.names:
        print("Provide one or more PDF names, or --all.")
        return 2
    deleted = _delete_docs(args.names)
    print(f"Deleted {deleted} chunks across {len(args.names)} doc(s).")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    on_disk: List[Path] = _disk_pdfs()
    on_disk_names: Set[str] = {p.name for p in on_disk}
    in_db_rows = _doc_counts()
    in_db_names: Set[str] = {r[0] for r in in_db_rows}

    to_remove = sorted(in_db_names - on_disk_names)
    to_add = [p for p in on_disk if p.name not in in_db_names]
    to_refresh: List[Path] = []
    if args.refresh:
        to_refresh = [p for p in on_disk if p.name in in_db_names]

    print(f"On disk: {len(on_disk_names)} PDF(s)")
    print(f"In DB : {len(in_db_names)} doc(s)")
    print(f"Will remove from DB: {len(to_remove)}")
    print(f"Will add to DB    : {len(to_add)}")
    if args.refresh:
        print(f"Will refresh      : {len(to_refresh)}")

    if to_remove:
        deleted = _delete_docs(to_remove)
        print(f"  removed {deleted} chunks for {len(to_remove)} doc(s):")
        for name in to_remove:
            print(f"    - {name}")

    if to_refresh:
        deleted = _delete_docs([p.name for p in to_refresh])
        print(f"  cleared {deleted} chunks before refresh.")

    targets = to_add + to_refresh
    total_chunks = 0
    for p in targets:
        n = ingest_pdf_file(p)
        print(f"  ingested {p.name}: {n} chunks")
        total_chunks += n
    print(f"Done. Inserted/updated {total_chunks} chunks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus admin CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List ingested PDFs and chunk counts")

    p_remove = sub.add_parser("remove", help="Delete chunks for one or more PDFs")
    p_remove.add_argument("names", nargs="*", help="PDF file names (e.g. paper.pdf)")
    p_remove.add_argument("--all", action="store_true", help="Delete ALL chunks")
    p_remove.set_defaults(func=cmd_remove)

    p_sync = sub.add_parser(
        "sync",
        help="Sync DB with data/pdfs/ (add new, remove orphans).",
    )
    p_sync.add_argument(
        "--refresh",
        action="store_true",
        help="Also re-ingest existing PDFs (recommended after editing files).",
    )
    p_sync.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    if args.cmd == "list":
        return cmd_list()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
