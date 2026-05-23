"""
Ingestion Pipeline — Master script that runs the full data processing pipeline.

Execution order:
  1. Parse all PDFs in data/raw_pdfs/ using pdf_parser.py
  2. Scrape all configured web URLs using web_scraper.py
  3. Chunk all extracted text using chunker.py
  4. Save combined output to data/chunks/all_chunks.jsonl
  5. Save image metadata to data/chunks/all_images.jsonl

Usage:
  python ingestion/ingest_pipeline.py                    # Run full pipeline
  python ingestion/ingest_pipeline.py --pdf-only         # Only parse PDFs
  python ingestion/ingest_pipeline.py --web-only         # Only scrape web
  python ingestion/ingest_pipeline.py --skip-web         # PDFs + chunking, skip web
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import sibling modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pdf_parser import parse_all_pdfs, parse_single_pdf
from ingestion.web_scraper import scrape_all_urls, ALL_URLS
from ingestion.markdown_parser import parse_rpi_docs
from ingestion.chunker import process_all_documents, OUTPUT_FILE


# --- Paths ---------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
CHUNKS_DIR = DATA_DIR / "chunks"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_METADATA_FILE = CHUNKS_DIR / "all_images.jsonl"


def _ensure_directories():
    """Create all required directories if they don't exist."""
    for d in [RAW_PDFS_DIR, CHUNKS_DIR, IMAGES_DIR / "arduino", IMAGES_DIR / "raspberry_pi"]:
        d.mkdir(parents=True, exist_ok=True)


def _save_images_metadata(all_results: list):
    """Save image metadata from all parsed documents to a JSONL file."""
    image_count = 0
    with open(IMAGES_METADATA_FILE, "w", encoding="utf-8") as f:
        for result in all_results:
            for image in result.get("images", []):
                # Clean image dict for serialization
                clean_image = {}
                for k, v in image.items():
                    if k == "bbox":
                        clean_image[k] = list(v) if isinstance(v, tuple) else v
                    else:
                        clean_image[k] = v
                
                # Add category and source info
                clean_image["category"] = result.get("category", "")
                
                f.write(json.dumps(clean_image, ensure_ascii=False) + "\n")
                image_count += 1
    
    print(f"\n  Image metadata saved: {image_count} entries -> {IMAGES_METADATA_FILE}")
    return image_count


def run_pipeline(
    parse_pdfs: bool = True,
    parse_rpi_repo: bool = True,
    scrape_web: bool = True,
    do_chunking: bool = True,
):
    """
    Run the full ingestion pipeline.
    
    Args:
        parse_pdfs: Whether to parse PDFs from data/raw_pdfs/
        parse_rpi_repo: Whether to parse Raspberry Pi asciidoc documentation
        scrape_web: Whether to scrape web URLs
        do_chunking: Whether to run chunking on the results
    """
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("  MULTIMODAL ELECTRONICS ASSISTANT - INGESTION PIPELINE")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    _ensure_directories()
    
    all_parse_results = []
    
    # --- Step 1: Parse PDFs ------------------------------------------
    if parse_pdfs:
        print("\n" + "-" * 70)
        print("  STEP 1: Parsing PDF files")
        print("-" * 70)
        
        pdf_files = sorted(RAW_PDFS_DIR.glob("*.pdf"))
        if pdf_files:
            pdf_results = parse_all_pdfs()
            all_parse_results.extend(pdf_results)
            print(f"\n  [OK] PDF parsing complete: {len(pdf_results)} document(s) processed")
        else:
            print(f"\n  [INFO] No PDF files found in {RAW_PDFS_DIR}")
            print(f"  Place your PDF files there and re-run the pipeline.")
    else:
        print("\n  [SKIP] PDF parsing skipped")
    
    # --- Step 2: Scrape Web ------------------------------------------
    if scrape_web:
        print("\n" + "-" * 70)
        print("  STEP 2: Scraping web documentation")
        print("-" * 70)
        
        web_results = scrape_all_urls()
        all_parse_results.extend(web_results)
        print(f"\n  [OK] Web scraping complete: {len(web_results)} page(s) scraped")
    else:
        print("\n  [SKIP] Web scraping skipped")

    # --- Step 2.5: Parse RPi Repository Docs -------------------------
    if parse_rpi_repo:
        print("\n" + "-" * 70)
        print("  STEP 2.5: Parsing RPi Repository AsciiDoc files")
        print("-" * 70)
        
        rpi_results = parse_rpi_docs()
        all_parse_results.extend(rpi_results)
        print(f"\n  [OK] RPi Docs parsing complete: {len(rpi_results)} document(s) processed")
    else:
        print("\n  [SKIP] RPi Docs parsing skipped")
    
    # --- Step 3: Save image metadata ---------------------------------
    if all_parse_results:
        print("\n" + "-" * 70)
        print("  STEP 3: Saving image metadata")
        print("-" * 70)
        
        image_count = _save_images_metadata(all_parse_results)
    
    # --- Step 4: Chunk all text --------------------------------------
    if do_chunking and all_parse_results:
        print("\n" + "-" * 70)
        print("  STEP 4: Chunking text into embedding-ready segments")
        print("-" * 70)
        
        all_chunks = process_all_documents(all_parse_results)
    elif not all_parse_results:
        print("\n  [WARN] No data to chunk — both PDF and web stages produced no results")
        all_chunks = []
    else:
        print("\n  [SKIP] Chunking skipped")
        all_chunks = []
    
    # --- Summary -----------------------------------------------------
    elapsed = time.time() - start_time
    
    # Count output files
    chunk_file_exists = OUTPUT_FILE.exists()
    chunk_count = 0
    if chunk_file_exists:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            chunk_count = sum(1 for _ in f)
    
    arduino_images = len(list((IMAGES_DIR / "arduino").glob("*.png"))) if (IMAGES_DIR / "arduino").exists() else 0
    rpi_images = len(list((IMAGES_DIR / "raspberry_pi").glob("*.png"))) if (IMAGES_DIR / "raspberry_pi").exists() else 0
    total_images = arduino_images + rpi_images
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Duration:            {elapsed:.1f} seconds")
    print(f"  Documents processed: {len(all_parse_results)}")
    print(f"  Chunks generated:    {chunk_count}")
    print(f"  Images extracted:    {total_images} ({arduino_images} Arduino, {rpi_images} RPi)")
    print(f"  Chunks file:         {OUTPUT_FILE} {'[OK]' if chunk_file_exists else '[MISSING]'}")
    print(f"  Images metadata:     {IMAGES_METADATA_FILE}")
    print("=" * 70)
    
    # Validation against Day 1 targets
    print("\n  Day 1 Target Validation:")
    if chunk_count >= 1000:
        print(f"    [OK] Chunks: {chunk_count} >= 1000 target")
    else:
        print(f"    [!!] Chunks: {chunk_count} < 1000 target - add more PDFs to data/raw_pdfs/")
    
    if total_images >= 100:
        print(f"    [OK] Images: {total_images} >= 100 target")
    else:
        print(f"    [!!] Images: {total_images} < 100 target - add more PDFs with diagrams")
    
    return {
        "documents_processed": len(all_parse_results),
        "chunks_generated": chunk_count,
        "images_extracted": total_images,
        "elapsed_seconds": elapsed,
    }


# ─── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multimodal electronics assistant ingestion pipeline"
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only parse PDFs, skip web scraping and RPi docs",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Only scrape web, skip PDF and RPi docs parsing",
    )
    parser.add_argument(
        "--rpi-only",
        action="store_true",
        help="Only parse RPi Repository AsciiDoc docs",
    )
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip web scraping (parse PDFs + RPi repo + chunk)",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip chunking step",
    )
    
    args = parser.parse_args()
    
    if args.pdf_only:
        run_pipeline(parse_pdfs=True, parse_rpi_repo=False, scrape_web=False, do_chunking=True)
    elif args.web_only:
        run_pipeline(parse_pdfs=False, parse_rpi_repo=False, scrape_web=True, do_chunking=True)
    elif args.rpi_only:
        run_pipeline(parse_pdfs=False, parse_rpi_repo=True, scrape_web=False, do_chunking=True)
    elif args.skip_web:
        run_pipeline(parse_pdfs=True, parse_rpi_repo=True, scrape_web=False, do_chunking=True)
    elif args.skip_chunk:
        run_pipeline(parse_pdfs=True, parse_rpi_repo=True, scrape_web=True, do_chunking=False)
    else:
        run_pipeline()
