"""
Semantic Text Chunker — Splits extracted text blocks into coherent, section-grounded,
and deduplicated chunks for vector embedding.

Features:
  - Heading-based Semantic Chunking (Priority 4): Flushes active chunks when a major H1/H2
    heading boundary is crossed, preventing semantic bleed between unrelated subjects.
  - Unified Document Objects (Priority 5): Chunks directly embed all associated image metadata
    (with captions and bboxes) into an internal list, becoming the canonical retrieval unit.
  - Deduplication (Priority 6): Drops duplicate chunks using md5 text hashing.
  - Tokenizer: tiktoken (cl100k_base).
"""

import json
import hashlib
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional


# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
OUTPUT_FILE = CHUNKS_DIR / "all_chunks.jsonl"

# --- Chunker defaults ----------------------------------------------------
CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS = 50

# --- Tokenizer -----------------------------------------------------------
_tokenizer = None


def _get_tokenizer():
    """Lazy-load the tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    enc = _get_tokenizer()
    return len(enc.encode(text))


def _generate_chunk_id(source_doc: str, section_name: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID.
    
    Format: '{clean_source}_{clean_section}_c{chunk:03d}'
    Example: 'arduino_uno_datasheet_features_c003'
    """
    stem = Path(source_doc).stem
    clean_src = "".join(c if c.isalnum() else "_" for c in stem.lower()).strip("_")
    while "__" in clean_src:
        clean_src = clean_src.replace("__", "_")
        
    clean_sect = "".join(c if c.isalnum() else "_" for c in section_name.lower()).strip("_")
    while "__" in clean_sect:
        clean_sect = clean_sect.replace("__", "_")
        
    # Truncate to keep IDs readable
    if len(clean_src) > 20:
        clean_src = clean_src[:20]
    if len(clean_sect) > 20:
        clean_sect = clean_sect[:20]
        
    return f"{clean_src}_{clean_sect}_c{chunk_index:03d}"


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[str]:
    """
    Split text into overlapping chunks based on token count.
    Splits along natural paragraph or sentence endings when possible.
    """
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    
    if len(tokens) <= chunk_size:
        return [text.strip()] if text.strip() else []
        
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens).strip()
        
        if chunk_text_str:
            # Look for natural sentence or paragraph breaks near the end
            if end < len(tokens):
                natural_breaks = [
                    chunk_text_str.rfind(".\n"),
                    chunk_text_str.rfind(". "),
                    chunk_text_str.rfind("\n\n"),
                    chunk_text_str.rfind("\n"),
                ]
                
                min_position = len(chunk_text_str) // 2
                best_break = -1
                for bp in natural_breaks:
                    if bp > min_position and bp > best_break:
                        best_break = bp
                        
                if best_break > 0:
                    chunk_text_str = chunk_text_str[:best_break + 1].strip()
                    actual_tokens = len(enc.encode(chunk_text_str))
                    end = start + actual_tokens
                    
            chunks.append(chunk_text_str)
            
        step = max(end - start - overlap, 1)
        start = start + step
        
    return chunks


def process_document(
    parse_result: Dict,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Groups text blocks by semantic headings (H1/H2 boundaries) and chunks them.
    Includes all associated images as structured inline metadata in Unified Document Objects.
    """
    source_doc = parse_result.get("source_document", "unknown")
    category = parse_result.get("category", "arduino")
    text_blocks = parse_result.get("text_blocks", [])
    images = parse_result.get("images", [])
    
    if not text_blocks:
        return []
        
    # Group text blocks by major heading (H1 or section)
    # When a block has is_heading or section changes, we group differently
    sections = []
    current_section_name = ""
    current_subsection_name = ""
    current_blocks = []
    
    for block in text_blocks:
        block_sect = block.get("section", "") or block.get("section_name", "") or "General"
        block_subsect = block.get("subsection", "")
        
        # Flush if we cross a section H1/H2 heading boundary
        if current_blocks and (block_sect != current_section_name or block_subsect != current_subsection_name):
            sections.append({
                "section": current_section_name,
                "subsection": current_subsection_name,
                "blocks": current_blocks
            })
            current_blocks = []
            
        current_section_name = block_sect
        current_subsection_name = block_subsect
        current_blocks.append(block)
        
    if current_blocks:
        sections.append({
            "section": current_section_name,
            "subsection": current_subsection_name,
            "blocks": current_blocks
        })
        
    all_chunks = []
    global_chunk_index = 0
    
    # Process each heading group independently to avoid semantic bleed
    for sect_group in sections:
        section_h1 = sect_group["section"]
        section_h2 = sect_group["subsection"]
        blocks = sect_group["blocks"]
        
        # Merge text blocks in this section
        section_text = "\n".join(b["text"] for b in blocks).strip()
        if not section_text or len(section_text) < 15:
            continue
            
        # Collect all images associated with this section
        sect_images = []
        for block in blocks:
            assoc_img_path = block.get("associated_image_path")
            if assoc_img_path:
                # Find corresponding full image metadata
                for img in images:
                    if img["image_path"] == assoc_img_path:
                        # Clean bbox tuple for serialization
                        clean_bbox = list(img["bbox"]) if isinstance(img["bbox"], tuple) else img["bbox"]
                        sect_images.append({
                            "image_id": img["image_id"],
                            "image_path": img["image_path"],
                            "caption": img.get("caption", ""),
                            "bbox": clean_bbox,
                            "width": img.get("width", 0),
                            "height": img.get("height", 0),
                        })
                        break
                        
        # Split section text into overlapping chunks
        text_chunks = chunk_text(section_text, chunk_size, overlap)
        
        for chunk_text_str in text_chunks:
            chunk_id = _generate_chunk_id(source_doc, section_h1 or "Intro", global_chunk_index)
            token_count = _count_tokens(chunk_text_str)
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "source_document": source_doc,
                "page_number": blocks[0].get("page_number", 1),
                "section": section_h1,
                "subsection": section_h2,
                "category": category,
                "token_count": token_count,
                "images": sect_images,  # Unified Retrieval Unit (Priority 5)
            }
            all_chunks.append(chunk_dict)
            global_chunk_index += 1
            
    return all_chunks


def process_all_documents(
    parse_results: List[Dict],
    output_file: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Process all parsed documents into chunks, deduplicate them (Priority 6), and save to JSONL.
    """
    if output_file is None:
        output_file = OUTPUT_FILE
    else:
        output_file = Path(output_file)
        
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SEMANTIC CHUNKER — Processing {len(parse_results)} document(s)")
    print(f"  Chunk size: {chunk_size} tokens")
    print(f"  Overlap:    {overlap} tokens")
    print(f"{'='*60}")
    
    raw_chunks = []
    for result in parse_results:
        source = result.get("source_document", "unknown")
        print(f"\n  Processing: {source}")
        
        chunks = process_document(result, chunk_size, overlap)
        raw_chunks.extend(chunks)
        print(f"    Raw chunks generated: {len(chunks)}")
        
    # Deduplicate chunks using Normalized Text Hash (Priority 6)
    seen_hashes = set()
    deduped_chunks = []
    
    for chunk in raw_chunks:
        # Normalize text to filter minor spacing/newline diffs
        normalized_text = "".join(chunk["text"].split()).lower()
        text_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduped_chunks.append(chunk)
            
    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in deduped_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    # Print summary
    total_tokens = sum(c.get("token_count", 0) for c in deduped_chunks)
    chunks_with_images = sum(1 for c in deduped_chunks if c.get("images"))
    
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"  Total raw chunks:    {len(raw_chunks)}")
    print(f"  Deduplicated chunks: {len(deduped_chunks)} (Dropped {len(raw_chunks) - len(deduped_chunks)} duplicates)")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"  Avg tokens/chunk:    {total_tokens // max(len(deduped_chunks), 1)}")
    print(f"  Chunks with images:  {chunks_with_images}")
    print(f"  Output file:         {output_file}")
    print(f"{'='*60}")
    
    return deduped_chunks


def load_chunks_from_jsonl(filepath: Optional[str] = None) -> List[Dict]:
    """Load chunks from the JSONL output file."""
    if filepath is None:
        filepath = OUTPUT_FILE
    else:
        filepath = Path(filepath)
        
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
