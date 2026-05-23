"""
PDF Parser — Extracts text blocks and images from PDF files using PyMuPDF.

Features:
  - Rich text block extraction using page.get_text("dict") to get font size, boldness, etc.
  - Section hierarchy tracking: H1, H2, H3 based on font size and boldness.
  - Noise filtering: removes page numbers, copyright markers, footers, dates.
  - Duplicate image removal: skips saving duplicate images using MD5 hashing.
  - Directional vertical proximity scoring for highly accurate image ↔ text associations.
"""

import os
import re
import json
import hashlib
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# --- Project root path ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDFS_DIR = DATA_DIR / "raw_pdfs"
IMAGES_DIR = DATA_DIR / "images"

# --- Noise Filtering Patterns --------------------------------------------
NOISE_PATTERNS = [
    re.compile(r"(?i)^page\s+\d+\s*(of\s*\d+)?$"),  # Page numbers
    re.compile(r"^\s*\d+\s*$"),                     # Single digit/number page marks
    re.compile(r"(?i)(copyright|\(c\)|all rights reserved)"), # Copyright info
    re.compile(r"(?i)(modified|date|updated):\s*\d+/\d+/\d+"), # Dates
    re.compile(r"(?i)^www\..*\.com$"),              # URL noise
]

# --- Global Image Deduplication registry ---------------------------------
_image_md5_registry = {}  # md5_hex -> relative_image_path


def _classify_source(pdf_path: str) -> str:
    """Determine if a PDF belongs to arduino or raspberry_pi category based on filename."""
    name_lower = Path(pdf_path).name.lower()
    if any(kw in name_lower for kw in ["arduino", "uno", "nano", "mega", "sensor"]):
        return "arduino"
    elif any(kw in name_lower for kw in ["raspberry", "rpi", "raspi", "pi_"]):
        return "raspberry_pi"
    return "arduino"


def _generate_image_id(source_doc: str, page_num: int, img_index: int) -> str:
    """Generate a unique image ID like 'arduino_projects_p042_img001'."""
    stem = Path(source_doc).stem.lower()
    clean_stem = "".join(c if c.isalnum() else "_" for c in stem).strip("_")
    while "__" in clean_stem:
        clean_stem = clean_stem.replace("__", "_")
    if len(clean_stem) > 25:
        clean_stem = clean_stem[:25]
    return f"{clean_stem}_p{page_num:03d}_img{img_index:03d}"


def clean_text(text: str) -> Optional[str]:
    """Clean block text and filter out noise (headers, footers, page numbers)."""
    text = text.strip()
    if not text or len(text) < 4:
        return None
        
    # Check against noise patterns
    for pattern in NOISE_PATTERNS:
        if pattern.search(text):
            return None
            
    return text


def extract_rich_text_blocks(page: fitz.Page, page_num: int, source_doc: str) -> Tuple[List[Dict], float]:
    """
    Extract text blocks along with font size, boldness, heading flag, and bbox.
    Uses get_text("dict") to get highly precise styling data.
    """
    page_dict = page.get_text("dict")
    text_blocks = []
    
    # Calculate average font size on this page to use as benchmark
    sizes = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:  # text
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text", "").strip()
                if span_text and len(span_text) > 3:
                    sizes.append(span.get("size", 10.0))
                    
    avg_font_size = sum(sizes) / len(sizes) if sizes else 10.0
    
    block_no = 0
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
            
        block_text_parts = []
        block_sizes = []
        is_bold = False
        
        for line in b.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_text += span_text
                
                span_size = span.get("size", 10.0)
                block_sizes.append(span_size)
                
                # Check for bold flag (flags bit 4 (16) is bold in PyMuPDF)
                if span.get("flags", 0) & 16 or "bold" in span.get("font", "").lower():
                    is_bold = True
            
            if line_text.strip():
                block_text_parts.append(line_text)
                
        text = "\n".join(block_text_parts).strip()
        cleaned = clean_text(text)
        if not cleaned:
            continue
            
        max_size = max(block_sizes) if block_sizes else avg_font_size
        
        # Heading detection heuristics
        is_heading = False
        if max_size > avg_font_size * 1.15 or (is_bold and len(cleaned) < 100 and not cleaned.endswith(".")):
            is_heading = True
            
        text_blocks.append({
            "text": cleaned,
            "bbox": tuple(b["bbox"]),
            "page_number": page_num,
            "source_document": source_doc,
            "block_index": block_no,
            "font_size": max_size,
            "is_bold": is_bold,
            "is_heading": is_heading,
            "section": "",
            "subsection": "",
            "subsubsection": "",
        })
        block_no += 1
        
    return text_blocks, avg_font_size


def extract_images(
    page: fitz.Page,
    page_num: int,
    source_doc: str,
    category: str,
    min_width: int = 50,
    min_height: int = 50,
) -> List[Dict]:
    """
    Extract embedded images and save them as PNGs.
    Implements MD5-based duplicate image detection to keep the dataset clean.
    """
    global _image_md5_registry
    
    image_list = page.get_images(full=True)
    extracted_images = []
    
    output_dir = IMAGES_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        
        try:
            base_image = page.parent.extract_image(xref)
            if base_image is None:
                continue
            
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]
            
            # Filter out tiny icon or decoration images
            if width < min_width or height < min_height:
                continue
            
            # 1. Deduplicate by MD5 hash
            img_md5 = hashlib.md5(image_bytes).hexdigest()
            
            if img_md5 in _image_md5_registry:
                # Duplicate found! Reuse existing image path
                relative_path = _image_md5_registry[img_md5]
                image_id = Path(relative_path).stem
            else:
                # Generate unique ID and save
                image_id = _generate_image_id(source_doc, page_num, img_index)
                filename = f"{image_id}.png"
                filepath = output_dir / filename
                
                # Convert colorspaces if CMYK or YCbCr
                from PIL import Image
                import io
                
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode in ('CMYK', 'YCbCr'):
                    img = img.convert('RGB')
                elif img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGBA')
                img.save(filepath, "PNG")
                
                relative_path = str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/")
                _image_md5_registry[img_md5] = relative_path
            
            # Estimate bounding box on the page
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                else:
                    pw, ph = page.rect.width, page.rect.height
                    bbox = (pw * 0.1, ph * 0.1, pw * 0.9, ph * 0.9)
            except Exception:
                pw, ph = page.rect.width, page.rect.height
                bbox = (pw * 0.1, ph * 0.1, pw * 0.9, ph * 0.9)
            
            extracted_images.append({
                "image_id": image_id,
                "image_path": relative_path,
                "bbox": bbox,
                "page_number": page_num,
                "source_document": source_doc,
                "width": width,
                "height": height,
                "caption": "",
            })
            
        except Exception as e:
            print(f"  [WARN] Failed to extract image {img_index} on page {page_num}: {e}")
            continue

    return extracted_images


def associate_images_with_text(
    text_blocks: List[Dict],
    images: List[Dict],
) -> List[Dict]:
    """
    Associate each image with the most semantically relevant text block on the same page.
    Scoring matches vertical alignment, caption keyword cues, and heading proximity.
    """
    # Regex to catch caption markers
    caption_pattern = re.compile(r"(?i)(figure|fig\.?|diagram|pinout|schematic|layout|wiring|table)\s*\d*")
    
    for image in images:
        img_page = image["page_number"]
        img_bbox = image["bbox"]
        
        # Calculate image center coordinates
        img_cx = (img_bbox[0] + img_bbox[2]) / 2
        img_cy = (img_bbox[1] + img_bbox[3]) / 2
        
        # Filter text blocks on this page
        page_blocks = [b for b in text_blocks if b["page_number"] == img_page]
        
        if not page_blocks:
            image["caption"] = ""
            image["associated_text_block_index"] = None
            continue
            
        best_score = -999999.0
        best_block = None
        
        for block in page_blocks:
            block_bbox = block["bbox"]
            block_cx = (block_bbox[0] + block_bbox[2]) / 2
            block_cy = (block_bbox[1] + block_bbox[3]) / 2
            
            dy = block_cy - img_cy  # Positive means block is BELOW image
            dx = abs(block_cx - img_cx)
            
            # 1. Proximity scoring (Captions are usually immediately BELOW or ABOVE)
            vertical_dist = abs(dy)
            if 0 < dy < 130:
                # Preferred: Directly below (within 130pt)
                vertical_weight = 150 - vertical_dist * 0.8
            elif -100 < dy < 0:
                # Acceptable: Directly above
                vertical_weight = 100 - vertical_dist * 0.8
            else:
                # Far away vertical penalty
                vertical_weight = -vertical_dist * 0.6
                
            # 2. Heading bonus
            heading_bonus = 50 if block.get("is_heading") else 0
            
            # 3. Caption text indicator bonus
            caption_bonus = 0
            block_text = block["text"]
            if caption_pattern.search(block_text):
                # Major bonus if starts with keyword, minor if contains keyword
                if caption_pattern.match(block_text.strip()):
                    caption_bonus = 200
                else:
                    caption_bonus = 100
                    
            # 4. Horizontal alignment penalty (captions should center alignment)
            horizontal_penalty = dx * 1.5
            
            # Final scoring
            score = vertical_weight + heading_bonus + caption_bonus - horizontal_penalty
            
            if score > best_score:
                best_score = score
                best_block = block
                
        if best_block:
            # Create two-way association
            best_block["associated_image_path"] = image["image_path"]
            
            # Store caption information
            caption_text = best_block["text"][:250].strip()
            image["caption"] = caption_text
            image["associated_text_block_index"] = best_block.get("block_index")
        else:
            image["caption"] = ""
            image["associated_text_block_index"] = None
            
    return images


def parse_single_pdf(pdf_path: str, category: Optional[str] = None) -> Dict:
    """Parse a single PDF file: extracts text blocks and images, and tracks structure."""
    pdf_path = Path(pdf_path).resolve()
    source_doc = pdf_path.name
    
    if category is None:
        category = _classify_source(str(pdf_path))
    
    print(f"\n{'='*60}")
    print(f"Parsing: {source_doc}")
    print(f"Category: {category}")
    print(f"{'='*60}")
    
    doc = fitz.open(str(pdf_path))
    all_text_blocks = []
    all_images = []
    
    # Running state for section hierarchy
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    
    page_count = len(doc)
    for page_num in range(page_count):
        page = doc[page_num]
        page_number = page_num + 1
        
        # 1. Extract Rich Text Blocks
        text_blocks, avg_font_size = extract_rich_text_blocks(page, page_number, source_doc)
        
        # 2. Update Section Hierarchy on the Page
        for block in text_blocks:
            if block.get("is_heading"):
                size = block.get("font_size", avg_font_size)
                # Determine H1/H2/H3 levels based on font size threshold comparison
                if size > avg_font_size * 1.4:
                    current_h1 = block["text"]
                    current_h2 = ""
                    current_h3 = ""
                elif size > avg_font_size * 1.2:
                    current_h2 = block["text"]
                    current_h3 = ""
                else:
                    current_h3 = block["text"]
            
            # Map running hierarchy state to block metadata
            block["section"] = current_h1
            block["subsection"] = current_h2
            block["subsubsection"] = current_h3
            
        all_text_blocks.extend(text_blocks)
        
        # 3. Extract Images
        images = extract_images(page, page_number, source_doc, category)
        all_images.extend(images)
        
        if page_number % 20 == 0:
            print(f"  Processed page {page_number}/{page_count}")
            
    doc.close()
    
    # 4. Associate images to text using directional vertical alignment scoring
    all_images = associate_images_with_text(all_text_blocks, all_images)
    
    stats = {
        "page_count": page_count,
        "text_block_count": len(all_text_blocks),
        "image_count": len(all_images),
    }
    
    print(f"\n  Results:")
    print(f"    Pages:       {stats['page_count']}")
    print(f"    Text blocks: {stats['text_block_count']}")
    print(f"    Images:      {stats['image_count']}")
    
    return {
        "source_document": source_doc,
        "category": category,
        "text_blocks": all_text_blocks,
        "images": all_images,
        "stats": stats,
    }


def parse_all_pdfs(pdf_dir: Optional[str] = None) -> List[Dict]:
    """Parse all PDF files in the raw_pdfs directory."""
    if pdf_dir is None:
        pdf_dir = RAW_PDFS_DIR
    else:
        pdf_dir = Path(pdf_dir)
        
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {pdf_dir}")
        return []
        
    results = []
    for pdf_file in pdf_files:
        try:
            result = parse_single_pdf(str(pdf_file))
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to parse {pdf_file.name}: {e}")
            continue
            
    return results
