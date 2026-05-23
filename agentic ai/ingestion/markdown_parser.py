"""
Markdown/AsciiDoc Parser - Extracts text sections and images from the
Raspberry Pi documentation GitHub repository.

Features:
  - Precise AsciiDoc (.adoc) heading tracking (H1, H2, H3).
  - Noise filtering: removes page markers, copyright info, URLs.
  - MD5-based duplicate image detection to ensure a clean visual dataset.
  - Directional, context-aware text-image associations.
"""

import os
import re
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# --- Project root path ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RPI_DOCS_DIR = DATA_DIR / "rpi_docs" / "documentation" / "asciidoc"
IMAGES_DIR = DATA_DIR / "images" / "raspberry_pi"

# --- AsciiDoc parsing patterns -------------------------------------------
HEADING_PATTERN = re.compile(r'^(={2,5})\s+(.+)$', re.MULTILINE)
IMAGE_BLOCK_PATTERN = re.compile(r'image::([^\[]+)\[([^\]]*)\]')
IMAGE_INLINE_PATTERN = re.compile(r'image:([^\[]+)\[([^\]]*)\]')
CODE_BLOCK_START = re.compile(r'^----\s*$')
INCLUDE_PATTERN = re.compile(r'^include::([^\[]+)\[([^\]]*)\]')
ADMONITION_PATTERN = re.compile(r'^(NOTE|WARNING|TIP|IMPORTANT|CAUTION):\s*(.*)', re.MULTILINE)

# --- Noise Filtering Patterns --------------------------------------------
NOISE_PATTERNS = [
    re.compile(r"(?i)^page\s+\d+\s*(of\s*\d+)?$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"(?i)(copyright|\(c\)|all rights reserved)"),
    re.compile(r"(?i)(modified|date|updated):\s*\d+/\d+/\d+"),
    re.compile(r"(?i)^www\..*\.com$"),
]

# Key directories to parse (most relevant for electronics/hardware content)
PRIORITY_DIRS = [
    "computers/raspberry-pi",
    "computers/getting-started",
    "computers/configuration",
    "computers/config_txt",
    "computers/camera",
    "computers/os",
    "computers/linux_kernel",
    "computers/remote-access",
    "computers/processors",
    "microcontrollers",
]

# --- Global Image Deduplication registry ---------------------------------
_image_md5_registry = {}  # md5_hex -> relative_image_path


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


def _extract_alt_text(attr_string: str) -> str:
    """Extract alt text from AsciiDoc image attributes."""
    alt_match = re.search(r'alt="([^"]*)"', attr_string)
    if alt_match:
        return alt_match.group(1)
    parts = attr_string.split(",")
    if parts and parts[0].strip():
        return parts[0].strip().strip('"')
    return ""


def _resolve_image_path(image_ref: str, adoc_file: Path) -> Optional[Path]:
    """Resolve an image reference to an actual file path."""
    adoc_dir = adoc_file.parent
    
    # Try relative to the adoc file's directory
    candidate = adoc_dir / image_ref
    if candidate.exists():
        return candidate
    
    # Try relative to the adoc file's parent (for included files)
    candidate = adoc_dir.parent / image_ref
    if candidate.exists():
        return candidate
    
    # Try looking in common image directories
    for parent in [adoc_dir, adoc_dir.parent, adoc_dir.parent.parent]:
        candidate = parent / "images" / Path(image_ref).name
        if candidate.exists():
            return candidate
            
    return None


def _copy_image_to_project(
    source_path: Path,
    image_id: str,
) -> Optional[str]:
    """
    Copy an image to data/images/raspberry_pi/.
    Implements MD5-based duplicate image detection.
    """
    global _image_md5_registry
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check duplicate using MD5
        img_bytes = source_path.read_bytes()
        img_md5 = hashlib.md5(img_bytes).hexdigest()
        
        if img_md5 in _image_md5_registry:
            return _image_md5_registry[img_md5]
            
        suffix = source_path.suffix.lower()
        target_name = f"{image_id}.png"
        target_path = IMAGES_DIR / target_name
        
        if suffix in ('.png',):
            shutil.copy2(source_path, target_path)
        elif suffix in ('.jpg', '.jpeg', '.gif', '.webp'):
            from PIL import Image
            img = Image.open(source_path)
            if img.mode in ('CMYK', 'YCbCr'):
                img = img.convert('RGB')
            elif img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
            img.save(target_path, 'PNG')
        elif suffix == '.svg':
            # Skip SVGs
            return None
        else:
            shutil.copy2(source_path, target_path)
            
        relative_path = str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        _image_md5_registry[img_md5] = relative_path
        return relative_path
        
    except Exception as e:
        print(f"  [WARN] Failed to copy image {source_path.name}: {e}")
        return None


def parse_adoc_file(adoc_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Parse a single AsciiDoc file, extracts rich text blocks and copies unique images."""
    try:
        content = adoc_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  [WARN] Cannot read {adoc_path.name}: {e}")
        return [], []
    
    text_blocks = []
    images = []
    
    try:
        rel_path = adoc_path.relative_to(RPI_DOCS_DIR)
    except ValueError:
        rel_path = adoc_path.relative_to(DATA_DIR)
    source_name = f"RPi Docs - {str(rel_path).replace(os.sep, '/')}"
    
    lines = content.split('\n')
    current_h1 = adoc_path.stem.replace('-', ' ').replace('_', ' ').title()
    current_h2 = ""
    current_h3 = ""
    
    current_text = []
    block_index = 0
    in_code_block = False
    page_num = 1
    img_index = 0
    
    # Keep track of where images are positioned relative to text blocks
    # We will temporarily store placeholder markers in the current text stream
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        
        if INCLUDE_PATTERN.match(stripped):
            continue
            
        if CODE_BLOCK_START.match(stripped):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            continue
            
        # 1. Heading mapping H1 / H2 / H3
        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            # Flush existing block
            if current_text:
                merged_text = '\n'.join(current_text).strip()
                cleaned = clean_text(merged_text)
                if cleaned:
                    text_blocks.append({
                        "text": cleaned,
                        "bbox": (0, 0, 0, 0),
                        "page_number": page_num,
                        "source_document": source_name,
                        "section": current_h1,
                        "subsection": current_h2,
                        "subsubsection": current_h3,
                        "block_index": block_index,
                        "is_heading": False,
                    })
                    block_index += 1
                current_text = []
            
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            
            if level == 2:  # == Heading -> H1
                current_h1 = heading_text
                current_h2 = ""
                current_h3 = ""
            elif level == 3:  # === Heading -> H2
                current_h2 = heading_text
                current_h3 = ""
            else:  # ==== Heading -> H3
                current_h3 = heading_text
                
            page_num = block_index + 1
            
            # Save heading itself as a semantic block
            text_blocks.append({
                "text": heading_text,
                "bbox": (0, 0, 0, 0),
                "page_number": page_num,
                "source_document": source_name,
                "section": current_h1,
                "subsection": current_h2,
                "subsubsection": current_h3,
                "block_index": block_index,
                "is_heading": True,
            })
            block_index += 1
            continue
            
        # 2. Image references
        img_found = False
        for pattern in [IMAGE_BLOCK_PATTERN, IMAGE_INLINE_PATTERN]:
            img_match = pattern.search(stripped)
            if img_match:
                image_ref = img_match.group(1).strip()
                image_attrs = img_match.group(2)
                alt_text = _extract_alt_text(image_attrs)
                
                real_path = _resolve_image_path(image_ref, adoc_path)
                if real_path and real_path.exists():
                    clean_section = re.sub(r'[^\w]', '_', current_h1).lower()[:20]
                    image_id = f"rpi_{clean_section}_img{img_index:03d}"
                    
                    project_path = _copy_image_to_project(real_path, image_id)
                    if project_path:
                        images.append({
                            "image_id": image_id,
                            "image_path": project_path,
                            "bbox": (0, 0, 0, 0),
                            "page_number": page_num,
                            "source_document": source_name,
                            "caption": alt_text or current_h1,
                            "section": current_h1,
                            "subsection": current_h2,
                            "subsubsection": current_h3,
                            "width": 0,
                            "height": 0,
                        })
                        img_index += 1
                        
                        # Add a temporary marker to associate with surrounding text block
                        current_text.append(f"__IMAGE_REF_MARKER_{image_id}__")
                        img_found = True
                        
        if img_found:
            continue
            
        if not stripped:
            if current_text:
                current_text.append('')
            continue
            
        if stripped.startswith(':') or stripped.startswith('[[') or stripped.startswith('ifdef::'):
            continue
            
        admon_match = ADMONITION_PATTERN.match(stripped)
        if admon_match:
            stripped = f"{admon_match.group(1)}: {admon_match.group(2)}"
            
        # Clean AsciiDoc tags
        stripped = re.sub(r'xref:[^\[]*\[([^\]]*)\]', r'\1', stripped)
        stripped = re.sub(r'https?://[^\[]*\[([^\]]*)\]', r'\1', stripped)
        stripped = stripped.replace('`', '').replace('*', '').replace('_', ' ')
        
        if stripped:
            current_text.append(stripped)
            
    # Save final text block
    if current_text:
        merged_text = '\n'.join(current_text).strip()
        cleaned = clean_text(merged_text)
        if cleaned:
            text_blocks.append({
                "text": cleaned,
                "bbox": (0, 0, 0, 0),
                "page_number": page_num,
                "source_document": source_name,
                "section": current_h1,
                "subsection": current_h2,
                "subsubsection": current_h3,
                "block_index": block_index,
                "is_heading": False,
            })
            
    # Context-aware horizontal text-image association
    # We resolve image markers placed inside text blocks and create the linkages
    for image in images:
        marker = f"__IMAGE_REF_MARKER_{image['image_id']}__"
        associated = False
        
        # Check text blocks for marker
        for block in text_blocks:
            if marker in block["text"]:
                # Clean marker out of the block's text
                block["text"] = block["text"].replace(marker, "").strip()
                block["associated_image_path"] = image["image_path"]
                
                # Update caption with nearest section details
                image["caption"] = f"{image['caption'] or block['section']}: {block['text'][:150]}"
                associated = True
                break
                
        if not associated:
            # Fallback to the last text block in the same section
            matching = [b for b in text_blocks if b["section"] == image["section"]]
            if matching:
                matching[-1]["associated_image_path"] = image["image_path"]
                
    # Clean any leftover markers in text blocks
    for block in text_blocks:
        block["text"] = re.sub(r'__IMAGE_REF_MARKER_[a-zA-Z0-9_]+__', '', block["text"]).strip()
        
    return text_blocks, images


def parse_rpi_docs(docs_dir: Optional[str] = None) -> List[Dict]:
    """Parse all priority AsciiDoc files from the Raspberry Pi documentation repo."""
    if docs_dir is None:
        docs_dir = RPI_DOCS_DIR
    else:
        docs_dir = Path(docs_dir)
        
    if not docs_dir.exists():
        print(f"[WARN] RPi docs directory not found: {docs_dir}")
        return []
        
    adoc_files = []
    for priority_dir in PRIORITY_DIRS:
        full_dir = docs_dir / priority_dir
        if full_dir.exists():
            for adoc_file in sorted(full_dir.rglob("*.adoc")):
                content = adoc_file.read_text(encoding='utf-8', errors='replace')
                non_include = [
                    l for l in content.strip().split('\n')
                    if l.strip() and not l.strip().startswith('include::')
                ]
                if len(non_include) > 2:
                    adoc_files.append(adoc_file)
                    
    results = []
    for adoc_file in adoc_files:
        text_blocks, images = parse_adoc_file(adoc_file)
        if text_blocks or images:
            try:
                rel_path = adoc_file.relative_to(RPI_DOCS_DIR)
            except ValueError:
                rel_path = adoc_file.name
            source_name = f"RPi Docs - {str(rel_path).replace(os.sep, '/')}"
            
            # Filter empty blocks
            text_blocks = [b for b in text_blocks if len(b["text"].strip()) >= 5]
            
            result = {
                "source_document": source_name,
                "category": "raspberry_pi",
                "text_blocks": text_blocks,
                "images": images,
                "stats": {
                    "page_count": max(b.get("page_number", 1) for b in text_blocks) if text_blocks else 0,
                    "text_block_count": len(text_blocks),
                    "image_count": len(images),
                },
            }
            results.append(result)
            
    return results
