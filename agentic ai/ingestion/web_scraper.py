"""
Web Scraper — Scrapes Arduino and Raspberry Pi documentation websites.

Uses BeautifulSoup to extract:
  - Structured text from article/main content tags
  - Image URLs, downloaded and saved as PNG files

Targets:
  - Arduino Official Docs: docs.arduino.cc
  - Raspberry Pi Documentation: raspberrypi.com/documentation
  - TutorialsPoint Arduino: tutorialspoint.com/arduino

Output:
  - Scraped text stored as list of dicts (same format as pdf_parser)
  - Images saved to data/images/{category}/
  - Raw HTML saved to data/raw_html/ for reproducibility
"""

import os
import re
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


# ─── Project root path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
IMAGES_DIR = DATA_DIR / "images"

# ─── Request configuration ──────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_TIMEOUT = 30  # seconds
DELAY_BETWEEN_REQUESTS = 2  # seconds — be polite to servers


# ─── Target URLs ─────────────────────────────────────────────────────
ARDUINO_URLS = [
    {
        "url": "https://docs.arduino.cc/hardware/uno-rev3/",
        "name": "Arduino UNO R3 Docs",
        "category": "arduino",
    },
    {
        "url": "https://docs.arduino.cc/learn/starting-guide/getting-started-arduino/",
        "name": "Arduino Getting Started",
        "category": "arduino",
    },
    {
        "url": "https://docs.arduino.cc/built-in-examples/",
        "name": "Arduino Built-in Examples",
        "category": "arduino",
    },
    {
        "url": "https://docs.arduino.cc/learn/electronics/lcd-displays/",
        "name": "Arduino LCD Displays",
        "category": "arduino",
    },
    {
        "url": "https://docs.arduino.cc/learn/communication/wire/",
        "name": "Arduino I2C Wire Library",
        "category": "arduino",
    },
]

RASPBERRY_PI_URLS = [
    {
        "url": "https://www.raspberrypi.com/documentation/computers/raspberry-pi.html",
        "name": "RPi Hardware Docs",
        "category": "raspberry_pi",
    },
    {
        "url": "https://www.raspberrypi.com/documentation/computers/configuration.html",
        "name": "RPi Configuration",
        "category": "raspberry_pi",
    },
    {
        "url": "https://www.raspberrypi.com/documentation/computers/os.html",
        "name": "RPi OS Docs",
        "category": "raspberry_pi",
    },
    {
        "url": "https://www.raspberrypi.com/documentation/computers/getting-started.html",
        "name": "RPi Getting Started",
        "category": "raspberry_pi",
    },
]

ALL_URLS = ARDUINO_URLS + RASPBERRY_PI_URLS


def _fetch_page(url: str) -> Optional[str]:
    """Fetch a web page and return the HTML content."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}")
        return None


def _save_raw_html(html: str, name: str) -> Path:
    """Save raw HTML to data/raw_html/ for reproducibility."""
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r'[^\w\-]', '_', name).lower()
    filepath = RAW_HTML_DIR / f"{clean_name}.html"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath


def _download_image(img_url: str, save_dir: Path, image_id: str) -> Optional[str]:
    """Download an image and save as PNG. Returns relative path or None on failure."""
    try:
        response = requests.get(img_url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type and not img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')):
            return None
        
        # Determine file extension
        ext = "png"
        if img_url.lower().endswith('.svg'):
            # Skip SVG files — we want raster images for CLIP embeddings
            return None
        
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{image_id}.{ext}"
        filepath = save_dir / filename
        
        # Save the image
        if img_url.lower().endswith(('.jpg', '.jpeg', '.gif', '.webp')):
            # Convert to PNG using Pillow
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(response.content))
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGBA')
            else:
                img = img.convert('RGB')
            img.save(filepath, "PNG")
        else:
            with open(filepath, "wb") as f:
                f.write(response.content)
        
        relative_path = str(filepath.relative_to(PROJECT_ROOT)).replace("\\", "/")
        return relative_path
        
    except Exception as e:
        print(f"  [WARN] Failed to download image {img_url}: {e}")
        return None


def _extract_content_from_html(
    html: str,
    url: str,
    source_name: str,
    category: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract structured text blocks and image metadata from HTML.
    
    Returns:
        (text_blocks, images) — same format as pdf_parser output
    """
    soup = BeautifulSoup(html, "html.parser")
    text_blocks = []
    images = []
    
    # Find the main content area (try multiple common selectors)
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="content")
        or soup.find("div", {"id": "content"})
        or soup.find("div", class_="documentation-content")
        or soup.find("div", class_="post-content")
        or soup.body
    )
    
    if main_content is None:
        print(f"  [WARN] No main content found in {url}")
        return text_blocks, images
    
    # ─── Extract text blocks ─────────────────────────────────
    # Walk through headings and paragraphs to maintain section structure
    current_section = source_name
    block_index = 0
    
    for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'pre', 'code', 'table']):
        tag = element.name
        
        # Update current section name from headings
        if tag in ['h1', 'h2', 'h3', 'h4']:
            heading_text = element.get_text(strip=True)
            if heading_text:
                current_section = heading_text
                # Also add the heading itself as a text block
                text_blocks.append({
                    "text": heading_text,
                    "bbox": (0, 0, 0, 0),  # No bounding box for web content
                    "page_number": 1,  # Web pages treated as single page
                    "source_document": source_name,
                    "section_name": current_section,
                    "source_url": url,
                    "block_index": block_index,
                    "tag": tag,
                })
                block_index += 1
            continue
        
        # Extract text from content elements
        text = element.get_text(strip=True)
        if not text or len(text) < 10:
            continue
        
        # For code blocks, preserve formatting
        if tag in ['pre', 'code']:
            text = element.get_text()  # Keep whitespace for code
        
        text_blocks.append({
            "text": text,
            "bbox": (0, 0, 0, 0),
            "page_number": 1,
            "source_document": source_name,
            "section_name": current_section,
            "source_url": url,
            "block_index": block_index,
            "tag": tag,
        })
        block_index += 1
    
    # ─── Extract images ──────────────────────────────────────
    img_dir = IMAGES_DIR / category
    img_tags = main_content.find_all("img")
    
    for img_index, img_tag in enumerate(img_tags):
        src = img_tag.get("src") or img_tag.get("data-src")
        if not src:
            continue
        
        # Build absolute URL
        abs_url = urljoin(url, src)
        
        # Skip tiny icons and tracking pixels
        width = img_tag.get("width")
        height = img_tag.get("height")
        if width and height:
            try:
                if int(width) < 50 or int(height) < 50:
                    continue
            except ValueError:
                pass
        
        # Generate image ID
        clean_name = re.sub(r'[^\w]', '_', source_name).lower()[:25]
        image_id = f"web_{clean_name}_img{img_index:03d}"
        
        # Download the image
        img_path = _download_image(abs_url, img_dir, image_id)
        if img_path is None:
            continue
        
        # Extract alt text / caption
        caption = (
            img_tag.get("alt", "")
            or img_tag.get("title", "")
            or ""
        ).strip()
        
        # If no alt text, try to find caption in nearby figcaption or paragraph
        if not caption:
            parent = img_tag.parent
            if parent and parent.name == "figure":
                figcaption = parent.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text(strip=True)
        
        images.append({
            "image_id": image_id,
            "image_path": img_path,
            "bbox": (0, 0, 0, 0),
            "page_number": 1,
            "source_document": source_name,
            "source_url": abs_url,
            "caption": caption[:500] if caption else "",
            "section_name": current_section,
            "width": int(width) if width and width.isdigit() else 0,
            "height": int(height) if height and height.isdigit() else 0,
        })
    
    return text_blocks, images


def scrape_single_url(url_config: Dict) -> Optional[Dict]:
    """
    Scrape a single URL and extract text blocks + images.
    
    Args:
        url_config: Dict with keys 'url', 'name', 'category'
    
    Returns:
        Dict with same structure as pdf_parser.parse_single_pdf output
    """
    url = url_config["url"]
    name = url_config["name"]
    category = url_config["category"]
    
    print(f"\n  Scraping: {name}")
    print(f"  URL: {url}")
    
    # Fetch the page
    html = _fetch_page(url)
    if html is None:
        return None
    
    # Save raw HTML
    _save_raw_html(html, name)
    
    # Extract content
    text_blocks, images = _extract_content_from_html(html, url, name, category)
    
    # Associate images with nearest text blocks (simplified for web content)
    # For web content, we associate images with the text block immediately before them
    for image in images:
        section = image.get("section_name", "")
        matching_blocks = [
            b for b in text_blocks
            if b.get("section_name") == section
        ]
        if matching_blocks:
            # Link the last text block in this section to this image
            matching_blocks[-1]["associated_image_path"] = image["image_path"]
    
    stats = {
        "page_count": 1,
        "text_block_count": len(text_blocks),
        "image_count": len(images),
    }
    
    print(f"    Text blocks: {stats['text_block_count']}")
    print(f"    Images:      {stats['image_count']}")
    
    return {
        "source_document": name,
        "source_url": url,
        "category": category,
        "text_blocks": text_blocks,
        "images": images,
        "stats": stats,
    }


def scrape_all_urls(urls: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Scrape all configured URLs.
    
    Args:
        urls: List of URL configs. Defaults to ALL_URLS.
    
    Returns:
        List of scrape results.
    """
    if urls is None:
        urls = ALL_URLS
    
    print(f"\n{'='*60}")
    print(f"WEB SCRAPER — Scraping {len(urls)} URL(s)")
    print(f"{'='*60}")
    
    results = []
    for i, url_config in enumerate(urls):
        try:
            result = scrape_single_url(url_config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed to scrape {url_config['name']}: {e}")
        
        # Polite delay between requests
        if i < len(urls) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Print summary
    total_blocks = sum(r["stats"]["text_block_count"] for r in results)
    total_images = sum(r["stats"]["image_count"] for r in results)
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"  URLs scraped:      {len(results)}/{len(urls)}")
    print(f"  Total text blocks: {total_blocks}")
    print(f"  Total images:      {total_images}")
    print(f"{'='*60}")
    
    return results


# ─── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Scrape a specific URL
        url = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "Custom Page"
        category = sys.argv[3] if len(sys.argv) > 3 else "arduino"
        results = [scrape_single_url({"url": url, "name": name, "category": category})]
        results = [r for r in results if r is not None]
    else:
        results = scrape_all_urls()
    
    if results:
        summary_path = DATA_DIR / "chunks" / "_scrape_summary.json"
        summary = {
            "total_urls": len(results),
            "documents": [
                {
                    "source": r["source_document"],
                    "url": r.get("source_url", ""),
                    "category": r["category"],
                    "stats": r["stats"],
                }
                for r in results
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
