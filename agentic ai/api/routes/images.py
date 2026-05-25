"""
Image Serving Endpoint — GET /images/{filepath:path}

Serves PNG files from data/images/ with subdirectory support:
  /images/arduino/a000066_datasheet_p006_img000.png
  /images/raspberry_pi/gpio_pinout.png

Security:
  - Path traversal protection (resolved path must stay within IMAGES_DIR)
  - Only serves actual files (resolved.is_file() check)
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# --- Config ------------------------------------------------------------------
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

router = APIRouter(tags=["images"])


# =============================================================================
#  GET /images/{filepath:path}
# =============================================================================

@router.get("/images/{filepath:path}")
async def get_image(filepath: str):
    """
    Serve a PNG image from data/images/.

    Args:
        filepath: Relative path within data/images/,
                  e.g. "arduino/a000066_datasheet_p006_img000.png"

    Returns:
        FileResponse with media_type="image/png"

    Raises:
        400: Path traversal attempt detected
        404: Image file not found
    """
    resolved = (IMAGES_DIR / filepath).resolve()

    # Security: ensure resolved path is still within IMAGES_DIR
    if not str(resolved).startswith(str(IMAGES_DIR.resolve())):
        log.warning(f"[IMAGES] Path traversal attempt blocked: {filepath}")
        raise HTTPException(status_code=400, detail="Invalid image path")

    # Must be an actual file on disk
    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {filepath}",
        )

    return FileResponse(resolved, media_type="image/png")
