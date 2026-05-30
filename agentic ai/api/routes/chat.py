"""
Chat Endpoint — POST /chat

Wires the FastAPI request to the LangGraph agent graph:
  1. Parse multipart/form-data (query, session_id, optional image)
  2. Save image to temp file if present
  3. Check response cache (skip pipeline on HIT — text-only requests)
  4. Fetch session history
  5. Invoke compiled graph via asyncio.to_thread()
  6. Transform AgentState → ChatResponse (Section 7.4 contract)
  7. Store response in cache (text-only requests)
  8. Update session store (with image context if applicable)
  9. Cleanup temp file
  10. Return response

Image Upload Upgrade:
  - Accepts multipart/form-data instead of JSON
  - Supports: text-only, image-only, image+text
  - Images saved to temp file → Vision Agent reads → temp file deleted
  - Max image size: 5 MB
  - Allowed formats: JPG, JPEG, PNG, WEBP
"""

import os
import sys
import uuid
import asyncio
import tempfile
import logging
from pathlib import Path
from copy import deepcopy

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

# Add project root to path for agent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.models import ChatResponse
from api import session
from api import cache
from agents.state import create_initial_state

# --- Config ------------------------------------------------------------------
log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Image upload constraints
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# =============================================================================
#  Helpers
# =============================================================================

def _path_to_url(image_path: str) -> str:
    """
    Convert a filesystem-relative image path to a serving URL.

    "data/images/arduino/file.png" → "/images/arduino/file.png"
    """
    return "/" + image_path.replace("data/", "", 1)


def _extract_top_image(image_results: list) -> tuple:
    """
    Pick the top-scoring image from retrieval results.

    Returns:
        (image_url, image_caption) or (None, None) if no images.
    """
    if not image_results:
        return None, None

    top = max(image_results, key=lambda x: x.get("score", 0))
    image_url = _path_to_url(top.get("image_path", ""))
    image_caption = top.get("caption", "")

    return image_url, image_caption


def _state_to_response(final_state: dict, session_id: str) -> ChatResponse:
    """
    Transform the LangGraph AgentState dict into a Section 7.4 ChatResponse.

    Mapping:
      intent == "clarify" → answer = clarification_question
      intent != "clarify" → answer = final_answer
      image_url           → top-scoring image from image_results
    """
    intent = final_state.get("intent", "text")

    # Answer: clarification question or synthesized answer
    if intent == "clarify":
        answer = final_state.get("clarification_question", "")
    else:
        answer = final_state.get("final_answer", "")

    # Top image
    image_url, image_caption = _extract_top_image(
        final_state.get("image_results", [])
    )

    # Sources
    sources = final_state.get("sources", [])

    return ChatResponse(
        session_id=session_id,
        intent=intent,
        answer=answer,
        image_url=image_url,
        image_caption=image_caption,
        sources=sources,
        coverage_found=final_state.get("coverage_found", True),
        offer_global_search=final_state.get("offer_global_search", False),
        global_search_requested=final_state.get("global_search_requested", False),
    )


def _validate_image(image: UploadFile) -> None:
    """
    Validate uploaded image file type and size.

    Raises HTTPException on invalid file.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file has no filename")

    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )


async def _save_temp_image(image: UploadFile) -> str:
    """
    Read uploaded image and save to a temporary file.

    Returns the temp file path. Caller is responsible for cleanup.
    Raises HTTPException if image exceeds size limit.
    """
    contents = await image.read()

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(contents) / 1024 / 1024:.1f} MB). Maximum: 5 MB",
        )

    suffix = Path(image.filename).suffix.lower() or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(PROJECT_ROOT)) as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    log.info(f"[CHAT] Image saved to temp file: {temp_path} ({len(contents)} bytes)")
    return temp_path


# =============================================================================
#  POST /chat
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: str = Form(""),
    session_id: str = Form(""),
    image: UploadFile | None = File(None),
    global_search_requested: bool = Form(False),
):
    """
    Process a user query through the full LangGraph agent pipeline.

    Accepts multipart/form-data with:
      - query:      Text question (optional if image provided)
      - session_id: Session identifier (optional, auto-generated if empty)
      - image:      Image file upload (optional, JPG/PNG/WEBP, max 5MB)

    Response:
        Section 7.4 ChatResponse JSON.
    """
    # 1. Session ID — use provided or generate new
    sid = session_id.strip() if session_id else ""
    sid = sid or str(uuid.uuid4())
    query_text = query.strip()

    # 2. Validate and save image to temp file
    temp_path = None
    has_image = image is not None and image.filename
    try:
        if has_image:
            _validate_image(image)
            temp_path = await _save_temp_image(image)

        # 3. Validate: must have query or image
        if not query_text and not has_image:
            raise HTTPException(
                status_code=400,
                detail="Either a text query or an image must be provided",
            )

        # 4. Fetch conversation history for multi-turn context
        history = session.get_history(sid)

        log.info(
            f"[CHAT] Query: '{query_text[:80]}' | "
            f"image: {'yes' if has_image else 'no'} | "
            f"session: {sid} | history: {len(history)} turns"
        )

        # 5. Cache check — only for text-only, first-turn standalone queries, skip if global search requested
        if not has_image and not history and query_text and not global_search_requested:
            cached = cache.get(query_text)
            if cached:
                cached.session_id = sid
                session.append_turn(sid, "user", query_text)
                session.append_turn(sid, "assistant", cached.answer or "")
                log.info(
                    f"[CHAT] Returning cached response | "
                    f"answer_len={len(cached.answer or '')} | "
                    f"image={'yes' if cached.image_url else 'no'}"
                )
                return cached

        # 6. Build initial state for the graph
        initial_state = create_initial_state(
            query=query_text,
            session_id=sid,
            session_history=history,
            image_path=temp_path,
        )
        initial_state["global_search_requested"] = global_search_requested

        # 7. Invoke graph — sync call, offload to thread
        from api.main import compiled_graph

        if compiled_graph is None:
            raise RuntimeError("Agent graph not initialized — server still starting up")

        final_state = await asyncio.to_thread(
            compiled_graph.invoke, initial_state
        )

        # 8. Transform state → response
        response = _state_to_response(final_state, sid)

        # 9. Cache store — only for text-only, first-turn, non-clarify, and when RAG coverage is found
        if (
            not has_image
            and response.intent != "clarify"
            and not history
            and query_text
            and response.coverage_found
            and not global_search_requested
        ):
            cache.put(query_text, response)

        # 10. Update session history (with image context if applicable)
        user_content = query_text
        if final_state.get("image_summary"):
            user_content += f"\n[Image: {final_state['image_summary']}]"
            components = final_state.get("components", [])
            if components:
                user_content += f"\n[Components: {', '.join(components)}]"
        session.append_turn(sid, "user", user_content)
        session.append_turn(sid, "assistant", response.answer or "")

        log.info(
            f"[CHAT] Response: intent={response.intent} | "
            f"answer_len={len(response.answer or '')} | "
            f"image={'yes' if response.image_url else 'no'} | "
            f"sources={len(response.sources)}"
        )

        return response

    except HTTPException:
        raise
    except RuntimeError as e:
        log.error(f"[CHAT] Graph not ready: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        log.exception(f"[CHAT] Agent processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}",
        )
    finally:
        # Always delete temp image file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            log.info(f"[CHAT] Temp image deleted: {temp_path}")
