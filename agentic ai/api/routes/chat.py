"""
Chat Endpoint — POST /chat

Wires the FastAPI request to the LangGraph agent graph:
  1. Parse multipart/form-data (query, session_id, optional image)
  2. Save image to temp file if present
  3. Detect follow-up vs standalone query
  4. Check versioned response cache (skip pipeline on HIT — standalone text-only)
  5. Fetch session history
  6. Invoke compiled graph via asyncio.to_thread()
  7. Transform AgentState → ChatResponse (Section 7.4 contract)
  8. Store response in versioned cache (standalone text-only with coverage)
  9. Update session store (with image context if applicable)
  10. Cleanup temp file
  11. Return response

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


def _reconstruct_state(initial_state: dict, output_list: list) -> dict:
    """
    Reconstruct the final state from a list of LangGraph stream updates.
    """
    state = deepcopy(initial_state)
    for item in output_list:
        if not isinstance(item, dict):
            continue
        
        # Check if the keys are node names (LangGraph stream format: {"node_name": {...}})
        is_node_format = False
        for k, v in item.items():
            if isinstance(v, dict) and k not in state:
                is_node_format = True
                break
                
        if is_node_format:
            for node_name, updates in item.items():
                if isinstance(updates, dict):
                    state.update(updates)
        else:
            state.update(item)
            
    return state


# --- Loading Status Variations ----------------------------------------------
NODE_STATUS_MAPPINGS = {
    "context_router": [
        "Understanding your question...",
        "Analyzing the request context...",
        "Determining the best processing path...",
    ],
    "hybrid_retrieval": [
        "Searching the knowledge base...",
        "Looking through technical documentation...",
        "Finding relevant information...",
        "Checking stored references...",
    ],
    "visual_retrieval": [
        "Searching for relevant technical diagrams...",
        "Checking schematics and illustrations...",
        "Retrieving visual references...",
    ],
    "synthesis": [
        "Preparing the answer...",
        "Combining retrieved information...",
        "Generating a response...",
    ],
    "general_response": [
        "Answering using general knowledge...",
        "Handling an out-of-scope query...",
        "Preparing a general response...",
    ],
    "vision_agent": [
        "Analyzing uploaded image...",
        "Identifying electronic components in image...",
        "Examining visual layout and connections...",
    ],
    "image_context_enrichment": [
        "Enriching search context with image details...",
        "Combining text query and visual observations...",
    ],
    "clarification": [
        "Formulating clarification question...",
        "Requesting more details about the setup...",
    ],
}


# =============================================================================
#  POST /chat
# =============================================================================

from fastapi import Request

@router.post("/chat")
async def chat(
    request: Request
):
    """
    Process a user query through the full LangGraph agent pipeline.

    Accepts:
      - JSON payload (for unit tests): {"query": "...", "session_id": "...", "global_search_requested": bool, "stream": bool}
      - multipart/form-data (for frontend): query, session_id, image (file), global_search_requested, stream

    Response:
        Section 7.4 ChatResponse JSON (or StreamingResponse if stream=True).
    """
    content_type = request.headers.get("content-type", "")
    
    query_text = ""
    session_id_val = ""
    image = None
    global_search_requested = False
    stream = False

    if "application/json" in content_type:
        body = await request.json()
        query_text = body.get("query", "") or ""
        session_id_val = body.get("session_id", "") or ""
        global_search_requested = bool(body.get("global_search_requested", False))
        stream = bool(body.get("stream", False))
    else:
        form = await request.form()
        query_text = form.get("query", "") or ""
        session_id_val = form.get("session_id", "") or ""
        
        gs_req = form.get("global_search_requested", "false")
        if isinstance(gs_req, str):
            global_search_requested = gs_req.lower() in ("true", "1", "yes")
        else:
            global_search_requested = bool(gs_req)
            
        stream_req = form.get("stream", "false")
        if isinstance(stream_req, str):
            stream = stream_req.lower() in ("true", "1", "yes")
        else:
            stream = bool(stream_req)

        img_field = form.get("image")
        if img_field is not None and not isinstance(img_field, str):
            image = img_field

    # 1. Session ID — use provided or generate new
    sid = session_id_val.strip() if session_id_val else ""
    sid = sid or str(uuid.uuid4())
    query_text = query_text.strip()

    # 2. Validate and save image to temp file
    temp_path = None
    has_image = image is not None and getattr(image, "filename", None)
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
            f"session: {sid} | history: {len(history)} turns | stream: {stream}"
        )

        # 5. Cache check — text-only standalone queries (works in any turn)
        if not has_image and query_text and not global_search_requested:
            if cache.is_standalone_query(query_text):
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
                    if stream:
                        # Stream cached response directly
                        import json
                        from fastapi.responses import StreamingResponse
                        async def cached_generator():
                            # Send initial loading status for UI consistency
                            yield f"data: {json.dumps({'type': 'status', 'text': 'Retrieving from cache...'})}\n\n"
                            await asyncio.sleep(0.2)
                            yield f"data: {json.dumps({'type': 'result', 'data': cached.dict()})}\n\n"
                        return StreamingResponse(cached_generator(), media_type="text/event-stream")
                    return cached

        # 6. Build initial state for the graph
        initial_state = create_initial_state(
            query=query_text,
            session_id=sid,
            session_history=history,
            image_path=temp_path,
        )
        initial_state["global_search_requested"] = global_search_requested

        # 7. Invoke graph
        from api.main import compiled_graph

        if compiled_graph is None:
            raise RuntimeError("Agent graph not initialized — server still starting up")

        if stream:
            from fastapi.responses import StreamingResponse
            import json
            import random

            # Capture temp path in a closure for cleanup
            captured_temp_path = temp_path
            # Set to None in parent scope so finally block won't delete prematurely
            temp_path = None

            async def event_generator():
                try:
                    active_nodes = set()
                    async for event in compiled_graph.astream_events(initial_state, version="v1"):
                        event_type = event.get("event")
                        node_name = event.get("name")

                        # Catch node start events
                        if (
                            event_type in ("on_chain_start", "on_node_start")
                            and node_name in NODE_STATUS_MAPPINGS
                            and node_name not in active_nodes
                        ):
                            active_nodes.add(node_name)
                            variants = NODE_STATUS_MAPPINGS[node_name]
                            status_text = random.choice(variants)
                            yield f"data: {json.dumps({'type': 'status', 'text': status_text})}\n\n"

                        # Catch graph execution end
                        elif event_type == "on_chain_end" and node_name == "LangGraph":
                            output_val = event["data"]["output"]
                            if isinstance(output_val, list):
                                final_state = _reconstruct_state(initial_state, output_val)
                            elif isinstance(output_val, dict):
                                final_state = output_val
                            else:
                                raise TypeError(
                                    f"Unexpected LangGraph output type: {type(output_val).__name__}. "
                                    f"Value snippet: {str(output_val)[:300]}"
                                )
                            response = _state_to_response(final_state, sid)

                            # Cache store — standalone, non-clarify queries with RAG coverage
                            if (
                                not has_image
                                and response.intent != "clarify"
                                and query_text
                                and response.coverage_found
                                and not global_search_requested
                                and cache.is_standalone_query(query_text)
                            ):
                                cache.put(query_text, response)

                            # Update session history (with image context if applicable)
                            user_content = query_text
                            if final_state.get("image_summary"):
                                user_content += f"\n[Image: {final_state['image_summary']}]"
                                components = final_state.get("components", [])
                                if components:
                                    user_content += f"\n[Components: {', '.join(components)}]"
                            session.append_turn(sid, "user", user_content)
                            session.append_turn(sid, "assistant", response.answer or "")

                            log.info(
                                f"[CHAT] Stream complete: intent={response.intent} | "
                                f"answer_len={len(response.answer or '')} | "
                                f"image={'yes' if response.image_url else 'no'} | "
                                f"sources={len(response.sources)}"
                            )

                            yield f"data: {json.dumps({'type': 'result', 'data': response.dict()})}\n\n"

                except Exception as e:
                    log.exception(f"[CHAT] Stream processing error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
                finally:
                    # Clean up temp image file inside stream worker
                    if captured_temp_path and os.path.exists(captured_temp_path):
                        try:
                            os.remove(captured_temp_path)
                            log.info(f"[CHAT] Temp image deleted inside stream worker: {captured_temp_path}")
                        except Exception as ce:
                            log.error(f"[CHAT] Failed to delete temp image in stream: {ce}")

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming fallback
        final_state = await asyncio.to_thread(
            compiled_graph.invoke, initial_state
        )

        # 8. Transform state → response
        response = _state_to_response(final_state, sid)

        # 9. Cache store — standalone, non-clarify queries with RAG coverage
        if (
            not has_image
            and response.intent != "clarify"
            and query_text
            and response.coverage_found
            and not global_search_requested
            and cache.is_standalone_query(query_text)
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
        # Always delete temp image file if not handled by streaming generator
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            log.info(f"[CHAT] Temp image deleted in finally: {temp_path}")
