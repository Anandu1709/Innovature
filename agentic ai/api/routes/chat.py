"""
Chat Endpoint — POST /chat

Wires the FastAPI request to the LangGraph agent graph:
  1. Parse ChatRequest (query, session_id)
  2. Fetch session history
  3. Invoke compiled graph via asyncio.to_thread()
  4. Transform AgentState → ChatResponse (Section 7.4 contract)
  5. Update session store
  6. Return response
"""

import sys
import uuid
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Add project root to path for agent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.models import ChatRequest, ChatResponse
from api import session
from agents.state import create_initial_state

# --- Config ------------------------------------------------------------------
log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


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
    )


# =============================================================================
#  POST /chat
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user query through the full LangGraph agent pipeline.

    Request:
        {"query": "...", "session_id": "..."}

    Response:
        Section 7.4 ChatResponse JSON.
    """
    # 1. Session ID — use provided or generate new
    session_id = request.session_id or str(uuid.uuid4())

    # 2. Fetch conversation history for multi-turn context
    history = session.get_history(session_id)

    log.info(
        f"[CHAT] Query: '{request.query}' | "
        f"session: {session_id} | history: {len(history)} turns"
    )

    # 3. Build initial state for the graph
    initial_state = create_initial_state(
        query=request.query,
        session_id=session_id,
        session_history=history,
    )

    # 4. Invoke graph — sync call, offload to thread to avoid blocking event loop
    try:
        # Import the pre-compiled graph from main.py
        from api.main import compiled_graph

        if compiled_graph is None:
            raise RuntimeError("Agent graph not initialized — server still starting up")

        final_state = await asyncio.to_thread(
            compiled_graph.invoke, initial_state
        )
    except RuntimeError as e:
        log.error(f"[CHAT] Graph not ready: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")
    except Exception as e:
        log.exception(f"[CHAT] Agent processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(e)}",
        )

    # 5. Transform state → response
    response = _state_to_response(final_state, session_id)

    # 6. Update session history
    session.append_turn(session_id, "user", request.query)
    session.append_turn(session_id, "assistant", response.answer or "")

    log.info(
        f"[CHAT] Response: intent={response.intent} | "
        f"answer_len={len(response.answer or '')} | "
        f"image={'yes' if response.image_url else 'no'} | "
        f"sources={len(response.sources)}"
    )

    return response
