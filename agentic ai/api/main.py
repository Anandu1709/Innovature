"""
FastAPI Application — Multimodal Electronics Assistant API.

Entry point: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Startup:
  - Eagerly loads all ML models (MiniLM, CLIP, BM25)
  - Connects to Milvus Lite and loads collections
  - Pre-compiles the LangGraph StateGraph

Endpoints:
  POST /chat              — Query the agent pipeline
  GET  /images/{filepath}  — Serve PNG diagram files
  GET  /health            — Rich status check
"""

import os
import sys
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- Project root setup ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import create_initial_state

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Pre-compiled graph (set during startup) ---------------------------------
compiled_graph = None

PRELOAD_QUERIES = []

# =============================================================================
#  App Initialization
# =============================================================================

app = FastAPI(
    title="Multimodal Electronics Assistant",
    description=(
        "AI-powered assistant for Arduino and Raspberry Pi documentation. "
        "Retrieves text explanations and technical diagrams simultaneously."
    ),
    version="1.0.0",
)

# --- CORS — permissive during development ------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
#  Startup — Eager Model Warmup
# =============================================================================

@app.on_event("startup")
async def warmup():
    """
    Eagerly load all ML models, indexes, and pre-compile the agent graph.

    This trades a slower server boot (~15-30s) for instant first-request
    latency. All resources are cached as module-level singletons in the
    agent modules, so this warmup benefits every subsequent request.
    """
    global compiled_graph

    log.info("[STARTUP] Warming up models and indexes...")

    try:
        # 1. Text embedding model (all-MiniLM-L6-v2, 384-dim)
        from agents.hybrid_retrieval import _get_text_model, _get_bm25
        _get_text_model()
        log.info("[STARTUP] Text embedding model loaded")

        # 2. BM25 keyword index (data/bm25_index.pkl)
        _get_bm25()
        log.info("[STARTUP] BM25 index loaded")

        # 3. Milvus Lite — centralized connection manager (text + image collections)
        from utils.milvus_manager import get_client, load_collection
        get_client()
        load_collection("text_collection")
        load_collection("image_collection")
        log.info("[STARTUP] Milvus text_collection and image_collection loaded")

        # 4. CLIP model (clip-ViT-B-32, 512-dim)
        from agents.visual_retrieval import _get_clip_model
        _get_clip_model()
        log.info("[STARTUP] CLIP model loaded")

        # 5. Pre-compile LangGraph StateGraph
        from agents.graph import build_graph
        compiled_graph = build_graph()
        log.info("[STARTUP] LangGraph agent graph compiled")

        # 6. Initialize SQLite document registry
        from admin.registry import init_db
        init_db()
        log.info("[STARTUP] SQLite document registry initialized")

    except Exception as e:
        log.exception(f"[STARTUP] Warmup failed: {e}")
        # Don't crash the server — let /health report the failure
        # and /chat will return 503 if graph is None

    log.info("[STARTUP] Warmup complete — starting cache preload...")

    # 7. Preload demo queries into cache (staggered to respect Gemini RPM limit)
    if compiled_graph is not None:
        import uuid
        import time as _time
        from api.routes.chat import _state_to_response
        from api import cache as _cache

        rpm_delay = 16  # seconds between queries (safe for 4 RPM limit)

        for i, query in enumerate(PRELOAD_QUERIES):
            try:
                # Respect RPM — wait between queries (skip delay before first)
                if i > 0:
                    log.info(f"[STARTUP] Waiting {rpm_delay}s for RPM limit...")
                    _time.sleep(rpm_delay)

                log.info(f"[STARTUP] Preloading ({i+1}/{len(PRELOAD_QUERIES)}): '{query}'")
                state = create_initial_state(
                    query=query,
                    session_id=str(uuid.uuid4()),
                    session_history=[],
                )
                final_state = compiled_graph.invoke(state)
                response = _state_to_response(final_state, "preload")
                if response.intent != "clarify":
                    _cache.put(query, response)
                    log.info(f"[STARTUP] ✓ Preloaded: '{query}'")
                else:
                    log.info(f"[STARTUP] ✗ Skipped (clarify intent): '{query}'")
            except Exception as e:
                log.warning(f"[STARTUP] ✗ Failed to preload '{query}': {e}")

        log.info(f"[STARTUP] Cache preload done — {_cache.stats()['size']} entries cached")
    else:
        log.warning("[STARTUP] Skipping cache preload — graph not available")


# =============================================================================
#  Health Check — Rich Status
# =============================================================================

@app.get("/health", tags=["health"])
async def health():
    """
    Rich health check reporting component status.

    Returns:
        {"status": "ok", "milvus": ..., "gemini": ..., "cache": {...}}
    """
    from api import cache as _cache

    checks = {
        "status": "ok",
        "milvus": "disconnected",
        "gemini": "disconnected",
        "cache": _cache.stats(),
    }

    # Check Milvus connection
    try:
        from agents.hybrid_retrieval import _milvus_client
        if _milvus_client is not None:
            checks["milvus"] = "connected"
    except Exception:
        pass

    # Check Gemini API key availability
    try:
        if os.getenv("GOOGLE_API_KEY"):
            checks["gemini"] = "connected"
    except Exception:
        pass

    return checks


@app.post("/cache/clear", tags=["admin"])
async def clear_cache():
    """
    Clear the response cache. Protected by DEBUG env var.

    Returns:
        {"cleared": <count>}
    """
    debug_mode = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    if not debug_mode:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="Cache clear is only available in DEBUG mode. Set DEBUG=true in .env",
        )

    from api import cache as _cache
    count = _cache.clear()
    return {"cleared": count}


# =============================================================================
#  Router Registration
# =============================================================================

from api.routes.chat import router as chat_router
from api.routes.images import router as images_router
from api.routes.admin import router as admin_router

app.include_router(chat_router)
app.include_router(images_router)
app.include_router(admin_router)

# Trigger reload comment v2
