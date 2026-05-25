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
        from agents.hybrid_retrieval import _get_text_model, _get_bm25, _get_milvus_client
        _get_text_model()
        log.info("[STARTUP] Text embedding model loaded")

        # 2. BM25 keyword index (data/bm25_index.pkl)
        _get_bm25()
        log.info("[STARTUP] BM25 index loaded")

        # 3. Milvus Lite (text_collection)
        _get_milvus_client()
        log.info("[STARTUP] Milvus text_collection loaded")

        # 4. CLIP model (clip-ViT-B-32, 512-dim) + image_collection
        from agents.visual_retrieval import _get_clip_model
        from agents.visual_retrieval import _get_milvus_client as _get_vis_milvus
        _get_clip_model()
        _get_vis_milvus()
        log.info("[STARTUP] CLIP model and image_collection loaded")

        # 5. Pre-compile LangGraph StateGraph
        from agents.graph import build_graph
        compiled_graph = build_graph()
        log.info("[STARTUP] LangGraph agent graph compiled")

    except Exception as e:
        log.exception(f"[STARTUP] Warmup failed: {e}")
        # Don't crash the server — let /health report the failure
        # and /chat will return 503 if graph is None

    log.info("[STARTUP] Warmup complete")


# =============================================================================
#  Health Check — Rich Status
# =============================================================================

@app.get("/health", tags=["health"])
async def health():
    """
    Rich health check reporting component status.

    Returns:
        {"status": "ok", "milvus": "connected"|"disconnected", "gemini": "connected"|"disconnected"}
    """
    checks = {
        "status": "ok",
        "milvus": "disconnected",
        "gemini": "disconnected",
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


# =============================================================================
#  Router Registration
# =============================================================================

from api.routes.chat import router as chat_router
from api.routes.images import router as images_router

app.include_router(chat_router)
app.include_router(images_router)
