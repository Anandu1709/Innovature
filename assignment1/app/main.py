"""
FastAPI application entry point.

Registers all routers and creates database tables on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import engine, Base
from app.routers import auth, csv_upload, csv_clean


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all database tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="CSV Manager API",
    description=(
        "A secure API for user authentication, CSV file upload with metadata extraction, "
        "and data cleaning operations. Built with FastAPI and JWT authentication."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Register Routers ─────────────────────────
app.include_router(auth.router)
app.include_router(csv_upload.router)
app.include_router(csv_clean.router)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "CSV Manager API is live. Visit /docs for interactive documentation.",
    }
