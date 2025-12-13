"""
RyzenAI-LocalLab - FastAPI Application Entry Point

Main application with all routes, middleware, and lifecycle events.
Ollama-only mode - no PyTorch required.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db


# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# =============================================================================
# Logging Configuration
# =============================================================================
def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# =============================================================================
# Application Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RyzenAI-LocalLab Starting... (Ollama-only mode)")
    logger.info("=" * 60)

    # Initialize database
    await init_db()
    logger.info("✓ Database initialized")

    # Show configuration
    logger.info(f"Ollama host: {settings.ollama_host}")
    logger.info(f"API: http://{settings.api_host}:{settings.api_port}")

    # Check Ollama
    from backend.services.ollama_engine import ollama_engine
    
    available = await ollama_engine.is_available()
    if available:
        logger.info("✓ Ollama connected")
        models = await ollama_engine.list_models()
        logger.info(f"  Available models: {len(models)}")
        for m in models[:5]:
            logger.info(f"    - {m.name} ({m.size / 1e9:.1f} GB)")
    else:
        logger.warning("⚠ Ollama not available - check if it's running")

    logger.info("=" * 60)
    logger.info("RyzenAI-LocalLab Ready!")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("RyzenAI-LocalLab Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="RyzenAI-LocalLab",
    description="Interface d'inférence HomeLab pour AMD Ryzen AI MAX+ (Ollama-only)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# =============================================================================
# CORS Middleware
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Include Routers
# =============================================================================
from backend.api import auth_routes, chat_routes, model_routes, openai_routes, system_routes, ollama_routes, unified_routes

app.include_router(auth_routes.router)
app.include_router(model_routes.router)
app.include_router(chat_routes.router)
app.include_router(openai_routes.router)
app.include_router(system_routes.router)
app.include_router(ollama_routes.router)
app.include_router(unified_routes.router)


# =============================================================================
# Root Endpoints
# =============================================================================
@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend application."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api")
async def api_info():
    """API information."""
    return {
        "name": "RyzenAI-LocalLab API",
        "version": "0.1.0",
        "mode": "ollama-only",
        "docs": "/docs",
        "openai_compatible": "/v1",
    }


@app.get("/api/system/health")
async def health():
    """Health check for Docker."""
    from backend.services.ollama_engine import ollama_engine
    available = await ollama_engine.is_available()
    return {"status": "ok" if available else "degraded", "ollama": available}


# =============================================================================
# Run with Uvicorn
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
