"""
RyzenAI-LocalLab - FastAPI Application Entry Point

Main application with all routes, middleware, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api import auth_routes, chat_routes, model_routes, openai_routes, system_routes
from backend.config import settings
from backend.database import init_db


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
    logging.getLogger("transformers").setLevel(logging.WARNING)


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
    logger.info("RyzenAI-LocalLab Starting...")
    logger.info("=" * 60)

    # Initialize database
    await init_db()
    logger.info("✓ Database initialized")

    # Show configuration
    logger.info(f"Models path: {settings.models_path}")
    logger.info(f"Device preference: {settings.device}")
    logger.info(f"API: http://{settings.api_host}:{settings.api_port}")

    # Check hardware
    from backend.services.hardware_monitor import hardware_monitor

    gpu_stats = hardware_monitor.get_gpu_stats()
    if gpu_stats.available:
        logger.info(f"✓ GPU detected: {gpu_stats.name}")
    else:
        logger.warning("⚠ No GPU detected, running on CPU")

    logger.info("=" * 60)
    logger.info("RyzenAI-LocalLab Ready!")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("RyzenAI-LocalLab Shutting down...")

    # Unload model if loaded
    from backend.services.inference_engine import inference_engine

    if inference_engine.is_loaded():
        await inference_engine.unload_model()
        logger.info("✓ Model unloaded")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="RyzenAI-LocalLab",
    description="Interface d'inférence HomeLab pour AMD Ryzen AI MAX+",
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
app.include_router(auth_routes.router)
app.include_router(model_routes.router)
app.include_router(chat_routes.router)
app.include_router(openai_routes.router)
app.include_router(system_routes.router)


# =============================================================================
# Root Endpoints
# =============================================================================
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api")
async def api_info():
    """API information."""
    return {
        "name": "RyzenAI-LocalLab API",
        "version": "0.1.0",
        "docs": "/docs",
        "openai_compatible": "/v1",
    }


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
