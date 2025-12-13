"""
RyzenAI-LocalLab Unified Engine Routes

API endpoints for the unified model manager.
Ensures only ONE model is loaded at a time across all backends.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.unified_engine import unified_engine

router = APIRouter(prefix="/api/engine", tags=["Unified Engine"])


# =============================================================================
# Schemas
# =============================================================================
class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_id: str = Field(..., description="Model ID with prefix: ollama:model, hf:repo/model, gguf:/path")


class EngineStatusResponse(BaseModel):
    """Current engine status."""
    backend: str
    model_id: str
    model_name: str
    status: str
    loaded_at: float
    error: Optional[str] = None


# =============================================================================
# Routes
# =============================================================================
@router.get("/status", response_model=EngineStatusResponse)
async def get_engine_status(current_user: User = Depends(get_current_active_user)):
    """
    Get the current unified engine status.
    
    Returns which backend is active and which model is loaded.
    """
    return unified_engine.get_status()


@router.post("/load")
async def load_model(request: LoadModelRequest, current_user: User = Depends(get_current_active_user)):
    """
    Load a model with streaming progress updates.
    
    Automatically unloads any previously loaded model first.
    
    Model ID formats:
    - ollama:qwen3:8b
    - hf:mistralai/Devstral
    - gguf:/path/to/model.gguf
    
    Returns Server-Sent Events with progress updates.
    """
    async def generate():
        import json
        async for progress in unified_engine.load_model(request.model_id):
            yield f"data: {json.dumps({
                'status': progress.status,
                'message': progress.message,
                'progress': progress.progress,
                'step': progress.step,
            })}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/unload")
async def unload_model(current_user: User = Depends(get_current_active_user)):
    """
    Unload the currently loaded model with streaming progress.
    
    Frees GPU memory and resets engine state.
    """
    async def generate():
        import json
        async for progress in unified_engine.unload_all():
            yield f"data: {json.dumps({
                'status': progress.status,
                'message': progress.message,
                'progress': progress.progress,
                'step': progress.step,
            })}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/generate")
async def generate(
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    current_user: User = Depends(get_current_active_user),
):
    """
    Generate a response using the currently loaded model.
    
    Works with any backend (Ollama, HuggingFace, llama.cpp) transparently.
    """
    if not unified_engine.is_loaded():
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Please load a model first."
        )
    
    async def generate_stream():
        import json
        async for chunk in unified_engine.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
