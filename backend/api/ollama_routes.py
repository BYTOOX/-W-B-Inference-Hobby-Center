"""
RyzenAI-LocalLab Ollama Routes

API endpoints for Ollama integration.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.ollama_engine import ollama_engine

router = APIRouter(prefix="/api/ollama", tags=["Ollama"])


# =============================================================================
# Schemas
# =============================================================================
class OllamaModelResponse(BaseModel):
    name: str
    size_gb: float
    modified_at: str


class PullRequest(BaseModel):
    model: str = Field(..., description="Model name (e.g., 'qwen3:8b')")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True


# =============================================================================
# Routes
# =============================================================================
@router.get("/status")
async def get_ollama_status(current_user: User = Depends(get_current_active_user)):
    """Check if Ollama is available."""
    available = await ollama_engine.is_available()
    return {
        "available": available,
        "base_url": ollama_engine.base_url,
        "current_model": ollama_engine.current_model,
    }


@router.get("/models", response_model=list[OllamaModelResponse])
async def list_ollama_models(current_user: User = Depends(get_current_active_user)):
    """List all locally available Ollama models."""
    models = await ollama_engine.list_models()
    return [
        OllamaModelResponse(
            name=m.name,
            size_gb=round(m.size / (1024**3), 2),
            modified_at=m.modified_at,
        )
        for m in models
    ]


@router.post("/pull")
async def pull_model(request: PullRequest, current_user: User = Depends(get_current_active_user)):
    """
    Pull/download an Ollama model with streaming progress.
    
    Example models: qwen3:8b, llama3.1:8b, mistral:7b
    """
    async def generate():
        import json
        async for progress in ollama_engine.pull_model(request.model):
            yield f"data: {json.dumps(progress)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.delete("/models/{model_name:path}")
async def delete_ollama_model(model_name: str, current_user: User = Depends(get_current_active_user)):
    """Delete an Ollama model."""
    success = await ollama_engine.delete_model(model_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    return {"message": "Model deleted", "model": model_name}


@router.post("/load")
async def load_model(request: PullRequest, current_user: User = Depends(get_current_active_user)):
    """Load/warm up a model into memory."""
    result = await ollama_engine.load_model(request.model)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


@router.post("/chat")
async def chat(request: ChatRequest, current_user: User = Depends(get_current_active_user)):
    """
    Chat with an Ollama model.
    
    Supports streaming responses.
    """
    if request.stream:
        async def generate():
            import json
            async for chunk in ollama_engine.generate_stream(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        content, stats = await ollama_engine.generate(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        return {
            "content": content,
            "stats": stats,
        }
