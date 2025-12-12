"""
RyzenAI-LocalLab OpenAI-Compatible API

Provides OpenAI-compatible endpoints for external tool integration.
Compatible with LangChain, Continue.dev, and other OpenAI SDK clients.
Ollama-only mode - no PyTorch required.
"""

import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.ollama_engine import ollama_engine

router = APIRouter(prefix="/v1", tags=["OpenAI-Compatible API"])


# =============================================================================
# OpenAI-Compatible Schemas
# =============================================================================
class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=32768)
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    """Streaming chat completion choice."""

    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    """Model information for /v1/models."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """Response for /v1/models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# Routes
# =============================================================================
@router.get("/models", response_model=ModelListResponse)
async def list_models(current_user: User = Depends(get_current_active_user)):
    """
    List available models.

    Compatible with OpenAI /v1/models endpoint.
    """
    ollama_models = await ollama_engine.list_models()

    models = [
        ModelInfo(
            id=m.name,
            object="model",
            created=int(time.time()),
            owned_by="ollama",
        )
        for m in ollama_models
    ]

    return ModelListResponse(object="list", data=models)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a chat completion.

    Compatible with OpenAI /v1/chat/completions endpoint.
    Uses Ollama for inference.
    """
    # Convert messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Determine model to use
    model = request.model
    if not model or model == "default":
        if ollama_engine.current_model:
            model = ollama_engine.current_model
        else:
            # Get first available model
            ollama_models = await ollama_engine.list_models()
            if ollama_models:
                model = ollama_models[0].name
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No models available. Pull a model first with: ollama pull qwen3:8b",
                )

    # Generate unique ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        # Streaming response
        async def generate():
            import json

            prompt_tokens = 0
            completion_tokens = 0

            async for chunk in ollama_engine.generate_stream(
                model=model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens or 2048,
            ):
                if chunk.get("error"):
                    yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                    return

                if chunk.get("content"):
                    response = ChatCompletionStreamResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": chunk["content"]},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {response.model_dump_json()}\n\n"

                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)

            # Final chunk with finish_reason
            final_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming response
        content, stats = await ollama_engine.generate(
            model=model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 2048,
        )

        prompt_tokens = stats.get("prompt_eval_count", 0)
        completion_tokens = stats.get("eval_count", 0)

        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


# =============================================================================
# Health Check
# =============================================================================
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    available = await ollama_engine.is_available()
    return {
        "status": "ok" if available else "degraded",
        "ollama_available": available,
        "current_model": ollama_engine.current_model,
    }
