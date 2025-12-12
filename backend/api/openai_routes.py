"""
RyzenAI-LocalLab OpenAI-Compatible API

Provides OpenAI-compatible endpoints for external tool integration.
Compatible with LangChain, Continue.dev, and other OpenAI SDK clients.
"""

import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.inference_engine import GenerationParams, inference_engine
from backend.services.model_manager import model_manager

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
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)


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
    local_models = model_manager.list_local_models()

    models = [
        ModelInfo(
            id=m.id,
            object="model",
            created=int(m.download_date.timestamp()) if m.download_date else int(time.time()),
            owned_by="local",
        )
        for m in local_models
    ]

    # Add currently loaded model as the first if loaded
    if inference_engine.is_loaded() and inference_engine.loaded_model_info:
        loaded_id = inference_engine.loaded_model_info.model_id
        models.insert(
            0,
            ModelInfo(
                id=f"{loaded_id} (loaded)",
                object="model",
                created=int(inference_engine.loaded_model_info.loaded_at),
                owned_by="local-loaded",
            ),
        )

    return ModelListResponse(object="list", data=models)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a chat completion.

    Compatible with OpenAI /v1/chat/completions endpoint.
    """
    # Check if model is loaded
    if not inference_engine.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Use the /api/chat/engine/load endpoint first.",
        )

    # Convert messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Extract system prompt if present
    system_prompt = None
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages = messages[1:]

    # Generation parameters
    params = GenerationParams(
        max_new_tokens=request.max_tokens or 2048,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=1.0 + request.frequency_penalty,
        stop_sequences=request.stop or [],
    )

    # Generate unique ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    model_id = inference_engine.loaded_model_info.model_id if inference_engine.loaded_model_info else "unknown"

    if request.stream:
        # Streaming response
        async def generate():
            import json

            full_response = ""
            final_stats = None

            async for token, stats in inference_engine.generate_stream(
                messages=messages,
                params=params,
                system_prompt=system_prompt,
            ):
                full_response += token
                if stats:
                    final_stats = stats

                if token:
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_id,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": token},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason
            final_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created,
                model=model_id,
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
        full_response, stats = await inference_engine.generate(
            messages=messages,
            params=params,
            system_prompt=system_prompt,
        )

        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_response),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=stats.prompt_tokens if stats else 0,
                completion_tokens=stats.completion_tokens if stats else 0,
                total_tokens=stats.total_tokens if stats else 0,
            ),
        )


# =============================================================================
# Health Check
# =============================================================================
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": inference_engine.is_loaded(),
        "device": inference_engine.device,
    }
