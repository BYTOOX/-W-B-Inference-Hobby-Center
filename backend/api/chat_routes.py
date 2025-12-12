"""
RyzenAI-LocalLab Chat Routes

Endpoints for chat sessions and message handling.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.auth import get_current_active_user
from backend.database import ChatMessage, ChatSession, User, get_db
from backend.services.inference_engine import GenerationParams, inference_engine

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# =============================================================================
# Schemas
# =============================================================================
class ChatMessageCreate(BaseModel):
    """Request to send a chat message."""

    content: str = Field(..., min_length=1)
    session_id: Optional[int] = None

    # Generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class ChatMessageResponse(BaseModel):
    """Chat message response."""

    id: int
    role: str
    content: str
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    generation_time: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionCreate(BaseModel):
    """Request to create a new chat session."""

    title: str = Field(default="New Chat", max_length=255)
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Chat session response."""

    id: int
    title: str
    model_name: Optional[str]
    system_prompt: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class ChatSessionDetail(ChatSessionResponse):
    """Chat session with messages."""

    messages: list[ChatMessageResponse] = []


class ModelLoadRequest(BaseModel):
    """Request to load a model."""

    model_path: str
    quantization: Optional[str] = Field(default=None, description="'4bit', '8bit', or None")


class EngineStatusResponse(BaseModel):
    """Inference engine status."""

    loaded: bool
    device: str
    model: Optional[dict] = None


# =============================================================================
# Engine Routes
# =============================================================================
@router.get("/engine/status", response_model=EngineStatusResponse)
async def get_engine_status(current_user: User = Depends(get_current_active_user)):
    """Get the current inference engine status."""
    return inference_engine.get_status()


@router.post("/engine/load")
async def load_model(request: ModelLoadRequest, current_user: User = Depends(get_current_active_user)):
    """Load a model for inference."""
    try:
        info = await inference_engine.load_model(
            model_path=request.model_path,
            quantization=request.quantization,
        )
        return {
            "status": "loaded",
            "model_id": info.model_id,
            "device": info.device,
            "dtype": info.dtype,
            "memory_gb": round(info.memory_used_gb, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/engine/unload")
async def unload_model(current_user: User = Depends(get_current_active_user)):
    """Unload the current model."""
    await inference_engine.unload_model()
    return {"status": "unloaded"}


# =============================================================================
# Session Routes
# =============================================================================
@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all chat sessions for the current user."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .options(selectinload(ChatSession.messages))
    )
    sessions = result.scalars().all()

    return [
        ChatSessionResponse(
            id=s.id,
            title=s.title,
            model_name=s.model_name,
            system_prompt=s.system_prompt,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=len(s.messages),
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat session."""
    session = ChatSession(
        user_id=current_user.id,
        title=session_data.title,
        model_name=session_data.model_name,
        system_prompt=session_data.system_prompt,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        model_name=session.model_name,
        system_prompt=session.system_prompt,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a chat session with all messages."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .options(selectinload(ChatSession.messages))
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return ChatSessionDetail(
        id=session.id,
        title=session.title,
        model_name=session.model_name,
        system_prompt=session.system_prompt,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
        messages=[
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                tokens_prompt=m.tokens_prompt,
                tokens_completion=m.tokens_completion,
                generation_time=m.generation_time,
                created_at=m.created_at,
            )
            for m in session.messages
        ],
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session."""
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    await db.delete(session)
    await db.commit()

    return {"message": "Session deleted"}


# =============================================================================
# Message Routes
# =============================================================================
@router.post("/send")
async def send_message(
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and get a streaming response.
    """
    # Check if model is loaded
    if not inference_engine.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first.",
        )

    # Get or create session
    session = None
    if message.session_id:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.id == message.session_id, ChatSession.user_id == current_user.id)
            .options(selectinload(ChatSession.messages))
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    else:
        # Create new session
        session = ChatSession(
            user_id=current_user.id,
            title=message.content[:50] + "..." if len(message.content) > 50 else message.content,
            model_name=inference_engine.loaded_model_info.model_id if inference_engine.loaded_model_info else None,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        # Reload with messages
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session.id).options(selectinload(ChatSession.messages))
        )
        session = result.scalar_one()

    # Save user message
    user_message = ChatMessage(session_id=session.id, role="user", content=message.content)
    db.add(user_message)
    await db.commit()

    # Build message history
    history = [{"role": m.role, "content": m.content} for m in session.messages]
    history.append({"role": "user", "content": message.content})

    # Generation parameters
    params = GenerationParams(
        max_new_tokens=message.max_tokens,
        temperature=message.temperature,
        top_p=message.top_p,
        top_k=message.top_k,
        repetition_penalty=message.repetition_penalty,
    )

    async def generate():
        import json

        full_response = ""
        final_stats = None

        async for token, stats in inference_engine.generate_stream(
            messages=history,
            params=params,
            system_prompt=session.system_prompt,
        ):
            full_response += token
            if stats:
                final_stats = stats

            # Send token
            yield f"data: {json.dumps({'token': token, 'done': stats is not None})}\n\n"

        # Save assistant message
        if full_response:
            async with db.begin():
                assistant_message = ChatMessage(
                    session_id=session.id,
                    role="assistant",
                    content=full_response,
                    tokens_prompt=final_stats.prompt_tokens if final_stats else None,
                    tokens_completion=final_stats.completion_tokens if final_stats else None,
                    generation_time=final_stats.generation_time if final_stats else None,
                )
                db.add(assistant_message)

            # Send final stats
            if final_stats:
                yield f"data: {json.dumps({'stats': final_stats.__dict__})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": str(session.id),
        },
    )
