"""
RyzenAI-LocalLab Chat Routes

Endpoints for chat sessions and message handling.
Ollama-only mode - no PyTorch required.
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
from backend.config import settings
from backend.database import ChatMessage, ChatSession, User, get_db
from backend.services.ollama_engine import ollama_engine

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
    max_tokens: int = Field(default=2048, ge=1, le=32768)


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


class EngineStatusResponse(BaseModel):
    """Inference engine status."""

    loaded: bool
    device: str
    model: Optional[str] = None


# =============================================================================
# Engine Routes (Ollama-based)
# =============================================================================
@router.get("/engine/status", response_model=EngineStatusResponse)
async def get_engine_status(current_user: User = Depends(get_current_active_user)):
    """Get the current inference engine status."""
    available = await ollama_engine.is_available()
    return EngineStatusResponse(
        loaded=ollama_engine.current_model is not None,
        device="gpu" if available else "unavailable",
        model=ollama_engine.current_model,
    )


@router.post("/engine/load")
async def load_model(request: ModelLoadRequest, current_user: User = Depends(get_current_active_user)):
    """Load an Ollama model for inference."""
    try:
        result = await ollama_engine.load_model(request.model_path)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return {
            "status": "loaded",
            "model_id": request.model_path,
            "device": "gpu",
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/engine/unload")
async def unload_model(current_user: User = Depends(get_current_active_user)):
    """Unload the current model."""
    ollama_engine.current_model = None
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
# Message Routes (Ollama streaming)
# =============================================================================
@router.post("/send")
async def send_message(
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and get a streaming response via Ollama.
    """
    # Check if model is loaded
    if not ollama_engine.current_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first via Models page.",
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
            model_name=ollama_engine.current_model,
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
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

    async def generate():
        import json

        full_response = ""
        final_stats = None

        async for chunk in ollama_engine.generate_stream(
            model=ollama_engine.current_model,
            messages=history,
            temperature=message.temperature,
            top_p=message.top_p,
            max_tokens=message.max_tokens,
        ):
            if chunk.get("error"):
                yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                return
            
            if chunk.get("content"):
                full_response += chunk["content"]
                yield f"data: {json.dumps({'token': chunk['content']})}\n\n"
            
            if chunk.get("done"):
                final_stats = chunk

        # Save assistant message
        if full_response:
            assistant_message = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=full_response,
                tokens_prompt=final_stats.get("prompt_eval_count") if final_stats else None,
                tokens_completion=final_stats.get("eval_count") if final_stats else None,
            )
            db.add(assistant_message)
            await db.commit()

            # Send final stats
            if final_stats:
                eval_duration = final_stats.get("eval_duration", 1)
                eval_count = final_stats.get("eval_count", 0)
                tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
                yield f"data: {json.dumps({'stats': {'tokens_per_second': tps, 'total_tokens': eval_count}})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": str(session.id),
        },
    )
