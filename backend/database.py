"""
RyzenAI-LocalLab Database Module

SQLAlchemy models and database session management.
Uses SQLite with async support via aiosqlite.
"""

import secrets
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from backend.config import settings


# =============================================================================
# Base Class
# =============================================================================
class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# =============================================================================
# Models
# =============================================================================
class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True, default=lambda: secrets.token_urlsafe(32)
    )
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', is_admin={self.is_admin})>"


class ChatSession(Base):
    """Chat session containing multiple messages."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="New Chat")
    model_name: Mapped[str] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title='{self.title}', model='{self.model_name}')>"


class ChatMessage(Base):
    """Individual message in a chat session."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_prompt: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_completion: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generation_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # seconds
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"<ChatMessage(id={self.id}, role='{self.role}', content='{preview}')>"


class LoadedModel(Base):
    """Track currently loaded models for quick reference."""

    __tablename__ = "loaded_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    model_path: Mapped[str] = mapped_column(String(512), nullable=False)
    loaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    device: Mapped[str] = mapped_column(String(20), default="auto")
    memory_used_gb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<LoadedModel(id={self.id}, model_id='{self.model_id}', device='{self.device}')>"


# =============================================================================
# Database Engine & Session
# =============================================================================

# Ensure data directory exists
settings.data_path.mkdir(parents=True, exist_ok=True)

# Async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level == "DEBUG",
    future=True,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions.

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize the database.

    Creates all tables if they don't exist.
    Creates the first admin user if no users exist.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create first admin user if needed
    from backend.auth import get_password_hash

    async with async_session_maker() as session:
        from sqlalchemy import select

        # Check if any users exist
        result = await session.execute(select(User).limit(1))
        existing_user = result.scalar_one_or_none()

        if existing_user is None:
            # Create first admin user
            admin_user = User(
                username=settings.first_admin_username,
                hashed_password=get_password_hash(settings.first_admin_password),
                is_admin=True,
                is_active=True,
            )
            session.add(admin_user)
            await session.commit()
            print(f"✓ Created admin user: {settings.first_admin_username}")
