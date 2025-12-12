"""
RyzenAI-LocalLab Authentication Routes

Endpoints for user registration, login, and API key management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import (
    Token,
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_current_admin_user,
    get_password_hash,
)
from backend.config import settings
from backend.database import User, get_db

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# =============================================================================
# Schemas
# =============================================================================
class UserCreate(BaseModel):
    """Schema for creating a new user."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """Schema for user response."""

    id: int
    username: str
    is_admin: bool
    is_active: bool
    api_key: str
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UserPublic(BaseModel):
    """Public user info (no API key)."""

    id: int
    username: str
    is_admin: bool
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Routes
# =============================================================================
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Register a new user.

    The first user registered becomes an admin.
    """
    # Check if username already exists
    result = await db.execute(select(User).where(User.username == user_data.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if this is the first user (make them admin)
    result = await db.execute(select(User).limit(1))
    is_first_user = result.scalar_one_or_none() is None

    # Create user
    user = User(
        username=user_data.username,
        hashed_password=get_password_hash(user_data.password),
        is_admin=is_first_user,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Login and get an access token.

    Uses OAuth2 password flow.
    """
    user = await authenticate_user(db, form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    # Create token
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user


@router.post("/regenerate-api-key", response_model=UserResponse)
async def regenerate_api_key(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Regenerate the current user's API key."""
    import secrets

    current_user.api_key = secrets.token_urlsafe(32)
    await db.commit()
    await db.refresh(current_user)

    return current_user


@router.get("/users", response_model=list[UserPublic])
async def list_users(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all users (admin only).
    """
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return users


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a user (admin only).
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.delete(user)
    await db.commit()

    return {"message": "User deleted"}
