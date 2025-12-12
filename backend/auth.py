"""
RyzenAI-LocalLab Authentication Module

JWT-based authentication and password hashing.
Supports both session tokens (UI) and API keys (OpenAI-compatible API).
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import User, get_db


# =============================================================================
# Password Hashing
# =============================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)


# =============================================================================
# JWT Token Management
# =============================================================================
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """Data extracted from JWT token."""

    username: Optional[str] = None
    user_id: Optional[int] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")

        if username is None:
            return None

        return TokenData(username=username, user_id=user_id)
    except JWTError:
        return None


# =============================================================================
# User Authentication
# =============================================================================
async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.

    Args:
        db: Database session
        username: User's username
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if user is None:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    return user


async def get_user_by_api_key(db: AsyncSession, api_key: str) -> Optional[User]:
    """
    Get a user by their API key.

    Args:
        db: Database session
        api_key: User's API key

    Returns:
        User object if found, None otherwise
    """
    result = await db.execute(select(User).where(User.api_key == api_key))
    return result.scalar_one_or_none()


# =============================================================================
# Dependency Injection
# =============================================================================
async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get the current authenticated user.

    Supports:
    - OAuth2 Bearer token (JWT)
    - API Key in Authorization header

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try OAuth2 token first
    if token:
        token_data = decode_access_token(token)
        if token_data and token_data.username:
            result = await db.execute(select(User).where(User.username == token_data.username))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return user

    # Try Bearer token (could be API key or JWT)
    if bearer and bearer.credentials:
        cred = bearer.credentials

        # Try as JWT first
        token_data = decode_access_token(cred)
        if token_data and token_data.username:
            result = await db.execute(select(User).where(User.username == token_data.username))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return user

        # Try as API key
        user = await get_user_by_api_key(db, cred)
        if user and user.is_active:
            return user

    raise credentials_exception


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify they are active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify they are an admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


# =============================================================================
# Optional Authentication (for public endpoints)
# =============================================================================
async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get the current user if authenticated, otherwise None.

    Used for endpoints that work both with and without authentication.
    """
    try:
        return await get_current_user(token, bearer, db)
    except HTTPException:
        return None
