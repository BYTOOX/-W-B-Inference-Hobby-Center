"""
RyzenAI-LocalLab Configuration Module

Centralized configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Paths
    # ==========================================================================
    models_path: Path = Field(
        default=Path("/srv/models"),
        description="Directory where AI models are stored",
    )
    data_path: Path = Field(
        default=Path("./data"),
        description="Directory for application data (SQLite, logs, etc.)",
    )

    # ==========================================================================
    # Server
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    ui_port: int = Field(default=8501, description="Streamlit UI port")

    # ==========================================================================
    # Security
    # ==========================================================================
    secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION",
        description="Secret key for JWT token signing",
    )
    access_token_expire_minutes: int = Field(
        default=60 * 24 * 7,  # 7 days
        description="JWT token expiration time in minutes",
    )
    first_admin_username: str = Field(
        default="admin",
        description="Username for the first admin user",
    )
    first_admin_password: str = Field(
        default="changeme",
        description="Password for the first admin user (change this!)",
    )

    # ==========================================================================
    # Hardware / Inference
    # ==========================================================================
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Device for inference: auto, cuda (ROCm), or cpu",
    )
    max_model_memory_fraction: float = Field(
        default=0.90,
        description="Maximum fraction of GPU memory to use for models",
        ge=0.1,
        le=1.0,
    )

    # ==========================================================================
    # Logging
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("models_path", "data_path", mode="before")
    @classmethod
    def resolve_path(cls, v):
        """Convert string paths to Path objects and resolve."""
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve()

    @field_validator("secret_key")
    @classmethod
    def check_secret_key(cls, v):
        """Warn if using default secret key."""
        if v == "CHANGE-ME-IN-PRODUCTION":
            import warnings

            warnings.warn(
                "Using default SECRET_KEY! Please set a secure key in .env",
                UserWarning,
                stacklevel=2,
            )
        return v

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def database_url(self) -> str:
        """SQLite database URL."""
        db_path = self.data_path / "locallab.db"
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def sync_database_url(self) -> str:
        """Synchronous SQLite database URL."""
        db_path = self.data_path / "locallab.db"
        return f"sqlite:///{db_path}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience export
settings = get_settings()
