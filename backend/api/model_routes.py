"""
RyzenAI-LocalLab Model Routes

Endpoints for model management: listing, downloading, and deletion.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.model_manager import (
    CompatibilityStatus,
    ModelFormat,
    model_manager,
)

router = APIRouter(prefix="/api/models", tags=["Models"])


# =============================================================================
# Schemas
# =============================================================================
class ModelInfoResponse(BaseModel):
    """Model information response."""

    id: str
    name: str
    path: str
    size_gb: float
    format: str
    downloaded: bool
    download_date: Optional[str] = None
    compatibility: str
    compatibility_message: str


class DownloadRequest(BaseModel):
    """Request to download a model."""

    repo_id: str = Field(..., description="HuggingFace repository ID (e.g., 'mistralai/Devstral-Small-2505')")


class StorageStatsResponse(BaseModel):
    """Storage statistics response."""

    models_count: int
    models_size_gb: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float


# =============================================================================
# Routes
# =============================================================================
@router.get("/", response_model=list[ModelInfoResponse])
async def list_models(current_user: User = Depends(get_current_active_user)):
    """List all locally downloaded models."""
    models = model_manager.list_local_models()

    return [
        ModelInfoResponse(
            id=m.id,
            name=m.name,
            path=str(m.path),
            size_gb=round(m.size_gb, 2),
            format=m.format.value,
            downloaded=m.downloaded,
            download_date=m.download_date.isoformat() if m.download_date else None,
            compatibility=m.compatibility.value,
            compatibility_message=m.compatibility_message,
        )
        for m in models
    ]


@router.get("/info/{repo_id:path}", response_model=ModelInfoResponse)
async def get_model_info(repo_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Get information about a model (local or remote).

    Args:
        repo_id: HuggingFace repository ID (e.g., 'mistralai/Devstral-Small-2505')
    """
    # Try local first
    models = model_manager.list_local_models()
    for m in models:
        if m.id == repo_id or m.id == repo_id.replace("/", "--"):
            return ModelInfoResponse(
                id=m.id,
                name=m.name,
                path=str(m.path),
                size_gb=round(m.size_gb, 2),
                format=m.format.value,
                downloaded=True,
                download_date=m.download_date.isoformat() if m.download_date else None,
                compatibility=m.compatibility.value,
                compatibility_message=m.compatibility_message,
            )

    # Try remote
    model_info = await model_manager.get_remote_model_info(repo_id)

    if not model_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    return ModelInfoResponse(
        id=model_info.id,
        name=model_info.name,
        path=str(model_info.path),
        size_gb=round(model_info.size_gb, 2),
        format=model_info.format.value,
        downloaded=model_info.downloaded,
        compatibility=model_info.compatibility.value,
        compatibility_message=model_info.compatibility_message,
    )


@router.post("/download")
async def download_model(request: DownloadRequest, current_user: User = Depends(get_current_active_user)):
    """
    Download a model from HuggingFace.

    Returns a streaming response with progress updates.
    """

    async def generate():
        import json

        async for progress in model_manager.download_model(request.repo_id):
            yield f"data: {json.dumps(progress.__dict__)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/download-sync")
async def download_model_sync(request: DownloadRequest, current_user: User = Depends(get_current_active_user)):
    """
    Download a model from HuggingFace (synchronous, blocking).

    Returns the path when complete.
    """
    try:
        path = model_manager.download_model_sync(request.repo_id)
        return {"path": str(path), "status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/{model_id:path}")
async def delete_model(model_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Delete a downloaded model.

    Args:
        model_id: Model ID (path relative to models directory)
    """
    success = model_manager.delete_model(model_id)

    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    return {"message": "Model deleted", "model_id": model_id}


@router.get("/storage", response_model=StorageStatsResponse)
async def get_storage_stats(current_user: User = Depends(get_current_active_user)):
    """Get storage statistics for the models directory."""
    stats = model_manager.get_storage_stats()
    return StorageStatsResponse(**stats)
