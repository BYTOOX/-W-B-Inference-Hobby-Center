"""
RyzenAI-LocalLab Model Routes

Endpoints for model management: listing, downloading, and deletion.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.model_manager import model_manager
from backend.services.download_manager import download_manager

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


class DownloadStatusResponse(BaseModel):
    """Download status response."""

    repo_id: str
    total_bytes: int
    current_bytes: int
    percent: float
    speed_mbps: float
    eta_seconds: int
    status: str
    error: Optional[str] = None


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


# =============================================================================
# Download Management with Persistent Tracking
# =============================================================================
@router.get("/downloads")
async def get_active_downloads(current_user: User = Depends(get_current_active_user)):
    """Get all active downloads (survives page refresh)."""
    return download_manager.get_active_downloads()


@router.get("/downloads/{repo_id:path}")
async def get_download_status(repo_id: str, current_user: User = Depends(get_current_active_user)):
    """Get status of a specific download."""
    status = download_manager.get_download(repo_id)
    if not status:
        raise HTTPException(status_code=404, detail="No active download for this model")
    return status


@router.post("/download")
async def start_download(request: DownloadRequest, current_user: User = Depends(get_current_active_user)):
    """
    Start a model download.
    
    Downloads run in background and can be tracked via GET /downloads.
    Progress persists across page refreshes.
    """
    result = await download_manager.start_download(request.repo_id)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result


@router.delete("/downloads/{repo_id:path}")
async def cancel_download(repo_id: str, current_user: User = Depends(get_current_active_user)):
    """Cancel an active download."""
    success = download_manager.cancel_download(repo_id)
    if not success:
        raise HTTPException(status_code=404, detail="No active download to cancel")
    return {"message": "Download cancelled", "repo_id": repo_id}


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


# =============================================================================
# GGUF Model Management
# =============================================================================
class GGUFListRequest(BaseModel):
    """Request to list GGUF files in a repository."""
    repo_id: str = Field(..., description="HuggingFace repository ID (e.g., 'bartowski/DeepSeek-R1-GGUF')")


class GGUFDownloadRequest(BaseModel):
    """Request to download a GGUF file."""
    repo_id: str = Field(..., description="HuggingFace repository ID")
    filename: str = Field(..., description="GGUF filename to download")


@router.post("/gguf/list")
async def list_gguf_files(request: GGUFListRequest, current_user: User = Depends(get_current_active_user)):
    """
    List all GGUF files available in a HuggingFace repository.
    
    Returns list of files with quantization info and sizes.
    """
    files = await model_manager.list_gguf_files(request.repo_id)
    return {"repo_id": request.repo_id, "files": files}


@router.get("/gguf/local")
async def list_local_gguf(current_user: User = Depends(get_current_active_user)):
    """List all locally downloaded GGUF files."""
    files = model_manager.list_local_gguf()
    return {"files": files}


@router.post("/gguf/download")
async def download_gguf(request: GGUFDownloadRequest, current_user: User = Depends(get_current_active_user)):
    """
    Download a specific GGUF file from HuggingFace.
    
    Returns SSE stream with progress updates.
    """
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        async for progress in model_manager.download_gguf(request.repo_id, request.filename):
            data = {
                "model_id": progress.model_id,
                "filename": progress.filename,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "percent": progress.percent,
                "speed_mbps": progress.speed_mbps,
                "eta_seconds": progress.eta_seconds,
                "status": progress.status,
                "error": progress.error,
            }
            yield f"data: {json.dumps(data)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete("/gguf/{filename}")
async def delete_gguf(filename: str, current_user: User = Depends(get_current_active_user)):
    """Delete a downloaded GGUF file."""
    import os
    from backend.config import settings
    
    gguf_path = settings.models_path / "gguf" / filename
    
    if not gguf_path.exists():
        raise HTTPException(status_code=404, detail="GGUF file not found")
    
    try:
        os.remove(gguf_path)
        return {"message": "GGUF file deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GGUFImportRequest(BaseModel):
    """Request to import GGUF to Ollama."""
    gguf_path: str
    model_name: str = None


@router.post("/gguf/import")
async def import_gguf_to_ollama(
    request: GGUFImportRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Import a GGUF file into Ollama for GPU acceleration."""
    
    async def generate():
        async for progress in model_manager.import_gguf_to_ollama(
            request.gguf_path, 
            request.model_name
        ):
            yield f"data: {json.dumps(progress)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
