"""
RyzenAI-LocalLab System Routes

Endpoints for system monitoring and health checks.
"""

from fastapi import APIRouter, Depends

from backend.auth import get_current_active_user
from backend.database import User
from backend.services.hardware_monitor import hardware_monitor

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/stats")
async def get_system_stats(current_user: User = Depends(get_current_active_user)):
    """Get real-time system statistics (CPU, RAM, GPU)."""
    return hardware_monitor.to_dict()


@router.get("/health")
async def health_check():
    """
    Health check endpoint (no auth required).

    Returns basic system status.
    """
    gpu_stats = hardware_monitor.get_gpu_stats()

    return {
        "status": "healthy",
        "gpu_available": gpu_stats.available,
        "gpu_name": gpu_stats.name if gpu_stats.available else None,
    }
