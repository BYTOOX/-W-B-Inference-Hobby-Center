"""
RyzenAI-LocalLab Model Manager

Handles downloading, listing, and managing AI models from HuggingFace.
Includes model compatibility detection and size estimation.
"""

import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from backend.config import settings
from backend.services.hardware_monitor import hardware_monitor


class ModelFormat(str, Enum):
    """Supported model formats."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class CompatibilityStatus(str, Enum):
    """Model compatibility with current hardware."""

    COMPATIBLE = "compatible"
    NEEDS_QUANTIZATION = "needs_quantization"
    TOO_LARGE = "too_large"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    path: Path
    size_gb: float
    format: ModelFormat
    downloaded: bool
    download_date: Optional[datetime] = None
    config: Optional[dict] = None
    compatibility: CompatibilityStatus = CompatibilityStatus.UNKNOWN
    compatibility_message: str = ""


@dataclass
class DownloadProgress:
    """Progress update for model downloads."""

    model_id: str
    filename: str
    downloaded_bytes: int
    total_bytes: int
    percent: float
    speed_mbps: float
    eta_seconds: int
    status: str  # "downloading", "completed", "error"
    error: Optional[str] = None


class ModelManager:
    """
    Model management service.

    Handles:
    - Listing local and remote models
    - Downloading from HuggingFace
    - Model compatibility detection
    - Storage management
    """

    def __init__(self, models_path: Optional[Path] = None):
        self.models_path = models_path or settings.models_path
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()
        self._download_progress: dict[str, DownloadProgress] = {}

    # =========================================================================
    # Model Listing
    # =========================================================================
    def list_local_models(self) -> list[ModelInfo]:
        """List all models downloaded locally."""
        models = []

        for model_dir in self.models_path.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("."):
                # Check for nested org/model structure
                if any(d.is_dir() for d in model_dir.iterdir()):
                    for sub_dir in model_dir.iterdir():
                        if sub_dir.is_dir():
                            info = self._get_model_info(sub_dir)
                            if info:
                                models.append(info)
                else:
                    info = self._get_model_info(model_dir)
                    if info:
                        models.append(info)

        return sorted(models, key=lambda m: m.name)

    def _get_model_info(self, model_path: Path) -> Optional[ModelInfo]:
        """Get information about a local model."""
        # Check for model files
        model_format = self._detect_format(model_path)
        if model_format == ModelFormat.UNKNOWN:
            return None

        # Calculate size
        size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024**3)

        # Get model ID from path
        rel_path = model_path.relative_to(self.models_path)
        model_id = str(rel_path).replace("\\", "/")

        # Load config if available
        config = self._load_config(model_path)

        # Check compatibility
        compatibility, message = self._check_compatibility(size_gb, config)

        return ModelInfo(
            id=model_id,
            name=model_path.name,
            path=model_path,
            size_gb=size_gb,
            format=model_format,
            downloaded=True,
            download_date=datetime.fromtimestamp(model_path.stat().st_mtime),
            config=config,
            compatibility=compatibility,
            compatibility_message=message,
        )

    def _detect_format(self, model_path: Path) -> ModelFormat:
        """Detect the format of a model."""
        files = list(model_path.rglob("*"))

        for f in files:
            if f.suffix == ".safetensors":
                return ModelFormat.SAFETENSORS
            elif f.suffix == ".bin" and "pytorch" in f.name.lower():
                return ModelFormat.PYTORCH
            elif f.suffix == ".gguf":
                return ModelFormat.GGUF

        # Check for .bin files (pytorch)
        if any(f.suffix == ".bin" for f in files):
            return ModelFormat.PYTORCH

        return ModelFormat.UNKNOWN

    def _load_config(self, model_path: Path) -> Optional[dict]:
        """Load model config.json if available."""
        config_path = model_path / "config.json"
        if config_path.exists():
            import json

            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    # =========================================================================
    # Compatibility Detection
    # =========================================================================
    def _check_compatibility(self, size_gb: float, config: Optional[dict]) -> tuple[CompatibilityStatus, str]:
        """
        Check if a model is compatible with current hardware.

        Returns:
            Tuple of (status, message)
        """
        # Get available memory
        mem_stats = hardware_monitor.get_memory_stats()
        gpu_stats = hardware_monitor.get_gpu_stats()

        # For unified memory systems, use total RAM
        if gpu_stats.available:
            available_memory = mem_stats.total_gb * settings.max_model_memory_fraction
        else:
            available_memory = mem_stats.total_gb * 0.8  # Reserve 20% for system

        # Estimate memory needed (rough: model size * 1.2 for overhead)
        estimated_memory = size_gb * 1.2

        # Check hidden size for more accurate estimation
        if config:
            hidden_size = config.get("hidden_size", 0)
            num_layers = config.get("num_hidden_layers", 0)
            vocab_size = config.get("vocab_size", 0)

            if hidden_size and num_layers:
                # Very rough parameter estimation
                params_b = (hidden_size * num_layers * 12) / 1e9
                # FP16 = 2 bytes per param
                estimated_memory = params_b * 2 * 1.2

        if estimated_memory > available_memory * 1.5:
            return (
                CompatibilityStatus.TOO_LARGE,
                f"Model requires ~{estimated_memory:.1f} GB, but only {available_memory:.1f} GB available. "
                f"Consider using a quantized version (GGUF Q4/Q5).",
            )
        elif estimated_memory > available_memory:
            return (
                CompatibilityStatus.NEEDS_QUANTIZATION,
                f"Model may be too large ({estimated_memory:.1f} GB). "
                f"Quantization or CPU offloading recommended.",
            )
        else:
            return (
                CompatibilityStatus.COMPATIBLE,
                f"Model should fit in memory ({estimated_memory:.1f} GB / {available_memory:.1f} GB available).",
            )

    # =========================================================================
    # Remote Model Info
    # =========================================================================
    async def get_remote_model_info(self, repo_id: str) -> Optional[ModelInfo]:
        """
        Get information about a model on HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "mistralai/Devstral-Small-2505")
        """
        try:
            # Get model info from HF API
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)

            # Calculate total size
            size_bytes = sum(f.size for f in model_info.siblings if f.size)
            size_gb = size_bytes / (1024**3)

            # Detect format from filenames
            model_format = ModelFormat.UNKNOWN
            for f in model_info.siblings:
                if f.rfilename.endswith(".safetensors"):
                    model_format = ModelFormat.SAFETENSORS
                    break
                elif f.rfilename.endswith(".gguf"):
                    model_format = ModelFormat.GGUF
                    break
                elif f.rfilename.endswith(".bin"):
                    model_format = ModelFormat.PYTORCH

            # Check if already downloaded
            local_path = self.models_path / repo_id.replace("/", "--")
            downloaded = local_path.exists()

            # Check compatibility
            compatibility, message = self._check_compatibility(size_gb, None)

            return ModelInfo(
                id=repo_id,
                name=repo_id.split("/")[-1],
                path=local_path,
                size_gb=size_gb,
                format=model_format,
                downloaded=downloaded,
                compatibility=compatibility,
                compatibility_message=message,
            )

        except HfHubHTTPError as e:
            print(f"Error fetching model info: {e}")
            return None

    # =========================================================================
    # Download Management
    # =========================================================================
    async def download_model(self, repo_id: str) -> AsyncGenerator[DownloadProgress, None]:
        """
        Download a model from HuggingFace with progress updates.

        Args:
            repo_id: HuggingFace repository ID

        Yields:
            DownloadProgress updates
        """
        import time

        local_dir = self.models_path / repo_id.replace("/", "--")
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get file list first
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)
            files = [f for f in model_info.siblings if f.size]
            total_size = sum(f.size for f in files)
            downloaded_size = 0

            start_time = time.time()

            for file_info in files:
                filename = file_info.rfilename
                file_size = file_info.size or 0

                yield DownloadProgress(
                    model_id=repo_id,
                    filename=filename,
                    downloaded_bytes=downloaded_size,
                    total_bytes=total_size,
                    percent=(downloaded_size / total_size * 100) if total_size > 0 else 0,
                    speed_mbps=0,
                    eta_seconds=0,
                    status="downloading",
                )

                # Download file
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )

                downloaded_size += file_size

                # Calculate speed and ETA
                elapsed = time.time() - start_time
                speed_bps = downloaded_size / elapsed if elapsed > 0 else 0
                speed_mbps = speed_bps / (1024 * 1024)
                remaining = total_size - downloaded_size
                eta = int(remaining / speed_bps) if speed_bps > 0 else 0

                yield DownloadProgress(
                    model_id=repo_id,
                    filename=filename,
                    downloaded_bytes=downloaded_size,
                    total_bytes=total_size,
                    percent=(downloaded_size / total_size * 100) if total_size > 0 else 0,
                    speed_mbps=speed_mbps,
                    eta_seconds=eta,
                    status="downloading",
                )

            yield DownloadProgress(
                model_id=repo_id,
                filename="",
                downloaded_bytes=total_size,
                total_bytes=total_size,
                percent=100,
                speed_mbps=0,
                eta_seconds=0,
                status="completed",
            )

        except Exception as e:
            yield DownloadProgress(
                model_id=repo_id,
                filename="",
                downloaded_bytes=0,
                total_bytes=0,
                percent=0,
                speed_mbps=0,
                eta_seconds=0,
                status="error",
                error=str(e),
            )

    def download_model_sync(self, repo_id: str, callback=None) -> Path:
        """
        Synchronous model download (for simpler use cases).

        Args:
            repo_id: HuggingFace repository ID
            callback: Optional callback function(filename, progress)

        Returns:
            Path to downloaded model
        """
        local_dir = self.models_path / repo_id.replace("/", "--")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        return local_dir

    # =========================================================================
    # Model Deletion
    # =========================================================================
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a downloaded model.

        Args:
            model_id: Model ID (path relative to models directory)

        Returns:
            True if deleted, False if not found
        """
        # Try both path formats
        model_path = self.models_path / model_id
        if not model_path.exists():
            model_path = self.models_path / model_id.replace("/", "--")

        if not model_path.exists():
            return False

        try:
            shutil.rmtree(model_path)
            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False

    # =========================================================================
    # Storage Stats
    # =========================================================================
    def get_storage_stats(self) -> dict:
        """Get storage statistics for the models directory."""
        total_size = 0
        model_count = 0

        for model in self.list_local_models():
            total_size += model.size_gb
            model_count += 1

        # Get disk usage
        disk_usage = shutil.disk_usage(self.models_path)

        return {
            "models_count": model_count,
            "models_size_gb": round(total_size, 2),
            "disk_total_gb": round(disk_usage.total / (1024**3), 2),
            "disk_used_gb": round(disk_usage.used / (1024**3), 2),
            "disk_free_gb": round(disk_usage.free / (1024**3), 2),
        }


# Singleton instance
model_manager = ModelManager()
