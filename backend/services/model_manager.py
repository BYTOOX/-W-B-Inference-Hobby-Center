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
        Uses snapshot_download with hf_transfer for maximum speed.

        Args:
            repo_id: HuggingFace repository ID

        Yields:
            DownloadProgress updates
        """
        import asyncio
        import os
        import time

        # Enable hf_transfer for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        local_dir = self.models_path / repo_id.replace("/", "--")
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get model info for size estimation
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)
            files = [f for f in model_info.siblings if f.size]
            total_size = sum(f.size for f in files)

            yield DownloadProgress(
                model_id=repo_id,
                filename="Initializing fast download...",
                downloaded_bytes=0,
                total_bytes=total_size,
                percent=0,
                speed_mbps=0,
                eta_seconds=0,
                status="downloading",
            )

            start_time = time.time()

            # Run snapshot_download in executor (it's blocking but fast)
            loop = asyncio.get_event_loop()

            def do_download():
                return snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            # Start download in background
            download_task = loop.run_in_executor(None, do_download)

            # Monitor progress by checking directory size
            while not download_task.done():
                await asyncio.sleep(2)

                # Calculate current size
                current_size = 0
                for f in local_dir.rglob("*"):
                    if f.is_file():
                        try:
                            current_size += f.stat().st_size
                        except OSError:
                            pass

                elapsed = time.time() - start_time
                speed_bps = current_size / elapsed if elapsed > 0 else 0
                speed_mbps = speed_bps / (1024 * 1024)
                remaining = total_size - current_size
                eta = int(remaining / speed_bps) if speed_bps > 0 else 0
                percent = (current_size / total_size * 100) if total_size > 0 else 0

                yield DownloadProgress(
                    model_id=repo_id,
                    filename=f"Downloading... {speed_mbps:.1f} MB/s",
                    downloaded_bytes=current_size,
                    total_bytes=total_size,
                    percent=min(percent, 99),  # Don't show 100% until complete
                    speed_mbps=speed_mbps,
                    eta_seconds=eta,
                    status="downloading",
                )

            # Wait for completion
            await download_task

            elapsed = time.time() - start_time
            avg_speed = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

            yield DownloadProgress(
                model_id=repo_id,
                filename=f"Complete! Avg: {avg_speed:.1f} MB/s",
                downloaded_bytes=total_size,
                total_bytes=total_size,
                percent=100,
                speed_mbps=avg_speed,
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
        Uses hf_transfer for maximum speed.

        Args:
            repo_id: HuggingFace repository ID
            callback: Optional callback function(filename, progress)

        Returns:
            Path to downloaded model
        """
        import os

        # Enable hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        local_dir = self.models_path / repo_id.replace("/", "--")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
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
    # GGUF Model Support
    # =========================================================================
    async def list_gguf_files(self, repo_id: str) -> list[dict]:
        """
        List GGUF files available in a HuggingFace repository.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "bartowski/DeepSeek-R1-GGUF")
            
        Returns:
            List of GGUF files with name, size, and quantization info
        """
        import re
        
        try:
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)
            gguf_files = []
            
            for f in model_info.siblings:
                if f.rfilename.endswith(".gguf"):
                    name = f.rfilename
                    
                    # Skip split files (multi-part models like -00001-of-00002)
                    if re.search(r'-\d{5}-of-\d{5}', name):
                        continue
                    
                    size_gb = f.size / (1024**3) if f.size else 0
                    
                    # Parse quantization from filename (comprehensive list)
                    quant = "unknown"
                    quant_patterns = [
                        # Standard K-quants
                        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", 
                        "Q4_K_S", "Q4_K_M", "Q4_K_L",
                        "Q5_K_S", "Q5_K_M", "Q5_K_L",
                        "Q6_K", "Q8_K",
                        # Legacy quants
                        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
                        # IQuants (newer)
                        "IQ1_S", "IQ1_M", "IQ2_S", "IQ2_M", "IQ2_XS", "IQ2_XXS",
                        "IQ3_S", "IQ3_M", "IQ3_XS", "IQ3_XXS",
                        "IQ4_XS", "IQ4_NL",
                        # FP formats
                        "F16", "F32", "BF16",
                    ]
                    
                    for q in quant_patterns:
                        # Case-insensitive match with word boundaries
                        if re.search(rf'[_\-\.]{q}[_\-\.]?', name, re.IGNORECASE) or \
                           name.upper().endswith(f"-{q}.GGUF") or \
                           f".{q}." in name.upper():
                            quant = q
                            break
                    
                    gguf_files.append({
                        "filename": name,
                        "size_gb": round(size_gb, 2),
                        "size_bytes": f.size or 0,
                        "quantization": quant,
                        "repo_id": repo_id,
                    })
            
            # Sort by size
            return sorted(gguf_files, key=lambda x: x["size_bytes"])
            
        except Exception as e:
            print(f"Error listing GGUF files: {e}")
            return []
    
    async def download_gguf(self, repo_id: str, filename: str) -> AsyncGenerator[DownloadProgress, None]:
        """
        Download a specific GGUF file from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            filename: Name of the GGUF file to download
            
        Yields:
            DownloadProgress updates
        """
        import asyncio
        import os
        import time
        
        # Enable hf_transfer for faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Create GGUF subdirectory
        gguf_dir = self.models_path / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        
        # Safe filename based on repo
        safe_name = f"{repo_id.replace('/', '_')}_{filename}"
        local_path = gguf_dir / filename
        
        try:
            # Get file size
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)
            file_info = next((f for f in model_info.siblings if f.rfilename == filename), None)
            
            if not file_info:
                yield DownloadProgress(
                    model_id=f"{repo_id}/{filename}",
                    filename=filename,
                    downloaded_bytes=0,
                    total_bytes=0,
                    percent=0,
                    speed_mbps=0,
                    eta_seconds=0,
                    status="error",
                    error=f"File {filename} not found in {repo_id}",
                )
                return
            
            total_size = file_info.size or 0
            
            yield DownloadProgress(
                model_id=f"{repo_id}/{filename}",
                filename=filename,
                downloaded_bytes=0,
                total_bytes=total_size,
                percent=0,
                speed_mbps=0,
                eta_seconds=0,
                status="downloading",
            )
            
            start_time = time.time()
            loop = asyncio.get_event_loop()
            
            # Calculate initial directory size BEFORE download starts
            initial_size = 0
            for f in gguf_dir.rglob("*"):
                if f.is_file():
                    try:
                        initial_size += f.stat().st_size
                    except OSError:
                        pass
            
            def do_download():
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=gguf_dir,
                )
            
            # Start download
            download_task = loop.run_in_executor(None, do_download)
            
            last_downloaded = 0
            last_time = start_time
            
            # Monitor progress
            while not download_task.done():
                await asyncio.sleep(0.5)  # Check more frequently
                
                # Calculate current total size
                current_total = 0
                for f in gguf_dir.rglob("*"):
                    if f.is_file():
                        try:
                            current_total += f.stat().st_size
                        except OSError:
                            pass
                
                # Downloaded = current - initial (delta)
                downloaded = max(0, current_total - initial_size)
                
                # Speed from last interval
                now = time.time()
                time_delta = now - last_time
                size_delta = downloaded - last_downloaded
                speed_bps = size_delta / time_delta if time_delta > 0 else 0
                speed_mbps = speed_bps / (1024 * 1024)
                
                last_downloaded = downloaded
                last_time = now
                
                remaining = max(0, total_size - downloaded)
                eta = int(remaining / speed_bps) if speed_bps > 0 else 0
                percent = (downloaded / total_size * 100) if total_size > 0 else 0
                
                yield DownloadProgress(
                    model_id=f"{repo_id}/{filename}",
                    filename=filename,
                    downloaded_bytes=downloaded,
                    total_bytes=total_size,
                    percent=min(max(percent, 0), 99),
                    speed_mbps=max(speed_mbps, 0),
                    eta_seconds=eta,
                    status="downloading",
                )
            
            # Complete
            result_path = await download_task
            elapsed = time.time() - start_time
            avg_speed = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            
            yield DownloadProgress(
                model_id=f"{repo_id}/{filename}",
                filename=filename,
                downloaded_bytes=total_size,
                total_bytes=total_size,
                percent=100,
                speed_mbps=avg_speed,
                eta_seconds=0,
                status="completed",
            )
            
        except Exception as e:
            yield DownloadProgress(
                model_id=f"{repo_id}/{filename}",
                filename=filename,
                downloaded_bytes=0,
                total_bytes=0,
                percent=0,
                speed_mbps=0,
                eta_seconds=0,
                status="error",
                error=str(e),
            )
    
    def list_local_gguf(self) -> list[dict]:
        """List all locally downloaded GGUF files."""
        gguf_dir = self.models_path / "gguf"
        if not gguf_dir.exists():
            return []
        
        gguf_files = []
        for f in gguf_dir.glob("*.gguf"):
            size_gb = f.stat().st_size / (1024**3)
            gguf_files.append({
                "filename": f.name,
                "path": str(f),
                "size_gb": round(size_gb, 2),
            })
        
        return sorted(gguf_files, key=lambda x: x["filename"])

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
