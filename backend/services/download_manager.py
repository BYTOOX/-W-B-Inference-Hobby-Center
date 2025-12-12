"""
RyzenAI-LocalLab - Download Manager Service

Uses huggingface-cli for reliable downloads with progress tracking.
"""

import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from backend.config import settings


@dataclass
class DownloadState:
    """State of an active download."""
    repo_id: str
    local_dir: Path
    total_bytes: int
    start_time: float
    status: str = "downloading"  # downloading, completed, error
    error: Optional[str] = None
    process: Optional[subprocess.Popen] = None
    output_lines: list = None
    
    def __post_init__(self):
        if self.output_lines is None:
            self.output_lines = []
    
    @property
    def current_bytes(self) -> int:
        """Calculate current downloaded size by checking directory."""
        if not self.local_dir.exists():
            return 0
        total = 0
        for f in self.local_dir.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total
    
    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0
        return min((self.current_bytes / self.total_bytes) * 100, 100)
    
    @property
    def speed_mbps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0
        return (self.current_bytes / (1024 * 1024)) / elapsed
    
    @property
    def eta_seconds(self) -> int:
        speed_bps = self.current_bytes / max(time.time() - self.start_time, 0.1)
        remaining = self.total_bytes - self.current_bytes
        if speed_bps <= 0:
            return 0
        return int(remaining / speed_bps)
    
    @property
    def last_output(self) -> str:
        if self.output_lines:
            return self.output_lines[-1]
        return ""
    
    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "total_bytes": self.total_bytes,
            "current_bytes": self.current_bytes,
            "percent": round(self.percent, 1),
            "speed_mbps": round(self.speed_mbps, 1),
            "eta_seconds": self.eta_seconds,
            "status": self.status,
            "error": self.error,
            "last_output": self.last_output,
        }


class DownloadManager:
    """
    Manages model downloads using huggingface-cli.
    
    Uses subprocess to run the CLI which is more reliable
    than the Python API for large downloads.
    """
    
    def __init__(self):
        self.active_downloads: dict[str, DownloadState] = {}
        self.hf_api = HfApi()
        self.models_path = Path(settings.models_path)
    
    def get_active_downloads(self) -> list[dict]:
        """Get all active downloads."""
        # Update status of all downloads
        for repo_id, state in list(self.active_downloads.items()):
            if state.process and state.status == "downloading":
                # Check if process is still running
                poll = state.process.poll()
                if poll is not None:
                    if poll == 0:
                        state.status = "completed"
                    else:
                        state.status = "error"
                        state.error = f"Process exited with code {poll}"
        
        return [d.to_dict() for d in self.active_downloads.values()]
    
    def get_download(self, repo_id: str) -> Optional[dict]:
        """Get a specific download status."""
        if repo_id in self.active_downloads:
            state = self.active_downloads[repo_id]
            # Update status
            if state.process and state.status == "downloading":
                poll = state.process.poll()
                if poll is not None:
                    state.status = "completed" if poll == 0 else "error"
            return state.to_dict()
        return None
    
    async def start_download(self, repo_id: str) -> dict:
        """
        Start a new download using huggingface-cli.
        
        Returns the initial download state.
        """
        # Check if already downloading
        if repo_id in self.active_downloads:
            state = self.active_downloads[repo_id]
            if state.status == "downloading" and state.process:
                if state.process.poll() is None:  # Still running
                    return state.to_dict()
        
        # Get model info for total size
        try:
            model_info = self.hf_api.model_info(repo_id, files_metadata=True)
            total_bytes = sum(f.size for f in model_info.siblings if f.size)
        except Exception as e:
            return {"error": str(e), "status": "error"}
        
        # Create local directory
        local_dir = self.models_path / repo_id.replace("/", "--")
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Build CLI command
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        cmd = [
            "huggingface-cli", "download",
            repo_id,
            "--local-dir", str(local_dir),
        ]
        
        # Start download process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
        except Exception as e:
            return {"error": f"Failed to start download: {e}", "status": "error"}
        
        # Create download state
        state = DownloadState(
            repo_id=repo_id,
            local_dir=local_dir,
            total_bytes=total_bytes,
            start_time=time.time(),
            process=process,
        )
        self.active_downloads[repo_id] = state
        
        # Start background task to read output
        asyncio.create_task(self._read_output(repo_id))
        
        return state.to_dict()
    
    async def _read_output(self, repo_id: str):
        """Read output from download process."""
        state = self.active_downloads.get(repo_id)
        if not state or not state.process:
            return
        
        loop = asyncio.get_event_loop()
        
        def read_line():
            if state.process and state.process.stdout:
                return state.process.stdout.readline()
            return ""
        
        while state.status == "downloading":
            try:
                line = await loop.run_in_executor(None, read_line)
                if line:
                    line = line.strip()
                    if line:
                        state.output_lines.append(line)
                        # Keep only last 10 lines
                        if len(state.output_lines) > 10:
                            state.output_lines.pop(0)
                else:
                    # Check if process ended
                    if state.process.poll() is not None:
                        state.status = "completed" if state.process.returncode == 0 else "error"
                        break
                await asyncio.sleep(0.1)
            except Exception:
                break
    
    def cancel_download(self, repo_id: str) -> bool:
        """Cancel an active download."""
        if repo_id in self.active_downloads:
            state = self.active_downloads[repo_id]
            if state.process and state.process.poll() is None:
                state.process.terminate()
                state.status = "cancelled"
                return True
        return False
    
    def clear_completed(self, repo_id: str):
        """Remove a completed/errored download from tracking."""
        if repo_id in self.active_downloads:
            state = self.active_downloads[repo_id]
            if state.status in ("completed", "error", "cancelled"):
                del self.active_downloads[repo_id]


# Global instance
download_manager = DownloadManager()
