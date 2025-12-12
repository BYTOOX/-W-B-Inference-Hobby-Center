"""
RyzenAI-LocalLab Hardware Monitor

Real-time system monitoring for CPU, RAM, and GPU (ROCm).
Provides stats for the HomeLab dashboard.
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class CPUStats:
    """CPU statistics."""

    load_percent: float
    load_per_core: list[float]
    frequency_mhz: float
    frequency_max_mhz: float
    core_count: int
    thread_count: int
    temperature: Optional[float] = None


@dataclass
class MemoryStats:
    """Memory (RAM) statistics."""

    total_gb: float
    used_gb: float
    available_gb: float
    percent_used: float


@dataclass
class GPUStats:
    """GPU statistics (ROCm)."""

    available: bool
    name: str = "Unknown"
    utilization_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    power_watts: float = 0.0
    driver_version: str = "Unknown"


@dataclass
class SystemStats:
    """Combined system statistics."""

    cpu: CPUStats
    memory: MemoryStats
    gpu: GPUStats


class HardwareMonitor:
    """
    Hardware monitoring service.

    Provides real-time stats for CPU, RAM, and GPU.
    GPU monitoring uses ROCm's rocm-smi tool.
    """

    def __init__(self):
        self._rocm_available: Optional[bool] = None
        self._gpu_name: Optional[str] = None

    def get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics."""
        # Get per-core load
        per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # Get frequency
        freq = psutil.cpu_freq()
        current_freq = freq.current if freq else 0
        max_freq = freq.max if freq else 0

        # Get core/thread count
        core_count = psutil.cpu_count(logical=False) or 1
        thread_count = psutil.cpu_count(logical=True) or 1

        # Try to get temperature (Linux only)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Look for CPU temperature
                for name in ["coretemp", "k10temp", "zenpower", "cpu_thermal"]:
                    if name in temps:
                        temperature = temps[name][0].current
                        break
        except Exception:
            pass

        return CPUStats(
            load_percent=sum(per_core) / len(per_core) if per_core else 0,
            load_per_core=per_core,
            frequency_mhz=current_freq,
            frequency_max_mhz=max_freq,
            core_count=core_count,
            thread_count=thread_count,
            temperature=temperature,
        )

    def get_memory_stats(self) -> MemoryStats:
        """Get memory (RAM) statistics."""
        mem = psutil.virtual_memory()

        return MemoryStats(
            total_gb=mem.total / (1024**3),
            used_gb=mem.used / (1024**3),
            available_gb=mem.available / (1024**3),
            percent_used=mem.percent,
        )

    def _check_rocm_available(self) -> bool:
        """Check if ROCm is available."""
        if self._rocm_available is not None:
            return self._rocm_available

        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._rocm_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._rocm_available = False

        return self._rocm_available

    def get_gpu_stats(self) -> GPUStats:
        """
        Get GPU statistics using ROCm.

        Falls back to PyTorch if rocm-smi is not available.
        """
        if not self._check_rocm_available():
            return self._get_gpu_stats_pytorch()

        try:
            # Get JSON output from rocm-smi
            result = subprocess.run(
                [
                    "rocm-smi",
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                    "--showtemp",
                    "--showpower",
                    "--showproductname",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return GPUStats(available=False)

            data = json.loads(result.stdout)

            # Parse first GPU (card0)
            gpu_data = data.get("card0", {})

            # Extract values
            name = gpu_data.get("Card Series", "AMD GPU")
            utilization = float(gpu_data.get("GPU use (%)", 0))

            # Memory (in bytes)
            mem_used = int(gpu_data.get("VRAM Total Used Memory (B)", 0))
            mem_total = int(gpu_data.get("VRAM Total Memory (B)", 0))

            # Temperature
            temp = float(gpu_data.get("Temperature (Sensor edge) (C)", 0))

            # Power
            power = float(gpu_data.get("Average Graphics Package Power (W)", 0))

            # Cache GPU name
            if self._gpu_name is None:
                self._gpu_name = name

            return GPUStats(
                available=True,
                name=name,
                utilization_percent=utilization,
                memory_used_gb=mem_used / (1024**3),
                memory_total_gb=mem_total / (1024**3),
                memory_percent=(mem_used / mem_total * 100) if mem_total > 0 else 0,
                temperature_c=temp,
                power_watts=power,
            )

        except Exception as e:
            print(f"ROCm monitoring error: {e}")
            return self._get_gpu_stats_pytorch()

    def _get_gpu_stats_pytorch(self) -> GPUStats:
        """Fallback GPU stats using PyTorch."""
        try:
            import torch

            if not torch.cuda.is_available():
                return GPUStats(available=False)

            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)

            # Memory info
            mem_total = props.total_memory
            mem_used = torch.cuda.memory_allocated(device)

            return GPUStats(
                available=True,
                name=name,
                utilization_percent=0,  # Not available via PyTorch
                memory_used_gb=mem_used / (1024**3),
                memory_total_gb=mem_total / (1024**3),
                memory_percent=(mem_used / mem_total * 100) if mem_total > 0 else 0,
            )

        except Exception:
            return GPUStats(available=False)

    def get_system_stats(self) -> SystemStats:
        """Get all system statistics."""
        return SystemStats(
            cpu=self.get_cpu_stats(),
            memory=self.get_memory_stats(),
            gpu=self.get_gpu_stats(),
        )

    def to_dict(self) -> dict:
        """Get all stats as a dictionary (for JSON API)."""
        stats = self.get_system_stats()

        return {
            "cpu": {
                "load_percent": round(stats.cpu.load_percent, 1),
                "load_per_core": [round(c, 1) for c in stats.cpu.load_per_core],
                "frequency_mhz": round(stats.cpu.frequency_mhz, 0),
                "frequency_max_mhz": round(stats.cpu.frequency_max_mhz, 0),
                "core_count": stats.cpu.core_count,
                "thread_count": stats.cpu.thread_count,
                "temperature_c": round(stats.cpu.temperature, 1) if stats.cpu.temperature else None,
            },
            "memory": {
                "total_gb": round(stats.memory.total_gb, 1),
                "used_gb": round(stats.memory.used_gb, 1),
                "available_gb": round(stats.memory.available_gb, 1),
                "percent_used": round(stats.memory.percent_used, 1),
            },
            "gpu": {
                "available": stats.gpu.available,
                "name": stats.gpu.name,
                "utilization_percent": round(stats.gpu.utilization_percent, 1),
                "memory_used_gb": round(stats.gpu.memory_used_gb, 2),
                "memory_total_gb": round(stats.gpu.memory_total_gb, 2),
                "memory_percent": round(stats.gpu.memory_percent, 1),
                "temperature_c": round(stats.gpu.temperature_c, 1),
                "power_watts": round(stats.gpu.power_watts, 1),
            },
        }


# Singleton instance
hardware_monitor = HardwareMonitor()
