"""
RyzenAI-LocalLab - Unified Model Manager

Central orchestrator that ensures only ONE model is loaded at a time
across all backends (Ollama, HuggingFace, llama.cpp).

Provides:
- Prefix-based backend detection (ollama:, hf:, gguf:)
- Automatic unloading of previous models before loading new ones
- Unified status tracking and events for frontend
"""

import asyncio
import gc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Callable, Optional

from backend.config import settings


class Backend(str, Enum):
    """Supported inference backends."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    NONE = "none"


@dataclass
class ModelState:
    """Current state of the loaded model."""
    backend: Backend = Backend.NONE
    model_id: str = ""
    model_name: str = ""  # Display name without prefix
    loaded_at: float = 0.0
    status: str = "idle"  # idle, loading, unloading, ready, error
    error: Optional[str] = None


@dataclass
class LoadProgress:
    """Progress update during model loading/unloading."""
    status: str
    message: str
    progress: float = 0.0  # 0.0 to 1.0
    step: str = ""  # Current step name


class UnifiedEngine:
    """
    Unified Model Manager - One model at a time across all backends.
    
    Usage:
        # Load Ollama model
        async for progress in unified_engine.load_model("ollama:qwen3:8b"):
            print(progress.message)
        
        # Load HuggingFace model  
        async for progress in unified_engine.load_model("hf:mistralai/Devstral"):
            print(progress.message)
    """
    
    def __init__(self):
        self.state = ModelState()
        self._loading_lock = asyncio.Lock()
        self._gguf_load_path: str = ""  # Full path for GGUF loading
        
        # Lazy imports to avoid circular dependencies
        self._ollama_engine = None
        self._inference_engine = None
        self._llama_engine = None
    
    @property
    def ollama_engine(self):
        """Lazy load Ollama engine."""
        if self._ollama_engine is None:
            from backend.services.ollama_engine import ollama_engine
            self._ollama_engine = ollama_engine
        return self._ollama_engine
    
    @property
    def inference_engine(self):
        """Lazy load HuggingFace inference engine."""
        if self._inference_engine is None:
            from backend.services.inference_engine import inference_engine
            self._inference_engine = inference_engine
        return self._inference_engine
    
    @property
    def llama_engine(self):
        """Lazy load llama.cpp engine."""
        if self._llama_engine is None:
            from backend.services.llama_engine import llama_engine
            self._llama_engine = llama_engine
        return self._llama_engine
    
    def parse_model_id(self, model_id: str) -> tuple[Backend, str]:
        """
        Parse model ID to extract backend and model name.
        
        Supported formats:
            ollama:qwen3:8b -> (OLLAMA, "qwen3:8b")
            hf:mistralai/Devstral -> (HUGGINGFACE, "mistralai/Devstral")
            gguf:/path/to/model.gguf -> (LLAMA_CPP, "model-name")
            
        Legacy (no prefix):
            If contains "/" -> assume HuggingFace
            Otherwise -> assume Ollama
        """
        if model_id.startswith("ollama:"):
            return Backend.OLLAMA, model_id[7:]
        elif model_id.startswith("hf:"):
            return Backend.HUGGINGFACE, model_id[3:]
        elif model_id.startswith("gguf:"):
            # Extract filename without extension for display
            from pathlib import Path
            path = model_id[5:]
            filename = Path(path).stem  # Gets filename without .gguf extension
            # Store full path in a separate attribute for loading
            self._gguf_load_path = path
            return Backend.LLAMA_CPP, filename
        else:
            # Legacy detection
            if "/" in model_id and not model_id.startswith("/"):
                return Backend.HUGGINGFACE, model_id
            else:
                return Backend.OLLAMA, model_id
    
    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "backend": self.state.backend.value,
            "model_id": self.state.model_id,
            "model_name": self.state.model_name,
            "status": self.state.status,
            "loaded_at": self.state.loaded_at,
            "error": self.state.error,
        }
    
    def is_loaded(self) -> bool:
        """Check if any model is currently loaded."""
        return self.state.backend != Backend.NONE and self.state.status == "ready"
    
    async def unload_all(self) -> AsyncGenerator[LoadProgress, None]:
        """
        Unload ALL backends. Yields progress updates.
        """
        if self.state.backend == Backend.NONE:
            yield LoadProgress(
                status="info",
                message="No model currently loaded",
                progress=1.0,
                step="check"
            )
            return
        
        self.state.status = "unloading"
        current_backend = self.state.backend
        current_model = self.state.model_name
        
        yield LoadProgress(
            status="unloading",
            message=f"⏳ Unloading {current_model} from {current_backend.value}...",
            progress=0.1,
            step="start"
        )
        
        try:
            # Unload based on current backend
            if current_backend == Backend.OLLAMA:
                yield LoadProgress(
                    status="unloading",
                    message=f"🦙 Releasing {current_model} from Ollama...",
                    progress=0.3,
                    step="ollama_unload"
                )
                await self._unload_ollama()
                
            elif current_backend == Backend.HUGGINGFACE:
                yield LoadProgress(
                    status="unloading",
                    message=f"🤗 Unloading {current_model} from HuggingFace...",
                    progress=0.3,
                    step="hf_unload"
                )
                await self.inference_engine.unload_model()
                
            elif current_backend == Backend.LLAMA_CPP:
                yield LoadProgress(
                    status="unloading",
                    message=f"🦙 Unloading {current_model} from llama.cpp...",
                    progress=0.3,
                    step="gguf_unload"
                )
                self.llama_engine.unload_model()
            
            # Force garbage collection
            yield LoadProgress(
                status="unloading",
                message="🧹 Cleaning up GPU memory...",
                progress=0.8,
                step="cleanup"
            )
            gc.collect()
            
            # Try to clear CUDA/ROCm cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            # Small delay to let GPU memory settle
            await asyncio.sleep(0.5)
            
            # Reset state
            self.state = ModelState()
            
            yield LoadProgress(
                status="success",
                message=f"✅ {current_model} unloaded successfully",
                progress=1.0,
                step="done"
            )
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            yield LoadProgress(
                status="error",
                message=f"❌ Failed to unload: {e}",
                progress=0.0,
                step="error"
            )
    
    async def _unload_ollama(self) -> None:
        """Unload current Ollama model using keep_alive=0."""
        if not self.ollama_engine.current_model:
            return
            
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self.ollama_engine.base_url}/api/generate",
                    json={
                        "model": self.ollama_engine.current_model,
                        "keep_alive": 0,
                        "prompt": ""
                    }
                )
            self.ollama_engine.current_model = None
        except Exception:
            # Even if unload fails, clear state
            self.ollama_engine.current_model = None
    
    async def load_model(self, model_id: str) -> AsyncGenerator[LoadProgress, None]:
        """
        Load a model, unloading any currently loaded model first.
        
        Args:
            model_id: Model identifier with optional prefix
                      (ollama:model, hf:repo/model, gguf:/path/model.gguf)
        
        Yields:
            LoadProgress updates for frontend display
        """
        # Prevent concurrent loads
        if self._loading_lock.locked():
            yield LoadProgress(
                status="error",
                message="⚠️ Another model is currently being loaded. Please wait.",
                progress=0.0,
                step="blocked"
            )
            return
        
        async with self._loading_lock:
            # Parse backend and model name
            backend, model_name = self.parse_model_id(model_id)
            
            yield LoadProgress(
                status="starting",
                message=f"🚀 Preparing to load {model_name}...",
                progress=0.05,
                step="parse"
            )
            
            # Unload current model first (if any)
            if self.state.backend != Backend.NONE:
                yield LoadProgress(
                    status="unloading",
                    message=f"⏳ Need to unload current model first...",
                    progress=0.1,
                    step="pre_unload"
                )
                async for progress in self.unload_all():
                    # Scale progress to 10-30%
                    progress.progress = 0.1 + progress.progress * 0.2
                    yield progress
            
            # Update state to loading
            self.state.status = "loading"
            self.state.backend = backend
            self.state.model_id = model_id
            self.state.model_name = model_name
            self.state.error = None
            
            try:
                # Load based on backend
                if backend == Backend.OLLAMA:
                    async for progress in self._load_ollama(model_name):
                        yield progress
                        
                elif backend == Backend.HUGGINGFACE:
                    async for progress in self._load_huggingface(model_name):
                        yield progress
                        
                elif backend == Backend.LLAMA_CPP:
                    async for progress in self._load_llama_cpp(model_name):
                        yield progress
                
                # Success
                self.state.status = "ready"
                self.state.loaded_at = time.time()
                
                yield LoadProgress(
                    status="success",
                    message=f"✅ {model_name} loaded and ready!",
                    progress=1.0,
                    step="ready"
                )
                
            except Exception as e:
                self.state.status = "error"
                self.state.error = str(e)
                self.state.backend = Backend.NONE
                
                yield LoadProgress(
                    status="error",
                    message=f"❌ Failed to load model: {e}",
                    progress=0.0,
                    step="error"
                )
    
    async def _load_ollama(self, model_name: str) -> AsyncGenerator[LoadProgress, None]:
        """Load an Ollama model."""
        yield LoadProgress(
            status="loading",
            message=f"🦙 Connecting to Ollama...",
            progress=0.35,
            step="ollama_connect"
        )
        
        # Check if Ollama is available
        if not await self.ollama_engine.is_available():
            raise RuntimeError("Ollama is not running. Please start Ollama first.")
        
        yield LoadProgress(
            status="loading",
            message=f"🦙 Loading {model_name} into GPU memory...",
            progress=0.5,
            step="ollama_load"
        )
        
        # Load model (warm up)
        result = await self.ollama_engine.load_model(model_name)
        
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "Unknown Ollama error"))
        
        yield LoadProgress(
            status="loading",
            message=f"🦙 {model_name} loaded, warming up...",
            progress=0.9,
            step="ollama_warmup"
        )
    
    async def _load_huggingface(self, model_path: str) -> AsyncGenerator[LoadProgress, None]:
        """Load a HuggingFace model."""
        yield LoadProgress(
            status="loading",
            message=f"🤗 Loading {model_path}...",
            progress=0.35,
            step="hf_start"
        )
        
        yield LoadProgress(
            status="loading",
            message=f"🤗 Loading tokenizer...",
            progress=0.45,
            step="hf_tokenizer"
        )
        
        yield LoadProgress(
            status="loading",
            message=f"🤗 Loading model weights (this may take a while)...",
            progress=0.55,
            step="hf_weights"
        )
        
        # Load model
        await self.inference_engine.load_model(model_path)
        
        yield LoadProgress(
            status="loading",
            message=f"🤗 Model loaded, preparing for inference...",
            progress=0.9,
            step="hf_ready"
        )
    
    async def _load_llama_cpp(self, model_path: str) -> AsyncGenerator[LoadProgress, None]:
        """Load a llama.cpp GGUF model."""
        # Use the stored full path for loading (model_path here is just display name)
        actual_path = getattr(self, '_gguf_load_path', model_path)
        
        yield LoadProgress(
            status="loading",
            message=f"📦 Loading GGUF model...",
            progress=0.35,
            step="gguf_start"
        )
        
        yield LoadProgress(
            status="loading",
            message=f"📦 Loading {model_path} into GPU...",
            progress=0.5,
            step="gguf_load"
        )
        
        # Load model (sync, run in executor)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.llama_engine.load_model(actual_path)
        )
        
        yield LoadProgress(
            status="loading",
            message=f"📦 GGUF model loaded and optimized",
            progress=0.9,
            step="gguf_ready"
        )
    
    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stream: bool = True,
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a response using the currently loaded model.
        
        Works with any backend transparently.
        """
        if not self.is_loaded():
            yield {"error": "No model loaded. Please load a model first."}
            return
        
        if self.state.backend == Backend.OLLAMA:
            async for chunk in self.ollama_engine.generate_stream(
                model=self.state.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                yield chunk
                
        elif self.state.backend == Backend.HUGGINGFACE:
            from backend.services.inference_engine import GenerationParams
            params = GenerationParams(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            async for token, stats in self.inference_engine.generate_stream(
                messages=messages,
                params=params,
            ):
                yield {"content": token, "stats": stats}
                
        elif self.state.backend == Backend.LLAMA_CPP:
            from backend.services.llama_engine import GenerationParams
            params = GenerationParams(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            async for token, stats in self.llama_engine.generate_stream(
                messages=messages,
                params=params,
            ):
                yield {"content": token, "stats": stats}


# Global singleton instance
unified_engine = UnifiedEngine()
