"""
RyzenAI-LocalLab - Ollama Backend Service

Integrates with Ollama for fast local inference.
GPU-only mode via Docker ROCm.
"""

import httpx
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from backend.config import settings


@dataclass
class OllamaModel:
    """Information about an Ollama model."""
    name: str
    size: int  # bytes
    modified_at: str
    digest: str


class OllamaEngine:
    """
    Inference engine using Ollama API.
    
    GPU-only mode via Docker container with ROCm support.
    
    Ollama handles:
    - Model downloading via `ollama pull`
    - Quantization and optimization
    - Memory management
    - GPU acceleration (AMD ROCm)
    """
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.ollama_host
        self.current_model: Optional[str] = None
    
    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[OllamaModel]:
        """List all locally available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return []
                
                data = response.json()
                models = []
                for model in data.get("models", []):
                    models.append(OllamaModel(
                        name=model.get("name", ""),
                        size=model.get("size", 0),
                        modified_at=model.get("modified_at", ""),
                        digest=model.get("digest", ""),
                    ))
                return models
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> AsyncGenerator[dict, None]:
        """
        Pull/download a model with progress updates.
        
        Args:
            model_name: Model name (e.g., "qwen3:8b", "llama3.1:8b")
            
        Yields:
            Progress updates with status, completed, total
        """
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            yield {
                                "status": data.get("status", ""),
                                "completed": data.get("completed", 0),
                                "total": data.get("total", 0),
                                "digest": data.get("digest", ""),
                            }
        except Exception as e:
            yield {"status": "error", "error": str(e)}
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.base_url}/api/delete",
                    json={"name": model_name}
                )
                return response.status_code == 200
        except Exception:
            return False
    
    async def load_model(self, model_name: str) -> dict:
        """
        Load/warm up a model.
        
        This sends an empty generate request to load the model into memory.
        """
        self.current_model = model_name
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "",
                        "keep_alive": "10m",
                    }
                )
                return {"status": "loaded", "model": model_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def generate_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a streaming response using chat format.
        
        Args:
            model: Model name
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks with 'content' and optional 'done' flag with stats
        """
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_tokens,
                        },
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            
                            if data.get("done"):
                                # Final message with stats
                                yield {
                                    "done": True,
                                    "total_duration": data.get("total_duration", 0),
                                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                                    "eval_count": data.get("eval_count", 0),
                                    "eval_duration": data.get("eval_duration", 0),
                                }
                            else:
                                # Streaming content
                                message = data.get("message", {})
                                content = message.get("content", "")
                                if content:
                                    yield {"content": content}
                                    
        except Exception as e:
            yield {"error": str(e)}
    
    async def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> tuple[str, dict]:
        """
        Generate a non-streaming response.
        
        Returns:
            Tuple of (response_text, stats)
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "num_predict": max_tokens,
                        },
                    },
                )
                
                data = response.json()
                message = data.get("message", {})
                content = message.get("content", "")
                
                stats = {
                    "total_duration": data.get("total_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                }
                
                return content, stats
                
        except Exception as e:
            return f"Error: {e}", {}
    
    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "engine": "ollama",
            "base_url": self.base_url,
            "current_model": self.current_model,
        }


# Global instance
ollama_engine = OllamaEngine()
