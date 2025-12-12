"""
RyzenAI-LocalLab - Llama.cpp Backend Service

Optimized inference using llama-cpp-python for GGUF models.
Best performance on AMD Ryzen AI MAX+ with ROCm.
"""

import asyncio
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from backend.config import settings


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationStats:
    """Statistics from text generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: float = 0.0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class LlamaModelInfo:
    """Information about a loaded llama.cpp model."""

    model_id: str
    model_path: Path
    context_length: int
    n_gpu_layers: int
    loaded_at: float


class LlamaCppEngine:
    """
    Inference engine using llama-cpp-python.

    Optimized for:
    - AMD Ryzen AI MAX+ with unified memory
    - GGUF model format
    - ROCm/HIP acceleration
    """

    def __init__(self):
        self.model = None
        self.loaded_model_info: Optional[LlamaModelInfo] = None

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None

    def get_status(self) -> dict:
        """Get engine status."""
        if not self.is_loaded():
            return {"loaded": False}

        return {
            "loaded": True,
            "model_id": self.loaded_model_info.model_id,
            "model_path": str(self.loaded_model_info.model_path),
            "context_length": self.loaded_model_info.context_length,
            "n_gpu_layers": self.loaded_model_info.n_gpu_layers,
        }

    async def load_model(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ) -> LlamaModelInfo:
        """
        Load a GGUF model.

        Args:
            model_path: Path to GGUF file or directory containing GGUF files
            n_ctx: Context length (max tokens)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)

        Returns:
            LlamaModelInfo with model details
        """
        from llama_cpp import Llama

        # Unload existing model
        if self.is_loaded():
            await self.unload_model()

        model_path = Path(model_path)

        # Find GGUF file
        if model_path.is_dir():
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {model_path}")
            # Prefer Q4_K_M or Q5_K_M quantizations
            preferred = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q8_0"]
            model_file = gguf_files[0]
            for pref in preferred:
                for f in gguf_files:
                    if pref in f.name:
                        model_file = f
                        break
        else:
            model_file = model_path

        print(f"Loading GGUF model: {model_file.name}")
        print(f"Context length: {n_ctx}")
        print(f"GPU layers: {n_gpu_layers} (-1 = all)")

        start_time = time.time()

        # Run model loading in executor (it's CPU-bound)
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=str(model_file),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True,
                n_threads=None,  # Auto-detect
                use_mmap=True,
                use_mlock=False,
            ),
        )

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.1f}s")

        self.loaded_model_info = LlamaModelInfo(
            model_id=model_file.stem,
            model_path=model_path,
            context_length=n_ctx,
            n_gpu_layers=n_gpu_layers,
            loaded_at=time.time(),
        )

        return self.loaded_model_info

    async def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        self.loaded_model_info = None
        gc.collect()
        print("Model unloaded")

    def _build_prompt(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build prompt from messages.
        Uses ChatML format which works with most models.
        """
        parts = []

        # System prompt
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # Messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add assistant start
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    async def generate_stream(
        self,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[tuple[str, Optional[GenerationStats]], None]:
        """
        Generate text with streaming.

        Args:
            messages: Chat messages in OpenAI format
            params: Generation parameters
            system_prompt: Optional system prompt

        Yields:
            Tuples of (token, stats) where stats is only set on the final yield
        """
        if not self.is_loaded():
            raise RuntimeError("No model loaded")

        prompt = self._build_prompt(messages, system_prompt)

        # Prepare stop sequences
        stop_sequences = params.stop_sequences.copy()
        stop_sequences.extend(["<|im_end|>", "<|im_start|>", "</s>"])

        generation_start = time.time()
        first_token_time = None
        completion_tokens = 0

        # Run generation in executor with streaming
        loop = asyncio.get_event_loop()

        def generate_sync():
            return self.model(
                prompt,
                max_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repetition_penalty,
                stop=stop_sequences,
                stream=True,
                echo=False,
            )

        # Start generation
        stream = await loop.run_in_executor(None, generate_sync)

        for output in stream:
            token = output["choices"][0]["text"]
            if token:
                if first_token_time is None:
                    first_token_time = time.time()

                completion_tokens += 1
                yield token, None

        # Calculate final stats
        generation_time = time.time() - generation_start
        ttft = (first_token_time - generation_start) if first_token_time else 0

        stats = GenerationStats(
            prompt_tokens=len(prompt.split()),  # Approximate
            completion_tokens=completion_tokens,
            total_tokens=len(prompt.split()) + completion_tokens,
            time_to_first_token=ttft,
            generation_time=generation_time,
            tokens_per_second=completion_tokens / generation_time if generation_time > 0 else 0,
        )

        yield "", stats

    async def generate(
        self,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, GenerationStats]:
        """
        Generate text without streaming.

        Returns:
            Tuple of (generated_text, stats)
        """
        full_text = ""
        stats = None

        async for token, final_stats in self.generate_stream(messages, params, system_prompt):
            full_text += token
            if final_stats:
                stats = final_stats

        return full_text, stats


# Global instance
llama_engine = LlamaCppEngine()
