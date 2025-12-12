"""
RyzenAI-LocalLab Inference Engine

Handles model loading and text generation using HuggingFace Transformers.
Optimized for AMD ROCm with unified memory architecture.
"""

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)

from backend.config import settings


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationStats:
    """Statistics from text generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: float = 0.0  # seconds
    generation_time: float = 0.0  # seconds
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0


@dataclass
class LoadedModelInfo:
    """Information about a loaded model."""

    model_id: str
    model_path: Path
    device: str
    dtype: str
    memory_used_gb: float
    loaded_at: float
    tokenizer_name: str


class InferenceEngine:
    """
    Inference engine for text generation.

    Handles:
    - Model loading with device detection
    - Streaming text generation
    - Performance metrics
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded_model_info: Optional[LoadedModelInfo] = None
        self._device: Optional[str] = None

    # =========================================================================
    # Device Detection
    # =========================================================================
    @property
    def device(self) -> str:
        """Get the compute device."""
        if self._device is not None:
            return self._device

        preference = settings.device

        if preference == "cpu":
            self._device = "cpu"
        elif preference in ("cuda", "auto"):
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = "cpu"

        return self._device

    def get_device_info(self) -> dict:
        """Get information about the compute device."""
        info = {
            "device": self.device,
            "device_name": "CPU",
            "memory_total_gb": 0,
            "memory_used_gb": 0,
            "cuda_available": torch.cuda.is_available(),
            "rocm": False,
        }

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = torch.cuda.get_device_name(0)
            info["memory_total_gb"] = props.total_memory / (1024**3)
            info["memory_used_gb"] = torch.cuda.memory_allocated(0) / (1024**3)

            # Check if using ROCm
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                info["rocm"] = True

        return info

    # =========================================================================
    # Model Loading
    # =========================================================================
    async def load_model(
        self,
        model_path: str | Path,
        quantization: Optional[str] = None,
        dtype: str = "auto",
    ) -> LoadedModelInfo:
        """
        Load a model for inference.

        Args:
            model_path: Path to the model directory or HuggingFace ID
            quantization: Optional quantization ("4bit", "8bit", None)
            dtype: Data type ("auto", "float16", "bfloat16", "float32")

        Returns:
            LoadedModelInfo with details about the loaded model
        """
        # Unload existing model
        if self.model is not None:
            await self.unload_model()

        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        model_id = model_path.name

        print(f"Loading model: {model_id}")
        print(f"Device: {self.device}")

        # Determine dtype
        if dtype == "auto":
            if self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Setup quantization config if requested
        quantization_config = None
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print("Loading model weights...")
        start_time = time.time()

        model_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            # For unified memory, offload everything to GPU
            model_kwargs["device_map"] = "auto"
            # Allow using almost all memory
            max_memory = {0: f"{int(settings.max_model_memory_fraction * 100)}%", "cpu": "32GB"}
            model_kwargs["max_memory"] = max_memory
        else:
            model_kwargs["device_map"] = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.1f}s")

        # Get memory usage
        memory_used = 0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / (1024**3)

        self.loaded_model_info = LoadedModelInfo(
            model_id=model_id,
            model_path=model_path,
            device=self.device,
            dtype=str(torch_dtype).replace("torch.", ""),
            memory_used_gb=memory_used,
            loaded_at=time.time(),
            tokenizer_name=self.tokenizer.name_or_path,
        )

        return self.loaded_model_info

    async def unload_model(self) -> None:
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.loaded_model_info = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        print("Model unloaded")

    # =========================================================================
    # Text Generation
    # =========================================================================
    async def generate_stream(
        self,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[tuple[str, Optional[GenerationStats]], None]:
        """
        Generate text with streaming output.

        Args:
            messages: List of message dicts with "role" and "content"
            params: Generation parameters
            system_prompt: Optional system prompt to prepend

        Yields:
            Tuples of (token_text, stats) where stats is only set on the last yield
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Build conversation
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        conversation.extend(messages)

        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for models without chat template
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in conversation])
            prompt += "\nassistant: "

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature if params.do_sample else 1.0,
            top_p=params.top_p if params.do_sample else 1.0,
            top_k=params.top_k if params.do_sample else 50,
            repetition_penalty=params.repetition_penalty,
            do_sample=params.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Start generation in background
        import threading

        generation_kwargs = {
            **inputs,
            "generation_config": gen_config,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream tokens
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        generated_text = ""

        for token in streamer:
            if first_token_time is None:
                first_token_time = time.time()

            completion_tokens += 1
            generated_text += token
            yield token, None

        thread.join()

        # Calculate stats
        end_time = time.time()
        generation_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0

        stats = GenerationStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token=ttft,
            generation_time=generation_time,
            tokens_per_second=completion_tokens / generation_time if generation_time > 0 else 0,
            prompt_tokens_per_second=prompt_tokens / ttft if ttft > 0 else 0,
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

        Args:
            messages: List of message dicts
            params: Generation parameters
            system_prompt: Optional system prompt

        Returns:
            Tuple of (generated_text, stats)
        """
        generated_text = ""
        stats = None

        async for token, token_stats in self.generate_stream(messages, params, system_prompt):
            generated_text += token
            if token_stats:
                stats = token_stats

        return generated_text, stats

    # =========================================================================
    # Status
    # =========================================================================
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None

    def get_status(self) -> dict:
        """Get current engine status."""
        status = {
            "loaded": self.is_loaded(),
            "device": self.device,
            "device_info": self.get_device_info(),
            "model": None,
        }

        if self.loaded_model_info:
            status["model"] = {
                "id": self.loaded_model_info.model_id,
                "path": str(self.loaded_model_info.model_path),
                "device": self.loaded_model_info.device,
                "dtype": self.loaded_model_info.dtype,
                "memory_gb": round(self.loaded_model_info.memory_used_gb, 2),
            }

        return status


# Singleton instance
inference_engine = InferenceEngine()
