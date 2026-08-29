"""Lazy optional inference adapters for public ReMEMber runs."""

from __future__ import annotations

import hashlib
import gc
import json
import os
from typing import Protocol, Sequence


class TextGenerator(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ) -> str: ...


class DeterministicFakeGenerator:
    """Offline smoke adapter; its output is fixed and contains no source text."""

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ) -> str:
        del model, temperature, max_tokens
        if "Reinforced Dialogue Summary" in prompt or "Dialogue Summary" in prompt:
            return json.dumps({"summary": "Deterministic fake summary."})
        return json.dumps({"gaps": [{
            "utterance_index": 0,
            "quote": "current window",
            "gap_reason": "The referenced context is not identified in the current window.",
            "evidence_need_query": "retrieve earlier dialogue identifying the referenced context",
            "keywords": ["context"],
        }]})


class DeterministicHashEmbedder:
    """Small deterministic CPU-only embedder intended for smoke tests only."""

    def __init__(self, dimensions: int = 32) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            vectors.append(vector)
        return vectors


class OpenAICompatibleGenerator:
    """Chat-completions adapter initialized only on the first request."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ) -> str:
        if not model:
            raise ValueError("model is required")
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        return message.content or getattr(message, "reasoning_content", None) or ""


class VLLMGenerator:
    """Lazy in-process vLLM adapter with device visibility set before import."""

    def __init__(
        self,
        *,
        gpu_devices: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        if gpu_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._tokenizer = None
        self._model: str | None = None

    def _load(self, model: str):
        if self._llm is not None and self._model == model:
            return self._llm, self._tokenizer
        if self._llm is not None:
            self._release()
        from vllm import LLM

        kwargs = {"model": model, "gpu_memory_utilization": self.gpu_memory_utilization}
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._model = model
        return self._llm, self._tokenizer

    def _release(self) -> None:
        llm = self._llm
        self._llm = None
        self._tokenizer = None
        self._model = None
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del engine, llm
        gc.collect()

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 768,
    ) -> str:
        if not model:
            raise ValueError("model is required")
        llm, tokenizer = self._load(model)
        from vllm import SamplingParams

        messages = [{"role": "user", "content": prompt}]
        options = {"tokenize": False, "add_generation_prompt": True}
        try:
            rendered = tokenizer.apply_chat_template(messages, **options, enable_thinking=False)
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, **options)
        [output] = llm.generate(
            [rendered],
            SamplingParams(max_tokens=max_tokens, temperature=temperature),
            use_tqdm=False,
        )
        return output.outputs[0].text


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers encoder with optional explicit device scope."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        gpu_devices: str | None = None,
        batch_size: int = 128,
    ) -> None:
        if gpu_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._encoder = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model, device=self.device)
        return self._encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


class WhitespaceTokenCounter:
    """Dependency-free approximate counter for smoke runs.

    It does NOT reproduce the paper's token accounting; runs that use it are
    labelled by `counter_adapter` in the output manifest.
    """

    def count(self, text: str) -> int:
        return max(1, len((text or "").split()))


class Cl100kTokenCounter:
    """Lazy exact token accounting retained from the paper implementation."""

    def __init__(self) -> None:
        self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        return len(self._encoding.encode(text or ""))
