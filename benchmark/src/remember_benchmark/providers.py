"""Injectable text providers; optional SDKs are imported only on first use."""

from __future__ import annotations

import os
import gc
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class StageProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class ScriptedProvider:
    responses: Sequence[str]
    calls: list[str] = field(default_factory=list)
    _index: int = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._index >= len(self.responses):
            raise RuntimeError("scripted provider has no response remaining")
        response = self.responses[self._index]
        self._index += 1
        return response


class OpenAICompatibleProvider:
    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = _temperature(temperature)
        self._client = None

    def complete(self, prompt: str) -> str:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("install remember-benchmark[openai] to use this provider") from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"missing API key environment variable {self.api_key_env}")
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("provider returned no text content")
        return content


class VLLMProvider:
    def __init__(self, model: str, *, device: str | None = None, temperature: float = 0) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self.device = device
        self.temperature = _temperature(temperature)
        self._llm = None
        self._sampling_params = None

    def complete(self, prompt: str) -> str:
        if self._llm is None:
            if self.device:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.device
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError("install remember-benchmark[vllm] to use this provider") from exc
            self._llm = LLM(model=self.model)
            self._sampling_params = SamplingParams(temperature=self.temperature)
        outputs = self._llm.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text

    def release(self) -> None:
        engine, self._llm = self._llm, None
        self._sampling_params = None
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                inner = getattr(engine, "llm_engine", None)
                shutdown = getattr(inner, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        gc.collect()


def _temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("temperature must be a non-negative number")
    return float(value)
