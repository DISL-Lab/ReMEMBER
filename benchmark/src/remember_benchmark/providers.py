"""Injectable text providers; optional SDKs are imported only on first use."""

from __future__ import annotations

import os
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
    def __init__(self, model: str, *, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY") -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
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
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("provider returned no text content")
        return content


class VLLMProvider:
    def __init__(self, model: str, *, device: str | None = None) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self.device = device
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
            self._sampling_params = SamplingParams(temperature=0)
        outputs = self._llm.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text
