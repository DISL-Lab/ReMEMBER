import sys
import types

from remember_benchmark.providers import OpenAICompatibleProvider, VLLMProvider


def test_openai_provider_applies_configured_temperature_lazily(monkeypatch):
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok"))])

    class OpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setenv("FIXTURE_KEY_ENV", "fixture-value")
    provider = OpenAICompatibleProvider("fixture/model", api_key_env="FIXTURE_KEY_ENV", temperature=0.25)
    assert provider.complete("prompt") == "ok"
    assert captured["temperature"] == 0.25


def test_vllm_provider_is_lazy_applies_temperature_and_releases_engine(monkeypatch):
    events = []

    class Engine:
        def shutdown(self):
            events.append("shutdown")

    class LLM:
        def __init__(self, **kwargs):
            events.append(("load", kwargs))
            self.engine = Engine()

        def generate(self, prompts, params):
            events.append(("generate", params.temperature))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="ok")])]

        def shutdown(self):
            self.engine.shutdown()

    class SamplingParams:
        def __init__(self, *, temperature):
            self.temperature = temperature

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    provider = VLLMProvider("fixture/model", temperature=0.4)
    assert events == []
    assert provider.complete("prompt") == "ok"
    assert ("generate", 0.4) in events
    provider.release()
    assert events.count("shutdown") == 1
    assert provider._llm is None and provider._sampling_params is None
