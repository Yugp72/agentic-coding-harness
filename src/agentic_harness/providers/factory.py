from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base import ModelProvider
from .mock import MockProvider
from .openai_provider import OpenAIProvider


def create_provider(name: str, model: str | None = None) -> ModelProvider:
    normalized = name.lower()
    if normalized == "mock":
        return MockProvider()
    if normalized in {"openai", "codex"}:
        return OpenAIProvider(model=model)
    if normalized in {"anthropic", "claude"}:
        return AnthropicProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")
