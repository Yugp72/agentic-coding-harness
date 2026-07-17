from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from agentic_harness.models import StreamEvent
from agentic_harness.providers.base import ModelProvider


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError("Install provider support with: pip install -e '.[anthropic]'") from exc
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield StreamEvent(text=text, timestamp_ns=time.perf_counter_ns())
