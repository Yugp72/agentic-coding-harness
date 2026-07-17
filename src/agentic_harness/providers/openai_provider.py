from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from agentic_harness.models import StreamEvent
from agentic_harness.providers.base import ModelProvider


class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Install provider support with: pip install -e '.[openai]'") from exc
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.1-codex")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        # Responses streaming keeps the adapter close to Codex-style tool-capable models while
        # leaving code execution under control of this harness rather than the model provider.
        stream = await self.client.responses.create(
            model=self.model,
            input=prompt,
            stream=True,
        )
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield StreamEvent(text=event.delta, timestamp_ns=time.perf_counter_ns())
