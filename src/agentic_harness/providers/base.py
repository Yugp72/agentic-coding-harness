from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from agentic_harness.models import StreamEvent


class ModelProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
