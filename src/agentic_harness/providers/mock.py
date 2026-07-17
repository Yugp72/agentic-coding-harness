from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from agentic_harness.models import StreamEvent
from agentic_harness.providers.base import ModelProvider


class MockProvider(ModelProvider):
    """Deterministic provider used for CI/demo runs without API keys."""

    name = "mock"
    model = "deterministic-fixture"

    def _answer(self, prompt: str) -> str:
        if "Task ID: fizzbuzz" in prompt:
            return '''def fizzbuzz(n: int) -> list[str]:\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append("FizzBuzz")\n        elif i % 3 == 0:\n            out.append("Fizz")\n        elif i % 5 == 0:\n            out.append("Buzz")\n        else:\n            out.append(str(i))\n    return out\n'''
        if "Task ID: balanced-brackets" in prompt:
            return '''def is_balanced(text: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in text:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n'''
        if "Task ID: lru-cache" in prompt:
            return '''from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        if capacity <= 0:\n            raise ValueError("capacity must be positive")\n        self.capacity = capacity\n        self._d = OrderedDict()\n\n    def get(self, key):\n        if key not in self._d:\n            return -1\n        self._d.move_to_end(key)\n        return self._d[key]\n\n    def put(self, key, value):\n        if key in self._d:\n            self._d.move_to_end(key)\n        self._d[key] = value\n        if len(self._d) > self.capacity:\n            self._d.popitem(last=False)\n'''
        return "def solve(*args, **kwargs):\n    raise NotImplementedError('mock has no fixture')\n"

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        answer = self._answer(prompt)
        # Stream in small word-like chunks so profiling is exercised in tests.
        parts = answer.split(" ")
        for i, part in enumerate(parts):
            await asyncio.sleep(0.001)
            suffix = " " if i < len(parts) - 1 else ""
            yield StreamEvent(text=part + suffix, timestamp_ns=time.perf_counter_ns())
