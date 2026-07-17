from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator

from .models import InferenceMetrics, StreamEvent

_TOKEN_RE = re.compile(r"\S+")


class StreamProfiler:
    """Wraps a streamed response and records TTFT, approximate token throughput and ITL.

    Provider SDKs expose different token-level callbacks, so the harness deliberately records
    event timing in a provider-neutral way. Each whitespace-delimited unit is treated as an
    approximate token for consistent cross-provider experimental comparisons.
    """

    def __init__(self) -> None:
        self.metrics = InferenceMetrics(started_ns=time.perf_counter_ns())

    async def collect(self, stream: AsyncIterator[StreamEvent]) -> str:
        chunks: list[str] = []
        async for event in stream:
            if self.metrics.first_token_ns is None and event.text:
                self.metrics.first_token_ns = event.timestamp_ns
            chunks.append(event.text)
            tokens = _TOKEN_RE.findall(event.text)
            self.metrics.approx_tokens += len(tokens)
            # When a provider returns multiple text units in one event, assign the same arrival
            # timestamp. This keeps measurement honest rather than inventing synthetic latency.
            self.metrics.token_timestamps_ns.extend([event.timestamp_ns] * len(tokens))
        self.metrics.finished_ns = time.perf_counter_ns()
        return "".join(chunks)
