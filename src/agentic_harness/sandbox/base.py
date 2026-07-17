from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_harness.models import BenchmarkTask, SandboxResult


class Sandbox(ABC):
    @abstractmethod
    async def run(self, task: BenchmarkTask, code: str) -> SandboxResult:
        raise NotImplementedError
