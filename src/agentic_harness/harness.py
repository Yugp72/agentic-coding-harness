from __future__ import annotations

import asyncio
import time

from .metrics import StreamProfiler
from .models import AgentAttempt, BenchmarkTask, HarnessConfig, RunResult
from .prompt_router import PromptRouter
from .providers.base import ModelProvider
from .sandbox.base import Sandbox


class CodingHarness:
    def __init__(
        self,
        provider: ModelProvider,
        sandbox: Sandbox,
        config: HarnessConfig | None = None,
    ) -> None:
        self.provider = provider
        self.sandbox = sandbox
        self.config = config or HarnessConfig()
        self.router = PromptRouter()

    async def _lane(self, task: BenchmarkTask, candidate: int) -> list[AgentAttempt]:
        attempts: list[AgentAttempt] = []
        previous_code = ""
        stdout = ""
        stderr = ""

        for iteration in range(1, self.config.max_iterations + 1):
            if iteration == 1:
                prompt = self.router.initial(task, candidate)
                prompt_kind = "initial"
            else:
                prompt = self.router.repair(task, previous_code, stdout, stderr, candidate)
                prompt_kind = "repair"

            profiler = StreamProfiler()
            code = (await profiler.collect(self.provider.stream(prompt))).strip()
            sandbox_result = await self.sandbox.run(task, code)
            attempt = AgentAttempt(
                attempt=iteration,
                candidate=candidate,
                code=code,
                provider=self.provider.name,
                model=self.provider.model,
                inference=profiler.metrics,
                sandbox=sandbox_result,
                prompt_kind=prompt_kind,
            )
            attempts.append(attempt)
            if sandbox_result.passed:
                break
            previous_code = code
            stdout = sandbox_result.stdout
            stderr = sandbox_result.stderr
        return attempts

    async def run(self, task: BenchmarkTask) -> RunResult:
        started = time.perf_counter_ns()
        lanes = await asyncio.gather(
            *[self._lane(task, i) for i in range(self.config.parallel_candidates)]
        )
        attempts = [attempt for lane in lanes for attempt in lane]
        successes = [attempt for attempt in attempts if attempt.sandbox.passed]
        if successes:
            # Prefer the successful candidate that required the fewest repair iterations, then
            # lower wall inference latency. This is an execution heuristic, not an LLM judge.
            best = min(
                successes,
                key=lambda a: (a.attempt, a.inference.elapsed_s or float("inf"), a.candidate),
            )
            success = True
        else:
            # Preserve the latest candidate for inspection even when all lanes fail.
            best = max(attempts, key=lambda a: (a.attempt, -a.candidate))
            success = False
        wall_time_ms = (time.perf_counter_ns() - started) / 1_000_000
        return RunResult(
            task=task,
            success=success,
            best_code=best.code,
            attempts=attempts,
            wall_time_ms=wall_time_ms,
        )
