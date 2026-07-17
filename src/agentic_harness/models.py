from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    id: str
    title: str
    prompt: str
    language: Literal["python"] = "python"
    entry_file: str = "solution.py"
    test_file: str = "test_solution.py"
    tests: str
    timeout_seconds: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    text: str
    timestamp_ns: int


@dataclass(slots=True)
class InferenceMetrics:
    started_ns: int
    first_token_ns: int | None = None
    finished_ns: int | None = None
    token_timestamps_ns: list[int] = field(default_factory=list)
    approx_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_ns is None:
            return None
        return (self.first_token_ns - self.started_ns) / 1_000_000

    @property
    def elapsed_s(self) -> float | None:
        if self.finished_ns is None:
            return None
        return (self.finished_ns - self.started_ns) / 1_000_000_000

    @property
    def tokens_per_second(self) -> float | None:
        elapsed = self.elapsed_s
        if not elapsed or elapsed <= 0:
            return None
        return self.approx_tokens / elapsed

    @property
    def mean_itl_ms(self) -> float | None:
        if len(self.token_timestamps_ns) < 2:
            return None
        gaps = [
            (b - a) / 1_000_000
            for a, b in zip(self.token_timestamps_ns, self.token_timestamps_ns[1:])
        ]
        return sum(gaps) / len(gaps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "elapsed_s": self.elapsed_s,
            "approx_tokens": self.approx_tokens,
            "tokens_per_second": self.tokens_per_second,
            "mean_itl_ms": self.mean_itl_ms,
        }


@dataclass(slots=True)
class SandboxResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False


@dataclass(slots=True)
class AgentAttempt:
    attempt: int
    candidate: int
    code: str
    provider: str
    model: str
    inference: InferenceMetrics
    sandbox: SandboxResult
    prompt_kind: str


@dataclass(slots=True)
class RunResult:
    task: BenchmarkTask
    success: bool
    best_code: str
    attempts: list[AgentAttempt]
    wall_time_ms: float

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "success": self.success,
            "wall_time_ms": self.wall_time_ms,
            "attempts": len(self.attempts),
            "best_code": self.best_code,
            "attempt_metrics": [
                {
                    "attempt": a.attempt,
                    "candidate": a.candidate,
                    "provider": a.provider,
                    "model": a.model,
                    "prompt_kind": a.prompt_kind,
                    "passed": a.sandbox.passed,
                    "sandbox_duration_ms": a.sandbox.duration_ms,
                    **a.inference.to_dict(),
                }
                for a in self.attempts
            ],
        }


@dataclass(slots=True)
class HarnessConfig:
    max_iterations: int = 3
    parallel_candidates: int = 2
    sandbox: Literal["docker", "local"] = "docker"
    run_dir: Path = Path("runs")
