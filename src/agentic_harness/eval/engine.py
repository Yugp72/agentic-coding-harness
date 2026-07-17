from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from agentic_harness.harness import CodingHarness
from agentic_harness.models import BenchmarkTask, RunResult


@dataclass(slots=True)
class EvalSummary:
    total: int
    passed: int
    success_rate: float
    mean_wall_time_ms: float
    mean_ttft_ms: float | None
    mean_itl_ms: float | None
    mean_tokens_per_second: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "success_rate": self.success_rate,
            "mean_wall_time_ms": self.mean_wall_time_ms,
            "mean_ttft_ms": self.mean_ttft_ms,
            "mean_itl_ms": self.mean_itl_ms,
            "mean_tokens_per_second": self.mean_tokens_per_second,
        }


class EvaluationEngine:
    def __init__(self, harness: CodingHarness, output_dir: Path) -> None:
        self.harness = harness
        self.output_dir = output_dir

    async def run(self, tasks: list[BenchmarkTask], task_parallelism: int = 4) -> tuple[list[RunResult], EvalSummary]:
        semaphore = asyncio.Semaphore(task_parallelism)

        async def one(task: BenchmarkTask) -> RunResult:
            async with semaphore:
                return await self.harness.run(task)

        results = await asyncio.gather(*(one(task) for task in tasks))
        summary = self._summarize(results)
        self._write(results, summary)
        return results, summary

    def _summarize(self, results: list[RunResult]) -> EvalSummary:
        attempts = [a for r in results for a in r.attempts]
        ttft = [a.inference.ttft_ms for a in attempts if a.inference.ttft_ms is not None]
        itl = [a.inference.mean_itl_ms for a in attempts if a.inference.mean_itl_ms is not None]
        tps = [a.inference.tokens_per_second for a in attempts if a.inference.tokens_per_second is not None]
        passed = sum(r.success for r in results)
        return EvalSummary(
            total=len(results),
            passed=passed,
            success_rate=passed / len(results) if results else 0.0,
            mean_wall_time_ms=mean(r.wall_time_ms for r in results) if results else 0.0,
            mean_ttft_ms=mean(ttft) if ttft else None,
            mean_itl_ms=mean(itl) if itl else None,
            mean_tokens_per_second=mean(tps) if tps else None,
        )

    def _write(self, results: list[RunResult], summary: EvalSummary) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps(result.summary()) + "\n")
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )
