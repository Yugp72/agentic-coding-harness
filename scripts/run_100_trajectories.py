"""Run repeated trajectories to produce enough samples for latency profiling."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic_harness.eval import EvaluationEngine
from agentic_harness.harness import CodingHarness
from agentic_harness.models import HarnessConfig
from agentic_harness.providers import create_provider
from agentic_harness.sandbox import create_sandbox
from agentic_harness.task_loader import load_tasks


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--sandbox", default="local")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--output", default="runs/trajectories")
    args = parser.parse_args()

    base = load_tasks(Path("benchmarks/tasks"))
    tasks = []
    for i in range(args.runs):
        source = base[i % len(base)]
        tasks.append(source.model_copy(update={"id": f"{source.id}-run-{i:03d}"}))

    harness = CodingHarness(
        create_provider(args.provider),
        create_sandbox(args.sandbox),
        HarnessConfig(max_iterations=3, parallel_candidates=args.parallel, sandbox=args.sandbox),
    )
    engine = EvaluationEngine(harness, Path(args.output))
    _, summary = await engine.run(tasks, task_parallelism=min(8, args.parallel * 2))
    print(summary.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
