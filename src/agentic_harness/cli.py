from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .eval import EvaluationEngine
from .harness import CodingHarness
from .models import HarnessConfig
from .providers import create_provider
from .sandbox import create_sandbox
from .task_loader import load_task, load_tasks

app = typer.Typer(no_args_is_help=True, help="Agentic coding harness and evaluation engine")
console = Console()


def build_harness(provider: str, model: str | None, sandbox: str, parallel: int, iterations: int) -> CodingHarness:
    return CodingHarness(
        create_provider(provider, model),
        create_sandbox(sandbox),
        HarnessConfig(max_iterations=iterations, parallel_candidates=parallel, sandbox=sandbox),
    )


@app.command()
def run(
    task_path: Path,
    provider: str = typer.Option("mock", help="mock, openai/codex, anthropic/claude"),
    model: str | None = typer.Option(None),
    sandbox: str = typer.Option("docker", help="docker or local"),
    parallel: int = typer.Option(2, min=1, max=16),
    iterations: int = typer.Option(3, min=1, max=10),
) -> None:
    """Run one benchmark task through parallel agent lanes."""
    task = load_task(task_path)
    harness = build_harness(provider, model, sandbox, parallel, iterations)
    result = asyncio.run(harness.run(task))
    console.print_json(json.dumps(result.summary()))
    console.print("\n[bold]Best candidate code[/bold]\n")
    console.print(result.best_code)
    raise typer.Exit(code=0 if result.success else 1)


@app.command(name="eval")
def eval_cmd(
    task_path: Path,
    provider: str = typer.Option("mock"),
    model: str | None = typer.Option(None),
    sandbox: str = typer.Option("docker"),
    parallel: int = typer.Option(2, min=1, max=16, help="Subagents per task"),
    iterations: int = typer.Option(3, min=1, max=10),
    task_parallelism: int = typer.Option(4, min=1, max=32),
    output: Path = typer.Option(Path("runs/latest")),
) -> None:
    """Evaluate a directory of tasks and emit JSONL telemetry."""
    tasks = load_tasks(task_path)
    if not tasks:
        raise typer.BadParameter("No .yaml/.yml/.json benchmark tasks found")
    harness = build_harness(provider, model, sandbox, parallel, iterations)
    engine = EvaluationEngine(harness, output)
    results, summary = asyncio.run(engine.run(tasks, task_parallelism=task_parallelism))

    table = Table(title="Evaluation")
    table.add_column("Task")
    table.add_column("Pass")
    table.add_column("Attempts", justify="right")
    table.add_column("Wall ms", justify="right")
    for result in results:
        table.add_row(
            result.task.id,
            "yes" if result.success else "no",
            str(len(result.attempts)),
            f"{result.wall_time_ms:.1f}",
        )
    console.print(table)
    console.print_json(json.dumps(summary.to_dict()))
    console.print(f"Saved telemetry to {output}")
    raise typer.Exit(code=0 if summary.passed == summary.total else 1)


if __name__ == "__main__":
    app()
