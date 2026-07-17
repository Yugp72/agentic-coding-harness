from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from agentic_harness.models import BenchmarkTask, SandboxResult
from agentic_harness.sandbox.base import Sandbox


class DockerSandbox(Sandbox):
    def __init__(self, image: str = "agentic-harness-sandbox:latest") -> None:
        self.image = image

    async def run(self, task: BenchmarkTask, code: str) -> SandboxResult:
        started = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="agentic-harness-") as tmp:
            root = Path(tmp)
            (root / task.entry_file).write_text(code, encoding="utf-8")
            (root / task.test_file).write_text(task.tests, encoding="utf-8")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                "512m",
                "--cpus",
                "1.0",
                "--pids-limit",
                "128",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{root}:/workspace:ro",
                "-w",
                "/workspace",
                self.image,
                "python",
                task.test_file,
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("Docker is not installed or not on PATH") from exc
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(), timeout=task.timeout_seconds + 5
                )
                timed_out = False
            except TimeoutError:
                process.kill()
                stdout_b, stderr_b = await process.communicate()
                timed_out = True
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            return SandboxResult(
                passed=process.returncode == 0 and not timed_out,
                exit_code=process.returncode if process.returncode is not None else -1,
                stdout=stdout_b.decode(errors="replace"),
                stderr=stderr_b.decode(errors="replace"),
                duration_ms=duration_ms,
                timed_out=timed_out,
            )
