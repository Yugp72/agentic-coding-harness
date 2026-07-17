from __future__ import annotations

from .models import BenchmarkTask


SYSTEM_PROMPT = """You are a coding agent inside an automated evaluation harness.
Return ONLY the complete contents of the requested solution file. Do not use markdown fences.
Do not modify tests. Prefer deterministic, dependency-free Python unless the task says otherwise.
"""


class PromptRouter:
    """Builds generation and repair prompts from task state."""

    def initial(self, task: BenchmarkTask, candidate: int) -> str:
        diversity = (
            "Use a straightforward correctness-first implementation."
            if candidate % 2 == 0
            else "Use a compact implementation with explicit edge-case handling."
        )
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Task ID: {task.id}\nTitle: {task.title}\n"
            f"Target file: {task.entry_file}\n\n{task.prompt}\n\n"
            f"Candidate strategy: {diversity}"
        )

    def repair(
        self,
        task: BenchmarkTask,
        previous_code: str,
        stdout: str,
        stderr: str,
        candidate: int,
    ) -> str:
        return (
            f"{SYSTEM_PROMPT}\n"
            f"Task ID: {task.id}\nTarget file: {task.entry_file}\n\n{task.prompt}\n\n"
            "The previous implementation failed the sandbox tests. Repair it.\n\n"
            f"Previous code:\n{previous_code}\n\n"
            f"Test stdout:\n{stdout[-5000:]}\n\n"
            f"Test stderr:\n{stderr[-5000:]}\n\n"
            f"Candidate lane: {candidate}. Return the full corrected file only."
        )
