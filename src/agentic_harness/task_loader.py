from __future__ import annotations

import json
from pathlib import Path

import yaml

from .models import BenchmarkTask


def load_task(path: Path) -> BenchmarkTask:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    return BenchmarkTask.model_validate(data)


def load_tasks(path: Path) -> list[BenchmarkTask]:
    if path.is_file():
        return [load_task(path)]
    files = sorted([*path.glob("*.yaml"), *path.glob("*.yml"), *path.glob("*.json")])
    return [load_task(file) for file in files]
