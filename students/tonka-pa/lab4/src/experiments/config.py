from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    task: str  # "density" | "classification" | "model_selection"
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        name=raw["name"],
        task=raw["task"],
        data=raw.get("data", {}),
        model=raw.get("model", {}),
        baseline=raw.get("baseline", {}),
        runs=raw.get("runs", []),
        artifacts=raw.get("artifacts", {}),
    )


def discover_configs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.yaml"))
