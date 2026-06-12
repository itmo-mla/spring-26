from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .config import discover_configs
from .runner import run_experiment

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4: EM-algorithm experiments")
    parser.add_argument("--config", type=str, help="path to a single YAML config")
    parser.add_argument("--run-all", action="store_true", help="run every config under configs/")
    args = parser.parse_args()

    if args.config:
        run_experiment(Path(args.config))
        return

    if args.run_all:
        configs = discover_configs(CONFIGS_DIR)
        console.print(f"[bold]found {len(configs)} configs[/bold]")
        for cfg in configs:
            run_experiment(cfg)
        return

    parser.error("expected --config PATH or --run-all")


if __name__ == "__main__":
    main()
