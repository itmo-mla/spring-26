from __future__ import annotations

import argparse

from rich.console import Console

from src.experiments.config import list_default_configs
from src.experiments.runner import run_experiment

console = Console()


def main() -> None:
    """Parse arguments and run configured experiments."""
    parser = argparse.ArgumentParser(description="Run Random Forest lab experiments.")
    parser.add_argument(
        "--config",
        help="Path to one YAML experiment config.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all default experiment configs.",
    )
    args = parser.parse_args()

    if args.run_all:
        for config_path in list_default_configs():
            run_experiment(config_path)
        return

    if args.config:
        run_experiment(args.config)
        return

    parser.error("Use --config configs/rf_default.yaml or --run-all.")


if __name__ == "__main__":
    main()
