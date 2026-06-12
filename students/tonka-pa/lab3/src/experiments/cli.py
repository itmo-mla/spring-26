import argparse
from pathlib import Path

from rich.console import Console

from src.experiments.config import list_config_paths
from src.experiments.runner import run_experiment


def main() -> None:
    """Run one or all configured experiments."""
    parser = argparse.ArgumentParser(description="Run gradient boosting lab experiments.")
    parser.add_argument("--config", type=Path, help="Path to one YAML config.")
    parser.add_argument("--run-all", action="store_true", help="Run all configs.")
    args = parser.parse_args()
    console = Console()

    if not args.config and not args.run_all:
        parser.error("Provide --config or --run-all.")

    paths = list_config_paths() if args.run_all else [args.config]
    for path in paths:
        console.rule(f"Running {path}")
        try:
            metrics = run_experiment(path)
            console.print(metrics[["model", "task", "fit_time", "predict_time"]])
        except Exception as exc:
            console.print(f"[red]FAILED {path}: {exc}[/red]")
            raise


if __name__ == "__main__":
    main()
