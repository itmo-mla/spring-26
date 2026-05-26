from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path("configs")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load and normalize one YAML experiment config."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config is empty or invalid: {config_path}")
    config.setdefault("name", config_path.stem)
    config.setdefault("dataset", {})
    config.setdefault("split", {})
    config.setdefault("model", {})
    config.setdefault("pruning", {})
    config["dataset"].setdefault("drop_measured", True)
    config["split"].setdefault("test_size", 0.25)
    config["split"].setdefault("validation_size", 0.25)
    config["split"].setdefault("random_state", 42)
    config["pruning"].setdefault("enabled", False)
    config["pruning"].setdefault("max_candidates", 15)
    if "task" not in config:
        raise ValueError(f"Config must define task: {config_path}")
    return config


def list_config_paths(config_dir: Path | str = CONFIG_DIR) -> list[Path]:
    """Return all experiment config paths."""
    return sorted(Path(config_dir).glob("*.yaml"))
