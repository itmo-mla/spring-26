from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path("configs")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load and normalize one YAML experiment config."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config is empty or invalid: {config_path}")
    config.setdefault("name", config_path.stem)
    config.setdefault("split", {})
    config.setdefault("model", {})
    config["split"].setdefault("test_size", 0.2)
    config["split"].setdefault("random_state", 42)
    config["split"].setdefault("cv_folds", 5)
    if "task" not in config:
        raise ValueError(f"Config must define task: {config_path}")
    return config


def list_config_paths(config_dir: Path | str = CONFIG_DIR) -> list[Path]:
    """Return all experiment config paths sorted alphabetically."""
    return sorted(Path(config_dir).glob("*.yaml"))
