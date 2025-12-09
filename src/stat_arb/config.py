from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# repo_root/stat_arb/config.py -> parents[2] = repo_root
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"


def load_yaml_config(name: str) -> Dict[str, Any]:
    """
    Load configs/{name}.yaml as a dict.

    Example:
        paths = load_yaml_config("paths")
        unhedged = load_yaml_config("unhedged_reversal")
    """
    cfg_path = CONFIG_DIR / f"{name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def get_paths_config() -> Dict[str, Any]:
    """Convenience alias for configs/paths.yaml."""
    return load_yaml_config("paths")


def get_strategy_config(name: str) -> Dict[str, Any]:
    """Convenience alias for configs/{name}.yaml."""
    return load_yaml_config(name)