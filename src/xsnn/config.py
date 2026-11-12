"""YAML configuration loading and path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_PATH_KEYS = (
    ("data", "cache_dir"),
    ("data", "pit_constituents_csv"),
    ("outputs", "checkpoints"),
    ("outputs", "activations"),
    ("outputs", "reports"),
    ("outputs", "figures"),
    ("outputs", "processed"),
    ("outputs", "logs"),
)


def project_root() -> Path:
    """Return the repository root.

    ``XSNN_ROOT`` wins when set. Otherwise the nearest parent of this file
    containing ``pyproject.toml`` is used, falling back to the working directory.
    """
    env = os.environ.get("XSNN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve relative paths against the project root."""
    root = project_root()
    config_path = Path(path) if path else root / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")
    cfg: dict[str, Any] = loaded
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(root)
    _resolve_paths(cfg, root)
    _validate(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any], root: Path) -> None:
    for section, key in _PATH_KEYS:
        section_cfg = cfg.get(section)
        if not isinstance(section_cfg, dict):
            continue
        value = section_cfg.get(key)
        if value is None or value == "":
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        section_cfg[key] = str(path.resolve())


def _validate(cfg: dict[str, Any]) -> None:
    required = ("data", "features", "models", "validation", "backtest", "outputs")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    validation = cfg["validation"]
    embargo = int(validation["embargo_months"])
    horizon = int(validation["horizon_months"])
    if embargo < 0:
        raise ValueError("embargo_months must be >= 0")
    if horizon < 1:
        raise ValueError("horizon_months must be >= 1")
    if int(validation["retrain_every_months"]) < 1:
        raise ValueError("retrain_every_months must be >= 1")
    if int(cfg["backtest"]["n_deciles"]) < 2:
        raise ValueError("n_deciles must be at least 2")
    hidden = cfg["models"]["mlp"]["hidden"]
    if list(hidden) != [64, 32, 16]:
        # The published architecture is part of the experiment definition.
        # Tests may override it only by passing a hand-built config that skips
        # this loader; the shipped YAML must keep the stated MLP.
        raise ValueError("MLP hidden sizes must stay [64, 32, 16] in shipped configs")
