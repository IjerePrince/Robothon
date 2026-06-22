"""
utils.py — Path & config helpers.

Provides:
    load_config(name)   — load YAML config from config/
    project_paths()     — return canonical project paths
    setup_logging()     — configure root logger
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict
import yaml

from . import PROJECT_ROOT, CONFIG_DIR


def project_paths() -> Dict[str, Path]:
    """Return canonical project subdirectory paths."""
    return {
        "root":         PROJECT_ROOT,
        "models":       PROJECT_ROOT / "models",
        "config":       CONFIG_DIR,
        "scripts":      PROJECT_ROOT / "scripts",
        "checkpoints":  PROJECT_ROOT / "checkpoints",
        "demos":        PROJECT_ROOT / "demos",
        "tests":        PROJECT_ROOT / "tests",
        "docs":         PROJECT_ROOT / "docs",
    }


def load_config(name: str) -> Dict[str, Any]:
    """Load a YAML config file from config/<name>.yaml (or .yml).

    Args:
        name: file basename without extension, e.g. "sim" or "rl"

    Returns:
        Parsed YAML content as a dict.
    """
    for ext in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name}{ext}"
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        f"Config file '{name}.yaml' not found in {CONFIG_DIR}"
    )


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with timestamp + level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("disaster_triage")


def ensure_dir(path: os.PathLike) -> Path:
    """Create directory if it doesn't exist; return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
