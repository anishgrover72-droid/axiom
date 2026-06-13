"""Programmatic Hydra config composition (for the server and notebooks)."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the top-level config with optional `key=value` overrides."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=overrides or [])
