"""Canonical artifact locations. One definition so trainers and consumers never drift."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def _exp(cfg: DictConfig) -> Path:
    return Path(cfg.paths.experiments)


def sft_dir(cfg: DictConfig) -> Path:
    return _exp(cfg) / cfg.sft.output_subdir / cfg.model.name


def prm_dir(cfg: DictConfig) -> Path:
    """The single trained verifier, shared across policy models."""
    return _exp(cfg) / cfg.prm.output_subdir


def grpo_dir(cfg: DictConfig) -> Path:
    return _exp(cfg) / cfg.grpo.output_subdir / cfg.model.name


def foundry_dir(cfg: DictConfig) -> Path:
    """Where the label foundry writes traces.jsonl + labels.jsonl."""
    return Path(cfg.paths.data) / "prm"
