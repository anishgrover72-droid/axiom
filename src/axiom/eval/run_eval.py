"""Run the full eval matrix (variants x datasets) and write results JSON.

Each cell is wrapped so a missing checkpoint (e.g. GRPO not yet trained) records an error
and the matrix continues rather than aborting.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from axiom.common.logging import get_logger
from axiom.eval.benchmarks import run_variant

log = get_logger(__name__)

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "data"


def run_suite(cfg: DictConfig) -> Path:
    """Evaluate every variant on every dataset; return the results path."""
    results: dict[str, dict] = {}
    for dataset in cfg.eval.datasets:
        data_cfg = OmegaConf.load(_CONFIG_ROOT / f"{dataset}.yaml")
        results[dataset] = {}
        for variant in cfg.eval.variants:
            name = variant["name"]
            try:
                results[dataset][name] = run_variant(cfg, variant, data_cfg)
            except Exception as exc:
                log.warning("variant %s on %s failed: %s", name, dataset, exc)
                results[dataset][name] = {"error": str(exc)}
            log.info("%s / %s -> %s", dataset, name, results[dataset][name])

    out = Path(cfg.paths.experiments) / "eval" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote eval matrix -> %s", out)
    return out
