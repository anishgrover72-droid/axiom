"""Inference API for a trained XD-PRM: score reasoning steps -> per-head + R_aggregate.

This is the single entry point every consumer uses (verifier decoding, adaptive depth,
the GRPO reward, eval). Load once, score many times.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from axiom.common.logging import get_logger
from axiom.common.steps import render_with_sentinels
from axiom.prm.model import domain_id, load_xdprm

log = get_logger(__name__)


@dataclass(slots=True)
class StepScores:
    """Per-head probabilities and aggregate reward for each scored step."""

    per_head: dict[str, list[float]]
    r_aggregate: list[float]

    @property
    def final(self) -> float:
        """Aggregate score of the last step (used for best-of-N reranking)."""
        return self.r_aggregate[-1] if self.r_aggregate else 0.0

    @property
    def mean(self) -> float:
        return sum(self.r_aggregate) / len(self.r_aggregate) if self.r_aggregate else 0.0


class PRMScorer:
    """Loads a trained XD-PRM and scores step lists."""

    def __init__(self, prm_cfg: DictConfig, ckpt_dir: str, device: str | None = None) -> None:
        import torch

        self.model, self.tok, self.sentinel_id = load_xdprm(prm_cfg, ckpt_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = prm_cfg.backbone.get("max_len", 4096)
        self.model.to(self.device).eval()

    def score(self, steps: list[str], domain: str) -> StepScores:
        """Score a reasoning trajectory's steps for one domain."""
        import torch

        if not steps:
            return StepScores({}, [])
        enc = self.tok(
            render_with_sentinels(steps),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        ).to(self.device)
        positions = (enc["input_ids"][0] == self.sentinel_id).nonzero(as_tuple=True)[0]
        n = int(positions.numel())
        if n != len(steps):
            log.warning(
                "scored %d of %d steps (trace truncated at max_len=%d); "
                "downstream indexing by step may misalign",
                n,
                len(steps),
                self.max_len,
            )
        dom = torch.full((n,), domain_id(domain), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"],
                enc["attention_mask"],
                torch.zeros(n, dtype=torch.long, device=self.device),
                positions,
                dom,
            )
            r_agg = self.model.aggregate(logits)
        per_head = {name: logit.sigmoid().tolist() for name, logit in logits.items()}
        return StepScores(per_head, r_agg.tolist())
