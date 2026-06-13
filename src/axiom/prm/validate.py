"""The G2 gate: prove XD-PRM is a trustworthy reward before anything consumes it.

Checks discrimination (AUC), head non-redundancy (max pairwise correlation), and
confidence calibration (ECE) on a held-out slice of the foundry's labels. Best-of-N lift
needs generation and is measured in eval, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from omegaconf import DictConfig

from axiom.common.errors import GateError
from axiom.common.logging import get_logger
from axiom.common.paths import foundry_dir
from axiom.eval.metrics import expected_calibration_error, pearson, roc_auc
from axiom.prm.dataset import load_labeled
from axiom.prm.score import PRMScorer

log = get_logger(__name__)


@dataclass(slots=True)
class GateReport:
    auc: float
    max_head_correlation: float
    ece: float
    n_steps: int
    passed: bool


def validate_prm(cfg: DictConfig, ckpt_dir: str, raise_on_fail: bool = True) -> GateReport:
    """Run the gate on the held-out split (never trained on); raise on failure."""
    scorer = PRMScorer(cfg.prm, ckpt_dir)
    traces, labels = load_labeled(foundry_dir(cfg), split="val")

    r_aggs: list[float] = []
    good: list[bool] = []
    confidences: list[float] = []
    head_series: dict[str, list[float]] = {}
    for trace in traces:
        scores = scorer.score([s.text for s in trace.steps], trace.domain)
        for k, r in enumerate(scores.r_aggregate):
            label = labels.get((trace.trace_id, k))
            if label is None or label.logic is None:
                continue
            r_aggs.append(r)
            good.append(label.logic >= 0.5)
            for head, series in scores.per_head.items():
                head_series.setdefault(head, []).append(series[k])
            if "confidence" in scores.per_head:
                confidences.append(scores.per_head["confidence"][k])

    auc = roc_auc(r_aggs, good)
    max_corr = max(
        (abs(pearson(head_series[a], head_series[b])) for a, b in combinations(head_series, 2)),
        default=0.0,
    )
    ece = expected_calibration_error(confidences, good) if confidences else 0.0
    gate = cfg.prm.gate
    passed = auc >= gate.min_auc and max_corr <= gate.max_head_correlation and ece <= gate.max_ece

    log.info(
        "GATE auc=%.3f (>=%.2f) max_head_corr=%.3f (<=%.2f) ece=%.3f (<=%.2f) -> %s",
        auc,
        gate.min_auc,
        max_corr,
        gate.max_head_correlation,
        ece,
        gate.max_ece,
        "PASS" if passed else "FAIL",
    )
    if not passed and raise_on_fail:
        raise GateError("XD-PRM failed the G2 gate; fix the foundry/PRM before downstream use")
    return GateReport(auc, max_corr, ece, len(r_aggs), passed)
