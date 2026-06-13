"""Adaptive Reasoning-Depth Controller (Innovation 3).

After each step, estimate uncertainty from the XD-PRM confidence head (falling back to
R_aggregate). Low uncertainty + a stated answer -> early exit; high uncertainty -> expand.
Inference-only; a hard depth/token cap bounds latency.
"""

from __future__ import annotations

from collections import Counter

from omegaconf import DictConfig

from axiom.common.answers import extract_prediction
from axiom.common.vllm_pool import Engine, GenConfig
from axiom.prm.score import StepScores

EXIT, EXPAND, CONTINUE = "exit", "expand", "continue"


def step_uncertainty(scores: StepScores | None) -> float:
    """Per-step uncertainty in [0,1]: prefer the confidence head, else 1 - R_aggregate."""
    if scores is None:
        return 1.0
    conf = scores.per_head.get("confidence")
    if conf:
        return 1.0 - conf[-1]
    return 1.0 - (scores.r_aggregate[-1] if scores.r_aggregate else 0.0)


def decide(uncertainty: float, has_answer: bool, cfg: DictConfig) -> str:
    """Map uncertainty to a control decision against the configured thresholds."""
    if has_answer and uncertainty < cfg.tau_low:
        return EXIT
    if uncertainty > cfg.tau_high:
        return EXPAND
    return CONTINUE


def self_consistency_uncertainty(
    engine: Engine, question: str, prior: list[str], answer_type: str, k: int
) -> float:
    """Disagreement across k full continuations: 1 - modal-answer agreement."""
    partial = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(prior))
    conv = [
        {
            "role": "user",
            "content": f"{question}\n\n{partial}\n\nFinish. End with 'The answer is X.'",
        }
    ]
    completions = engine.chat([conv], GenConfig(n=k, temperature=0.8, max_tokens=256))[0]
    preds = [extract_prediction(c, answer_type) for c in completions]
    valid = [p for p in preds if p is not None]
    if not valid:
        return 1.0
    return 1.0 - Counter(valid).most_common(1)[0][1] / len(valid)
