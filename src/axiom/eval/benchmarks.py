"""Run one (variant x dataset) cell of the eval matrix and return its metrics."""

from __future__ import annotations

import copy
import time

from omegaconf import DictConfig

from axiom.common.answers import grade
from axiom.common.paths import prm_dir
from axiom.eval.metrics import mean
from axiom.inference.engine import ReasoningEngine

_PRM_METRICS = {"mean_step_reward", "contradiction_rate"}


def _with_overrides(cfg: DictConfig, decode: str, adaptive: bool) -> DictConfig:
    local = copy.deepcopy(cfg)
    local.infer.verifier_decode.enabled = decode == "verifier"
    local.infer.adaptive_depth.enabled = bool(adaptive)
    return local


def run_variant(cfg: DictConfig, variant: dict, data_cfg: DictConfig) -> dict[str, float]:
    """Evaluate one variant on one dataset; return the requested metrics."""
    from axiom.data.loaders import load_examples

    prm_ready = (prm_dir(cfg) / "xdprm.pt").exists()
    local = _with_overrides(cfg, variant["decode"], variant["adaptive"])
    engine = ReasoningEngine(local, variant["policy"], use_prm=prm_ready)

    correct, tokens, latency, step_reward, contradiction = [], [], [], [], []
    for ex in load_examples(data_cfg, "eval", cfg.eval.limit):
        t0 = time.perf_counter()
        res = engine.reason(ex.question, ex.domain, ex.answer_type)
        latency.append(time.perf_counter() - t0)
        verdict = grade(
            " ".join(s.text for s in res.steps), ex.gold_answer, ex.answer_type, ex.choices
        )
        correct.append(verdict.correct)
        tokens.append(res.n_tokens)
        step_reward.append(mean([s.r_aggregate for s in res.steps]))
        cons = [s.per_head["consistency"] for s in res.steps if "consistency" in s.per_head]
        contradiction.append(mean([float(c < 0.5) for c in cons]) if cons else 0.0)

    pool = {
        "accuracy": mean([float(c) for c in correct]),
        "tokens_per_answer": mean(tokens),
        "latency_s": mean(latency),
        "mean_step_reward": mean(step_reward),
        "contradiction_rate": mean(contradiction),
    }
    return {
        m: pool[m] for m in cfg.eval.metrics if m in pool and (prm_ready or m not in _PRM_METRICS)
    }
