"""Unified Efficient Inference Engine.

Combines the base policy, verifier-guided decoding, and the adaptive-depth controller into
one step-by-step reasoner that also emits the per-head XD-PRM trace the demo renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig

from axiom.common.answers import extract_prediction, has_answer
from axiom.common.logging import get_logger
from axiom.common.paths import grpo_dir, prm_dir, sft_dir
from axiom.common.tokens import count
from axiom.common.vllm_pool import get_engine
from axiom.inference.adaptive_depth import EXIT, EXPAND, decide, step_uncertainty
from axiom.inference.verifier_decode import propose_next
from axiom.prm.score import PRMScorer

log = get_logger(__name__)


@dataclass(slots=True)
class ReasoningStep:
    idx: int
    text: str
    per_head: dict[str, float]
    r_aggregate: float
    uncertainty: float
    decision: str
    surviving: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningResult:
    question: str
    steps: list[ReasoningStep]
    final_answer: str | None
    n_tokens: int


def resolve_policy(cfg: DictConfig, variant: str) -> str:
    """Return a vLLM-servable model path for a variant (merging adapters once if needed)."""
    if variant == "base":
        return cfg.model.hf_id
    adapter = sft_dir(cfg) if variant == "sft" else grpo_dir(cfg)
    merged = Path(cfg.paths.experiments) / "merged" / f"{cfg.model.name}-{variant}"
    if not merged.exists():
        _merge_adapter(cfg.model.hf_id, adapter, merged)
    return str(merged)


def _merge_adapter(base_id: str, adapter: Path, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base_id).save_pretrained(str(out))
    log.info("merged %s + %s -> %s", base_id, adapter, out)


class ReasoningEngine:
    """Step-by-step reasoner: policy + (optional) verifier decoding + (optional) adaptive depth."""

    def __init__(self, cfg: DictConfig, variant: str = "sft", use_prm: bool = True) -> None:
        self.cfg = cfg
        self.infer = cfg.infer
        self.policy = get_engine(
            resolve_policy(cfg, variant), cfg.model.dtype, cfg.model.max_seq_len
        )
        self.scorer = PRMScorer(cfg.prm, str(prm_dir(cfg))) if use_prm else None

    def _step_cap(self) -> int:
        return (
            self.infer.adaptive_depth.max_depth
            if self.infer.adaptive_depth.enabled
            else self.infer.verifier_decode.max_steps
        )

    def iter_steps(self, question: str, domain: str):
        """Yield each ReasoningStep as it is produced (drives both reason() and streaming)."""
        vd, ad = self.infer.verifier_decode, self.infer.adaptive_depth
        prior: list[str] = []
        branch = vd.branch_factor if vd.enabled else 1
        for idx in range(self._step_cap()):
            proposal = propose_next(
                self.policy,
                self.scorer,
                question,
                prior,
                domain,
                branch,
                vd.keep_top_k,
                vd.temperature,
                vd.max_step_tokens,
            )
            prior.append(proposal.text)
            answered = has_answer(proposal.text)
            uncertainty = step_uncertainty(proposal.scores)
            decision = (
                decide(uncertainty, answered, ad)
                if ad.enabled
                else (EXIT if answered else "continue")
            )

            scores = proposal.scores
            yield ReasoningStep(
                idx=idx,
                text=proposal.text,
                per_head={h: v[-1] for h, v in scores.per_head.items()} if scores else {},
                r_aggregate=scores.final if scores else 0.0,
                uncertainty=uncertainty,
                decision=decision,
                surviving=proposal.surviving,
                pruned=proposal.pruned,
            )
            if decision == EXIT:
                return
            if decision == EXPAND:
                branch = min(branch * 2, 2 * vd.branch_factor)

    def reason(self, question: str, domain: str, answer_type: str) -> ReasoningResult:
        """Run to completion and return the full traced result."""
        steps = list(self.iter_steps(question, domain))
        last = steps[-1].text if steps else ""
        n_tokens = sum(count(s.text, self.cfg.model.hf_id) for s in steps)
        return ReasoningResult(question, steps, extract_prediction(last, answer_type), n_tokens)
