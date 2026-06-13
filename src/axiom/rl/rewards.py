"""The GRPO composite reward.

    R = a*correctness + b*R_aggregate(XD-PRM) - c*length - d*repetition

Correctness (verifiable) dominates so the policy cannot win by gaming the learned reward;
the length and repetition terms are explicit anti-hacking guards. The XD-PRM is frozen.
"""

from __future__ import annotations

from itertools import pairwise

from omegaconf import DictConfig

from axiom.common.answers import grade
from axiom.common.steps import segment
from axiom.common.tokens import count
from axiom.data.schemas import RewardBreakdown
from axiom.prm.score import PRMScorer


def _text(completion) -> str:
    """GRPO completions are strings or chat-message lists; normalize to text."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _repetition_penalty(text: str) -> float:
    """1 - distinct-2: high when the model loops the same bigrams."""
    toks = text.split()
    if len(toks) < 2:
        return 0.0
    bigrams = list(pairwise(toks))
    return 1.0 - len(set(bigrams)) / len(bigrams)


class CompositeReward:
    """Callable reward for TRL's GRPOTrainer. Extra dataset columns arrive as kwargs."""

    def __init__(
        self, cfg: DictConfig, scorer: PRMScorer | None, hf_id: str, max_new_tokens: int
    ) -> None:
        self.cfg = cfg
        self.scorer = scorer
        self.hf_id = hf_id
        self.max_new_tokens = max_new_tokens
        self.last: list[RewardBreakdown] = []  # populated for periodic audits
        self.__name__ = "axiom_composite_reward"  # TRL reads this for reward logging

    def breakdown(
        self, text: str, gold: str, answer_type: str, domain: str, choices: dict | None = None
    ) -> RewardBreakdown:
        correct = float(grade(text, gold, answer_type, choices).correct)
        steps = segment(text)
        process = self.scorer.score(steps, domain).mean if self.scorer and steps else 0.0
        length = min(1.0, count(text, self.hf_id) / self.max_new_tokens)
        repetition = _repetition_penalty(text)
        total = (
            self.cfg.correctness * correct
            + self.cfg.process * process
            - self.cfg.length_penalty * length
            - self.cfg.repetition_penalty * repetition
        )
        return RewardBreakdown(
            correctness=correct,
            process=process,
            length_penalty=length,
            repetition_penalty=repetition,
            total=total,
        )

    def __call__(
        self, prompts, completions, gold, answer_type, domain, choices=None, **_
    ) -> list[float]:
        chs = choices if choices is not None else [None] * len(completions)
        self.last = [
            self.breakdown(_text(c), g, at, d, ch)
            for c, g, at, d, ch in zip(completions, gold, answer_type, domain, chs, strict=True)
        ]
        return [b.total for b in self.last]
