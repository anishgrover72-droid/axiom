"""Difficulty scoring and curriculum bucketing for the five cross-domain datasets.

Difficulty is derived from observable properties of the example (op count, dataset,
domain) rather than model performance, so it is computed once and used for both
curriculum staging and the release dataset annotations.
"""

from __future__ import annotations

import re

from axiom.data.schemas import Example

# Canonical stages in increasing difficulty; curriculum uses this order.
STAGES: tuple[str, ...] = ("easy", "medium", "multihop", "adversarial")

# GSM8K annotates each arithmetic operation as <<expr=result>>.
_OPS = re.compile(r"<<[^>]+>>")

# Datasets whose questions inherently require multi-hop retrieval.
_MULTIHOP_DATASETS = frozenset({"hotpotqa", "hotpot_qa"})

# Datasets with adversarial distractors designed to fool naive pattern-matching.
_ADVERSARIAL_DATASETS = frozenset({"arc_challenge"})


def score_difficulty(example: Example) -> str:
    """Return the difficulty stage for a single example."""
    if example.dataset in _ADVERSARIAL_DATASETS:
        return "adversarial"
    if example.dataset in _MULTIHOP_DATASETS or example.domain == "multihop":
        return "multihop"
    if example.dataset == "gsm8k":
        n_ops = len(_OPS.findall(example.gold_answer))
        if n_ops >= 4:
            return "multihop"
        if n_ops >= 2:
            return "medium"
        return "easy"
    # Commonsense and OpenBookQA: single-hop but require world knowledge.
    if example.answer_type == "mcq":
        return "medium"
    return "easy"


def bucket(examples: list[Example]) -> dict[str, list[Example]]:
    """Group examples by difficulty stage; all stages are always present as keys."""
    result: dict[str, list[Example]] = {s: [] for s in STAGES}
    for ex in examples:
        result[score_difficulty(ex)].append(ex)
    return result
