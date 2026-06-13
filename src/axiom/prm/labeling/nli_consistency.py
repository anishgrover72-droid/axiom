"""CONSISTENCY and COMMONSENSE labels from one off-the-shelf NLI model (free).

Prior steps are the premise, the current step the hypothesis. Consistency = 1 -
P(contradiction) (the deck's "contradiction detection"); commonsense plausibility =
P(entailment) (the free fallback used when no teacher judge is configured). Both reuse the
same cached model, so the 5-head PRM stays fully supervised at zero API cost.
"""

from __future__ import annotations

from functools import lru_cache

from axiom.common.errors import ConfigError


@lru_cache(maxsize=1)
def _nli(model_name: str):
    """Load and cache the NLI model with its lowercased label->index map."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    idx = {lbl.lower(): i for i, lbl in model.config.id2label.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), idx, device


def _class_index(idx: dict[str, int], needle: str) -> int:
    """Resolve an NLI class column by label substring; fail loud on an unknown schema.

    Label order is checkpoint-specific, so guessing a default would silently invert the
    consistency/plausibility signal — corrupting the labels with no error.
    """
    for label, i in idx.items():
        if needle in label:
            return int(i)
    raise ConfigError(f"NLI model has no '{needle}' class in labels {sorted(idx)}")


def _step_probs(steps: list[str], model_name: str):
    """Softmax NLI probabilities for (prior-steps -> step) over steps 1..n-1."""
    import torch

    tok, model, idx, device = _nli(model_name)
    premises = ["\n".join(steps[:k]) for k in range(1, len(steps))]
    enc = tok(
        premises, steps[1:], return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        return model(**enc).logits.softmax(-1), idx


def nli_consistency_labels(steps: list[str], model_name: str) -> list[float]:
    """Per-step 1 - P(contradiction | prior_steps, step); first step is 1.0."""
    if len(steps) <= 1:
        return [1.0] * len(steps)
    probs, idx = _step_probs(steps, model_name)
    return [1.0, *(1.0 - probs[:, _class_index(idx, "contradiction")]).tolist()]


def nli_entailment_labels(steps: list[str], model_name: str) -> list[float]:
    """Per-step P(entailment | prior_steps, step) as a free commonsense-plausibility proxy."""
    if len(steps) <= 1:
        return [1.0] * len(steps)
    probs, idx = _step_probs(steps, model_name)
    return [1.0, *probs[:, _class_index(idx, "entail")].tolist()]
