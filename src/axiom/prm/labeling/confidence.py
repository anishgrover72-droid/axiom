"""CONFIDENCE labels from self-consistency of the MC-rollout samples (free).

Reuses the answers sampled in mc_rollout. Modal agreement across the k samples is a
calibrated proxy for the model's per-step certainty: unanimous -> 1.0, fully split -> low.
"""

from __future__ import annotations

from collections import Counter


def confidence_from_answers(answers_per_step: list[list[str | None]]) -> list[float | None]:
    """Per-step confidence = fraction of samples agreeing with the modal answer.

    A step with no parseable answers is left unmeasured (None -> masked in the loss), not
    forced to 0.0, which would teach the confidence head that unmeasurable == uncertain.
    """
    out: list[float | None] = []
    for answers in answers_per_step:
        valid = [a for a in answers if a is not None]
        out.append(Counter(valid).most_common(1)[0][1] / len(valid) if valid else None)
    return out
