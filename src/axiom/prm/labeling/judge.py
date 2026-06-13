"""COMMONSENSE (+causal) labels via a teacher-LLM judge.

The only paid label stream, so it is budget-capped to a subset (see aggregate_release).
The judge rates each step for real-world plausibility and causal coherence in one call,
returning a JSON array. Parse failures degrade to `None` (masked in loss), never crash.
"""

from __future__ import annotations

import json
import re

from axiom.common.logging import get_logger
from axiom.distill.teacher import TeacherClient

log = get_logger(__name__)

_SYSTEM = (
    "You are a strict reasoning verifier. For each numbered step, rate its real-world "
    "plausibility and causal coherence from 0.0 (implausible/incoherent) to 1.0 (sound). "
    "Reply with ONLY a JSON array of numbers, one per step."
)
_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def judge_commonsense_labels(
    client: TeacherClient, question: str, steps: list[str]
) -> list[float | None]:
    """Per-step plausibility/causal score in [0,1], or `None` per step on failure."""
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    try:
        raw = client.complete(f"Question: {question}\n\nSteps:\n{numbered}", system=_SYSTEM)
        match = _ARRAY.search(raw)
        values = json.loads(match.group(0)) if match else []
        scores = [min(1.0, max(0.0, float(v))) for v in values[: len(steps)]]
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # OSError covers urllib URLError/HTTPError/timeouts: a transient API failure must
        # mask this step (None), never abort the foundry mid-run.
        log.warning("judge call/parse failed: %s", exc)
        return [None] * len(steps)
    return scores + [None] * (len(steps) - len(scores))
