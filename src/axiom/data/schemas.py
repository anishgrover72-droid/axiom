"""Pydantic data contracts — the vocabulary every module speaks.

Import these; never pass bare dicts across module boundaries.
"""

from __future__ import annotations

from pydantic import BaseModel

# Canonical head order: training, scoring, and the collator must all agree.
HEAD_NAMES: tuple[str, ...] = (
    "logic",
    "commonsense",
    "consistency",
    "efficiency",
    "confidence",
)


class Example(BaseModel):
    """A single benchmark problem with its gold answer."""

    id: str
    dataset: str
    domain: str
    question: str
    gold_answer: str
    answer_type: str  # "numeric" | "mcq" | "span"
    choices: dict | None = None  # label+text lists for MCQ datasets


class Step(BaseModel):
    """One reasoning step within a trace."""

    idx: int
    text: str


class Trace(BaseModel):
    """A full reasoning trajectory produced by a model for one example."""

    trace_id: str
    example_id: str
    dataset: str
    domain: str
    model: str
    question: str
    steps: list[Step]
    final_answer: str | None = None
    gold_answer: str | None = None
    answer_type: str | None = None
    n_tokens: int | None = None


class StepLabel(BaseModel):
    """Per-step ground-truth labels for one or more XD-PRM heads.

    A head value of None means the label is absent for this step; the training
    collator masks it out of that head's loss (not 0-labeled).
    """

    trace_id: str
    step_idx: int
    domain: str
    logic: float | None = None
    commonsense: float | None = None
    consistency: float | None = None
    efficiency: float | None = None
    confidence: float | None = None

    def head_values(self) -> dict[str, float | None]:
        """Return head labels in canonical HEAD_NAMES order (None = unlabeled)."""
        return {h: getattr(self, h) for h in HEAD_NAMES}


class RewardBreakdown(BaseModel):
    """Decomposed GRPO composite reward for one model completion."""

    correctness: float
    process: float
    length_penalty: float
    repetition_penalty: float
    total: float
