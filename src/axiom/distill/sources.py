"""Load existing open distilled reasoning traces into the `Trace` schema (primary path).

Avoids generating teacher traces from scratch. Field names come from the source config.
"""

from __future__ import annotations

from collections.abc import Iterator

from omegaconf import DictConfig

from axiom.common.steps import segment
from axiom.data.schemas import Step, Trace


def _first(value: object) -> str:
    """Some datasets store multiple generations per row; take the first."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def load_open_traces(src: DictConfig, *, dataset: str, domain: str) -> Iterator[Trace]:
    """Stream traces from an open dataset, segmenting each reasoning text into steps."""
    from datasets import load_dataset

    ds = load_dataset(src.open_id, split=src.open_split)
    f = src.fields
    for i, row in enumerate(ds):
        if src.limit is not None and i >= src.limit:
            break
        steps = [Step(idx=j, text=t) for j, t in enumerate(segment(_first(row[f.reasoning])))]
        if not steps:
            continue
        yield Trace(
            trace_id=f"{src.open_id}:{i}",
            example_id=str(i),
            dataset=dataset,
            domain=domain,
            model="teacher-open",
            question=str(row[f.question]),
            steps=steps,
            final_answer=_first(row[f.answer]) if f.get("answer") else None,
        )
