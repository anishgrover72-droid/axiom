"""Build the reasoning-trace corpus: load open traces, or generate them from a teacher."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig

from axiom.common.io import write_records
from axiom.common.logging import get_logger
from axiom.common.steps import segment
from axiom.data.loaders import load_examples
from axiom.data.schemas import Step, Trace
from axiom.distill.sources import load_open_traces
from axiom.distill.teacher import TeacherClient

log = get_logger(__name__)

_COT_SYSTEM = "Solve step by step. Number each step. End with 'The answer is <answer>.'"


def _teacher_traces(cfg: DictConfig) -> Iterator[Trace]:
    client = TeacherClient(cfg.teacher)
    examples = list(load_examples(cfg.data, "train", cfg.distill.source.limit))
    completions = client.complete_batch([e.question for e in examples], system=_COT_SYSTEM)
    for ex, text in zip(examples, completions, strict=True):
        steps = [Step(idx=j, text=t) for j, t in enumerate(segment(text))]
        if steps:
            yield Trace(
                trace_id=f"{ex.id}:teacher",
                example_id=ex.id,
                dataset=ex.dataset,
                domain=ex.domain,
                model=cfg.teacher.model,
                question=ex.question,
                steps=steps,
                final_answer=ex.gold_answer,
            )


def build_traces(cfg: DictConfig) -> Path:
    """Write the trace corpus for the configured dataset; return the output path."""
    src = cfg.distill.source
    out = Path(cfg.paths.data) / "traces" / f"{cfg.data.name}.jsonl"
    traces = (
        load_open_traces(src, dataset=cfg.data.name, domain=cfg.data.domain)
        if src.mode == "open"
        else _teacher_traces(cfg)
    )
    n = write_records(out, traces)
    log.info("wrote %d traces -> %s", n, out)
    return out
