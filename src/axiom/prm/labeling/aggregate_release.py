"""The label foundry orchestrator.

Generates one student trace per example, runs every labeler, and writes the XD-PRM
training data (traces.jsonl + labels.jsonl) plus the publishable Compact Distilled
Reasoning Trace Dataset (data/release/). One pass produces both.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from axiom.common.io import write_records
from axiom.common.logging import get_logger
from axiom.common.paths import foundry_dir
from axiom.common.seed import set_seed
from axiom.common.steps import segment
from axiom.common.vllm_pool import GenConfig, get_engine
from axiom.data.loaders import load_examples
from axiom.data.schemas import HEAD_NAMES, Step, StepLabel, Trace
from axiom.prm.labeling.confidence import confidence_from_answers
from axiom.prm.labeling.efficiency import efficiency_labels
from axiom.prm.labeling.mc_rollout import mc_logic_labels
from axiom.prm.labeling.nli_consistency import nli_consistency_labels, nli_entailment_labels

log = get_logger(__name__)

_SOLVE = "Solve step by step. Number each step. End with 'The answer is X.'"
_JUDGED_DOMAINS = {"commonsense", "science"}


def _maybe_judge_client(cfg: DictConfig):
    """A teacher judge only if the commonsense head is on and credentials exist."""
    if not cfg.prm.heads.commonsense.enabled:
        return None
    from axiom.common.errors import ConfigError
    from axiom.distill.teacher import TeacherClient

    try:
        return TeacherClient(cfg.teacher)
    except ConfigError:
        log.warning("no judge credentials; commonsense head falls back to unlabeled")
        return None


def run_foundry(cfg: DictConfig) -> Path:
    """Produce labeled steps for the configured dataset; return the data/prm dir."""
    set_seed(cfg.seed)
    lab = cfg.prm.labeling
    engine = get_engine(cfg.model.hf_id, cfg.model.dtype, cfg.model.max_seq_len)
    judge = _maybe_judge_client(cfg)
    need_mc = cfg.prm.heads.logic.enabled or cfg.prm.heads.confidence.enabled

    examples = list(load_examples(cfg.data, "train", cfg.distill.source.limit))
    mains = engine.chat(
        [
            [{"role": "system", "content": _SOLVE}, {"role": "user", "content": e.question}]
            for e in examples
        ],
        GenConfig(n=1, temperature=0.7, max_tokens=cfg.model.max_seq_len // 4, seed=cfg.seed),
    )

    traces: list[Trace] = []
    labels: list[StepLabel] = []
    judged = 0
    for ex, main in zip(examples, mains, strict=True):
        texts = segment(main[0])
        if not texts:
            continue
        trace = Trace(
            trace_id=f"{ex.id}:0",
            example_id=ex.id,
            dataset=ex.dataset,
            domain=ex.domain,
            model=cfg.model.name,
            question=ex.question,
            steps=[Step(idx=i, text=t) for i, t in enumerate(texts)],
            final_answer=ex.gold_answer,
        )

        logic, answers = (
            mc_logic_labels(
                engine,
                ex.question,
                texts,
                ex.gold_answer,
                ex.answer_type,
                lab.logic,
                ex.choices,
                cfg.seed,
            )
            if need_mc
            else ([None] * len(texts), [[] for _ in texts])
        )
        confidence = (
            confidence_from_answers(answers)
            if cfg.prm.heads.confidence.enabled
            else [None] * len(texts)
        )
        consistency = (
            nli_consistency_labels(texts, lab.consistency.nli_model)
            if cfg.prm.heads.consistency.enabled
            else [None] * len(texts)
        )
        efficiency = (
            efficiency_labels(texts, lab.efficiency.embedder, cfg.model.hf_id)
            if cfg.prm.heads.efficiency.enabled
            else [None] * len(texts)
        )
        commonsense = [None] * len(texts)
        if judge and ex.domain in _JUDGED_DOMAINS and judged < lab.commonsense.subset_size:
            from axiom.prm.labeling.judge import judge_commonsense_labels

            commonsense = judge_commonsense_labels(judge, ex.question, texts)
            judged += 1
        elif cfg.prm.heads.commonsense.enabled:
            # Free fallback: NLI entailment of each step is a plausibility proxy (no API cost).
            commonsense = nli_entailment_labels(texts, lab.consistency.nli_model)

        for k in range(len(texts)):
            labels.append(
                StepLabel(
                    trace_id=trace.trace_id,
                    step_idx=k,
                    domain=ex.domain,
                    logic=logic[k],
                    commonsense=commonsense[k],
                    consistency=consistency[k],
                    efficiency=efficiency[k],
                    confidence=confidence[k],
                )
            )
        traces.append(trace)

    out_dir = foundry_dir(cfg) / cfg.data.name  # per-dataset => cross-domain accumulation
    write_records(out_dir / "traces.jsonl", traces)
    write_records(out_dir / "labels.jsonl", labels)
    _write_release(Path(cfg.paths.data) / "release" / f"{cfg.data.name}.jsonl", traces, labels)
    log.info("foundry: %d traces, %d labeled steps (%d judged)", len(traces), len(labels), judged)
    return out_dir


def _write_release(out_path: Path, traces: list[Trace], labels: list[StepLabel]) -> None:
    """Merge traces + labels into the publishable dataset shard (one row per step)."""
    import json

    by_id = {(label.trace_id, label.step_idx): label for label in labels}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for trace in traces:
            for step in trace.steps:
                label = by_id.get((trace.trace_id, step.idx))
                f.write(
                    json.dumps(
                        {
                            "dataset": trace.dataset,
                            "domain": trace.domain,
                            "question": trace.question,
                            "step_idx": step.idx,
                            "step": step.text,
                            "labels": label.head_values() if label else dict.fromkeys(HEAD_NAMES),
                        }
                    )
                    + "\n"
                )
