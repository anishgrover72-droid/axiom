"""Step-labeled dataset for XD-PRM training.

A trace is rendered with step sentinels and tokenized; the k-th sentinel marks the end of
step k. Steps are flattened across the batch in the collator so heads see one
[num_steps, hidden] tensor. Missing per-head labels become mask=0 (excluded from loss).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from axiom.common.logging import get_logger
from axiom.common.steps import render_with_sentinels
from axiom.data.schemas import HEAD_NAMES, StepLabel, Trace
from axiom.prm.model import domain_id

log = get_logger(__name__)


class StepLabeledDataset(Dataset):
    """Yields one tokenized trace with its per-step, per-head targets and masks."""

    def __init__(
        self,
        traces: list[Trace],
        labels: dict[tuple[str, int], StepLabel],
        tokenizer,
        sentinel_id: int,
        max_len: int,
    ) -> None:
        self.traces = traces
        self.labels = labels
        self.tok = tokenizer
        self.sentinel_id = sentinel_id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, i: int) -> dict:
        trace = self.traces[i]
        text = render_with_sentinels([s.text for s in trace.steps])
        ids = self.tok(text, truncation=True, max_length=self.max_len)["input_ids"]
        sent_pos = [p for p, t in enumerate(ids) if t == self.sentinel_id]
        if len(sent_pos) != len(trace.steps):
            log.warning(
                "trace %s truncated: %d of %d steps labeled (max_len=%d)",
                trace.trace_id,
                len(sent_pos),
                len(trace.steps),
                self.max_len,
            )

        targets = {h: [] for h in HEAD_NAMES}
        masks = {h: [] for h in HEAD_NAMES}
        for k in range(len(sent_pos)):
            label = self.labels.get((trace.trace_id, k))
            for h in HEAD_NAMES:
                value = getattr(label, h) if label else None
                masks[h].append(value is not None)
                targets[h].append(float(value) if value is not None else 0.0)
        return {
            "ids": ids,
            "sent_pos": sent_pos,
            "domain": domain_id(trace.domain),
            "targets": targets,
            "masks": masks,
        }


def make_collate(pad_id: int):
    """Build a collate_fn that pads inputs and flattens steps across the batch."""

    def collate(batch: list[dict]) -> dict:
        max_t = max(len(b["ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_t), dtype=torch.long)
        s_batch, s_tok, s_dom = [], [], []
        targets = {h: [] for h in HEAD_NAMES}
        masks = {h: [] for h in HEAD_NAMES}
        for bi, b in enumerate(batch):
            n = len(b["ids"])
            input_ids[bi, :n] = torch.tensor(b["ids"])
            attention[bi, :n] = 1
            for k, pos in enumerate(b["sent_pos"]):
                s_batch.append(bi)
                s_tok.append(pos)
                s_dom.append(b["domain"])
                for h in HEAD_NAMES:
                    targets[h].append(b["targets"][h][k])
                    masks[h].append(b["masks"][h][k])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "step_batch_idx": torch.tensor(s_batch, dtype=torch.long),
            "step_tok_idx": torch.tensor(s_tok, dtype=torch.long),
            "step_domain": torch.tensor(s_dom, dtype=torch.long),
            "targets": {h: torch.tensor(v, dtype=torch.float) for h, v in targets.items()},
            "masks": {h: torch.tensor(v, dtype=torch.bool) for h, v in masks.items()},
        }

    return collate


def is_holdout(trace_id: str) -> bool:
    """Deterministic ~20% PRM-validation holdout, shared by train_prm and validate_prm.

    Keeping one definition here is what prevents the gate from scoring trained-on traces.
    """
    return int(hashlib.md5(trace_id.encode()).hexdigest(), 16) % 5 == 0


def load_labeled(
    prm_root: Path, split: str = "all"
) -> tuple[list[Trace], dict[tuple[str, int], StepLabel]]:
    """Read and merge every per-dataset foundry output under `prm_root`.

    Each foundry run writes `prm_root/<dataset>/{traces,labels}.jsonl`, so the cross-domain
    PRM trains on all domains at once. `split` selects 'train' (non-holdout), 'val'
    (holdout), or 'all'.
    """
    from axiom.common.io import read_records

    trace_files = sorted(prm_root.glob("*/traces.jsonl"))
    if not trace_files and (prm_root / "traces.jsonl").exists():
        trace_files = [prm_root / "traces.jsonl"]  # single-dataset / legacy layout

    traces: list[Trace] = []
    labels: dict[tuple[str, int], StepLabel] = {}
    for tf in trace_files:
        for trace in read_records(tf, Trace):
            if split == "train" and is_holdout(trace.trace_id):
                continue
            if split == "val" and not is_holdout(trace.trace_id):
                continue
            traces.append(trace)
        label_file = tf.with_name("labels.jsonl")
        if label_file.exists():
            for label in read_records(label_file, StepLabel):
                labels[(label.trace_id, label.step_idx)] = label
    return traces, labels
