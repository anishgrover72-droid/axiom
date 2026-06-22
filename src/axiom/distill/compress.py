"""Sparse CoT Compression (Innovation 2).

Data-side, embedding-driven compression: prune redundant steps and merge tiny ones until
the trace fits a token budget, always keeping the first and final (answer) step. The GRPO
efficiency reward continues compression later; this just builds the SFT corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig

from axiom.common.answers import grade
from axiom.common.embed import embed
from axiom.common.errors import ConfigError
from axiom.common.io import read_records, write_records
from axiom.common.logging import get_logger
from axiom.common.tokens import count_batch, reduction_ratio
from axiom.data.schemas import Step, Trace

log = get_logger(__name__)


def _answer_preserved(steps: list[str], gold: str | None, answer_type: str | None) -> bool:
    """True if the compressed reasoning still grades to the trace's own final answer."""
    if not gold or not answer_type:
        return True  # nothing to preserve against; accept the compression
    try:
        return grade("\n".join(steps), gold, answer_type).correct
    except ConfigError:
        # Gold doesn't match answer_type (e.g. MCQ letter in a numeric-typed run).
        # Can't validate; don't block compression.
        return True


def _total_tokens(texts: list[str], hf_id: str) -> int:
    return sum(count_batch(texts, hf_id)) if texts else 0


def _merge_short(texts: list[str], min_tokens: int, hf_id: str) -> list[str]:
    counts = count_batch(texts, hf_id)
    out: list[str] = []
    for text, n in zip(texts, counts, strict=True):
        if out and n < min_tokens:
            out[-1] = f"{out[-1]} {text}".strip()
        else:
            out.append(text)
    return out


def compress_trace(
    trace: Trace, cfg: DictConfig, hf_id: str, answer_type: str | None = None
) -> Trace:
    """Return a token-compressed copy of `trace`, reverting if it breaks answer derivability."""
    texts = [s.text for s in trace.steps]
    n = len(texts)
    if n <= 2:
        return trace

    emb = embed(texts, cfg.embedder)
    keep = [True] * n
    kept_idx = [0]  # first step always survives
    for i in range(1, n - 1):
        if max(float(emb[i] @ emb[j]) for j in kept_idx) >= cfg.novelty_threshold:
            keep[i] = False  # redundant with an earlier kept step
        else:
            kept_idx.append(i)

    budget = int(_total_tokens(texts, hf_id) * cfg.target_ratio)
    interior = [i for i in range(1, n - 1) if keep[i]]
    while interior and _total_tokens([texts[i] for i in range(n) if keep[i]], hf_id) > budget:
        worst = max(
            interior,
            key=lambda i: max(float(emb[i] @ emb[j]) for j in range(n) if keep[j] and j != i),
        )
        keep[worst] = False
        interior.remove(worst)

    kept = _merge_short([texts[i] for i in range(n) if keep[i]], cfg.min_step_tokens, hf_id)
    if cfg.preserve_answer and not _answer_preserved(kept, trace.final_answer, answer_type):
        log.debug("compression reverted (answer not derivable): %s", trace.trace_id)
        return trace
    steps = [Step(idx=j, text=t) for j, t in enumerate(kept)]
    return trace.model_copy(update={"steps": steps, "n_tokens": _total_tokens(kept, hf_id)})


def compress_corpus(cfg: DictConfig) -> Path:
    """Compress every trace for the configured dataset; log the achieved token reduction."""
    in_path = Path(cfg.paths.data) / "traces" / f"{cfg.data.name}.jsonl"
    out_path = Path(cfg.paths.data) / "compressed" / f"{cfg.data.name}.jsonl"
    hf_id = cfg.model.hf_id
    before = after = 0

    def _stream() -> Iterator[Trace]:
        nonlocal before, after
        for trace in read_records(in_path, Trace):
            before += _total_tokens([s.text for s in trace.steps], hf_id)
            out = compress_trace(trace, cfg.distill.compress, hf_id, cfg.data.answer_type)
            after += out.n_tokens or 0
            yield out

    n = write_records(out_path, _stream())
    log.info(
        "compressed %d traces; token reduction %.1f%% -> %s",
        n,
        100 * reduction_ratio(before, after),
        out_path,
    )
    return out_path
