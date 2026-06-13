"""EFFICIENCY labels via a free heuristic.

A compact, efficient step is novel (low similarity to prior steps) and not bloated. We
combine a novelty term with a length factor that penalizes steps far longer than a target.
"""

from __future__ import annotations

from axiom.common.embed import embed
from axiom.common.tokens import count_batch

_TARGET_TOKENS = 32  # a tight, information-dense reasoning step


def efficiency_labels(steps: list[str], embedder: str, hf_id: str) -> list[float]:
    """Per-step efficiency in [0,1] = novelty x length_factor."""
    if not steps:
        return []
    emb = embed(steps, embedder)
    lengths = count_batch(steps, hf_id)
    out: list[float] = []
    for i, n_tokens in enumerate(lengths):
        novelty = 1.0 if i == 0 else 1.0 - max(float(emb[i] @ emb[j]) for j in range(i))
        length_factor = min(1.0, _TARGET_TOKENS / max(n_tokens, 1))
        # cosine can be negative -> novelty can exceed 1; clamp so the head target stays in [0,1]
        out.append(min(1.0, max(0.0, novelty)) * length_factor)
    return out
