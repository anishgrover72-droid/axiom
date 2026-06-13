"""Verifier-Guided Decoding (Innovation 4).

At each reasoning step, sample B candidate next-steps, score each with the frozen XD-PRM,
advance the best, and prune the rest. Inference-only: no training. When no scorer is given
this degrades to ordinary single-sample decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

from axiom.common.steps import segment
from axiom.common.vllm_pool import Engine, GenConfig
from axiom.prm.score import PRMScorer, StepScores

_STEP_SYSTEM = "Continue the solution. Output ONLY the next single reasoning step, then stop."


@dataclass(slots=True)
class StepProposal:
    """The advanced next-step plus its score and the pruned/surviving alternatives."""

    text: str
    scores: StepScores | None
    surviving: list[str]  # top-k candidates kept (for the explainability view)
    pruned: list[str]  # candidates dropped


def _candidate_prompt(question: str, prior: list[str]) -> list[dict]:
    partial = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(prior)) or "(none yet)"
    user = f"{question}\n\nSteps so far:\n{partial}\n\nNext step:"
    return [{"role": "system", "content": _STEP_SYSTEM}, {"role": "user", "content": user}]


def generate_candidates(
    engine: Engine,
    question: str,
    prior: list[str],
    n: int,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """Sample `n` candidate next-steps, each trimmed to a single segmented step."""
    gen = GenConfig(n=n, temperature=temperature, max_tokens=max_tokens, stop=["\n\n"])
    completions = engine.chat([_candidate_prompt(question, prior)], gen)[0]
    return [seg[0] if (seg := segment(c)) else c.strip() for c in completions]


def propose_next(
    engine: Engine,
    scorer: PRMScorer | None,
    question: str,
    prior: list[str],
    domain: str,
    branch_factor: int,
    keep_top_k: int,
    temperature: float,
    max_tokens: int,
) -> StepProposal:
    """Generate, score, and select the next step (best-of-B with XD-PRM pruning)."""
    candidates = generate_candidates(
        engine, question, prior, branch_factor, temperature, max_tokens
    )
    if not scorer:
        return StepProposal(candidates[0], None, candidates[:1], candidates[1:])

    scored = [scorer.score([*prior, c], domain) for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: scored[i].final, reverse=True)
    best = order[0]
    surviving = [candidates[i] for i in order[:keep_top_k]]
    pruned = [candidates[i] for i in order[keep_top_k:]]
    return StepProposal(candidates[best], scored[best], surviving, pruned)
