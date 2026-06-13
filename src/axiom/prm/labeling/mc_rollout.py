"""LOGIC labels via Monte-Carlo rollouts (Math-Shepherd style).

For each step prefix, sample k continuations from the student and measure the fraction
that reach the gold answer: a soft, verifiable, fully-automatic p(prefix -> correct). The
same k samples feed the Confidence head (see confidence.py), so one expensive op yields
two label streams.
"""

from __future__ import annotations

from omegaconf import DictConfig

from axiom.common.answers import grade
from axiom.common.vllm_pool import Engine, GenConfig

_SYSTEM = "You are finishing a partial step-by-step solution. End with 'The answer is X.'"


def _prefix_prompt(question: str, steps: list[str]) -> list[dict]:
    partial = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    user = f"{question}\n\nPartial solution:\n{partial}\n\nContinue from here and finish."
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def mc_logic_labels(
    engine: Engine,
    question: str,
    steps: list[str],
    gold: str,
    answer_type: str,
    cfg: DictConfig,
    choices: dict | None = None,
    seed: int | None = None,
) -> tuple[list[float], list[list[str | None]]]:
    """Return (logic[step], sampled_answers[step][sample]) over all step prefixes."""
    conversations = [_prefix_prompt(question, steps[: k + 1]) for k in range(len(steps))]
    gen = GenConfig(
        n=cfg.rollouts_k, temperature=cfg.temperature, max_tokens=cfg.max_new_tokens, seed=seed
    )
    completions = engine.chat(conversations, gen)  # [n_steps][k]

    logic: list[float] = []
    answers: list[list[str | None]] = []
    for comps in completions:
        verdicts = [grade(c, gold, answer_type, choices) for c in comps]
        logic.append(sum(v.correct for v in verdicts) / len(verdicts) if verdicts else 0.0)
        answers.append([v.pred for v in verdicts])
    return logic, answers
