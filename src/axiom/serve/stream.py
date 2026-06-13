"""Turn a reasoning run into a sequence of serializable SSE events."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict

from axiom.common.answers import extract_prediction
from axiom.common.tokens import count
from axiom.inference.engine import ReasoningEngine


def reason_events(
    engine: ReasoningEngine, question: str, domain: str, answer_type: str
) -> Iterator[dict]:
    """Yield one `step` event per reasoning step, then a final `done` event."""
    texts: list[str] = []
    for step in engine.iter_steps(question, domain):
        texts.append(step.text)
        yield {"event": "step", "data": asdict(step)}

    last = texts[-1] if texts else ""
    yield {
        "event": "done",
        "data": {
            "answer": extract_prediction(last, answer_type),
            "n_steps": len(texts),
            "n_tokens": sum(count(t, engine.cfg.model.hf_id) for t in texts),
        },
    }
