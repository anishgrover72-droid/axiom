"""FastAPI service: stream verifier-scored reasoning steps for the explainability demo.

POST /reason streams Server-Sent Events; each `step` event carries the per-head XD-PRM
scores, the adaptive-depth decision, and the pruned candidates the frontend renders.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from axiom.common.config import load_config
from axiom.inference.engine import ReasoningEngine
from axiom.serve.stream import reason_events

app = FastAPI(title="AXIOM")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_engine: ReasoningEngine | None = None


class ReasonRequest(BaseModel):
    question: str
    domain: str = "commonsense"
    answer_type: str = "span"


def _get_engine() -> ReasoningEngine:
    """Build the engine once, from env-selected variant (default SFT policy)."""
    global _engine
    if _engine is None:
        cfg = load_config(os.environ.get("AXIOM_OVERRIDES", "").split() or None)
        _engine = ReasoningEngine(cfg, variant=os.environ.get("AXIOM_VARIANT", "sft"))
    return _engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reason")
async def reason(req: ReasonRequest) -> EventSourceResponse:
    engine = _get_engine()
    events = reason_events(engine, req.question, req.domain, req.answer_type)

    async def emit():
        # Pull each blocking step off the thread pool so steps stream as they are produced.
        while (evt := await run_in_threadpool(next, events, None)) is not None:
            yield {"event": evt["event"], "data": json.dumps(evt["data"])}

    return EventSourceResponse(emit())
