# AXIOM

**Adaptive eXplainable Intelligence for Optimized Micro-Reasoning** — a cross-domain,
verifier-centric framework that turns Small Language Models into efficient, self-checking,
adaptive reasoners.

- **What we build:** `PLAN.md`
- **How we write it:** `CLAUDE.md` (binding engineering charter)

## Quickstart

```bash
make dev          # CPU subset: work on the shared contracts + tests
make test         # run the fast test suite
make install      # full GPU stack (A100/H100 box)
make serve        # FastAPI reasoning service on :8000
```

Frontend (explainability demo): `cd frontend && npm install && npm run dev`.

## Pipeline

```
traces → compress → sft → prm-label → prm-train → grpo → eval → serve
```

Each stage is a thin CLI in `scripts/` over logic in `src/axiom/`, driven by Hydra
configs in `configs/`. The XD-PRM step-level verifier (`src/axiom/prm/`) is the hub:
GRPO, verifier-guided decoding, and the adaptive-depth controller all consume it.

## Layout

`src/axiom/{common,data,distill,sft,prm,rl,inference,eval,serve}` — see `PLAN.md` §2.
