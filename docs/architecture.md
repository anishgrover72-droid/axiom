# Technical Architecture

AXIOM is a **verifier-centric** reasoning framework for Small Language Models (SLMs). A single
cross-domain process reward model (**XD-PRM**) sits at the centre and is consumed by four of the
five subsystems, which makes it both the headline novelty and the critical path.

## 1. System overview

```
                         ┌──────────────────────────────────────────────┐
                         │                  XD-PRM                       │
                         │  backbone + 5 scalar heads per reasoning step │
                         │  logic · commonsense · consistency ·         │
                         │  efficiency · confidence                      │
                         └───────▲───────────▲───────────▲──────────────┘
                                 │           │           │
            training reward ─────┘   decode score        confidence signal
                                 │           │           │
   ┌──────────┐   ┌──────────┐   │   ┌───────┴───────┐   │   ┌────────────────────┐
   │  Data &  │──▶│  SFT      │──▶│──▶│ GRPO (RL)     │──▶│──▶│ Adaptive-depth +    │
   │  Distill │   │ (QLoRA)   │       │ composite rwd │       │ verifier decoding   │
   └──────────┘   └──────────┘       └───────────────┘       └─────────┬──────────┘
        │                                                              │
        ▼                                                              ▼
  Sparse compression                                          FastAPI + SSE  ──▶  React console
  (~40% fewer tokens)
```

## 2. Data flow (9 stages)

| # | Stage | Module | In → Out |
|---|---|---|---|
| 00 | Ingest & difficulty-tag | `data/` | 5 datasets → unified `Example` schema |
| 01 | Distill teacher traces | `distill/` | questions → step-segmented reasoning `Trace`s |
| 02 | Sparse compression | `distill/compress` | verbose traces → sparse traces (~40% fewer tokens) |
| 03 | QLoRA SFT | `sft/` | sparse traces → fine-tuned student |
| 04 | PRM label foundry | `prm/labeling` | student + questions → per-step `StepLabel`s |
| 05 | Train XD-PRM | `prm/` | step labels → 5-head verifier + **G2 gate** |
| 06 | GRPO RL | `rl/` | student + XD-PRM → RL-tuned policy |
| 07 | Evaluate | `eval/` | policy + XD-PRM → AUC / best-of-N / ECE |
| 08 | Serve | `inference/` + `serve/` | policy + XD-PRM → live SSE reasoning |

## 3. Shared contracts (the spine)

To keep training, scoring and decoding identical, five contracts live in exactly one place each
and are imported everywhere (never re-implemented):

| Contract | File | Role |
|---|---|---|
| Step segmentation | `common/steps.py` | the one definition of "a step" + boundary sentinel; idempotent |
| Answer matching | `common/answers.py` | per-dataset gold extraction + verifiable equality |
| Token accounting | `common/tokens.py` | the single tokenizer-aware counter behind every efficiency number |
| Rollout engine | `common/vllm_pool.py` | one shared vLLM pool (MC labeling, GRPO sampling, self-consistency, decode) |
| Schemas | `data/schemas.py` | `Example, Trace, Step, StepLabel, RewardBreakdown` |

## 4. XD-PRM — the core model

- **Backbone:** a small instruct model (Qwen2.5-0.5B/1.5B for debug, Phi-4-mini for the headline),
  loaded 4-bit (QLoRA).
- **Heads:** the pooled hidden state at each step's boundary sentinel feeds **five scalar heads**
  (`logic, commonsense, consistency, efficiency, confidence`). A domain embedding conditions the
  backbone so one model serves every domain.
- **Aggregate reward** `R_aggregate` is a weighted combination of the heads, used as the step score.

### The label foundry (the real engineering)

Per-step training labels come from **Monte-Carlo rollouts** through the shared vLLM pool: from a
step prefix we sample *k* completions and measure how often they reach the gold answer
(Math-Shepherd-style soft value). One rollout batch yields the **logic**, **confidence**
(self-consistency / entropy) and **efficiency** signals at once — *compute once, derive thrice*.
Commonsense uses a free open NLI proxy; consistency checks contradiction against prior steps.

## 5. How the verifier is reused

- **GRPO reward** — composite of correctness + `R_aggregate` + efficiency penalty.
- **Verifier-guided decoding** — score candidate steps with XD-PRM, keep the best (KV-cache reused).
- **Adaptive-depth controller** — the confidence head's uncertainty decides `continue / expand / exit`,
  so easy questions exit early and hard ones get more compute.

## 6. Serving & UI

- `serve/` is a FastAPI app that streams reasoning over **SSE** (`event: step` / `event: done`),
  emitting per-step text, the 5 head scores, `R_aggregate`, uncertainty and the depth decision.
- `frontend/` renders this as a live **reasoning console** (streaming ledger + verifier telemetry),
  plus marketing/explainer pages (pipeline flowchart, tech & datasets). See
  [user-guide.md](user-guide.md).

## 7. Determinism & config

Every run is a composed **Hydra** config; ablations are CLI overrides
(`prm.heads.efficiency.enabled=false`), never code edits. All seeds flow from `common/seed.py`, so
anything affecting PRM labels or eval is reproducible from config.
