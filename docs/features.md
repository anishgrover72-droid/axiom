# Salient Features

## Novelty

1. **XD-PRM — one cross-domain process verifier.** A single backbone with **five scalar heads**
   (logic, commonsense, consistency, efficiency, confidence) scores *every reasoning step* across
   math, science, commonsense and multi-hop. Most PRMs are single-domain and single-signal; XD-PRM
   is cross-domain and multi-signal, and it is reused by four subsystems.

2. **MC-rollout label foundry — compute once, derive thrice.** Per-step labels come from
   Monte-Carlo rollouts; one rollout batch yields the logic, confidence and efficiency signals
   simultaneously, making the expensive labeling stage far cheaper.

3. **Sparse reasoning compression (~40% fewer tokens).** Prunes filler and merges redundant steps
   under a token budget with an **answer-preservation check**, then RL continues the compression.

4. **Composite-reward GRPO.** Reinforcement learning on a reward that blends correctness, the
   verifier's step quality, and an efficiency penalty — reasoning that is correct *and* compact.

5. **Verifier-guided adaptive-depth decoding.** The confidence head's uncertainty drives a
   `continue / expand / exit` controller, so easy questions exit early and hard ones get more
   compute — tokens spent where they matter.

## Engineering

- **Shared contracts, one source of truth** — step segmentation, answer matching, token counting,
  the rollout engine and the schemas each exist in exactly one place and are imported everywhere
  (identical at train, score and decode time). Unit-tested for idempotence/determinism.
- **Config over code** — every tunable is a Hydra config; ablations are CLI overrides, never edits.
- **One shared vLLM pool** — MC labeling, GRPO sampling, self-consistency and verifier-decode all
  batch through the same KV-cache-reusing pool.
- **Reproducible** — all seeds flow from one module; anything affecting labels/eval is reproducible
  from config. Hard **G2 gate** blocks the pipeline unless the verifier passes (AUC/ECE/head-corr).
- **Tested** — fast contract suite (30 passing) plus a slow GPU smoke test.

## Product / demo

- **Live explainability console** — watch the model reason step-by-step with per-step 5-head
  telemetry, an aggregate-reward dial, and the depth decisions, streamed over SSE.
- **Self-demonstrating UI** — falls back to a labelled sample trace when no GPU backend is up, so
  it's always presentable.
- **Open & self-contained** — open-weight models, public datasets, public-domain imagery; no
  proprietary API required to run the pipeline.

## Deliverables

- Trained **XD-PRM** + SFT/GRPO **student** (published on Hugging Face — see README).
- **Compact Distilled Reasoning Trace Dataset** (`data/release/`, published on Hugging Face).
- Reproducible training/eval (`notebooks/colab_run.ipynb`, `RUNBOOK.md`).
- React explainability app.
