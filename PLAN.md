# AXIOM — Master Execution Plan

**Adaptive eXplainable Intelligence for Optimized Micro-Reasoning**
A cross-domain, verifier-centric framework that turns Small Language Models into *efficient, self-checking, adaptive* reasoners.

> **Constraints locked:** 1× A100/H100 (40–80 GB) · QLoRA/LoRA · **XD-PRM is the novelty centerpiece** · hackathon horizon · **the whole system is in scope — no phased scope-cutting.** Development is organized as parallel **workstream chunks** joined by a dependency DAG and **capability gates**, not calendar phases.

---

## 0. The thesis in one diagram

The deck's 9-stage pipeline, redrawn as the **actual data/control flow** we will build, with the dependency that everything hinges on made explicit:

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                    XD-PRM (5-head verifier)               │
                         │   the hub: produces step-level reward signal R_aggregate  │
                         └───────▲───────────────▲───────────────▲──────────────────┘
                                 │ trains on     │ rewards        │ scores at decode
                                 │ step labels   │ rollouts       │ + at inference
        ┌──────────┐   ┌─────────┴──┐   ┌────────┴────┐   ┌───────┴────────┐   ┌──────────────┐
 Teacher│ Trace    │   │ Sparse CoT │   │ Difficulty- │   │ SLM Fine-Tune  │   │ GRPO / RLVR  │
 traces→│Distillat.│ → │ Compression│ → │ Aware       │ → │ (QLoRA SFT)    │ → │ Optimization │
        └──────────┘   └────────────┘   │ Curriculum  │   └────────────────┘   └──────┬───────┘
                                         └─────────────┘                               │
                                                                                       ▼
                         ┌──────────────────────────── Efficient Inference Engine ───────────────────────────┐
                         │  Verifier-Guided Decoding  +  Adaptive Reasoning-Depth Controller  +  base policy   │
                         └───────────────────────────────────────────────────────────────────────────────────┘
```

**The single most important fact about this project:** XD-PRM is simultaneously (a) the headline novelty, (b) the training signal for GRPO, (c) the scoring function for verifier-guided decoding, and (d) a source of the confidence signal for the adaptive-depth controller. **Four of the five components consume XD-PRM.** Therefore XD-PRM — and specifically the *pipeline that produces its per-step training labels* — is the critical path. The model architecture (a backbone + 5 scalar heads) is trivial; the **label foundry** is the real engineering.

---

## 1. What the deck commits us to (faithful spec)

### 1.1 The five components (and what each actually is)
| # | Deck name | Concrete realization | Consumes | Produces |
|---|-----------|----------------------|----------|----------|
| 1 | Distilled Sparse Reasoning Traces | Token-budget-aware compression of teacher CoT → SFT data | teacher traces | compressed traces, SFT corpus |
| 2 | GRPO / RLVR Optimization | TRL `GRPOTrainer`, verifiable + process + efficiency reward | SFT model, **XD-PRM** | RL-tuned policy |
| 3 | **XD-PRM (Cross-Domain Process Reward Model)** | backbone + **5 heads** → per-step `r_h` → `R_aggregate` | step-labeled data | step rewards (the hub) |
| 4 | Adaptive Reasoning-Depth Controller | entropy/uncertainty/confidence-gated early-exit vs. expand | policy, **XD-PRM** confidence | depth decisions |
| 5 | Verifier-Guided Decoding | step-level beam; prune low-`R_aggregate` branches | policy, **XD-PRM** | pruned reasoning |

### 1.2 The five XD-PRM heads — and where the deck's "causal/contradiction" claims live
The novelty slide promises *commonsense-aware verification, causal coherence validation, semantic contradiction detection*. These are **not** extra heads — they map onto the 5:

| Head | Output `r_h ∈ [0,1]` | Covers deck claim | Cross-domain role |
|------|----------------------|-------------------|-------------------|
| **Logic** | logical validity / leads-to-correct | math correctness | math, formal |
| **Commonsense** | real-world plausibility + **causal coherence** | "causal coherence validation" | commonsense, causal, science |
| **Consistency** | alignment w/ prior steps + **contradiction detection** | "semantic contradiction detection" | all domains |
| **Efficiency** | compactness / non-redundancy | the "~40%" token story | all domains |
| **Confidence** | calibrated uncertainty | feeds adaptive depth | all domains |

`R_aggregate(step) = Σ_h w_h · r_h`, weights fixed in v1, made learnable (or domain-conditioned) in v2.

### 1.3 The four innovations → measurable claims (we report numbers, not adjectives)
| Innovation | Claim in deck | How we measure it | Honest status |
|---|---|---|---|
| XD-PRM | cross-domain step verification | ROC-AUC on good-vs-bad steps; best-of-N lift | core deliverable |
| Sparse Reasoning Compression | "~40% fewer tokens, quality preserved" | tokens/answer ratio @ matched accuracy | **target, to be measured** |
| Adaptive Compute Allocation | deep reasoning only when needed | tokens & latency vs. fixed-depth @ equal accuracy | core deliverable |
| Verifier-Guided Decoding | suppress weak trajectories, less hallucination | accuracy↑, contradiction-rate↓ | core deliverable |

### 1.4 Datasets & the dataset contribution
- **Use/eval:** GSM8K (math), CommonsenseQA + OpenBookQA (commonsense/science), ARC-Challenge (hard tier), HotpotQA (multi-hop). Synthetic CoT = fine-tuning data.
- **Contribution (deliverable):** *Compact Distilled Reasoning Trace Dataset* = sparse traces + verifier-aligned per-step annotations + commonsense process labels. **This is a byproduct of the label foundry (§4) — the foundry's outputs are designed to be directly publishable.**

---

## 2. Repository structure (the full file tree)

```
axiom/
├── README.md · PLAN.md · pyproject.toml · requirements.txt · Makefile · .env.example · .gitignore
│
├── configs/                              # Hydra config groups — every run is a composed config
│   ├── config.yaml
│   ├── model/        {qwen2_5_0_5b, qwen2_5_1_5b, qwen2_5_7b, phi4_mini}.yaml
│   ├── data/         {gsm8k, commonsenseqa, openbookqa, arc_challenge, hotpotqa}.yaml
│   ├── distill/      {sources, compress}.yaml
│   ├── sft/          lora.yaml
│   ├── prm/          xdprm.yaml          # heads, label-source weights, loss weights, aggregation
│   ├── grpo/         grpo.yaml           # group size, reward weights, KL, anti-hacking
│   ├── infer/        {adaptive_depth, verifier_decode, engine}.yaml
│   └── eval/         suite.yaml
│
├── data/                                 # gitignored
│   ├── raw/  traces/  compressed/  curriculum/  prm/  release/   # release/ = the publishable dataset
│
├── src/axiom/
│   ├── common/
│   │   ├── steps.py        # ★ THE shared step-segmentation contract (one definition, used everywhere)
│   │   ├── answers.py      # per-dataset gold extraction + verifiable answer matching
│   │   ├── tokens.py       # token counting / budget accounting (the efficiency metric)
│   │   ├── vllm_pool.py    # ★ shared vLLM engine for fast batched rollouts (the compute bottleneck)
│   │   ├── seed.py · io.py · logging.py · registry.py
│   ├── data/
│   │   ├── schemas.py      # pydantic: Example, Trace, Step, StepLabel, RewardBreakdown
│   │   ├── loaders.py      # HF datasets → unified schema
│   │   └── difficulty.py   # difficulty scoring + Easy/Inter/Multi-hop/Adversarial buckets
│   ├── distill/
│   │   ├── sources.py      # PRIMARY: load existing open reasoning-trace datasets
│   │   ├── teacher.py      # OPTIONAL: generate traces from a teacher (vLLM/API)
│   │   ├── generate.py     # entrypoint (optional path)
│   │   └── compress.py     # ★ Sparse CoT Compression (informativeness prune/merge + answer-preservation check)
│   ├── sft/
│   │   └── train_sft.py    # QLoRA SFT on compressed traces, curriculum-staged
│   ├── prm/                # ★★★ XD-PRM — centerpiece
│   │   ├── model.py        # backbone + 5 heads, step-pooling, R_aggregate
│   │   ├── heads.py        # head modules + per-head loss
│   │   ├── dataset.py      # step-labeled dataset; per-head label masking
│   │   ├── train_prm.py    # multi-task training
│   │   ├── score.py        # inference API: score_steps(text) → per-step/per-head/R_aggregate
│   │   ├── validate.py     # ★ gate checks: AUC, best-of-N, calibration, head-correlation
│   │   └── labeling/       # ★ THE LABEL FOUNDRY (the real work + the publishable dataset)
│   │       ├── mc_rollout.py      # LOGIC: Math-Shepherd-style p(prefix→correct) via vLLM rollouts
│   │       ├── nli_consistency.py # CONSISTENCY/contradiction: off-the-shelf NLI vs prior steps
│   │       ├── efficiency.py      # EFFICIENCY: novel-info / redundancy / length heuristic
│   │       ├── judge.py           # COMMONSENSE+causal: teacher-LLM judge (budget-capped)
│   │       ├── confidence.py      # CONFIDENCE: token entropy / self-consistency variance
│   │       └── aggregate_release.py # merge all labels → data/release/ (publishable)
│   ├── rl/
│   │   ├── rewards.py      # composite reward + normalization + anti-hacking terms
│   │   └── train_grpo.py   # TRL GRPOTrainer wiring (PRM reward server)
│   ├── inference/
│   │   ├── adaptive_depth.py   # ★ Innovation 3: entropy/confidence-gated exit/expand + budget cap
│   │   ├── verifier_decode.py  # ★ Innovation 4: step-level beam; prune low-R_aggregate branches
│   │   └── engine.py           # unified engine: policy + decode strategy + depth ctrl + PRM trace
│   ├── eval/
│   │   ├── benchmarks.py · metrics.py · run_eval.py   # matrix + ablation harness
│   └── serve/
│       ├── app.py · stream.py   # FastAPI, SSE streaming steps + per-head scores + depth decisions
│
├── scripts/      00_download → 01_traces → 02_compress → 03_sft → 04_prm_label → 05_prm_train
│                 → 06_grpo → 07_eval → 08_serve   (thin CLIs over src/)
├── frontend/     React + Vite + shadcn/ui + React Flow (ReasoningGraph, HeadScores, DepthTrace, Demo)
├── notebooks/    label-quality audits, ablation plots, reward-hacking forensics
├── tests/        steps.py, answers.py, schemas, reward shapes, PRM scoring determinism
└── experiments/  gitignored: checkpoints, W&B, eval JSON, run cards
```

**Three invariants the tree enforces:**
1. **One step definition** (`common/steps.py`) shared by PRM-train, PRM-score, verifier-decode, and the frontend. Drift here silently corrupts the PRM → unit-tested, single source of truth.
2. **One rollout engine** (`common/vllm_pool.py`). MC-rollout labeling, GRPO sampling, self-consistency, and verifier-decode all generate from the same vLLM pool → batched, KV-cached, the only way the compute budget closes.
3. **The label foundry emits the publishable dataset directly** (`labeling/aggregate_release.py → data/release/`). The contribution is free.

---

## 3. Workstream chunks & dependency DAG (this replaces "phases")

Seven chunks. Each is an independent unit of work with a crisp "done" definition. The DAG says what unblocks what; **chunks on different branches run in parallel.**

```
WS-A  Infra & Contracts ───────────────────────────────────────────┐ (unblocks all)
        steps.py · schemas · answers · tokens · vllm_pool · configs · eval skeleton · tests
                                                                     │
WS-B  Data Foundry ──────────────┬───────────────────────────────────┤
   B1 trace acquisition (open)   │                                    │
   B2 sparse compression  ───────┼──► (SFT corpus)                    │
   B3 difficulty/curriculum      │                                    │
   B4 ★ PRM label foundry  ──────┼──────────────► WS-C                │
                                 │                                    │
WS-C  ★ XD-PRM ──────────────────┼───────────► WS-D(GRPO), WS-E       │
   C1 model+heads  C2 train  C3 validate(GATE)                        │
                                 │                                    │
WS-D  Student Training           │                                    │
   D1 SFT  ◄── B2,B3 ────────────┘                                    │
   D2 GRPO ◄── C(validated) + D1                                      │
                                                                     │
WS-E  Inference Engine ◄── C(validated) + D1                          │
   E1 adaptive depth   E2 verifier decode   E3 unified engine         │
                                                                     │
WS-F  Eval & Ablation ◄── E (+D2)                                     │
WS-G  Serve & Explainability Demo ◄── E ◄────────────────────────────┘
```

**Critical path:** `A → B4 → C2 → C3(GATE) → E → F/G`. Everything off this path (B1, B2, B3, D1, frontend shell, eval harness) is built **in parallel** while the foundry+PRM cook. D1 (SFT) finishing early matters because it provides both the GRPO init *and* the policy the inference engine wraps.

### Capability gates (system states, not dates)
- **G0 — Skeleton alive:** every module imports; toy example flows end-to-end; `make test` green; one real SFT step runs.
- **G1 — Foundry online:** PRM label foundry produces labeled steps for ≥2 domains; label-quality audit passes (spot-check + automatic sanity stats).
- **G2 — PRM validated (the hard gate):** XD-PRM passes §4.4 (AUC > 0.7, best-of-N lift, confidence calibrated). *Nothing downstream that consumes PRM is trusted until G2.*
- **G3 — Engine beats baseline:** unified inference (SFT policy + verifier decode + adaptive depth) beats vanilla CoT on accuracy **and** tokens/answer.
- **G4 — RL closes the loop:** GRPO with composite reward ≥ SFT policy (or is cleanly diagnosed and the SFT+engine result stands as headline).
- **G5 — Story complete:** full eval matrix + ablations + live demo + `data/release/` exported.

---

## 4. ★ The label foundry & XD-PRM (deepest section — this is where we win or lose)

### 4.1 Why this is the crux
No public dataset (GSM8K/CSQA/ARC/HotpotQA) ships **per-step process labels**. A 5-head PRM needs 5 label streams. The foundry's job: manufacture those labels **cheaply and at scale**, mostly without human or paid-LLM annotation.

### 4.2 Per-head label sources (engineered for cost)
| Head | Source | Cost | Mechanism | Output |
|------|--------|------|-----------|--------|
| **Logic** | MC rollout (Math-Shepherd / "process-as-soft-value") | compute-only | For each step prefix, sample *k* (≈8–16) continuations from the student via vLLM; `label = fraction reaching gold answer`. Verifiable, automatic. | soft `p(correct)` per step |
| **Consistency** | off-the-shelf NLI (e.g. DeBERTa-MNLI) | ~free | `r = 1 − P(contradiction | prior_steps ⊢ step)`; gives **contradiction detection** for free | per step |
| **Efficiency** | heuristic | free | normalized: `novel_info_ratio − redundancy_penalty − length_overrun`; novelty via embedding-sim vs prior steps | per step |
| **Commonsense (+causal)** | teacher-LLM judge | **paid, capped** | rubric-scored plausibility + causal-coherence on a **subset** of CSQA/OBQA/HotpotQA steps; the only paid stream | per step (subset) |
| **Confidence** | self-consistency / entropy | free (reuses Logic rollouts) | answer-variance across the *k* samples already drawn, or mean token entropy → calibration target | per step |

**Key efficiency move:** the Logic head's *k* rollouts are the project's main compute cost — and they **simultaneously** produce the Confidence label (variance of the same samples) and the Efficiency baseline. One expensive operation, three labels. The Commonsense judge is the *only* paid stream and is budget-capped to a subset; if budget is tight it degrades gracefully to an NLI/plausibility proxy.

**Honest scoping:** Logic + Consistency + Efficiency + Confidence are **fully supervised for free**; Commonsense is **partially** supervised (paid subset + proxy). We ship a genuine 5-head model and *document head strength explicitly* — no claim of uniform label quality.

### 4.3 XD-PRM model (`prm/model.py`)
```
text w/ step sentinels → backbone (Qwen2.5-0.5B/1.5B, QLoRA)  → hidden states
   → gather hidden at each step's end-sentinel position (step-pooling)
   → 5 parallel MLP heads → r_logic, r_cs, r_cons, r_eff, r_conf ∈ [0,1]
   → R_aggregate = Σ_h w_h · r_h
```
- **Backbone choice:** same family/tokenizer as the student → no token-alignment pain, and the student's hidden representations transfer. (DeBERTa-v3 is a cheaper fallback if PRM-inference latency dominates decoding.)
- **Cross-domain handling:** a small **domain embedding** added to head inputs (math/commonsense/causal/science/multi-hop) so heads share a backbone but specialize per domain — this is the literal "Cross-Domain" in XD-PRM.
- **Loss:** `Σ_h mask_h · BCE/MSE(r_h, label_h)` with per-head masking (a step missing the commonsense label simply doesn't backprop that head). Class-imbalance handled per head.

### 4.4 Validation = the G2 gate (`prm/validate.py`)
The PRM is a *reward*; a bad reward poisons GRPO and verifier-decoding. So before anything consumes it:
1. **Discrimination** — held-out GSM8K: mean `R_aggregate`(correct-prefix) > (incorrect-prefix), **ROC-AUC > 0.7**.
2. **Best-of-N lift** — reranking *N* student samples by terminal `R_aggregate` beats random/self-consistency on GSM8K.
3. **Calibration** — confidence head's predicted correctness ≈ empirical (low ECE).
4. **Head non-redundancy** — pairwise head correlation < ~0.9 (else heads are duplicates and we prune/merge — a real risk for a 5-head design).

Fail any → fix the foundry/PRM before G3. This gate is non-negotiable.

---

## 5. The other components (concrete designs)

### 5.1 Sparse CoT Compression (`distill/compress.py`)
Algorithm: segment teacher trace → score each step's *informativeness* (does removing it change derivability of the answer? embedding-novelty + LLM/heuristic) → **prune filler, merge redundant steps** under a token budget → **answer-preservation check** (compressed trace must still yield the gold answer when re-derived). Output: SFT corpus. The GRPO efficiency reward continues compression during RL. **Metric:** `tokens/answer(compressed) ÷ tokens/answer(verbose)` at matched accuracy → report against the ~40% target.

### 5.2 SFT (`sft/train_sft.py`)
QLoRA on compressed traces, **curriculum-staged** (Easy→Intermediate→Multi-hop→Adversarial; advance on val-acc/verifier-confidence threshold). Produces the baseline policy + GRPO init.

> **Current implementation:** curriculum is a deterministic *length-ordered* proxy (short/easy traces first); difficulty-bucketed stages with val-gated promotion are planned, not yet built. Results from this path must be labelled "length-ordered SFT," not "staged curriculum."

### 5.3 GRPO / RLVR (`rl/train_grpo.py`, `rl/rewards.py`)
TRL `GRPOTrainer`; group size *G*≈8; KL to SFT ref. **Composite reward:**
```
R = α·correctness(verifiable match) + β·R_aggregate(XD-PRM, process) + γ·length_penalty
    − δ·repetition_penalty            (anti-hacking)
```
- **Reward-hacking guards (critical — RL + a learned reward = Goodhart):** repetition/length penalties, KL leash, periodic *PRM-vs-ground-truth audits* during training, and freezing PRM weights during RL (never co-train reward + policy without a leash).
- **Compute:** vLLM-backed rollouts; PRM served as a fast scoring endpoint; QLoRA so policy+ref+rollouts fit one GPU.

### 5.4 Adaptive Reasoning-Depth Controller (`inference/adaptive_depth.py`) — inference-only
After each step, compute an uncertainty signal `u` = mix of (token entropy, XD-PRM confidence head, self-consistency variance). **Stop** (early-exit) when `u < τ_low` *and* answer is stable; **expand** (continue / widen verifier beam / raise budget) when `u > τ_high`; else continue normally. A hard `max_depth`/token-budget cap guarantees latency bounds. `τ` calibrated on a dev set per domain. This is the token-efficiency + latency story, with **zero extra training**.

### 5.5 Verifier-Guided Decoding (`inference/verifier_decode.py`) — inference-only
Step-level search: at each reasoning step, sample *B* candidate next-steps, score with frozen XD-PRM, keep top-*k*, prune the rest (beam/best-of-N over *steps*, not tokens). KV-cache reuse across candidates keeps it affordable. Combined with §5.4, the **unified engine** (`inference/engine.py`) = base policy + verifier-guided decode + adaptive depth + emits the full per-step score trace for the demo.

> **Current implementation:** this is *best-of-B per step* — the single highest-`R_aggregate` candidate is advanced; `keep_top_k` only selects which siblings the demo renders as surviving-vs-pruned (no true multi-beam frontier yet). KV reuse exists at *generation* (vLLM prefix caching), but PRM *scoring* re-runs the growing prefix per candidate (O(B·D²)); shared-prefix score caching is planned. Report this row as "best-of-B verifier decode," not "beam search."

---

## 6. Evaluation & ablation matrix (`eval/`)
**Benchmarks:** GSM8K, CommonsenseQA, OpenBookQA, ARC-Challenge, HotpotQA (EM/F1).
**Metrics:** accuracy/EM/F1 · **tokens/answer** · **latency** · process-quality (mean step `R_aggregate`, or judged) · **hallucination/contradiction rate** (NLI-based).
**Matrix (each row = a Hydra config + W&B run):**

| Config | Isolates |
|---|---|
| base SLM (zero-shot CoT) | floor |
| + SFT on compressed traces | distillation + compression value |
| + verifier-guided decoding | XD-PRM @ inference |
| + adaptive depth | efficiency/latency gain |
| + GRPO (**full AXIOM**) | RL value |
| **ablations:** drop each PRM head · verbose-vs-sparse traces · fixed-vs-adaptive depth · domain-embedding on/off | component & novelty attribution |

This matrix *is* the results section of both the slides and any paper.

---

## 7. Serving & explainability demo (`serve/`, `frontend/`)
Reproduces the deck's exact flow for **"Can a vegetarian survive on Mars using current technology?"**: FastAPI `POST /reason` streams steps over SSE, each carrying `{text, r_logic, r_cs, r_cons, r_eff, r_conf, R_aggregate, depth_decision, pruned_siblings}`. React + shadcn + React Flow render reasoning as a graph (nodes colored by `R_aggregate`), a 5-head bar panel per step, a depth-trace ribbon (expand/exit), and ghosted pruned branches. **This visualization is the "X — eXplainable" in AXIOM** and the strongest live differentiator.

---

## 8. Critical risks & engineered mitigations (no scope-cuts — these are fixes)
1. **Label noise in the foundry** → MC rollouts noisy at low *k*. Mitigation: tune *k*, soft labels, audit notebook, NLI/heuristic cross-checks; heads are independently maskable.
2. **Goodhart / reward hacking under GRPO** → policy games the PRM. Mitigation: KL leash, length/repetition penalties, frozen PRM, periodic PRM-vs-truth audits, keep verifiable-correctness as the dominant reward term.
3. **5 heads may be redundant** → high inter-head correlation. Mitigation: the G2 non-redundancy check; prune/merge correlated heads, keep the architecture honest.
4. **"~40%" and "5-head" are hypotheses** → reported as measured outcomes, framed as targets in the deck.
5. **Step-segmentation drift** → silent PRM corruption. Mitigation: one tested definition, used everywhere.
6. **PRM inference cost inside decoding** → could dominate latency. Mitigation: small/distilled PRM backbone, batched scoring, KV reuse, cap candidates *B*.
7. **Compute bottleneck = rollouts** → Mitigation: single shared vLLM pool, batched, reused across labeling/RL/self-consistency.

---

## 9. Open questions (answer → I freeze configs and we start at G0)
1. **Judge budget** for the Commonsense/causal head — API (DeepSeek/OpenAI/Anthropic) + rough $ cap? (decides commonsense-head strength vs. proxy fallback)
2. **Trace source** — OK to standardize on existing open distilled traces (OpenR1 / Bespoke-Stratos / OpenThoughts) instead of generating? (strongly recommended)
3. **Models** — Phi-4-mini as headline; which Qwen2.5 size as the cross-model comparison (1.5B/3B/7B)? Debugging happens on 1.5B regardless.
4. **A100 host** (Colab Pro+ / Lambda / RunPod / cluster) — affects persistent storage, vLLM setup, and how `scripts/` launch.
5. **Demo bar** — is the React Flow explainability UI a judged deliverable, or is a Gradio/CLI demo sufficient? (decides frontend budget)
```
