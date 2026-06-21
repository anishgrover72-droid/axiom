# AXIOM Technical Architecture

**Adaptive eXplainable Intelligence for Optimized Micro-Reasoning**
Samsung ennovateX AX Hackathon — Problem Statement 06

---

## 1. System Overview

AXIOM turns Small Language Models (SLMs) into efficient, self-checking, adaptive
reasoners by wrapping them in four tightly-coupled innovations:

1. **Distilled Sparse Reasoning Traces** — token-budget-aware compression of teacher
   chain-of-thought, used to build the SFT corpus.
2. **XD-PRM (Cross-Domain Process Reward Model)** — a 5-head step-level verifier that
   scores reasoning quality along five orthogonal dimensions. This is the architectural
   centrepiece: it provides the training signal for GRPO, the pruning signal for
   verifier-guided decoding, and the uncertainty estimate for the adaptive depth
   controller.
3. **Adaptive Reasoning-Depth Controller** — entropy/confidence-gated early-exit or
   step-expansion at inference time, bounded by a hard token budget.
4. **Verifier-Guided Decoding** — beam-style step-level search where the XD-PRM prunes
   low-reward branches, suppressing hallucinated or contradictory reasoning paths.

### Agentic framing

The system can be understood as three cooperating agents operating at inference time:

| Agent | Role | Implemented in |
|---|---|---|
| **GRPO-agent** (policy) | Generates candidate reasoning steps | `src/axiom/rl/train_grpo.py`, `src/axiom/inference/engine.py` |
| **XD-PRM-verifier-agent** | Scores each step across 5 heads, emits R_aggregate | `src/axiom/prm/score.py` |
| **Adaptive-controller-agent** | Reads confidence head; decides exit / continue / expand | `src/axiom/inference/adaptive_depth.py` |

The three agents run in a loop: policy generates a step → verifier scores it → controller
decides whether to accept, expand, or terminate. The loop is bounded by
`infer.max_steps` and `infer.max_new_tokens`.

---

## 2. Data Flow

```
raw datasets (GSM8K, CSQA, ARC-C, HotpotQA, OBQA)
        │
        ▼  00_download_data.py
   data/raw/<dataset>.jsonl          (axiom.data.loaders.prefetch)
        │
        ▼  01_build_traces.py
   data/traces/<dataset>.jsonl       (axiom.distill.generate: open trace sources or teacher)
        │
        ▼  02_compress.py
   data/compressed/<dataset>.jsonl   (axiom.distill.compress: novelty-threshold prune + budget cap)
        │
        ▼  03_sft.py
   experiments/sft/                  (axiom.sft.train_sft: QLoRA fine-tune on compressed traces)
        │
        ▼  04_prm_label.py ─────── THE LABEL FOUNDRY ──────────────────────────────────┐
   data/prm/<dataset>/               logic (MC-rollout), consistency (NLI),            │
   data/release/                     efficiency (embedding novelty), commonsense        │
        │                            (teacher judge, budget-capped), confidence         │
        │                            (rollout variance) — merged + released             │
        ▼  05_prm_train.py
   experiments/xdprm/                (axiom.prm.train_prm: multi-task on 5 heads)
        │  [G2 gate validates: AUC > 0.7, best-of-N lift, ECE < 0.15, head correlation < 0.9]
        ▼  06_grpo.py
   experiments/grpo/                 (axiom.rl.train_grpo: TRL GRPOTrainer + composite reward)
        │
        ▼  07_eval.py               (axiom.eval.run_eval: accuracy + tokens/answer vs. baseline)
        ▼  08_serve.py              (axiom.serve.app: FastAPI + SSE streaming)
```

---

## 3. XD-PRM: Architecture

**Backbone:** any causal HF model (default: `Qwen/Qwen2.5-0.5B-Instruct` for the
verifier; the student is `Qwen/Qwen2.5-1.5B-Instruct`). Loaded in 4-bit (QLoRA) on
GPU, float32 on CPU.

**Step pooling:** a `<|step|>` sentinel token is appended to each reasoning step. The
PRM reads the hidden state at each sentinel position (`hidden_pool: last_sentinel`) and
feeds it to the five reward heads. This lets the backbone share computation across the
full trace while providing per-step representations.

**Domain conditioning:** a learned `nn.Embedding(4, hidden_size)` over four domains
(math / commonsense / science / multihop) is added to each step's hidden state before
the heads. This is the "XD" (cross-domain) in XD-PRM.

**The five heads** (all small MLPs: `linear → GELU → dropout → linear → scalar logit`):

| Head | What it measures | Label source | Loss |
|---|---|---|---|
| **Logic** | P(prefix → correct final answer) | Math-Shepherd MC rollouts (k=8 via vLLM) | MSE on sigmoid |
| **Commonsense** | Real-world plausibility + causal coherence | Teacher-LLM judge (budget-capped at 2000 examples) | MSE |
| **Consistency** | 1 − P(contradiction with prior steps) | Off-the-shelf NLI model (cross-encoder/nli-deberta-v3-base) | BCE |
| **Efficiency** | Compactness / low redundancy | Embedding cosine novelty (all-MiniLM-L6-v2) | MSE |
| **Confidence** | Calibrated uncertainty | Variance across MC-rollout answer pool | MSE |

`R_aggregate(step) = Σ_h w_h · σ(logit_h)` with `w_logic = w_commonsense = w_consistency = 1.0`,
`w_efficiency = w_confidence = 0.5`.

---

## 4. Label Foundry Details

The foundry (`src/axiom/prm/labeling/`) manufactures per-step labels for all 5 heads at
near-zero annotation cost:

- **Logic (`mc_rollout.py`):** For each prefix ending at step k, sample k=8 continuations
  with the student model (via vLLM). Label = fraction of continuations that reach the
  correct final answer (Math-Shepherd soft value).
- **Consistency (`nli_consistency.py`):** Run cross-encoder NLI on (prior steps, step k).
  Consistency label = 1 − P(contradiction). Commonsense fallback = P(entailment) when
  no teacher judge is configured.
- **Efficiency (`efficiency.py`):** Cosine similarity between step k's embedding and
  prior-step embeddings. Label = 1 − max_similarity (higher = more novel = more efficient).
- **Commonsense (`judge.py`):** Teacher LLM (configurable; DeepSeek by default) scores
  plausibility and causal correctness. Capped at 2000 examples to stay within compute budget.
- **Confidence (`confidence.py`):** Reuses the MC-rollout answer pool. Label = modal
  answer agreement fraction. Steps with no parseable answers are masked (None → loss mask).

All label streams are merged and written to `data/release/` by
`aggregate_release.py` — the publishable dataset contribution.

---

## 5. GRPO Training

**Framework:** TRL `GRPOTrainer` (≥ 0.16). The XD-PRM is frozen; only the student
policy receives gradient updates.

**Composite reward:**
```
R = 1.0 × correctness  +  0.5 × R_aggregate(XD-PRM)
  − 0.1 × length_ratio  −  0.2 × repetition_penalty
```

- `correctness`: 1 if the final answer matches gold (via `axiom.common.answers.grade`), 0 otherwise.
- `R_aggregate`: mean XD-PRM score across all steps (process quality signal).
- `length_ratio`: `token_count / max_new_tokens` — penalises unnecessary verbosity.
- `repetition_penalty`: `1 − distinct-2 bigram ratio` — penalises looping.

**Anti-hacking:** KL coefficient `kl_coef = 0.04` against the SFT reference policy.
Group size G=8 samples per prompt; advantages standardised within the group by TRL.

---

## 6. Adaptive Depth Controller

At each decode step, `adaptive_depth.decide()` reads the XD-PRM confidence head (or
falls back to R_aggregate) and applies two thresholds:

| Condition | Decision |
|---|---|
| uncertainty < τ_low **and** a final answer is present | `EXIT` (early-exit) |
| uncertainty > τ_high | `EXPAND` (re-sample the step with a higher temperature) |
| otherwise | `CONTINUE` |

Self-consistency uncertainty (majority-vote disagreement across k=5 full continuations)
is available as a richer but costlier alternative for boundary cases.

---

## 7. Verifier-Guided Decoding

At each step, `propose_next()` samples B=4 candidate next-steps from the policy,
scores each with the frozen XD-PRM, and advances the highest-scoring candidate.
Low-scoring branches are pruned and logged to the SSE stream for the explainability
frontend. When no scorer is provided (e.g. T4 runs without a trained PRM), the decoder
falls back to single-sample greedy.

---

## 8. Inference Engine (`src/axiom/inference/engine.py`)

The unified engine composes:
1. **Policy** — the GRPO-tuned SFT model (or base SFT model for ablation).
2. **Verifier-guided decode** — branch-factor B, keep-top-k K.
3. **Adaptive depth controller** — tau_low / tau_high / max_steps.
4. **XD-PRM scorer** — loaded once, shared by decode and controller.

The FastAPI service (`src/axiom/serve/app.py`) exposes SSE events for each step,
carrying the step text, per-head scores, and the depth controller's decision. This
drives the React + React Flow explainability frontend.

---

## 9. Datasets

| Dataset | HF path | Domain | Answer type | Used for |
|---|---|---|---|---|
| GSM8K | `openai/gsm8k` | math | numeric | Primary train + eval |
| CommonsenseQA | `tau/commonsense_qa` | commonsense | mcq | Train + commonsense head labels |
| ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | science | mcq | Adversarial eval + train |
| HotpotQA | `hotpotqa/hotpot_qa` (fullwiki) | multihop | span | Multi-hop train + eval |
| OpenBookQA | `allenai/openbookqa` (main) | science | mcq | Science eval + train |

---

## 10. Models

| Role | Model | HF ID |
|---|---|---|
| Student (headline) | Phi-4-mini | `microsoft/Phi-4-mini-instruct` |
| Student (fast-iter / T4 run) | Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| PRM backbone | Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` |
| Teacher judge | DeepSeek-R1 (API) | configurable via `configs/teacher/` |

---

## 11. Honest Status: What Worked, What Didn't

### What worked
- **Label foundry without API cost:** The NLI consistency head and embedding efficiency
  head produce clean labels at zero cost. The confidence head reuses MC-rollout samples
  for free. Together these three heads are fully supervised with no paid annotation.
- **Modular step contract:** `axiom.common.steps.segment()` is idempotent and shared
  by PRM training, scoring, and inference. Zero drift between training-time and
  inference-time step boundaries — a common failure mode in PRM systems.
- **Test suite:** 30/30 fast tests pass covering the core contracts (step segmentation,
  answer matching, schema round-trip, foundry labels, reward metrics, difficulty buckets).
- **Config-driven ablations:** Any head can be disabled or re-weighted with a single
  CLI override (`prm.heads.efficiency.enabled=false`), enabling clean ablation studies.

### What was scaled back due to T4 / time constraints
- **MC-rollout labeling (Logic head) on T4:** vLLM is not reliably available on Kaggle
  T4 GPU instances (CUDA 12.1, 16 GB VRAM). The Kaggle run uses a rule-based fallback
  for the Logic head: correctness of a single greedy sample instead of k=8 rollouts.
  This reduces label quality for the logic head but does not affect the other four heads.
- **No vLLM on T4:** `common/vllm_pool.py` includes a CPU fallback `HFEngine`
  (`AutoModelForCausalLM` + manual generate loop) that activates when vLLM is unavailable. The verifier-guided
  decoder degrades to single-sample greedy in this configuration.
- **Phi-4-mini not used on Kaggle:** Phi-4-mini (3.8B) does not fit alongside the
  5-head PRM in 16 GB with QLoRA overhead. The T4 Kaggle run uses Qwen2.5-1.5B-Instruct.
- **Teacher judge for commonsense head:** The DeepSeek judge requires an API key not
  available during the Kaggle run. The commonsense head falls back to NLI entailment
  scores (free, already computed for consistency).
- **Full GRPO run:** The complete GRPO training (500 steps, G=8, batch=8) takes ~4-6 h
  on a single A100. The Kaggle T4 run uses `max_steps=100-200` and `group_size=4` to
  produce a demonstrable result within the 12-hour session limit.

---

## 12. Reproducibility

All hyperparameters live in `configs/`. A full headline run is reproduced with:

```bash
python -m scripts.00_download_data data=gsm8k
python -m scripts.01_build_traces  data=gsm8k distill.source.mode=open
python -m scripts.02_compress
python -m scripts.03_sft
python -m scripts.04_prm_label
python -m scripts.05_prm_train     # gate validates before returning
python -m scripts.06_grpo          model=phi4_mini
python -m scripts.07_eval
```

Override `model=qwen2_5_1_5b` for the T4-compatible fast-iteration run.
Seeds are global (`seed: 0` in `config.yaml`) and propagated via `axiom.common.seed`.
