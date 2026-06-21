# AXIOM – Adaptive eXplainable Intelligence for Optimized Micro-Reasoning

- **Problem Statement Number** - 06
- **Problem Statement Title** - Enhancing Reasoning Capabilities in Small Language Models (SLMs) using Reinforcement Learning
- **Team name** - AXIOM
- **Team members (Names)** - Prabinder Singh, Anish Grover
- **Institute/College Name** - Thapar Institute of Engineering & Technology, Patiala
- **Final Presentation Google Drive Link** - [TO BE ADDED]
- **Full Submission Demo Video Link** - [TO BE ADDED]
- **Setup & Result Reproducibility Video Link** - [TO BE ADDED]

---

## What we built

AXIOM wraps a Small Language Model in four tightly-coupled innovations:

1. **Distilled Sparse Reasoning Traces** — teacher chain-of-thought compressed to a
   token budget while preserving answer derivability. Builds the SFT corpus.
2. **XD-PRM (Cross-Domain Process Reward Model)** — a backbone + 5 reward heads that
   score each reasoning step along Logic, Commonsense, Consistency, Efficiency, and
   Confidence dimensions. The hub: four downstream components consume its output.
3. **Adaptive Reasoning-Depth Controller** — reads the XD-PRM confidence head after
   each step; exits early when certain, expands when uncertain, bounded by a token cap.
4. **Verifier-Guided Decoding** — samples B candidate next-steps per position, prunes
   the lowest-scoring branches with the frozen XD-PRM, advances the best.

### Project Artefacts

- **Technical Documentation** - See [/docs/ax.md](docs/ax.md) for full architecture,
  label foundry design, GRPO reward decomposition, and honest status.
- **Source Code** - See [/src/axiom/](src/axiom/) — all logic lives here.
  `scripts/` are thin numbered CLIs (`00_download_data` → `08_serve`).
- **Models Used**
  - Student (headline run): `microsoft/Phi-4-mini-instruct` — https://huggingface.co/microsoft/Phi-4-mini-instruct
  - Student (T4 / fast-iter): `Qwen/Qwen2.5-1.5B-Instruct` — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
  - Student (ablation small): `Qwen/Qwen2.5-0.5B-Instruct` — https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
  - Student (7B upper bound): `Qwen/Qwen2.5-7B-Instruct` — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
  - PRM backbone: `Qwen/Qwen2.5-0.5B-Instruct` (same family as student — shared tokenizer)
  - Teacher judge: DeepSeek-chat via OpenAI-compatible API (commonsense head labeling only)
- **Models Published** - [TO BE ADDED after Kaggle run + HF upload]
- **Datasets Used**
  - GSM8K — `openai/gsm8k` (math, numeric)
  - CommonsenseQA — `tau/commonsense_qa` (commonsense, MCQ)
  - ARC-Challenge — `allenai/ai2_arc` / ARC-Challenge (science, MCQ)
  - HotpotQA — `hotpotqa/hotpot_qa` / fullwiki (multihop, span)
  - OpenBookQA — `allenai/openbookqa` / main (science, MCQ)
- **Datasets Published** - [TO BE ADDED — synthetic distilled reasoning traces + per-step
  XD-PRM labels will be uploaded to HuggingFace after the Kaggle training run]

---

## Quickstart

```bash
# CPU dev (contracts + fast tests only — no GPU required)
pip install pydantic hydra-core omegaconf datasets sentence-transformers numpy pytest
pip install -e .
pytest -v .                      # 30 tests, all fast, no model download

# Full GPU stack (A100/H100, install vLLM first — it pins a compatible torch)
pip install vllm
pip install -r requirements.txt
pip install -e .

# Pipeline (each script is a thin Hydra CLI over src/axiom/)
python -m scripts.00_download_data
python -m scripts.01_build_traces
python -m scripts.02_compress
python -m scripts.03_sft
python -m scripts.04_prm_label
python -m scripts.05_prm_train       # G2 gate: blocks downstream if AUC < 0.7
python -m scripts.06_grpo  model=phi4_mini
python -m scripts.07_eval
python -m scripts.08_serve           # FastAPI + SSE on :8000

# Frontend explainability demo
cd frontend && npm install && npm run dev
```

Config overrides (Hydra):
```bash
python -m scripts.06_grpo model=qwen2_5_1_5b grpo.train.steps=100   # T4-compatible
python -m scripts.05_prm_train prm.heads.efficiency.enabled=false    # head ablation
```

---

## Pipeline

```
raw data  →  traces  →  compress  →  SFT  →  prm-label  →  prm-train  →  GRPO  →  eval  →  serve
                                                  ↑ label foundry (5 heads, zero API cost for 4/5)
```

The XD-PRM is the hub: trained on foundry labels, then frozen and consumed by GRPO
(process reward), verifier-guided decode (step pruning), and the adaptive depth
controller (confidence-gated exit).

---

## Repository layout

```
axiom/
├── configs/          Hydra config groups (model/, data/, prm/, grpo/, sft/, …)
├── src/axiom/
│   ├── common/       shared contracts: steps, answers, tokens, vllm_pool, io, seed
│   ├── data/         schemas (pydantic), loaders (HF→Example), difficulty buckets
│   ├── distill/      trace sources, teacher generation, sparse CoT compression
│   ├── sft/          QLoRA SFT trainer
│   ├── prm/          XD-PRM model, heads, dataset, train, score, validate, labeling/
│   ├── rl/           composite reward, GRPO (TRL GRPOTrainer)
│   ├── inference/    adaptive_depth, verifier_decode, unified engine
│   ├── eval/         benchmarks, metrics, eval harness
│   └── serve/        FastAPI + SSE stream
├── scripts/          thin numbered CLIs (00–08), ≤30 lines each
├── frontend/         React + Vite + shadcn/ui + React Flow (explainability demo)
├── tests/            30 fast unit tests (pytest), no GPU required
└── docs/             ax.md (full architecture doc)
```

---

## Attribution

Base framework: original AXIOM design by Prabinder Singh and Anish Grover.
GRPO implementation uses Hugging Face TRL (`GRPOTrainer`). XD-PRM label foundry
builds on Math-Shepherd (MC-rollout logic labels), SQuAD-style span-F1, and
cross-encoder NLI for consistency detection. No proprietary API dependencies in
any runtime path; the teacher judge (DeepSeek) is optional and only used for
the commonsense head during labeling.
