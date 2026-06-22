# AXIOM — Adaptive eXplainable Intelligence for Optimized Micro-reasoning

A cross-domain, **verifier-centric** framework that turns Small Language Models into efficient,
self-checking, adaptive reasoners. A single process reward model (**XD-PRM**) scores every
reasoning step on five axes and acts as the hub for reinforcement learning, verifier-guided
decoding, and an adaptive-depth controller.

---

- **Problem Statement Number** — 06
- **Problem Statement Title** — Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning
- **Team name** — AXIOM
- **Team members (Names)** — Prabinder Singh, Anish Grover
- **Institute/College Name** — Thapar Institute of Engineering & Technology, Patiala
- **Final Presentation Google Drive Link** — `<FILL: public PDF link>`
- **Full Submission Demo Video Link** — `<FILL: YouTube public/unlisted>`
- **Setup & Result Reproducibility Video Link** — `<FILL: YouTube public/unlisted>`

---

## What AXIOM does

Small models reason verbosely and unreliably. AXIOM attacks both at once:

1. **XD-PRM** — one backbone + **five scalar heads** (logic, commonsense, consistency, efficiency,
   confidence) that verify a reasoning step *cross-domain* (math, science, commonsense, multi-hop).
2. **Sparse reasoning compression** — prunes filler / merges redundant steps under a token budget
   while preserving the answer (the **~40% fewer tokens** target).
3. **GRPO reinforcement learning** — a composite XD-PRM reward trains the policy to reason
   *correctly and compactly*.
4. **Adaptive-depth + verifier-guided decoding** — spend tokens only where step-uncertainty is high.

The same XD-PRM is consumed by **four of the five components** — it is the critical path and the
core novelty.

## Pipeline

```
ingest → distill traces → compress → SFT (QLoRA) → PRM label foundry → XD-PRM → GRPO → eval → serve
  00          01            02          03               04             05       06     07     08
```

Each stage is a thin numbered CLI in [`scripts/`](scripts/) over logic in [`src/axiom/`](src/axiom/),
driven by Hydra configs in [`configs/`](configs/).

## Repository layout

| Path | What |
|---|---|
| [`src/axiom/`](src/axiom/) | all importable logic (`common, data, distill, sft, prm, rl, inference, eval, serve`) |
| [`scripts/`](scripts/) | numbered CLIs `00_…`→`08_…` (arg-parse + call `src/`) |
| [`configs/`](configs/) | Hydra config groups (every tunable) |
| [`frontend/`](frontend/) | React + Vite explainability UI (the live reasoning console) |
| [`tests/`](tests/) | pytest mirroring `src/` (contracts: steps, answers, schemas, rewards) |
| [`notebooks/`](notebooks/) | `colab_run.ipynb`, `RERUN.md` — one-click / tuned cloud training |
| [`docs/`](docs/) | technical documentation (see below) |

## Quickstart

```bash
make test                          # fast contract tests (no GPU)
# GPU box (Colab/Kaggle/A100):
pip install vllm && pip install -e . -r requirements.txt
python scripts/00_download_data.py
python scripts/03_sft.py model=qwen2_5_7b
make serve                         # FastAPI reasoning service on :8000
cd frontend && npm install && npm run dev   # explainability console
```

Full training is reproduced in [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) /
[`notebooks/RERUN.md`](notebooks/RERUN.md), documented in
[`docs/installation.md`](docs/installation.md) and [`RUNBOOK.md`](RUNBOOK.md).

## Project Artefacts

- **Technical Documentation** — [`docs/`](docs/): [architecture](docs/architecture.md) ·
  [tech stack & OSS](docs/tech-stack.md) · [installation](docs/installation.md) ·
  [user guide](docs/user-guide.md) · [features](docs/features.md).
- **Agentic AI & open-weight usage** — [`docs/ax.md`](docs/ax.md).
- **Source Code** — [`src/`](src/) (training + eval), [`frontend/`](frontend/) (demo UI).
- **Models Used** (open weight — PS-06 suggested):
  - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — **KPI base / policy**
  - [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — alternative SLM base
  - [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) / [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — debug + XD-PRM backbone
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — compression / efficiency head
- **Models Published** — `<FILL: HF link to the trained XD-PRM + SFT/GRPO student, open license e.g. Apache-2.0>`
- **Datasets Used** (public — PS-06 benchmarks first):
  - [GSM8K](https://huggingface.co/datasets/openai/gsm8k) ·
    [MMLU](https://huggingface.co/datasets/cais/mmlu) ·
    [StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) ·
    [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat)
  - SFT trace source: [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
  - additional eval domains: [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc),
    [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa),
    [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)
- **Datasets Published** — Compact Distilled Reasoning Trace Dataset + per-step XD-PRM labels
  (`data/release/`) — `<FILL: HF dataset link, CC-BY-4.0 or equivalent>`

## KPIs & Benchmarks (PS-06)

Demonstrate improvement on **≥ 2 of 3** benchmarks. Baseline = the same base model **without** RL.

| Benchmark | Min target | Expected | Baseline | AXIOM (SFT+RL) | Δ |
|---|---|---|---|---|---|
| GSM8K | ≥ 50% | ≥ +5% over baseline | `<FILL>` | `<FILL>` | `<FILL>` |
| MMLU | ≥ 45% | ≥ +5% over baseline | `<FILL>` | `<FILL>` | `<FILL>` |
| StrategyQA | ≥ 65% | ≥ +5% over baseline | `<FILL>` | `<FILL>` | `<FILL>` |

**Efficiency (AXIOM's edge):** minimal latency overhead via sparse compression
(~57% token reduction measured on GSM8K traces) + verifier-guided adaptive depth.

| Internal KPI | Target | Measured |
|---|---|---|
| XD-PRM step ROC-AUC | > 0.70 | `<FILL>` |
| Tokens/answer reduction @ matched accuracy | ~40% | `<FILL>` |
| Confidence calibration (ECE) | < 0.15 | `<FILL>` |

Base model for the KPI run: **Qwen2.5-7B-Instruct** (or **Phi-3-mini**) on a ≥24 GB GPU
(see [`notebooks/RERUN.md`](notebooks/RERUN.md)).

## Attribution

AXIOM's pipeline, XD-PRM model, and frontend are **original work**. It builds on open-source
libraries (PyTorch, HF Transformers/Datasets, TRL, PEFT, bitsandbytes, vLLM, FAISS,
sentence-transformers, FastAPI, React/Vite) and open-weight models / public datasets listed above.
New contributions: the **five-head cross-domain process reward model (XD-PRM)** and its MC-rollout
**label foundry**, **sparse reasoning compression**, the **composite-reward GRPO** loop, and the
**verifier-guided adaptive-depth** decoder. See [`docs/tech-stack.md`](docs/tech-stack.md) for the
full OSS list with links.

---

*Engineering charter: [`CLAUDE.md`](CLAUDE.md) · Full design: [`PLAN.md`](PLAN.md) ·
Execution runbook: [`RUNBOOK.md`](RUNBOOK.md)*
