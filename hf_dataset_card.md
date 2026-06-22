---
language:
- en
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
tags:
- reasoning
- process-reward-model
- reinforcement-learning
- small-language-models
- chain-of-thought
- step-level-labels
- grpo
- gsm8k
- math-reasoning
- distillation
pretty_name: AXIOM Compact Distilled Reasoning Traces
size_categories:
- n<1K
---

# AXIOM Compact Distilled Reasoning Traces

A dataset of compressed, step-segmented math reasoning traces with per-step process reward labels
across five quality axes (logic, commonsense, consistency, efficiency, confidence). Created as part
of the **AXIOM** framework for training cross-domain Process Reward Models (XD-PRM) and
GRPO-tuned Small Language Models.

## Repositories & Resources

| Resource | Link |
|---|---|
| GitHub (AXIOM) | https://github.com/anishgrover72-droid/axiom |
| Trained Model | https://huggingface.co/prabindersinghh/axiom-qwen2.5-1.5b-reasoning |
| Dataset (this) | https://huggingface.co/datasets/prabindersinghh/axiom-reasoning-traces |
| Demo Video | https://youtu.be/xdE6rI9mULU?si=v4dzUQCTxAbbht6g |

---

## Dataset Overview

AXIOM Compact Distilled Reasoning Traces contains **38 SFT-ready traces** and **46 compressed traces**
derived from GSM8K-domain math problems. Each record is a reasoning trace that has been:

1. Sourced from `open-r1/OpenR1-Math-220k` (DeepSeek-R1 distilled, Apache 2.0)
2. Segmented into numbered reasoning steps using the shared AXIOM step contract
3. Compressed via novelty-threshold pruning + token-budget capping (59.5% token reduction)
4. Labeled per-step on five reward axes by the AXIOM label foundry

The dataset is designed for:
- **QLoRA SFT** of small language models on compact, verified reasoning traces
- **XD-PRM training** using per-step step-level labels
- **Research** into reasoning compression and process reward modeling

---

## Motivation

Large reasoning models (e.g., DeepSeek-R1) generate verbose traces with redundant steps. Training
smaller models directly on these traces transfers both the reasoning content *and* the verbosity.
AXIOM Compact Distilled Reasoning Traces addresses this by:

- **Compressing** traces to ~40% fewer tokens while preserving the correct final answer
- **Labeling** every remaining step on five quality axes so a Process Reward Model can distinguish
  useful steps from filler
- Providing a **reproducible pipeline** so anyone can regenerate the dataset at any scale

The result is a compact, quality-filtered dataset suited for training 1–4B parameter models on
constrained compute (single T4 GPU).

---

## Dataset Creation Pipeline

```
OpenR1-Math-220k (300 rows)
       │
       ▼  [max_reasoning_chars=6000 filter]
       │  → filters out 254 competition-level traces (avg 20k chars)
       ▼
  46 traces segmented into steps via axiom.common.steps.segment()
       │
       ▼  [Sparse compression]
       │  novelty_threshold=0.92 cosine pruning
       │  target_ratio=0.6 token-budget cap
       │  answer-preservation check (grade())
       ▼
  46 compressed traces (59.5% token reduction)
       │
       ▼  [SFT filter: max_seq_tokens=2048]
       │  → drops 8 traces still over the token cap post-compression
       ▼
  38 SFT-ready traces
       │
       ▼  [PRM label foundry]
       │  Logic: MC rollouts (k=1 rule-based fallback)
       │  Consistency: cross-encoder NLI
       │  Commonsense: NLI entailment proxy
       │  Efficiency: cosine novelty score
       │  Confidence: rollout answer variance
       ▼
  38 PRM-labeled traces → per-step StepLabel records
```

---

## Source Dataset

**OpenR1-Math-220k** (`open-r1/OpenR1-Math-220k`, Apache 2.0)

DeepSeek-R1 distilled reasoning traces over mathematics problems. AXIOM uses a 300-row sample
filtered to `max_reasoning_chars=6000` to exclude competition-level traces incompatible with
GSM8K-style numeric SFT.

| Property | Value |
|---|---|
| Source rows scanned | 300 |
| Character limit applied | 6,000 chars/trace |
| Traces retained after char filter | 46 |
| License | Apache 2.0 |

---

## Trace Generation

Raw reasoning text from OpenR1-Math-220k is segmented into numbered steps by
`axiom.common.steps.segment()` — the single canonical step-boundary function used at
SFT, PRM labeling, PRM scoring, and decoding time. This prevents boundary drift across pipeline
stages.

The step sentinel `<|step|>` is appended to each step; the PRM reads the hidden state at each
sentinel position during training and inference.

Average trace properties (pre-compression):
- Steps per trace: ~5–8 (after `max_reasoning_chars=6000` filter)
- Median token count: ~450 tokens per trace
- Question domain: arithmetic word problems (GSM8K-type)

---

## Trace Compression

`scripts/02_compress.py` applies two passes:

**Pass 1 — Novelty-threshold pruning**
Each step is embedded with `all-MiniLM-L6-v2` (Apache 2.0). Any step whose cosine similarity
with a previously kept step exceeds `novelty_threshold=0.92` is dropped as redundant.

**Pass 2 — Token-budget cap**
Iteratively drops the most-redundant remaining step until the trace fits `target_ratio=0.6`
of its original token count.

**Answer-preservation check**
After compression, `axiom.common.answers.grade()` validates that the compressed trace still
reaches the correct final answer. If the check fails, the original uncompressed trace is
kept (no silent correctness loss).

**Measured result:** 59.5% token reduction across 46 GSM8K traces (target was ~40%; higher
reduction is attributable to the verbosity of competition-level OpenR1-Math traces).

---

## SFT Dataset Generation

Compressed traces are formatted using the Qwen2.5 chat template:

```
<|im_start|>system
You are a helpful math reasoning assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
Step 1: {step_1_text}<|step|>
Step 2: {step_2_text}<|step|>
...
The answer is {answer}.<|im_end|>
```

A `max_seq_tokens=2048` filter drops any formatted trace that still exceeds the token cap
(8 of 46 traces dropped). The 38 retained traces are curriculum-ordered (shortest first as a
length proxy for difficulty).

---

## PRM Label Generation

`scripts/04_prm_label.py` generates per-step labels for all five XD-PRM heads:

| Head | Signal source | Description |
|---|---|---|
| **logic** | MC rollouts (k=1 rule-based) | Binary correct/incorrect based on greedy rollout from this step |
| **consistency** | Cross-encoder NLI (cross-encoder/nli-deberta-v3-base) | Entailment probability between step k and prior steps |
| **commonsense** | NLI entailment proxy | Factual plausibility; falls back to unlabeled if no judge available |
| **efficiency** | Cosine novelty (all-MiniLM-L6-v2) | 1 − max cosine sim to prior steps; high = novel/informative |
| **confidence** | Rollout answer variance | Agreement of k continuations; single rollout gives binary signal |

Label values are floats in [0, 1]. Unlabeled steps (commonsense fallback) carry `label=None` and
are masked during XD-PRM training.

---

## Dataset Structure

```
axiom-reasoning-traces/
├── README.md                     (this file)
├── data/
│   ├── traces.jsonl              compressed traces (46 records)
│   ├── sft_ready.jsonl           SFT-formatted traces (38 records)
│   └── prm_labels.jsonl          per-step PRM labels (38 traces × ~5 steps)
```

---

## File Formats

### `traces.jsonl` — one JSON object per line

```json
{
  "id": "gsm8k_train_042",
  "question": "Janet's ducks lay 16 eggs per day...",
  "answer": "9",
  "answer_type": "numeric",
  "steps": [
    {"idx": 0, "text": "Janet's ducks lay 16 eggs per day."},
    {"idx": 1, "text": "She eats 3 for breakfast and uses 4933828 to bake muffins, leaving 16 − 3 − 4933828 = ... "}
  ],
  "original_token_count": 312,
  "compressed_token_count": 127,
  "compression_ratio": 0.407,
  "source": "open-r1/OpenR1-Math-220k"
}
```

### `sft_ready.jsonl` — chat-formatted records

```json
{
  "id": "gsm8k_train_042",
  "text": "<|im_start|>system\nYou are a helpful math reasoning assistant.<|im_end|>\n<|im_start|>user\nJanet's ducks lay 16 eggs per day...<|im_end|>\n<|im_start|>assistant\nStep 1: ...<|step|>\nThe answer is 9.<|im_end|>",
  "token_count": 187
}
```

### `prm_labels.jsonl` — per-step label records

```json
{
  "trace_id": "gsm8k_train_042",
  "step_idx": 0,
  "step_text": "Janet's ducks lay 16 eggs per day.",
  "labels": {
    "logic": 0.83,
    "consistency": 0.91,
    "commonsense": null,
    "efficiency": 0.74,
    "confidence": 1.0
  },
  "domain": "math"
}
```

---

## Example Record

**Question:** A store sells apples for $0.80 each and oranges for $1.20 each. If Maria buys 5 apples
and 3 oranges, how much does she pay in total?

**Compressed trace (3 steps, down from 6 original):**

```
Step 1: Cost of 5 apples = 5 × $0.80 = $4.00.<|step|>
Step 2: Cost of 3 oranges = 3 × $1.20 = $3.60.<|step|>
Step 3: Total = $4.00 + $3.60 = $7.60.<|step|>
The answer is 7.6.
```

**PRM labels:**

| Step | Logic | Consistency | Commonsense | Efficiency | Confidence |
|---|---|---|---|---|---|
| 1 | 0.89 | 0.95 | null | 0.82 | 1.0 |
| 2 | 0.91 | 0.93 | null | 0.78 | 1.0 |
| 3 | 0.95 | 0.97 | null | 0.65 | 1.0 |

*(commonsense=null indicates the NLI fallback was active; step is masked in PRM loss)*

---

## Statistics

| Metric | Value |
|---|---|
| Source rows scanned | 300 |
| Traces after `max_reasoning_chars=6000` | 46 |
| Traces in `sft_ready.jsonl` (after `max_seq_tokens=2048`) | 38 |
| Traces dropped at SFT filter | 8 |
| Mean steps per compressed trace | ~4.8 |
| Mean compressed token count | ~187 tokens |
| Mean original token count | ~462 tokens |
| Token reduction (measured) | **59.5%** |
| Fraction of steps with commonsense label | ~0% (NLI fallback active; no judge) |
| Fraction of steps with logic label (rollouts_k=1) | ~100% (binary) |
| Question domain | Arithmetic word problems (GSM8K-type) |
| Language | English |

---

## Usage Instructions

### Load the SFT dataset

```python
from datasets import load_dataset

ds = load_dataset("prabindersinghh/axiom-reasoning-traces", data_files="data/sft_ready.jsonl", split="train")
print(ds[0]["text"])
```

### Fine-tune with TRL SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        max_seq_length=2048,
        dataset_text_field="text",
        output_dir="./axiom-sft-output",
    ),
)
trainer.train()
```

### Load PRM labels

```python
import json

labels = []
with open("prm_labels.jsonl") as f:
    for line in f:
        labels.append(json.loads(line))

# Access per-step logic scores
for record in labels:
    print(record["trace_id"], record["step_idx"], record["labels"]["logic"])
```

### Reproduce the full dataset from source

```bash
git clone https://github.com/anishgrover72-droid/axiom.git axiom && cd axiom
pip install -e .
python scripts/00_download_data.py data=gsm8k
python scripts/01_build_traces.py  data=gsm8k distill.source.limit=300 distill.source.max_reasoning_chars=6000
python scripts/02_compress.py      data=gsm8k
python scripts/03_sft.py           model=qwen2_5_1_5b sft.train.max_seq_tokens=2048
python scripts/04_prm_label.py     model=qwen2_5_1_5b
```

See [notebooks/kaggle_run.ipynb](https://github.com/anishgrover72-droid/axiom/blob/main/notebooks/kaggle_run.ipynb) for a runnable end-to-end notebook (Kaggle T4).

---

## Limitations

- **Small corpus.** Only 38 SFT-ready / 46 compressed traces — generated from 300 rows of
  OpenR1-Math-220k under a 6,000-character filter to exclude incompatible competition traces.
  A production dataset would use a full corpus or a domain-matched source.

- **Logic labels are binary (rollouts_k=1).** The Math-Shepherd method ideally uses k=8 MC
  rollouts to estimate step-level correctness probabilities. With k=1 (greedy rollout), the
  signal is binary (correct/incorrect) rather than a calibrated probability, reducing PRM
  training quality.

- **Commonsense labels absent.** The NLI-based commonsense head fell back to unlabeled on this
  run. Steps are masked in the PRM loss for this head.

- **Single domain.** Only arithmetic word problems (GSM8K-type) are covered. The AXIOM
  architecture supports cross-domain training (math, science, commonsense, multi-hop) but this
  dataset release covers math only.

- **No train/val/test split.** The corpus is too small for meaningful held-out splits.
  Evaluation is done on the original GSM8K test set (1,319 examples) rather than held-out traces.

---

## Citation

If you use this dataset, please cite the AXIOM submission:

```bibtex
@misc{axiom2026,
  title     = {AXIOM: Adaptive eXplainable Intelligence for Optimized Micro-Reasoning},
  author    = {Prabinder Singh and Anish Grover},
  year      = {2026},
  note      = {Samsung ennovateX AX Hackathon 2026, Problem Statement 06},
  url       = {https://github.com/anishgrover72-droid/axiom}
}
```

---

## Related Repositories

| Resource | URL |
|---|---|
| AXIOM GitHub (code + configs) | https://github.com/anishgrover72-droid/axiom |
| AXIOM Trained Model (HF) | https://huggingface.co/prabindersinghh/axiom-qwen2.5-1.5b-reasoning |
| AXIOM Dataset (this) | https://huggingface.co/datasets/prabindersinghh/axiom-reasoning-traces |
| Demo Video | https://youtu.be/xdE6rI9mULU?si=v4dzUQCTxAbbht6g |
| Source Traces (OpenR1-Math-220k) | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k |
| Base Model (Qwen2.5-1.5B-Instruct) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

---

*AXIOM — Samsung ennovateX AX Hackathon 2026 · Problem Statement 06 ·
Thapar Institute of Engineering & Technology, Patiala*
