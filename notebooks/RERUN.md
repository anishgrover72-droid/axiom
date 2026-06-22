# Rerun cells — finish the PS-06 pipeline (2×T4 / A100)

Paste these into the Kaggle/Colab notebook **after** the clone+install cells. They run the stages
that never completed: **XD-PRM → GRPO (RL) → eval**, plus the SFT/foundry fixes.

> First: `pip install vllm` on the box (needed for real MC rollouts), and
> `git -C <repo> pull origin main` so MMLU/StrategyQA + the eval suite are present.

## Cell A — pick GPU preset + build the override string

```python
import os
GPU = "A100"   # "A100" (single 40/80GB)  or  "T4x2" (Kaggle 2×T4)

if GPU == "A100":
    MODEL  = "qwen2_5_7b"          # or "phi3_mini"
    PRESET = (
        "sft.train.batch_size=4 sft.train.grad_accum=8 "
        "sft.train.max_seq_tokens=4096 model.max_seq_len=4096 "
        "prm.labeling.logic.rollouts_k=8 "
        "grpo.train.steps=300 grpo.train.batch_size=8 grpo.train.group_size=8"
    )
    TRACE_LIMIT, FOUNDRY_LIMIT, EVAL_LIMIT = 2000, 1500, 300
else:  # T4x2 — tighter memory; smaller model
    MODEL  = "phi3_mini"           # or "qwen2_5_1_5b" if Phi-3 OOMs
    PRESET = (
        "sft.train.batch_size=1 sft.train.grad_accum=16 "
        "sft.train.max_seq_tokens=2048 model.max_seq_len=2048 "
        "prm.labeling.logic.rollouts_k=4 "
        "grpo.train.steps=150 grpo.train.batch_size=2 grpo.train.group_size=4"
    )
    TRACE_LIMIT, FOUNDRY_LIMIT, EVAL_LIMIT = 800, 800, 200

BASE = f"model={MODEL} data=gsm8k {PRESET} wandb.enabled=false"
print("MODEL:", MODEL, "| GPU:", GPU)
print("BASE:", BASE)
```

## Cell B — data → traces → compress → SFT

```python
!python scripts/00_download_data.py {BASE}
!python scripts/01_build_traces.py  {BASE} distill.source.limit={TRACE_LIMIT}
!python scripts/02_compress.py      {BASE} distill.source.limit={TRACE_LIMIT}
!python scripts/03_sft.py           {BASE}
```

## Cell C — PRM label foundry (one run per benchmark domain)

```python
for D in ["gsm8k", "mmlu", "strategyqa"]:
    print("=== foundry:", D, "===")
    !python scripts/04_prm_label.py {BASE} data={D} distill.source.limit={FOUNDRY_LIMIT}
```

## Cell D — train XD-PRM + G2 gate  (must print `GATE ... PASS`)

```python
!python scripts/05_prm_train.py {BASE}
# If it raises GateError (AUC<0.70): rerun Cell C with prm.labeling.logic.rollouts_k=8,
# or disable a redundant head, e.g. append  prm.heads.consistency.enabled=false  to BASE.
```

## Cell E — GRPO (the RL stage) + final eval on the PS-06 benchmarks

```python
!python scripts/06_grpo.py {BASE}
!python scripts/07_eval.py {BASE} eval.limit={EVAL_LIMIT}
import json, os, glob
p = sorted(glob.glob(os.environ["AXIOM_EXP"] + "/eval/*.json"))
print(json.dumps(json.load(open(p[-1])), indent=2))   # base vs sft vs axiom_full per dataset
```

## Cell F — publish (after gates pass)

```python
# Set HF_TOKEN, then upload with your HF cell (or huggingface_hub):
#   - the merged policy (LoRA merged into the base) under an Apache-2.0/MIT repo
#   - the step-label dataset from $AXIOM_DATA/prm/*/labels.jsonl (CC-BY)
from huggingface_hub import HfApi  # example; fill repo ids
```

### Notes
- **Latency/efficiency KPI**: compression already measured **~57% token reduction**; the eval also
  reports `tokens_per_answer` per variant — cite `axiom_full` vs `base`.
- **Improvement = `axiom_full` (or `sft_verifier_adp`) minus `base`** on each of GSM8K / MMLU /
  StrategyQA. Need **+5% on ≥2 of 3** (GSM8K ≥50%, MMLU ≥45%, StrategyQA ≥65%).
- If a stage dies on memory: drop `*_LIMIT`, lower `sft.train.batch_size`, or switch `MODEL` down
  (`qwen2_5_7b → phi3_mini → qwen2_5_1_5b`).
```
