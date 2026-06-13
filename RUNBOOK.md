# AXIOM — Execution Runbook (cloud A100)

The plan to take AXIOM from code → a trained, gate-passing model + demo. Scope locked for
v1: **GSM8K + CommonsenseQA + OpenBookQA + ARC-Challenge** (HotpotQA deferred), **debug on
Qwen2.5-1.5B then headline on Phi-4-mini**, **commonsense head via free NLI proxy (no judge
API)**.

---

## 0. Compute recommendation

| Option | Why | Rough $ |
|---|---|---|
| **RunPod, 1× A100 80GB (Community Cloud)** ✅ | cheapest, per-second billing, pick the PyTorch/CUDA 12.1 template | ~$1.2–1.9/hr |
| Lambda Cloud, 1× A100 80GB | simplest UX, reliable on-demand | ~$1.1–1.8/hr |
| Colab Pro+ | only if you must; A100 not guaranteed, sessions drop on long runs | subscription |

Pick **A100 80GB + ~150GB disk**. Estimated v1 wall-clock: **~½ day on 1.5B**, **+½ day for the
Phi-4-mini headline** (foundry MC-rollouts dominate). Budget ~$30–60.

---

## 1. One-time setup

```bash
git clone <your-repo> axiom && cd axiom

# vLLM FIRST — it pins a compatible torch. Then the rest, then freeze.
pip install vllm
pip install -e . && pip install -r requirements.txt
pip freeze > requirements.lock.txt        # capture the exact working set

cp .env.example .env                       # set HF_TOKEN (leave JUDGE_* blank = free fallback)
huggingface-cli login                      # or rely on HF_TOKEN
python -c "import vllm, trl, peft; from trl import GRPOTrainer; print('stack OK')"
pytest -m "not slow"                       # contracts must pass (fast)
```

**Phi-4-mini gating:** accept its license on the HF model page once, under the account whose
`HF_TOKEN` you use.

---

## 2. Run order (Hydra overrides select model/data per stage)

Work on **`model=qwen2_5_1_5b`** end-to-end first; only then repeat with `model=phi4_mini`.
`distill.source.limit` caps examples per stage — start small, scale once green.

### Stage 1 — SFT (math-style CoT distillation)
```bash
python scripts/00_download_data.py
python scripts/01_build_traces.py distill.source.limit=20000     # OpenR1 math traces
python scripts/02_compress.py                                    # logs the % token reduction
python scripts/03_sft.py model=qwen2_5_1_5b
```
**Gate:** SFT loss falls; `compress` reports a real reduction (~target). *Note: SFT teaches
math-style CoT only; commonsense/science reasoning comes from the verifier + RL.*

### Stage 2 — Label foundry, one run per domain (accumulates under `data/prm/<dataset>/`)
```bash
for D in gsm8k commonsenseqa openbookqa arc_challenge; do
  python scripts/04_prm_label.py model=qwen2_5_1_5b data=$D distill.source.limit=1500
done
```
**Gate:** spot-check `data/prm/<dataset>/labels.jsonl` — logic labels in [0,1], consistency
high on coherent steps. *This is the expensive stage (MC rollouts).*

### Stage 3 — Train XD-PRM on all domains + G2 gate
```bash
python scripts/05_prm_train.py model=qwen2_5_1_5b
```
**Hard gate (G2):** must print `GATE ... PASS` (AUC>0.70, head-corr<0.90, ECE<0.15). If it
fails: raise `labeling.logic.rollouts_k`, add foundry examples, or disable a redundant head
(`prm.heads.<name>.enabled=false`). **Nothing downstream is trusted until this passes.**

### Stage 4 — Inference eval (verifier-guided + adaptive depth)
```bash
python scripts/07_eval.py model=qwen2_5_1_5b eval.limit=200
```
**Gate:** `sft_verifier` / `sft_verifier_adp` beat `sft` on accuracy **and** tokens/answer.

### Stage 5 — GRPO (optional, last, cuttable)
```bash
python scripts/06_grpo.py model=qwen2_5_1_5b grpo.train.steps=300
python scripts/07_eval.py model=qwen2_5_1_5b eval.limit=200     # axiom_full row
```
**Gate:** `axiom_full` ≥ `sft_verifier_adp`, or diagnose reward hacking (watch the audit logs)
and keep the inference-only result as headline.

### Stage 6 — Headline model
Repeat Stages 1–5 with `model=phi4_mini` (the foundry/PRM from 1.5B can be reused; retrain the
policy). Foundry can stay on the 1.5B generator to save compute — labels describe step quality
regardless of generator.

### Stage 7 — Demo + dataset release
```bash
make serve                                   # FastAPI :8000 (set AXIOM_VARIANT=grpo|sft)
cd frontend && npm install && npm run dev     # the React Flow explainability UI
# data/release/<dataset>.jsonl  = the Compact Distilled Reasoning Trace Dataset
```

---

## 3. Smoke before the real run
```bash
pip install torch transformers peft           # if not already
pytest -m slow                                 # runs the full PRM train→score on a tiny model
```

---

## 4. What I need from you to proceed
1. **The box** — RunPod/Lambda chosen + access (you run commands; paste logs/errors back).
2. **`HF_TOKEN`** in `.env` + Phi-4-mini license accepted.
3. **(optional) `WANDB_API_KEY`** for live training curves.
4. Confirm **per-domain foundry size** (default 1500) and **SFT corpus size** (default 20k) — raise/lower for time vs quality.

Everything else (datasets, configs, fixes) is in place. Open the box and we start at Stage 1.
```
