# Installation & Reproducibility

Two environments: a **CPU dev/test** path (anywhere) and a **GPU training** path (CUDA — Colab/A100).
vLLM and bitsandbytes are **CUDA-only**, so training does not run on macOS/CPU.

## A. CPU — contracts & tests (no GPU)

```bash
git clone <repo> axiom && cd axiom
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (pydantic, hydra, numpy, datasets)
make test                  # or: pytest -m "not slow"
```

Expected: the fast contract suite passes (steps, answers, schemas, metrics, compress, difficulty,
foundry labels). The single skipped test needs `torch` and is the GPU smoke test.

## B. GPU — full training/eval (the reproducibility path)

### Easiest: Google Colab (recommended for reproduction)

Open [`notebooks/colab_run.ipynb`](../notebooks/colab_run.ipynb) and:

1. **Runtime → Change runtime type → GPU** (T4 works for 0.5B; L4/A100 for 1.5B+/Phi-4-mini).
2. Zip & upload the project to Drive:
   ```bash
   cd ~/Desktop && zip -r axiom.zip axiom -x '*/.venv*' '*/node_modules/*' '*/__pycache__/*' '*/.git/*'
   ```
3. Run cells **1–7** (mount Drive, unzip, install, verify `cuda True`, set paths).
4. Put a free **HF token** ([hf.co/settings/tokens](https://huggingface.co/settings/tokens)) in cell 6.
5. Run **Track 1** (cells `00→13`): data → traces → compress → SFT → eval. *Guarantees a trained
   model + `base`/`sft` numbers.*
6. Run **Track 2** (cells `15→17`): label foundry → XD-PRM (**G2 gate**) → GRPO → full eval.

Outputs persist to Drive (`axiom_out/`), so a disconnect won't lose finished stages — re-run
cells 1–7 and use the resume-status cell.

### On your own GPU box

```bash
pip install vllm                      # FIRST — pins a compatible torch
pip install -e . && pip install -r requirements.txt
cp .env.example .env                  # set HF_TOKEN
python -c "import vllm, trl, peft; print('stack OK')"
pytest -m "not slow"
```

Then run the pipeline (smallest/fastest shown; drop limits / change `model=` to scale):

```bash
python scripts/00_download_data.py
python scripts/01_build_traces.py  model=qwen2_5_0_5b distill.source.limit=300
python scripts/02_compress.py      model=qwen2_5_0_5b distill.source.limit=300
python scripts/03_sft.py           model=qwen2_5_0_5b distill.source.limit=300
python scripts/04_prm_label.py     model=qwen2_5_0_5b distill.source.limit=300
python scripts/05_prm_train.py     model=qwen2_5_0_5b      # runs the G2 gate
python scripts/06_grpo.py          model=qwen2_5_0_5b grpo.train.steps=300
python scripts/07_eval.py          model=qwen2_5_0_5b eval.limit=200
```

Headline run: repeat with `model=phi4_mini`. Full procedure & gates: [`RUNBOOK.md`](../RUNBOOK.md).

## C. Serve the demo

```bash
make serve                                  # FastAPI on :8000 (AXIOM_VARIANT=grpo|sft)
cd frontend && npm install && npm run dev    # explainability console on :5173
```

The console auto-falls back to a labelled **sample reasoning trace** if the backend isn't running,
so the UI is demonstrable without a GPU.

## Configuration

All tunables are Hydra configs in [`configs/`](../configs/); override on the CLI
(`prm.heads.efficiency.enabled=false`, `eval.limit=200`). Environment: `HF_TOKEN` (required),
`AXIOM_DATA` / `AXIOM_EXP` (output paths), optional `WANDB_API_KEY`.
