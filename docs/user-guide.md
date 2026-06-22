# User Guide

AXIOM has two faces: a **training/eval pipeline** (CLI) and an **explainability web app**.

> Screenshots: add captured images under `docs/images/` and reference them here, e.g.
> `![Home](images/home.png)`. Suggested shots: hero, pipeline flowchart, datasets table,
> live reasoning console.

## The web app

Run `cd frontend && npm run dev` → open http://localhost:5173.

### Pages

1. **Home** — hero (Mars/space theme), animated stat band (5 heads · ~40% fewer tokens ·
   9 stages · >0.70 AUC), the value props, the **five verifier heads**, and an **auto-playing live
   reasoning demo** that streams a sample trace as you scroll.
   `![Home](images/home.png)`
2. **Pipeline** — a horizontal **flowchart** of the 9 stages, phase-grouped
   (Data → Compress → Train → Deliver), each node showing the input → output transformation.
   `![Pipeline](images/pipeline.png)`
3. **Tech** — colour-coded **stack** cards, an animated **datasets table** (domain tags + size
   bars), and **target metrics**.
   `![Tech](images/tech.png)`
4. **Console** — the live reasoning demo.

### Using the Console

1. Type a question (or click an **example chip**) and pick a domain.
2. Press **Reason**. Steps stream into the **Reasoning log**; the left rail of each step glows by
   its `R_aggregate` (signal intensity).
3. Click any step to inspect **Verifier telemetry**: the aggregate-reward dial, the five head
   channels, uncertainty, and the depth decision (`continue / expand / exit`).
4. The footer shows the final answer, step count and token count.

If the FastAPI backend isn't running, the console streams a clearly **labelled sample trace** so
the experience is always demonstrable. To get live reasoning, run `make serve` (see
[installation.md](installation.md)).
`![Console](images/console.png)`

## The CLI pipeline

Each stage is `python scripts/NN_*.py [hydra overrides]`. Common overrides:

| Override | Effect |
|---|---|
| `model=qwen2_5_0_5b\|qwen2_5_1_5b\|phi4_mini` | pick the model |
| `data=gsm8k\|commonsenseqa\|openbookqa\|arc_challenge` | pick the dataset |
| `distill.source.limit=N` | cap examples per stage (start small) |
| `eval.limit=N` | cap eval examples |
| `prm.heads.<name>.enabled=false` | ablate a verifier head |
| `wandb.enabled=false` | disable tracking |

Read the eval JSON at `${AXIOM_EXP}/eval/results.json` for the `base / sft / sft_verifier /
sft_verifier_adp / axiom_full` rows (accuracy, tokens/answer, latency, contradiction-rate).

## Interpreting results

- **XD-PRM ROC-AUC** — does the verifier separate good vs bad steps (target > 0.70).
- **tokens/answer ratio** — compression vs the verbose baseline at matched accuracy (~40% target).
- **ECE** — is the confidence head calibrated (target < 0.15).
- **best-of-N lift** — does verifier-guided decoding beat plain sampling.
