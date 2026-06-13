# CLAUDE.md — Engineering Charter for AXIOM

Read this before writing or changing any code. It is binding. `PLAN.md` is *what* we build; this is *how* we write it. When the two conflict, ask — don't guess.

**Prime directive:** write the *minimum* correct, optimised code that satisfies the task. No filler, no speculative abstractions, no files nobody asked for. Every line must earn its place.

---

## 1. The ten rules (non-negotiable)

1. **No unnecessary lines.** If a line can be removed without losing correctness or clarity, remove it. No dead code, no commented-out code, no unused imports/vars/args, no `print` debugging left behind.
2. **No unwanted files.** Create a file only if `PLAN.md`'s tree calls for it or the task genuinely requires it. Never scaffold "for later." No `utils.py` dumping grounds, no `tmp_*.py`, no duplicate scripts. Delete what you obsolete.
3. **One responsibility per module.** A file does one thing named by its path. If you're tempted to add an unrelated function, it belongs in another file (or doesn't exist yet).
4. **Don't repeat — reuse the contracts.** Step segmentation, schemas, answer-matching, token counting, and the rollout engine each exist in exactly one place (§4). Import them. Never re-implement.
5. **Config over constants.** No magic numbers or hardcoded paths/hyperparameters in `src/`. They live in `configs/*.yaml` and arrive via Hydra. Code reads config; it doesn't embed values.
6. **Type everything public.** Every function signature in `src/axiom/` is fully type-annotated. Data crossing module boundaries uses the pydantic schemas in `data/schemas.py`, not loose dicts.
7. **Fail loud, fail early.** Validate inputs at boundaries; raise specific exceptions. No silent `except: pass`, no returning `None` to mean "error."
8. **Optimise the hot path, not the cold one.** Batch GPU/IO work, reuse the vLLM pool and KV cache, vectorise. Do *not* micro-optimise glue code at the cost of readability.
9. **Test the contracts.** `common/steps.py`, `common/answers.py`, `data/schemas.py`, and reward shapes have unit tests. Determinism-sensitive code (step segmentation, PRM scoring) is tested for idempotence.
10. **Self-documenting names > comments.** Comment *why*, never *what*. If code needs a "what" comment, rename or simplify it instead.

---

## 2. Canonical repository structure

This is the **only** allowed top-level layout. Do not invent siblings. (Full file list in `PLAN.md` §2.)

```
axiom/
├── configs/        Hydra YAML config groups          (all tunables live here)
├── data/           gitignored datasets + data/release/ (publishable dataset)
├── src/axiom/      ALL importable logic
│   ├── common/     shared contracts (steps, answers, tokens, vllm_pool, registry, io, seed, logging)
│   ├── data/       schemas, loaders, difficulty
│   ├── distill/    trace sources/teacher, sparse compression
│   ├── sft/        QLoRA SFT
│   ├── prm/        ★ XD-PRM: model, heads, dataset, train, score, validate, labeling/
│   ├── rl/         rewards, GRPO
│   ├── inference/  adaptive_depth, verifier_decode, engine
│   ├── eval/       benchmarks, metrics, run_eval
│   └── serve/      FastAPI app + SSE stream
├── scripts/        thin numbered CLIs (00_…→08_…) that ONLY parse args + call src/
├── frontend/       React + Vite + shadcn/ui + React Flow
├── notebooks/      exploration & plots only (never imported by src/)
├── tests/          pytest mirroring src/ layout
└── experiments/    gitignored: checkpoints, W&B, eval JSON
```

**Placement rules:**
- Logic goes in `src/axiom/`. `scripts/` are ≤ ~30 lines: build config, call one `src/` entrypoint, exit. **No business logic in scripts.**
- Notebooks never get imported by `src/`. If notebook code is worth keeping, promote it into `src/` with tests.
- `experiments/` and `data/` are outputs — never import from them, never commit them.

---

## 3. Python style

- **Target:** Python 3.10+. Formatter **ruff format** (black-compatible), linter **ruff** — both must pass clean. Line length 100.
- **Imports:** stdlib / third-party / local, grouped; absolute imports (`from axiom.common.steps import segment`). No wildcard imports. No unused imports (ruff enforces).
- **Typing:** full annotations on public functions; `from __future__ import annotations` at top of every module; prefer `list`/`dict`/`X | None` over `typing.List`/`Optional`.
- **Data structures:** pydantic models (`data/schemas.py`) for anything persisted or crossing modules; `@dataclass(slots=True)` for internal value objects. Never pass around bare dicts as ad-hoc records.
- **Docstrings:** one-line imperative summary for every public function/class. Add an Args/Returns block **only** when types aren't self-explanatory. No docstrings on trivial private helpers.
- **Comments:** rare, and only explain *why* / non-obvious math / a cited algorithm (e.g. `# Math-Shepherd soft value: P(prefix → gold)`).  Delete narration.
- **Functions:** small, single-purpose, early-return over nested `if`. If a function exceeds ~40 lines or 3 nesting levels, split it.
- **Errors:** specific exceptions (define a small `axiom.common.errors` if needed). Validate at boundaries; trust internally.
- **Randomness/determinism:** all seeds flow from `common/seed.py`. Anything affecting PRM labels or eval must be reproducible from config.
- **No:** global mutable state, `*args/**kwargs` pass-through to dodge typing, monkey-patching, `os.system`, hardcoded absolute paths.

---

## 4. The shared contracts (import, never reinvent)

These five are the spine. Re-implementing any of them is a bug.

| Contract | File | Rule |
|---|---|---|
| **Step segmentation** | `common/steps.py` | The *only* definition of "a step" + boundary sentinel. Identical at PRM-train, PRM-score, and decode time. Idempotent. Unit-tested. |
| **Answer matching** | `common/answers.py` | Per-dataset gold extraction + verifiable equality. All correctness rewards & eval use this. |
| **Token accounting** | `common/tokens.py` | The one tokenizer-aware counter behind every efficiency/`~40%` number. |
| **Rollout engine** | `common/vllm_pool.py` | The single shared vLLM pool. MC labeling, GRPO sampling, self-consistency, verifier-decode all go through it — batched, KV-reused. |
| **Schemas** | `data/schemas.py` | `Example, Trace, Step, StepLabel, RewardBreakdown`. The vocabulary every module speaks. |

---

## 5. Configs (Hydra)

- Every script run = a composed Hydra config. No argparse hyperparameters beyond `--config-name`/overrides.
- Config groups mirror `PLAN.md` §2 (`model/`, `data/`, `prm/`, `grpo/`, …). One concept per file.
- An ablation is a CLI override (`prm.heads.efficiency.enabled=false`), **never** a code edit.
- Defaults live in YAML; code accesses via the typed config object. If you add a tunable, add it to the YAML and document it inline in the YAML, not in code.

---

## 6. Performance discipline

- **Batch everything GPU/IO-bound.** Never loop single examples through the model when a batch works. Rollouts, PRM scoring, and embeddings are always batched.
- **Reuse compute.** The MC-rollout samples produce Logic *and* Confidence *and* Efficiency signals — compute once, derive thrice (`PLAN.md` §4.2). KV-cache reuse across verifier-decode candidates.
- **Memory:** QLoRA (4-bit) by default; gradient checkpointing on for training; stream datasets, don't load whole corpora into RAM.
- **Measure before optimising** anything non-obvious; put the numbers in the run card. Don't pre-optimise cold glue code.

---

## 7. Testing

- `pytest`; tests mirror `src/` paths. Fast by default; mark GPU/integration tests `@pytest.mark.slow`.
- **Must-have tests:** step-segmentation idempotence, answer-matching per dataset, schema round-trip, reward tensor shapes/ranges, PRM scoring determinism.
- A bug fix lands with the regression test that would have caught it.
- Don't test trivial getters or third-party behaviour.

---

## 8. Dependencies

- Add a dependency only if it removes materially more code than it adds. Prefer the stack already chosen (PyTorch, HF Transformers/Datasets, TRL, PEFT, vLLM, FAISS, pydantic, hydra-core, FastAPI, wandb).
- Pin in `requirements.txt`. No overlapping libs that do the same job. Justify any new dependency in the PR description.

---

## 9. Git & change hygiene

- Small, focused commits; imperative subject (`Add MC-rollout logic labeler`).
- A change includes its tests and any config it introduces — no orphan code.
- Never commit `data/`, `experiments/`, checkpoints, secrets, or `node_modules/` (enforced by `.gitignore`).
- Branch off the default branch; don't commit/push unless asked.

---

## 10. The "don't" list (quick scan before you submit)

- ❌ Files/abstractions not required by the current task.
- ❌ Dead code, commented-out blocks, leftover prints, TODOs without an owner.
- ❌ Hardcoded paths, magic numbers, duplicated step/answer/token logic.
- ❌ Bare dicts across module boundaries; untyped public functions.
- ❌ Logic inside `scripts/` or `notebooks/`.
- ❌ Silent excepts; returning `None` for errors.
- ❌ Re-implementing a `common/` contract.
- ❌ Unbatched GPU calls in a hot path.
- ❌ New dependency without justification.

**Before finishing any task, re-read this list and the §1 rules. If anything you wrote violates them, fix it before reporting done.**
