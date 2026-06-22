# AX — Open-Weight Models & Agentic Development

This document explains (A) how AXIOM uses **open-weight models**, and (B) the **agentic AI
development setup** used to build the solution — including, honestly, **what did not work**.

---

## A. Open-weight models in the solution

AXIOM uses **only open-weight models** and public datasets — no proprietary model weights.

| Model | Weight licence | Where it's used |
|---|---|---|
| Qwen2.5-0.5B / 1.5B / 7B-Instruct | Apache-2.0 | student policy **and** XD-PRM backbone (debug → comparison) |
| Phi-4-mini-instruct | MIT | headline policy model |
| all-MiniLM-L6-v2 | Apache-2.0 | sentence embeddings for sparse compression + the efficiency head |
| (free open NLI proxy) | open | the commonsense head signal — no judge API needed |

A teacher path (`distill.source.mode=teacher`) exists but the **default uses an open distilled
trace dataset** (OpenR1-Math-220k), so the pipeline runs end-to-end without any closed model.

### Agentic behaviour *inside* the product

AXIOM is itself an agentic reasoning system at inference time:
- **Reasoning + planning** — the policy emits step-segmented reasoning; the **adaptive-depth
  controller** (`continue / expand / exit`) plans how much to think per question.
- **Tool/verifier chaining** — **verifier-guided decoding** scores candidate steps with XD-PRM and
  selects the best, chaining generation ↔ verification step by step.
- **Multi-signal self-check** — five heads independently judge each step (a lightweight
  "panel-of-critics" over a single reasoning trace).

---

## B. Agentic development setup (how we built it)

The codebase and especially the **frontend** were built with an **agentic coding assistant
(Claude Code — Anthropic Claude)** operating directly in the repository.

### Harness / coding agent
- **Claude Code** in the IDE: the agent reads/edits files, runs shell commands, searches the repo,
  and verifies its own work (build/test) in a loop — not just code suggestions.
- **`CLAUDE.md` as the agent instruction file** (the "agents.md" equivalent): a binding engineering
  charter (10 rules, canonical layout, the five shared contracts). Giving the agent explicit,
  enforceable house rules was the single highest-leverage thing — it kept generated code
  minimal, typed, and consistent across many sessions.

### Tool use / tool chaining
Representative real chains from this build:
- **Repo-grounded docs**: `grep`/read configs → extract exact HF IDs and dataset sizes → write docs
  (so every link/number is real, not hallucinated).
- **Asset pipeline**: query the NASA Images API → download public-domain Mars/galaxy originals →
  `sips` downscale/flip → wire into CSS → `npm run build` to verify. Fully tool-chained.
- **Build-verify loop**: every change was followed by `npm run build` / `pytest`. This caught a
  stray `arimport` typo that broke compilation, and confirmed the contract tests (30 passing).

### Reasoning & planning
- The agent planned multi-step work (multi-page frontend, the 9-stage docs) before editing, and
  used clarifying questions when a request was genuinely ambiguous (e.g. "tech console" → Console
  vs Tech page) instead of guessing.

### MCP servers & skills
- **MCP — Claude Design / DesignSync**: used to read an existing brand design system before deciding
  the visual direction (we ultimately kept AXIOM's identity independent).
- **MCP — context7**: pulling current library/framework docs on demand.
- **Skills** available in-harness (e.g. code-review, run/verify) for quality passes.
- **Memory / context handling**: a persistent project-memory plus the long-context window let the
  agent carry decisions (theme, fonts, data contracts) across a long iterative session; when context
  filled, it was summarised and continued without losing the thread.

### Multi-agent orchestration
- The harness supports spawning sub-agents and a deterministic **Workflow** orchestrator
  (fan-out/verify/synthesize). For this project the work was mostly single-agent iterative; the
  product's own 5-head verifier is the multi-critic pattern applied at inference.

---

## What worked

- **A written charter (`CLAUDE.md`)** → consistent, minimal, typed code across sessions.
- **Grounding docs in the actual repo** (grep configs for IDs/sizes) → zero fabricated facts.
- **Tight build/test-after-every-change loop** → fast detection of breakages (e.g. the typo).
- **Tool-chained asset sourcing** (NASA API → `sips` → CSS → build) → sharp, licence-safe visuals.
- **Self-demonstrating UI** (sample-trace fallback) so the demo never depends on a live GPU.
- **Clarify-then-act** on ambiguous design asks saved re-work.

## What did NOT work (honest)

- **Procedural / CSS-gradient planet looked fake.** An attempt to "generate our own" Mars with CSS
  radial-gradients was visually poor; we reverted to a real NASA image **circular-clipped** to remove
  its black box. *Lesson: agents over-reach toward clever-but-ugly; a human visual check is essential.*
- **Font choices needed several human corrections.** The agent's first typography read as
  "AI-generated"; it took explicit user feedback to land on Clash Display / Satoshi / Geist Mono.
  *Taste is hard to specify; previews + iteration beat one-shot guessing.*
- **No local training.** The dev machine is Apple-Silicon; **vLLM + bitsandbytes are CUDA-only** and
  `torch` wasn't even installed, so the model **cannot be trained locally**. We moved training to
  **Google Colab** (`notebooks/colab_run.ipynb`). *Plan for the actual target hardware early.*
- **Model not fully trained at write-time.** KPIs are stated as **targets/gates** until the Colab run
  completes; the live console therefore ships a **labelled sample trace** rather than implying
  measured results. *We chose honesty over fabricated numbers.*
- **Ambiguous one-word requests** ("tech console") caused a near-miss; resolved by asking rather than
  assuming.

---

*The agentic tooling accelerated implementation and documentation substantially, but visual taste,
hardware constraints, and honest KPI reporting still required human judgement at every step.*
