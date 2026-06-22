# Technical Stack & Open-Source Components

Everything AXIOM is built on is open source. Versions are pinned in
[`requirements.txt`](../requirements.txt) (Python) and [`frontend/package.json`](../frontend/package.json) (UI).

## Open-weight models

| Model | Role | License | Link |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | smallest student / fast XD-PRM backbone (smoke + debug) | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct |
| Qwen2.5-1.5B-Instruct | debug student / backbone | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| Qwen2.5-7B-Instruct | cross-model comparison | Apache-2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Phi-4-mini-instruct | headline policy model | MIT | https://huggingface.co/microsoft/Phi-4-mini-instruct |
| all-MiniLM-L6-v2 | sentence embeddings for sparse compression + efficiency head | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |

> The commonsense head uses a **free open NLI proxy** by default — no proprietary judge API is
> required to run the pipeline.

## Datasets (public)

| Dataset | Domain | Link |
|---|---|---|
| GSM8K | math | https://huggingface.co/datasets/openai/gsm8k |
| ARC-Challenge | science | https://huggingface.co/datasets/allenai/ai2_arc |
| CommonsenseQA | commonsense | https://huggingface.co/datasets/tau/commonsense_qa |
| OpenBookQA | science | https://huggingface.co/datasets/allenai/openbookqa |
| HotpotQA | multi-hop | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| OpenR1-Math-220k | SFT reasoning-trace source | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k |

## Python / ML libraries

| Library | Why | Link |
|---|---|---|
| PyTorch | training/inference tensor backend | https://pytorch.org |
| Hugging Face Transformers | model loading + tokenizers | https://github.com/huggingface/transformers |
| Hugging Face Datasets | dataset loading/streaming | https://github.com/huggingface/datasets |
| TRL | GRPO trainer (RL) | https://github.com/huggingface/trl |
| PEFT | QLoRA / LoRA adapters | https://github.com/huggingface/peft |
| bitsandbytes | 4-bit quantization | https://github.com/bitsandbytes-foundation/bitsandbytes |
| Accelerate | distributed/precision plumbing | https://github.com/huggingface/accelerate |
| vLLM | fast batched rollouts / serving (the shared pool) | https://github.com/vllm-project/vllm |
| sentence-transformers | embeddings for compression / efficiency | https://github.com/UKPLab/sentence-transformers |
| FAISS | vector similarity (retrieval/dedup) | https://github.com/facebookresearch/faiss |
| Hydra + OmegaConf | composed configuration | https://github.com/facebookresearch/hydra |
| Pydantic | typed schemas across module boundaries | https://github.com/pydantic/pydantic |
| FastAPI + Uvicorn + sse-starlette | SSE reasoning service | https://github.com/fastapi/fastapi |
| Weights & Biases | experiment tracking | https://github.com/wandb/wandb |
| pytest + ruff | tests + lint/format | https://github.com/pytest-dev/pytest · https://github.com/astral-sh/ruff |

## Frontend libraries

| Library | Why | Link |
|---|---|---|
| React 18 | UI | https://github.com/facebook/react |
| Vite 5 | dev server / bundler | https://github.com/vitejs/vite |
| TypeScript | typing | https://github.com/microsoft/TypeScript |

Fonts: **Clash Display** + **Satoshi** (Fontshare) and **Geist Mono** (Google Fonts).
Hero/galaxy imagery: **NASA/USGS public-domain** Viking Mars mosaic (`PIA00407`) and Spitzer
Milky Way panorama (`PIA10955`), self-hosted in `frontend/public/`.

## Languages & platforms

Python 3.10+ (training/serving), TypeScript/React (UI). Trains on a single CUDA GPU
(Colab T4/L4/A100); the contract tests run anywhere (no GPU). vLLM and bitsandbytes are CUDA-only.
