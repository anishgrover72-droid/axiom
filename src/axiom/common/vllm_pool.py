"""The single shared rollout engine.

MC-rollout labeling, GRPO sampling, self-consistency, and verifier-guided decoding all
generate through one cached engine per model — batched, KV-cached. Generating is the
project's main compute cost; routing it through here is how the budget closes.

`get_engine` uses vLLM when available (the A100/H100 path) and transparently falls back to
a transformers backend on machines without vLLM/CUDA (CPU/MPS), so the same code runs in
both places. Both backends are imported lazily so non-GPU code can import this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm import LLM


@dataclass(slots=True)
class GenConfig:
    """Sampling settings for one generate call."""

    n: int = 1  # samples per prompt (MC rollouts, group sampling)
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


class Engine(Protocol):
    """The interface every backend exposes; consumers depend on this, not the backend."""

    hf_id: str

    def generate(self, prompts: list[str], cfg: GenConfig) -> list[list[str]]: ...

    def chat(self, conversations: list[list[dict]], cfg: GenConfig) -> list[list[str]]: ...


class RolloutEngine:
    """Thin batched wrapper over a vLLM `LLM`."""

    def __init__(
        self,
        hf_id: str,
        *,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from vllm import LLM

        self.hf_id = hf_id
        self._llm: LLM = LLM(
            model=hf_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,  # reuse KV across shared prompt prefixes
        )

    def generate(self, prompts: list[str], cfg: GenConfig) -> list[list[str]]:
        """Return `cfg.n` completions per prompt, in one batched call."""
        from vllm import SamplingParams

        params = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop or None,
            seed=cfg.seed,
        )
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outputs]

    def chat(self, conversations: list[list[dict]], cfg: GenConfig) -> list[list[str]]:
        """Apply the model's chat template, then generate."""
        tok = self._llm.get_tokenizer()
        prompts = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        return self.generate(prompts, cfg)


class HFEngine:
    """Transformers fallback for machines without vLLM/CUDA (CPU/MPS). Slow; for smoke runs."""

    def __init__(self, hf_id: str, *, dtype: str = "float32", max_model_len: int = 4096) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.max_model_len = max_model_len
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=getattr(torch, dtype))
            .to(self.device)
            .eval()
        )

    def generate(self, prompts: list[str], cfg: GenConfig) -> list[list[str]]:
        """One prompt at a time (no paged attention); `cfg.stop` is handled by re-segmentation."""
        import torch

        pad_id = (
            self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        )
        out: list[list[str]] = []
        for prompt in prompts:
            enc = self.tok(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_model_len
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    do_sample=cfg.temperature > 0 or cfg.n > 1,
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_tokens,
                    num_return_sequences=cfg.n,
                    pad_token_id=pad_id,
                )
            start = enc["input_ids"].shape[1]
            out.append([self.tok.decode(g[start:], skip_special_tokens=True) for g in gen])
        return out

    def chat(self, conversations: list[list[dict]], cfg: GenConfig) -> list[list[str]]:
        prompts = [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        return self.generate(prompts, cfg)


@lru_cache(maxsize=2)
def get_engine(hf_id: str, dtype: str = "bfloat16", max_model_len: int = 4096) -> Engine:
    """Cached engine per (model, dtype, length): vLLM if installed, else the HF fallback."""
    try:
        import vllm  # noqa: F401
    except ImportError:
        return HFEngine(hf_id, dtype=dtype, max_model_len=max_model_len)
    return RolloutEngine(hf_id, dtype=dtype, max_model_len=max_model_len)
