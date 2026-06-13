"""Tokenizer-aware token counting. The single source of every efficiency number.

`transformers` is imported lazily so the pure-python contracts stay importable without
the GPU stack. Tokenizers are cached per model id.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@lru_cache(maxsize=8)
def get_tokenizer(hf_id: str) -> PreTrainedTokenizerBase:
    """Load and cache a fast tokenizer for `hf_id`."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, use_fast=True)


def count(text: str, hf_id: str) -> int:
    """Number of tokens `hf_id`'s tokenizer assigns to `text` (no special tokens)."""
    return len(get_tokenizer(hf_id).encode(text, add_special_tokens=False))


def count_batch(texts: list[str], hf_id: str) -> list[int]:
    """Token counts for a batch, in one tokenizer call."""
    enc = get_tokenizer(hf_id)(texts, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in enc]


def reduction_ratio(verbose: int, compressed: int) -> float:
    """Fraction of tokens removed: 0.40 means a 40% reduction."""
    return 0.0 if verbose <= 0 else 1.0 - compressed / verbose
