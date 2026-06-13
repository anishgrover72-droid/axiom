"""Cached sentence embeddings, shared by Sparse CoT compression and the efficiency head."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_embedder(name: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], name: str) -> np.ndarray:
    """L2-normalized embeddings; dot product equals cosine similarity."""
    return get_embedder(name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
