"""Single entry point for determinism. All randomness flows from here."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU+CUDA) if available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
