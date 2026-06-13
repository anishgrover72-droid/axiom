"""The XD-PRM reward heads and their masked multi-task losses.

Each head is a small MLP over a step's pooled hidden state, emitting a logit. Labels in
[0,1] are matched with BCE (binary signals) or MSE-on-sigmoid (soft targets). A `None`
label for a head masks that step out of the head's loss.
"""

from __future__ import annotations

import torch
from torch import nn

from axiom.data.schemas import HEAD_NAMES


class RewardHead(nn.Module):
    """MLP: pooled step hidden state -> scalar logit."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [S] logits


def masked_head_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str
) -> torch.Tensor:
    """Per-head loss over labeled steps only. Returns 0 if the head has no labels."""
    if mask.sum() == 0:
        return logits.sum() * 0.0  # keep graph connected, contribute nothing
    logits, target = logits[mask], target[mask]
    if kind == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
    return nn.functional.mse_loss(torch.sigmoid(logits), target)


def build_heads(head_cfgs: dict, hidden_size: int) -> nn.ModuleDict:
    """Construct the enabled heads, keyed by canonical head name."""
    return nn.ModuleDict(
        {name: RewardHead(hidden_size) for name in HEAD_NAMES if head_cfgs[name].enabled}
    )
