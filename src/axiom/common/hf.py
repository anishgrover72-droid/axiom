"""HuggingFace QLoRA/LoRA config builders, shared by SFT and GRPO.

`transformers`/`peft` are imported lazily so non-GPU code stays importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig


def bnb_config(model_cfg: DictConfig) -> BitsAndBytesConfig | None:
    """4-bit NF4 quantization config for QLoRA, or None when disabled."""
    if not model_cfg.load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.dtype),
    )


def lora_config(lora_cfg: DictConfig, task_type: str = "CAUSAL_LM") -> LoraConfig:
    """PEFT LoRA config from the `lora` config block."""
    from peft import LoraConfig

    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias="none",
        task_type=task_type,
    )
