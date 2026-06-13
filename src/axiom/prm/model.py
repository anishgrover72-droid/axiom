"""XD-PRM: backbone + 5 cross-domain heads producing per-step R_aggregate.

The backbone reads a reasoning trace rendered with step sentinels; each step's hidden
state at its sentinel position is pooled and fed to the heads. A domain embedding makes
the heads cross-domain (the "XD"). Steps are flattened across the batch so heads operate
on a single [num_steps, hidden] tensor regardless of how many steps each trace has.
"""

from __future__ import annotations

from omegaconf import DictConfig
from torch import Tensor, nn

from axiom.common.errors import ConfigError
from axiom.common.steps import STEP_SENTINEL
from axiom.prm.heads import build_heads

DOMAINS: tuple[str, ...] = ("math", "commonsense", "science", "multihop")


def domain_id(domain: str) -> int:
    """Map a domain string to its embedding index (unknown -> commonsense)."""
    return DOMAINS.index(domain) if domain in DOMAINS else DOMAINS.index("commonsense")


class XDPRM(nn.Module):
    """Cross-domain, multi-head, step-level process reward model."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        head_cfgs: DictConfig,
        use_domain_embedding: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = build_heads(head_cfgs, hidden_size)
        self.head_weights = {name: float(head_cfgs[name].weight) for name in self.heads}
        self.domain_emb = nn.Embedding(len(DOMAINS), hidden_size) if use_domain_embedding else None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        step_batch_idx: Tensor,
        step_tok_idx: Tensor,
        step_domain: Tensor,
    ) -> dict[str, Tensor]:
        """Return {head_name: logits[num_steps]} for the flattened steps in the batch."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        step_h = hidden[step_batch_idx, step_tok_idx]  # [num_steps, hidden]
        if self.domain_emb is not None:
            step_h = step_h + self.domain_emb(step_domain)
        return {name: head(step_h) for name, head in self.heads.items()}

    def aggregate(self, logits: dict[str, Tensor]) -> Tensor:
        """Weighted sum of per-head probabilities -> R_aggregate[num_steps]."""
        return sum(self.head_weights[name] * logit.sigmoid() for name, logit in logits.items())


def build_xdprm(prm_cfg: DictConfig):
    """Construct (model, tokenizer, sentinel_id) from the prm config."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from axiom.common.hf import bnb_config, lora_config

    bb = prm_cfg.backbone
    tokenizer = AutoTokenizer.from_pretrained(bb.hf_id)
    tokenizer.add_special_tokens({"additional_special_tokens": [STEP_SENTINEL]})

    backbone = AutoModel.from_pretrained(
        bb.hf_id,
        quantization_config=bnb_config(bb),
        torch_dtype=getattr(torch, bb.get("dtype", "bfloat16")),
    )
    backbone.resize_token_embeddings(len(tokenizer))
    if bb.get("lora"):
        from peft import get_peft_model

        backbone = get_peft_model(backbone, lora_config(bb.lora, task_type="FEATURE_EXTRACTION"))

    model = XDPRM(backbone, backbone.config.hidden_size, prm_cfg.heads, bb.domain_embedding)
    sentinel_id = tokenizer.convert_tokens_to_ids(STEP_SENTINEL)
    if sentinel_id is None or sentinel_id == tokenizer.unk_token_id:
        raise ConfigError(
            f"step sentinel {STEP_SENTINEL!r} did not register as a token id "
            f"(got {sentinel_id}); step pooling would silently misalign."
        )
    return model, tokenizer, sentinel_id


# Only the trained pieces are checkpointed; the quantized base reloads from the hub.
_TRAINABLE_KEYS = ("heads.", "domain_emb", "lora_", "embed_tokens", "wte")


def save_xdprm(model: XDPRM, tokenizer, out_dir) -> None:
    """Save heads, domain embedding, LoRA deltas, and resized token embeddings."""
    import torch

    from axiom.common.io import ensure_dir

    out = ensure_dir(out_dir)
    state = {k: v for k, v in model.state_dict().items() if any(t in k for t in _TRAINABLE_KEYS)}
    torch.save(state, out / "xdprm.pt")
    tokenizer.save_pretrained(out)


def load_xdprm(prm_cfg: DictConfig, ckpt_dir):
    """Rebuild XD-PRM and load its trained deltas (strict=False over the frozen base)."""
    import torch

    model, tokenizer, sentinel_id = build_xdprm(prm_cfg)
    state = torch.load(f"{ckpt_dir}/xdprm.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model, tokenizer, sentinel_id
