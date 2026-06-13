"""Multi-task training of XD-PRM over flattened, per-head-masked steps."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from axiom.common.logging import get_logger
from axiom.common.paths import foundry_dir, prm_dir
from axiom.common.seed import set_seed
from axiom.prm.dataset import StepLabeledDataset, load_labeled, make_collate
from axiom.prm.heads import masked_head_loss
from axiom.prm.model import build_xdprm, save_xdprm

log = get_logger(__name__)


def train_prm(cfg: DictConfig) -> Path:
    """Train the verifier on the foundry's labels; return the checkpoint dir."""
    set_seed(cfg.seed)
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, sentinel_id = build_xdprm(cfg.prm)
    model.to(device)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    traces, labels = load_labeled(foundry_dir(cfg), split="train")  # hold out the gate's slice
    # Truncate at the PRM backbone's length, NOT the policy's, so train/score drop the same
    # trailing steps and the sentinel/step-count contract holds at inference (see score.py).
    dataset = StepLabeledDataset(traces, labels, tokenizer, sentinel_id, cfg.prm.backbone.max_len)
    loader = DataLoader(
        dataset, batch_size=cfg.prm.train.batch_size, shuffle=True, collate_fn=make_collate(pad_id)
    )
    log.info("XD-PRM training on %d traces", len(dataset))

    kinds = {name: cfg.prm.heads[name].loss for name in model.heads}
    weights = model.head_weights
    optim = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.prm.train.lr)
    accum = cfg.prm.train.grad_accum

    model.train()
    for epoch in range(cfg.prm.train.epochs):
        optim.zero_grad()
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(
                ids,
                attn,
                batch["step_batch_idx"].to(device),
                batch["step_tok_idx"].to(device),
                batch["step_domain"].to(device),
            )
            loss = sum(
                weights[name]
                * masked_head_loss(
                    logit,
                    batch["targets"][name].to(device),
                    batch["masks"][name].to(device),
                    kinds[name],
                )
                for name, logit in logits.items()
            )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.prm.train.max_grad_norm)
                optim.step()
                optim.zero_grad()
            if step % 20 == 0:
                log.info("epoch %d step %d loss %.4f", epoch, step, loss.detach().item())

    out_dir = prm_dir(cfg)
    save_xdprm(model, tokenizer, out_dir)
    log.info("saved XD-PRM -> %s", out_dir)
    return out_dir
