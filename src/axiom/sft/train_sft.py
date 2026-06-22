"""QLoRA SFT of the student on compressed traces.

Curriculum is realized as length-ordered training (short, easy traces first → long,
hard ones), the cheap, deterministic proxy for Easy→…→Adversarial staging.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from axiom.common.hf import bnb_config, lora_config
from axiom.common.io import read_records
from axiom.common.logging import get_logger
from axiom.common.paths import sft_dir
from axiom.common.seed import set_seed
from axiom.data.schemas import Trace

log = get_logger(__name__)


def _format(trace: Trace, tokenizer) -> str:
    reasoning = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(trace.steps))
    if trace.final_answer:
        reasoning += f"\nThe answer is {trace.final_answer}."
    messages = [
        {"role": "user", "content": trace.question},
        {"role": "assistant", "content": reasoning},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def train_sft(cfg: DictConfig) -> Path:
    """Fine-tune the student on the compressed corpus; return the checkpoint dir."""
    set_seed(cfg.seed)
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_cfg, sft = cfg.model, cfg.sft
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)

    traces = list(
        read_records(Path(cfg.paths.data) / "compressed" / f"{cfg.data.name}.jsonl", Trace)
    )
    if sft.curriculum.enabled:
        traces.sort(key=lambda t: len(t.steps))
    texts = [_format(t, tokenizer) for t in traces]
    cap: int | None = sft.train.max_seq_tokens
    if cap is not None:
        n_before = len(texts)
        texts = [t for t in texts if len(tokenizer.encode(t)) <= cap]
        log.info("dropped %d/%d traces over max_seq_tokens=%d", n_before - len(texts), n_before, cap)
    dataset = Dataset.from_dict({"text": texts})
    log.info("SFT on %d traces", len(dataset))

    out_dir = sft_dir(cfg)
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=sft.train.epochs,
        per_device_train_batch_size=sft.train.batch_size,
        gradient_accumulation_steps=sft.train.grad_accum,
        learning_rate=sft.train.lr,
        warmup_ratio=sft.train.warmup_ratio,
        max_grad_norm=sft.train.max_grad_norm,
        gradient_checkpointing=sft.train.gradient_checkpointing,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb" if cfg.wandb.enabled else "none",
        dataset_text_field="text",
        max_seq_length=model_cfg.max_seq_len,
        model_init_kwargs={
            "quantization_config": bnb_config(model_cfg),
            "torch_dtype": getattr(torch, model_cfg.dtype),
        },
    )
    trainer = SFTTrainer(
        model=model_cfg.hf_id,
        args=args,
        train_dataset=dataset,
        peft_config=lora_config(sft.lora),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("saved SFT adapter -> %s", out_dir)
    return out_dir
