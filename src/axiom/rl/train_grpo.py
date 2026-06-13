"""GRPO/RLVR fine-tuning with the XD-PRM composite reward (verifier frozen)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from axiom.common.hf import bnb_config, lora_config
from axiom.common.logging import get_logger
from axiom.common.paths import grpo_dir, prm_dir
from axiom.common.seed import set_seed
from axiom.data.loaders import load_examples
from axiom.prm.score import PRMScorer
from axiom.rl.rewards import CompositeReward

log = get_logger(__name__)

_SOLVE = "Solve step by step. Number each step. End with 'The answer is X.'"


def train_grpo(cfg: DictConfig) -> Path:
    """RL-tune the SFT policy against the frozen verifier; return the checkpoint dir."""
    set_seed(cfg.seed)
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id)

    def to_prompt(question: str) -> str:
        messages = [{"role": "system", "content": _SOLVE}, {"role": "user", "content": question}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    examples = load_examples(cfg.data, "train", cfg.distill.source.limit)
    dataset = Dataset.from_list(
        [
            {
                "prompt": to_prompt(e.question),
                "gold": e.gold_answer,
                "answer_type": e.answer_type,
                "domain": e.domain,
                "choices": e.choices,
            }
            for e in examples
        ]
    )
    log.info("GRPO on %d prompts", len(dataset))

    scorer = PRMScorer(cfg.prm, str(prm_dir(cfg))) if cfg.grpo.reward.process > 0 else None
    reward = CompositeReward(
        cfg.grpo.reward, scorer, cfg.model.hf_id, cfg.grpo.train.max_new_tokens
    )

    out_dir = grpo_dir(cfg)
    args = GRPOConfig(
        output_dir=str(out_dir),
        num_generations=cfg.grpo.train.group_size,
        per_device_train_batch_size=cfg.grpo.train.batch_size,
        learning_rate=cfg.grpo.train.lr,
        beta=cfg.grpo.train.kl_coef,
        max_prompt_length=cfg.grpo.train.max_prompt_len,
        max_completion_length=cfg.grpo.train.max_new_tokens,
        max_steps=cfg.grpo.train.steps,
        bf16=True,
        logging_steps=10,
        save_steps=cfg.grpo.train.steps,
        report_to="wandb" if cfg.wandb.enabled else "none",
        model_init_kwargs={
            "quantization_config": bnb_config(cfg.model),
            "torch_dtype": getattr(torch, cfg.model.dtype),
        },
    )
    trainer = GRPOTrainer(
        model=cfg.model.hf_id,
        args=args,
        reward_funcs=[reward],
        train_dataset=dataset,
        peft_config=lora_config(cfg.sft.lora),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    log.info("saved GRPO policy -> %s", out_dir)
    return out_dir
