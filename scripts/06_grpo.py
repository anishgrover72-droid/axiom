"""GRPO/RLVR fine-tuning against the frozen XD-PRM reward."""

import hydra
from omegaconf import DictConfig

from axiom.rl.train_grpo import train_grpo


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_grpo(cfg)


if __name__ == "__main__":
    main()
