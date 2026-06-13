"""QLoRA SFT of the student on compressed traces."""

import hydra
from omegaconf import DictConfig

from axiom.sft.train_sft import train_sft


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_sft(cfg)


if __name__ == "__main__":
    main()
