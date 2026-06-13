"""Warm the HuggingFace cache for the selected dataset and trace source."""

import hydra
from omegaconf import DictConfig

from axiom.data.loaders import prefetch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    prefetch(cfg)


if __name__ == "__main__":
    main()
