"""Build the reasoning-trace corpus (open dataset or teacher generation)."""

import hydra
from omegaconf import DictConfig

from axiom.distill.generate import build_traces


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    build_traces(cfg)


if __name__ == "__main__":
    main()
