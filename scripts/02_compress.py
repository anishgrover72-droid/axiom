"""Sparse CoT compression of the trace corpus."""

import hydra
from omegaconf import DictConfig

from axiom.distill.compress import compress_corpus


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    compress_corpus(cfg)


if __name__ == "__main__":
    main()
