"""Run the eval matrix (variants x datasets) and write results JSON."""

import hydra
from omegaconf import DictConfig

from axiom.eval.run_eval import run_suite


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_suite(cfg)


if __name__ == "__main__":
    main()
