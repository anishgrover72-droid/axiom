"""Run the label foundry: produce XD-PRM training data + the publishable release."""

import hydra
from omegaconf import DictConfig

from axiom.prm.labeling.aggregate_release import run_foundry


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_foundry(cfg)


if __name__ == "__main__":
    main()
