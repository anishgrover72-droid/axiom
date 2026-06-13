"""Train XD-PRM, then run the G2 gate (raises if the verifier is not trustworthy)."""

import hydra
from omegaconf import DictConfig

from axiom.prm.train_prm import train_prm
from axiom.prm.validate import validate_prm


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = train_prm(cfg)
    validate_prm(cfg, str(ckpt))  # G2 gate: blocks downstream use on failure


if __name__ == "__main__":
    main()
