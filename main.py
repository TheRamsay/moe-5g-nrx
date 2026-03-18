import os
from collections.abc import Callable
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["TF_USE_LEGACY_KERAS"] = "1"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Loaded config: {cfg.model.name} ({cfg.model.family})")
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    cast("Callable[[], None]", main)()
