import os
from collections.abc import Callable
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import build_dataloader, peek_batch

os.environ["TF_USE_LEGACY_KERAS"] = "1"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Loaded config: {cfg.model.name} ({cfg.model.family})")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    dataloader = build_dataloader(cfg)
    if cfg.dataset.diagnostics.sample_preview:
        batch = peek_batch(dataloader)
        print(
            "Sample batch -> inputs:",
            tuple(batch.inputs.shape),
            "bits:",
            tuple(batch.bit_labels.shape),
        )
        snr_min = float(batch.snr_db.amin().item())
        snr_max = float(batch.snr_db.amax().item())
        print(
            "SNR range:",
            f"[{snr_min:.2f}, {snr_max:.2f}] dB",
            "channel:",
            batch.channel_profile,
            "modulation:",
            batch.modulation,
        )


if __name__ == "__main__":
    cast("Callable[[], None]", main)()
