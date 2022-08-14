import random

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils import get_logger

log = get_logger(__name__)


def train(config: DictConfig) -> None:
    # Seeding
    log.info("Seeding.")
    seed = config.get("seed", random.randint(0, 1_000_000))
    pl.seed_everything(seed, workers=True)

    # Model
    log.info("Creating LitModel.")
    model: pl.LightningModule = hydra.utils.instantiate(config.lit_model, _recursive_=False)

    # Callbacks
    callbacks = []
    if config.get("callbacks"):
        for callback_name, callback_conf in config.callbacks.items():
            assert "_target_" in callback_conf, "Must specify _target_ for callbacks."
            log.info(f"Instantiating callback {callback_name} -> {callback_conf._target_}")
            callbacks.append(hydra.utils.instantiate(callback_conf))

    if config.lit_model.get("scheduler_config"):
        log.info(f"Instantiating callback LearningRateMonitor -> {pl.callbacks.LearningRateMonitor}")
        callbacks.append(pl.callbacks.LearningRateMonitor())

    # WandbLogger
    log.info("Creating WandbLogger.")
    logger: pl.loggers.wandb.WandbLogger = hydra.utils.instantiate(config.logger)

    # DataModule
    log.info("Creating DataModule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    # Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)
