from typing import List, Optional
from pytorch_lightning.core.saving import convert
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from src.utils import utils
try:
    from habana_frameworks.torch import hpu
    hpu.init()
except:
    pass

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating data <{config.data._target_}>")
    data: LightningDataModule = hydra.utils.instantiate(config.data, _recursive_=False)

    # Init Lightning model
    log.info(f"Instantiating task <{config.task._target_}>")
    task: LightningModule = hydra.utils.instantiate(config.task, _recursive_=False)

    checkpoint = config.get("checkpoint")
    if checkpoint:
        task = task.load_from_checkpoint(checkpoint)
        log.info(f"Loaded checkpoint {checkpoint}")

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning plugins
    plugins: List = []
    if "plugins" in config:
        for _, plug_conf in config["plugins"].items():
            if "_target_" in plug_conf:
                log.info(f"Instantiating plugin <{plug_conf._target_}>")
                plugins.append(hydra.utils.instantiate(plug_conf))

    # Init Lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks,
        logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        task=task,
        data=data,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Evaluate model checkpoint
    if config.get("evaluate"):
        log.info("Validating pretrained model")
        trainer.validate(task, datamodule=data,
                         ckpt_path=config.get("evaluate"))
    else:  # Train the model
        resume_checkpoint = config.get("resume_checkpoint", None)
        log.info("Starting training!")
        trainer.fit(task, datamodule=data, ckpt_path=resume_checkpoint)

        # Evaluate model on test set after training
        if not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            trainer.test(task, datamodule=data)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        task=task,
        data=data,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
