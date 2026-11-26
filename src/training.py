"""Glue between Hydra and Lightning: every object is built from its block of the config.

    cfg.data       -> LightningDataModule
    cfg.model      -> LightningModule
    cfg.callbacks  -> list of callbacks (EarlyStopping, ModelCheckpoint, ...)
    cfg.logger     -> WandbLogger
    cfg.trainer    -> Trainer, receiving the callbacks and the logger

The LightningModules should call self.save_hyperparameters(logger=False): the full resolved config
is logged to W&B here, and the module hparams would overwrite some of its keys (e.g. name).
"""
import json
import time
from pathlib import Path

import lightning as L
import wandb
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from src.utils import PROJECT_ROOT, resolve_path


class EpochSummary(Callback):
    """Keeps the train and val metrics of every epoch and prints one line per epoch."""

    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(("train/", "val/"))}
        metrics["epoch"] = trainer.current_epoch + 1
        self.history.append(metrics)
        values = " | ".join(f"{k} {metrics[k]:.4f}" for k in sorted(metrics) if k != "epoch")
        print(f"epoca {metrics['epoch']:3d} | {values}", flush=True)


def build_callbacks(cfg):
    """One callback per block of cfg.callbacks, keyed by its name in the YAML."""
    callbacks = {}
    for name, block in cfg.callbacks.items():
        # ModelCheckpoint resolves a relative dirpath against the current folder when it is created,
        # and the notebook runs from notebooks/. The path of the config is relative to the repository root.
        extra = {"dirpath": str(resolve_path(block.dirpath))} if "dirpath" in block else {}
        callbacks[name] = instantiate(block, **extra)
    return callbacks


def build_logger(cfg):
    """WandbLogger from cfg.logger, with the full resolved config attached to the run."""
    logger = instantiate(cfg.logger, save_dir=str(resolve_path(cfg.logger.save_dir)))
    logger._wandb_init.pop("anonymous", None)  # Lightning always sends it, wandb >= 0.31 warns that it is deprecated
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    return logger


def build_trainer(cfg, callbacks, logger=None):
    """lightning.Trainer from cfg.trainer, adding the callbacks and the logger."""
    return instantiate(cfg.trainer, callbacks=list(callbacks.values()),
                       logger=logger if logger is not None else False)


def run_experiment(cfg, retrain=False):
    """Train one configuration (fit, then test with the best checkpoint) and save a JSON summary.

    If the summary of cfg.name already exists it is loaded instead of training again.
    """
    results_file = resolve_path(cfg.paths.results_dir) / f"{cfg.name}.json"
    if results_file.exists() and not retrain:
        print(f"{cfg.name}: ya entrenado, se cargan sus resultados de {results_file.name}")
        return json.loads(results_file.read_text(encoding="utf-8"))

    L.seed_everything(cfg.seed, workers=True, verbose=False)
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model, _convert_="all")
    callbacks = build_callbacks(cfg)
    logger = build_logger(cfg)
    trainer = build_trainer(cfg, callbacks, logger)
    print(f">> {cfg.name} | {cfg.description} | {logger.experiment.url}", flush=True)

    start = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    minutes = (time.perf_counter() - start) / 60
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path="best", verbose=False)[0]

    history = callbacks["epoch_summary"].history
    monitor = callbacks["checkpoint"].monitor
    best = min(history, key=lambda h: h[monitor])
    results = {
        "run": cfg.name,
        "model": cfg.model.name,
        "description": cfg.description,
        "epochs": len(history),
        "stopped_early": len(history) < cfg.trainer.max_epochs,
        "best_epoch": best["epoch"],
        "best": best,
        "minutes": round(minutes, 2),
        "checkpoint": Path(callbacks["checkpoint"].best_model_path).relative_to(PROJECT_ROOT).as_posix(),
        "wandb_url": logger.experiment.url,
        "test": {k: float(v) for k, v in test_metrics.items()},
        "history": history,
    }
    wandb.finish()
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"OK {cfg.name}: {results['epochs']} epocas en {minutes:.1f} min | mejor epoca {best['epoch']}"
          f" ({monitor} {best[monitor]:.4f}) | test {results['test']}", flush=True)
    return results
