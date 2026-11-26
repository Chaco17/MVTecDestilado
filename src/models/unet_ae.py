"""Model C: deterministic autoencoder based on U-Net, trained from scratch.

PLACEHOLDER for the person in charge of model C. Work here, in conf/model/unet_ae.yaml,
in conf/experiment/unet_ae/ and in notebooks/modelo_c.ipynb.

Interface expected by the rest of the project (the same for the three models):

- The class is built with hydra.utils.instantiate(cfg.model), so __init__ receives exactly the
  keys of conf/model/unet_ae.yaml (name, z_dim, base_channels, optimizer, scheduler, and any key
  that is added there). Call self.save_hyperparameters(logger=False): src/training.py already logs
  the full config to W&B.
- Batches are dicts {"image": [B, 3, 224, 224] normalized with ImageNet mean/std,
  "category": [B] index of the category, "label": [B] 0 normal / 1 anomaly (only test has 1)}.
  Training and validation only contain normal images.
- get_embedding(x) returns the bottleneck vector [B, z_dim]. The anomaly detector
  (notebooks/anomalias.ipynb) fits the normal distribution on these vectors.
- Log the reconstruction loss as train/loss, val/loss and test/loss: EarlyStopping and
  ModelCheckpoint (conf/callbacks/default.yaml) monitor val/loss.
- configure_optimizers() can return src.models.common.build_optimizers(self.parameters(),
  self.optimizer_factory, self.scheduler_factory).
- Be careful with the skip connections: if they pass too much information the network copies the
  input without using the bottleneck and the embedding learns nothing useful (it happened in
  Tarea 03, where the full U-Net reached an AUROC close to 0.5).
"""
import lightning as L


class UNetAutoencoderModule(L.LightningModule):
    """Placeholder of model C. See the module docstring for the expected interface."""

    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Model C is not implemented yet (src/models/unet_ae.py)")
