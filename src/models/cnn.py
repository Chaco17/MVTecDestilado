"""Model A: CNN classifier trained from scratch.

Backbone with the first three stages of ResNet-18 (conv1, conv2_x, conv3_x) and random weights,
followed by an FC classifier over the 10 categories. Config: conf/model/cnn_scratch.yaml.

To be implemented in the model A step (notebooks/modelos_ab.ipynb).
"""
import lightning as L


class CNNClassifierModule(L.LightningModule):
    """Placeholder. Interface expected by the rest of the project:

    __init__   receives the keys of conf/model/cnn_scratch.yaml and calls save_hyperparameters(logger=False)
    get_embedding(x)   returns the embedding [B, d] used by the anomaly detector
    training_step, validation_step, test_step   log train/loss, val/loss and test/loss (also */acc)
    configure_optimizers   uses src.models.common.build_optimizers
    """

    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Model A is not implemented yet (src/models/cnn.py)")
