"""Model B: the same student as model A, trained by distillation from a full ResNet-18 (teacher).

Config: conf/model/cnn_distill.yaml (inherits from cnn_scratch.yaml and adds the teacher).
The distillation technique is decided and implemented in the model B step (notebooks/modelos_ab.ipynb).
"""
import lightning as L


class DistillModule(L.LightningModule):
    """Placeholder. Same interface as src.models.cnn.CNNClassifierModule (get_embedding, */loss, */acc)."""

    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Model B is not implemented yet (src/models/distill.py)")
