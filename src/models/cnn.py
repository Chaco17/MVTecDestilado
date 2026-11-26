"""Model A: CNN classifier trained from scratch.

Backbone with the first three stages of ResNet-18 (conv1, conv2_x, conv3_x) and random weights,
followed by an FC classifier over the 10 categories. Config: conf/model/cnn_scratch.yaml.
Model B (src/models/distill.py) reuses the same network as student.
"""
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src.models.common import build_optimizers


class ResNet18Stem(nn.Module):
    """First three stages of ResNet-18, with the layers of torchvision so the structure is identical
    to the teacher of model B.

    torchvision names: conv1 = conv1 + bn1 + relu, conv2_x = maxpool + layer1, conv3_x = layer2.
    With a 224x224 input: conv1 -> 64x112x112, conv2_x -> 64x56x56, conv3_x -> 128x28x28.
    """

    out_channels = 128

    def __init__(self, weights=None):
        super().__init__()
        resnet = resnet18(weights=weights)  # weights=None: random initialization (Kaiming)
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.conv2_x = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.conv3_x = resnet.layer2

    def forward_features(self, x):
        """Output of every stage, keyed by its name (model B compares them with the teacher)."""
        features = {}
        for name in ("conv1", "conv2_x", "conv3_x"):
            x = getattr(self, name)(x)
            features[name] = x
        return features

    def forward(self, x):
        return self.conv3_x(self.conv2_x(self.conv1(x)))


class CNNClassifier(nn.Module):
    """Backbone + global average pooling (embedding) + FC classifier.

    The embedding is the average of each of the 128 channels of conv3_x over the 28x28 positions.
    If z_dim is given, a linear projection 128 -> z_dim is added and the embedding is taken after it.
    """

    def __init__(self, num_classes, hidden_dim=256, dropout=0.3, z_dim=None):
        super().__init__()
        self.backbone = ResNet18Stem()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(ResNet18Stem.out_channels, z_dim) if z_dim else nn.Identity()
        self.embedding_dim = z_dim or ResNet18Stem.out_channels
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def get_embedding(self, x):
        return self.projection(self.pool(self.backbone(x)).flatten(1))

    def forward(self, x):
        return self.head(self.get_embedding(x))


class CNNClassifierModule(L.LightningModule):
    """LightningModule of model A. It classifies the category of normal images with cross entropy.

    The arguments are the keys of conf/model/cnn_scratch.yaml. optimizer and scheduler are
    functools.partial objects built by Hydra (_partial_: true).
    """

    def __init__(self, name, num_classes, head, optimizer, scheduler=None, z_dim=None):
        super().__init__()
        # logger=False: src/training.py already logs the full config to W&B.
        # The optimizer and scheduler are partials, they are rebuilt from the config, not from the checkpoint.
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        self.net = CNNClassifier(num_classes, z_dim=z_dim, **head)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        return self.net.get_embedding(x)

    def shared_step(self, batch, stage):
        x, y = batch["image"], batch["category"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log_dict({f"{stage}/loss": loss, f"{stage}/acc": acc},
                      on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        return build_optimizers(self.parameters(), self.optimizer_factory, self.scheduler_factory)
