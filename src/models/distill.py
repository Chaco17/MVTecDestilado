"""Model B: the same student as model A, trained by distillation from a pretrained ResNet-18 (teacher).

Distillation method: STFPM (Wang et al., 2021, Student-Teacher Feature Pyramid Matching).
The student imitates the intermediate feature maps of the frozen teacher at several stages.
For every position (h, w) of a stage l, both feature vectors are L2-normalized along the channels
and compared with half the squared Euclidean distance, which equals 1 - cos(F_t, F_s):

    loss_l = mean over images and positions of 1/2 * || F_t / ||F_t|| - F_s / ||F_s|| ||^2

The distillation loss is the sum over the chosen stages. On top of it the classification cross entropy
of model A is kept, so model B is still a classifier and the only difference with A is the teacher.

Config: conf/model/cnn_distill.yaml (inherits from cnn_scratch.yaml).
"""
import lightning as L
import torch
import torch.nn.functional as F

from src.models.cnn import CNNClassifier, ResNet18Stem
from src.models.common import build_optimizers


def stfpm_loss(student_features, teacher_features):
    """STFPM loss of one stage: features [B, C, H, W] normalized along C, 1/2 squared L2 per position,
    averaged over the positions and the images. It is computed in float32 (mixed precision is on)."""
    s = F.normalize(student_features.float(), dim=1)
    t = F.normalize(teacher_features.float(), dim=1)
    return 0.5 * (t - s).pow(2).sum(dim=1).mean()


class DistillModule(L.LightningModule):
    """LightningModule of model B.

    The arguments are the keys of conf/model/cnn_distill.yaml:
    teacher   {"weights": torchvision weights of the ResNet-18, e.g. "IMAGENET1K_V1"}
    distill   {"layers": stages compared with the teacher, "weight": weight of the distillation loss}
    The rest are the same as model A.
    """

    # Stages with the same shape in student and teacher (conv1 is left out: STFPM starts at conv2_x)
    COMPARABLE_LAYERS = ("conv2_x", "conv3_x")

    def __init__(self, name, num_classes, head, optimizer, teacher, distill, scheduler=None, z_dim=None):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["optimizer", "scheduler"])
        self.net = CNNClassifier(num_classes, z_dim=z_dim, **head)  # student, random weights
        # The teacher is a pretrained ResNet-18. Only its first three stages produce the features that
        # are compared, so conv4_x, conv5_x and the classifier of the teacher are never computed.
        self.teacher = ResNet18Stem(weights=teacher["weights"])
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.layers = list(distill["layers"])
        self.distill_weight = distill["weight"]
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler

    def train(self, mode=True):
        """Lightning calls model.train() at the start of every training epoch. The teacher must stay in
        eval mode, otherwise its BatchNorm layers would update their ImageNet statistics with our images."""
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x):
        return self.net(x)

    def get_embedding(self, x):
        return self.net.get_embedding(x)

    def shared_step(self, batch, stage):
        x, y = batch["image"], batch["category"]
        student = self.net.backbone.forward_features(x)
        with torch.no_grad():
            teacher = self.teacher.forward_features(x)
        logits = self.net.head(self.net.projection(self.net.pool(student["conv3_x"]).flatten(1)))

        ce = F.cross_entropy(logits, y)
        # The loss of every shared stage is logged, also the ones not used for training, so runs that
        # distill different stages can be compared with the same metric (e.g. val/distill_conv3_x)
        per_layer = {name: stfpm_loss(student[name], teacher[name]) for name in self.COMPARABLE_LAYERS}
        distill = sum(per_layer[name] for name in self.layers)
        loss = ce + self.distill_weight * distill
        acc = (logits.argmax(dim=1) == y).float().mean()
        metrics = {f"{stage}/loss": loss, f"{stage}/ce": ce, f"{stage}/distill": distill, f"{stage}/acc": acc}
        metrics.update({f"{stage}/distill_{name}": value for name, value in per_layer.items()})
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        # Only the student is optimized, the teacher stays frozen
        return build_optimizers(self.net.parameters(), self.optimizer_factory, self.scheduler_factory)
