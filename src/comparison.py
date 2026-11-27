"""Measurements to compare models: cost (parameters, FLOPs, inference time) and similarity of the
features of a student with the ones of the teacher."""
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.models.distill import stfpm_loss


def count_flops(fn, image_size=224, device="cpu"):
    """Floating point operations of fn for one image (a multiply and an add count as 2)."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        fn(x)
    return counter.get_total_flops()


@torch.no_grad()
def inference_time_ms(fn, batch_size, device, image_size=224, warmup=10, repeats=50):
    """Median time in milliseconds of fn on a random batch, after a warmup.

    On GPU the CUDA kernels run asynchronously, so torch.cuda.synchronize() is called before reading
    the clock. The median is less sensitive than the mean to isolated slow repetitions.
    """
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return float(torch.tensor(times).median())


@torch.no_grad()
def teacher_similarity(student_backbone, teacher_backbone, loader, layers=("conv2_x", "conv3_x"), device="cuda"):
    """Mean STFPM loss (1 - cosine similarity per position) between the features of a backbone and the
    ones of the teacher, over all the images of a loader. Both backbones are ResNet18Stem."""
    student_backbone.to(device).eval()
    teacher_backbone.to(device).eval()
    totals = {name: 0.0 for name in layers}
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        s = student_backbone.forward_features(x)
        t = teacher_backbone.forward_features(x)
        for name in layers:
            totals[name] += stfpm_loss(s[name], t[name]).item() * x.size(0)
        n += x.size(0)
    return {name: total / n for name, total in totals.items()}
