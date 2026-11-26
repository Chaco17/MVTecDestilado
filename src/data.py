"""MVTec AD loading: index of the images, Dataset over an in-memory cache and LightningDataModule."""
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as TF
from tqdm.auto import tqdm

from src.utils import resolve_path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def build_index(root, categories):
    """List the images of train/ and test/ of each category.

    label is 0 for normal images (good folder) and 1 for anomalies. path is relative to root.
    """
    root = Path(root)
    rows = []
    for category in categories:
        for split in ("train", "test"):
            split_dir = root / category / split
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for path in sorted(defect_dir.iterdir()):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        rows.append({
                            "category": category,
                            "split": split,
                            "defect": defect_dir.name,
                            "label": int(defect_dir.name != "good"),
                            "path": path.relative_to(root).as_posix(),
                        })
    return pd.DataFrame(rows)


def load_image(path, image_size):
    """Open an image as RGB, resize it and return it as a uint8 tensor [3, H, W]."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return TF.pil_to_tensor(img)


class MVTecDataset(Dataset):
    """Subset of the in-memory image cache.

    images is the uint8 tensor with every image [N, 3, H, W].
    index has the rows of this subset, its pos column is the position of each image in images.
    """

    def __init__(self, images, index, mean, std):
        self.images = images
        self.index = index.reset_index(drop=True)
        self.positions = torch.tensor(self.index["pos"].to_numpy())
        self.categories = torch.tensor(self.index["category_idx"].to_numpy())
        self.labels = torch.tensor(self.index["label"].to_numpy())
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        image = self.images[self.positions[i]].float().div(255)
        image = (image - self.mean) / self.std
        return {"image": image, "category": self.categories[i], "label": self.labels[i]}


class MVTecDataModule(L.LightningDataModule):
    """train and val come only from train/good (split stratified by category), test is test/."""

    def __init__(self, root, categories, image_size=224, batch_size=32, num_workers=0,
                 cache_dir="outputs/cache", val_split=0.2,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.root = resolve_path(root)
        self.categories = list(categories)
        cache_dir = resolve_path(cache_dir)
        self.cache_file = cache_dir / f"mvtec_{image_size}px.pt"
        self.index_file = cache_dir / f"mvtec_{image_size}px.csv"

    def prepare_data(self):
        """Build the on-disk cache if it does not exist or if the list of images changed."""
        index = build_index(self.root, self.categories)
        if self.cache_file.exists() and self.index_file.exists():
            cached = pd.read_csv(self.index_file)
            if cached["path"].tolist() == index["path"].tolist():
                return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        images = torch.stack([
            load_image(self.root / p, self.hparams.image_size)
            for p in tqdm(index["path"], desc="Construyendo cache")
        ])
        torch.save(images, self.cache_file)
        index.to_csv(self.index_file, index=False)

    def setup(self, stage=None):
        if hasattr(self, "test_set"):
            return
        index = pd.read_csv(self.index_file)
        images = torch.load(self.cache_file)
        index["pos"] = np.arange(len(index))
        index["category_idx"] = index["category"].map({c: i for i, c in enumerate(self.categories)})

        # Validation: val_split of train/good, stratified by category and with a fixed seed
        train_good = index[index.split == "train"]
        train_df, val_df = train_test_split(
            train_good,
            test_size=self.hparams.val_split,
            stratify=train_good["category"],
            random_state=self.hparams.seed,
        )
        test_df = index[index.split == "test"]

        def make(df):
            return MVTecDataset(images, df.sort_values("pos"), self.hparams.mean, self.hparams.std)

        self.train_set = make(train_df)
        self.val_set = make(val_df)
        self.test_set = make(test_df)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)
