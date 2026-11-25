"""Image manifest loading, preprocessing and data augmentation."""

from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from ecovoz.config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MANIFESTS, PROCESSED
from ecovoz.utils import read_csv


def eval_transform() -> v2.Compose:
    """Same in evaluation and in the application: short side to 256 and a 224 center crop."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256, antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> v2.Compose:
    """Augmentation that simulates a camera: framing, rotation, perspective, lighting and blur."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),  # waste has no natural orientation
            v2.RandomApply([v2.RandomRotation(20)], p=0.4),
            v2.RandomPerspective(distortion_scale=0.2, p=0.2),
            v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.25, 0.04)], p=0.8),
            v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.5))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ]
    )


class WasteDataset(Dataset):
    def __init__(self, split: str, augment: bool = False) -> None:
        self.rows = [row for row in read_csv(MANIFESTS / "images.csv") if row["split"] == split]
        self.transform = train_transform() if augment else eval_transform()
        self.labels = [CLASSES.index(row["label"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        with Image.open(PROCESSED / self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]
