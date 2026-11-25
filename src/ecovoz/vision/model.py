"""The two compared architectures and their frozen or fine-tuned blocks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

WEIGHT_FILES = {
    "resnet50": "resnet50-11ad3fa6.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-5c1a4163.pth",
}


def build_model(arch: str, num_classes: int, weights_dir: Path | None = None) -> nn.Module:
    """Builds the network; with `weights_dir` it loads the ImageNet weights before replacing the head."""
    if arch not in WEIGHT_FILES:
        raise ValueError(f"Unknown architecture: {arch}")
    model = getattr(models, arch)(weights=None)
    if weights_dir is not None:
        state = torch.load(weights_dir / WEIGHT_FILES[arch], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    if arch == "resnet50":
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    else:
        model.classifier[2] = nn.Dropout(0.3)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def head_of(model: nn.Module, arch: str) -> nn.Module:
    return model.fc if arch == "resnet50" else model.classifier


def finetune_blocks(model: nn.Module, arch: str) -> list[nn.Module]:
    """Last blocks of the feature extractor, the ones with the most specific features."""
    if arch == "resnet50":
        return [model.layer3, model.layer4]
    return [model.features[index] for index in range(11, len(model.features))]


def set_trainable(model: nn.Module, modules: list[nn.Module]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def train_mode(model: nn.Module, modules: list[nn.Module]) -> None:
    """Only the trainable blocks update BatchNorm and use dropout."""
    model.eval()
    for module in modules:
        module.train()
