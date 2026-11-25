"""Trains both architectures with four transfer learning strategies.

The strategy for each architecture is chosen by macro F1 on validation.
This script never reads the test split.

Strategies:
- head: frozen feature extractor, only the head is trained.
- partial_noaug: the last blocks are fine-tuned, without data augmentation.
- partial_aug: the last blocks are fine-tuned, with data augmentation.
- full_aug: the whole network is fine-tuned, with a lower learning rate in the extractor.

Usage: python scripts/train_vision.py [--arch A] [--strategy S]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ecovoz.config import ARCHITECTURES, CLASSES, DOWNLOADS, MODELS, REPORTS, RUNS, SEED
from ecovoz.utils import get_device, load_checkpoint, read_json, save_half_precision, set_seed, write_json
from ecovoz.vision.data import WasteDataset
from ecovoz.vision.model import build_model, finetune_blocks, head_of, set_trainable, train_mode

STRATEGIES = {
    "head": {"unfreeze": "none", "augment": True},
    "partial_noaug": {"unfreeze": "partial", "augment": False},
    "partial_aug": {"unfreeze": "partial", "augment": True},
    "full_aug": {"unfreeze": "all", "augment": True},
}
HEAD_EPOCHS, HEAD_LR = 4, 1e-3
TUNE_EPOCHS, TUNE_HEAD_LR = 30, 5e-4
BACKBONE_LR = {"partial": 1.5e-4, "all": 1e-4}
PATIENCE = 8
BATCH_SIZE = 32


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None) -> dict:
    total, predictions, targets = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(labels)
            predictions.append(logits.argmax(1).cpu())
            targets.append(labels.cpu())
    predicted, target = torch.cat(predictions).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total / len(target),
        "accuracy": float((predicted == target).mean()),
        "f1": float(f1_score(target, predicted, average="macro")),
    }


def make_loaders(augment: bool) -> tuple[DataLoader, DataLoader]:
    common = {"batch_size": BATCH_SIZE, "pin_memory": True}
    generator = torch.Generator().manual_seed(SEED)
    train = DataLoader(
        WasteDataset("train", augment), shuffle=True, num_workers=2, persistent_workers=True, generator=generator, **common
    )
    val = DataLoader(WasteDataset("val"), **common)
    return train, val


def train(arch: str, strategy: str, device: torch.device) -> dict:
    set_seed(SEED)
    options = STRATEGIES[strategy]
    train_loader, val_loader = make_loaders(options["augment"])
    model = build_model(arch, len(CLASSES), DOWNLOADS).to(device)
    head = head_of(model, arch)

    counts = np.bincount(train_loader.dataset.labels, minlength=len(CLASSES))
    weights = torch.tensor(np.sqrt(counts.mean() / counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    phases = [("head", HEAD_EPOCHS if options["unfreeze"] != "none" else TUNE_EPOCHS)]
    if options["unfreeze"] != "none":
        phases.append(("finetune", TUNE_EPOCHS))

    best = {"f1": -1.0}
    history = []
    target = RUNS / "vision" / f"{arch}_{strategy}.pt"
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    started = time.perf_counter()

    for phase, epochs in phases:
        if phase == "head":
            trainable = [head]
            groups = [{"params": list(head.parameters()), "lr": HEAD_LR}]
        else:
            whole = options["unfreeze"] == "all"
            trainable = [model] if whole else [*finetune_blocks(model, arch), head]
            in_head = {id(parameter) for parameter in head.parameters()}
            backbone = [p for module in trainable for p in module.parameters() if id(p) not in in_head]
            groups = [
                {"params": backbone, "lr": BACKBONE_LR[options["unfreeze"]]},
                {"params": list(head.parameters()), "lr": TUNE_HEAD_LR},
            ]
        set_trainable(model, trainable)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch, n=epochs: 0.02 + 0.98 * (1 + math.cos(math.pi * epoch / n)) / 2
        )
        stale = 0
        for _ in range(epochs):
            train_mode(model, trainable)
            fitted = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            model.eval()
            checked = run_epoch(model, val_loader, criterion, device)
            schedule.step()
            history.append(
                {
                    "epoch": len(history) + 1,
                    "phase": phase,
                    **{f"train_{key}": value for key, value in fitted.items()},
                    **{f"val_{key}": value for key, value in checked.items()},
                }
            )
            print(
                f"{arch} {strategy} {len(history):02d} train_f1={fitted['f1']:.4f} val_f1={checked['f1']:.4f}",
                flush=True,
            )
            stale += 1
            if checked["f1"] > best["f1"]:
                best = {**checked, "epoch": len(history)}
                stale = 0
                state = {"arch": arch, "strategy": strategy, "classes": list(CLASSES)}
                save_half_precision({**state, "state": model.state_dict()}, target)
            if phase == phases[-1][0] and stale >= PATIENCE:
                break

    result = {
        "arch": arch,
        "strategy": strategy,
        "best": best,
        "seconds": round(time.perf_counter() - started, 1),
        "history": history,
    }
    write_json(REPORTS / "vision_runs" / f"{arch}_{strategy}.json", result)
    return result


def export_best(arch: str) -> None:
    runs = [read_json(path) for path in sorted((REPORTS / "vision_runs").glob(f"{arch}_*.json"))]
    if not runs:
        return
    winner = max(runs, key=lambda run: run["best"]["f1"])
    checkpoint = load_checkpoint(RUNS / "vision" / f"{arch}_{winner['strategy']}.pt")
    save_half_precision(checkpoint, MODELS / f"{arch}.pt")
    print(f"CHOSEN {arch}: {winner['strategy']} val_f1={winner['best']['f1']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=ARCHITECTURES)
    parser.add_argument("--strategy", choices=STRATEGIES)
    args = parser.parse_args()
    device = get_device()
    for arch in [args.arch] if args.arch else ARCHITECTURES:
        for strategy in [args.strategy] if args.strategy else STRATEGIES:
            train(arch, strategy, device)
        export_best(arch)


if __name__ == "__main__":
    main()
