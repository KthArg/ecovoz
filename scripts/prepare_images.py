"""Prepara las imágenes y las particiona 70/15/15 sin fuga de datos.

1. Une TrashNet y las clases equivalentes de RealWaste.
2. Elimina los duplicados exactos, comparando los píxeles decodificados.
3. Agrupa las fotos casi idénticas (el mismo objeto fotografiado varias veces)
   con la similitud coseno de ResNet-50 preentrenada.
4. Asigna grupos completos a una sola partición, estratificando por clase y fuente.

Uso: python scripts/prepare_images.py [--threshold T] [--inspect]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageOps

from ecovoz.config import (
    CLASSES,
    DOWNLOADS,
    FIGURES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFESTS,
    PROCESSED,
    REPORTS,
    RUNS,
    SEED,
    STORED_SIDE,
)
from ecovoz.utils import get_device, read_json, write_csv, write_json
from ecovoz.vision.model import build_model

SOURCES = {
    "trashnet": ("dataset-resized.zip", {name: name for name in CLASSES}),
    "realwaste": (
        "realwaste.zip",
        {
            "Cardboard": "cardboard",
            "Glass": "glass",
            "Metal": "metal",
            "Paper": "paper",
            "Plastic": "plastic",
            "Miscellaneous Trash": "trash",
        },
    ),
}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def normalize_images() -> tuple[list[dict], dict]:
    """Guarda cada imagen como JPEG RGB con lado corto de 256 px."""
    seen: dict[str, dict] = {}
    conflicts: set[str] = set()
    duplicates = 0
    for source, (archive, mapping) in SOURCES.items():
        with zipfile.ZipFile(DOWNLOADS / archive) as bundle:
            for member in sorted(bundle.namelist()):
                path = PurePosixPath(member)
                if member.startswith("__MACOSX") or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = mapping.get(path.parent.name)
                if label is None:
                    continue
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(bundle.read(member)))).convert("RGB")
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest in seen:
                    duplicates += 1
                    if seen[digest]["label"] != label:
                        conflicts.add(digest)
                    continue
                relative = f"images/{label}/{source}_{digest[:16]}.jpg"
                target = PROCESSED / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    scale = STORED_SIDE / min(image.size)
                    size = (round(image.width * scale), round(image.height * scale))
                    image.resize(size, Image.Resampling.BICUBIC).save(target, quality=92)
                seen[digest] = {
                    "path": relative,
                    "label": label,
                    "source": source,
                    "original": member,
                    "sha256_pixels": digest,
                }
    rows = [row for digest, row in seen.items() if digest not in conflicts]
    return rows, {"exact_duplicates": duplicates, "label_conflicts": len(conflicts)}


@torch.inference_mode()
def embed(rows: list[dict], device: torch.device) -> np.ndarray:
    model = build_model("resnet50", len(CLASSES), DOWNLOADS)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[:, None, None]
    output = []
    for start in range(0, len(rows), 64):
        batch = []
        for row in rows[start : start + 64]:
            image = Image.open(PROCESSED / row["path"]).resize((IMAGE_SIZE, IMAGE_SIZE))
            batch.append(np.array(image))
        images = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2).float() / 255
        features = model((images - mean) / std)
        output.append(torch.nn.functional.normalize(features).cpu().numpy())
    return np.concatenate(output)


def find_groups(rows: list[dict], similarity: np.ndarray, threshold: float) -> list[int]:
    """Componentes conexas de los pares de una misma fuente por encima del umbral."""
    parent = list(range(len(rows)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in np.argwhere(np.triu(similarity >= threshold, 1)):
        if rows[a]["source"] == rows[b]["source"]:
            parent[root(int(a))] = root(int(b))
    return [root(index) for index in range(len(rows))]


def split_groups(rows: list[dict], groups: list[int]) -> list[str]:
    """Reparte los grupos, de mayor a menor, a la partición más lejos de su cuota."""
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    order = sorted(members)
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda group: -len(members[group]))

    strata = [(row["label"], row["source"]) for row in rows]
    totals = Counter(strata)
    filled = Counter()
    assigned = [""] * len(rows)
    for group in order:
        need = Counter(strata[index] for index in members[group])
        split = max(
            RATIOS,
            key=lambda name: sum(
                count * (RATIOS[name] * totals[key] - filled[name, key]) / (RATIOS[name] * totals[key])
                for key, count in need.items()
            ),
        )
        for index in members[group]:
            assigned[index] = split
            filled[split, strata[index]] += 1
    return assigned


def save_inspection(rows: list[dict], similarity: np.ndarray) -> None:
    """Pares de muestra por rango de similitud, para elegir el umbral mirando las fotos."""
    import matplotlib.pyplot as plt

    levels = (0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85)
    upper = np.triu(similarity, 1)
    generator = np.random.default_rng(SEED)
    figure, axes = plt.subplots(len(levels), 8, figsize=(16, 2 * len(levels)))
    for level, row_axes in zip(levels, axes, strict=True):
        pairs = np.argwhere((upper >= level) & (upper < level + 0.01))
        generator.shuffle(pairs)
        for axis in row_axes:
            axis.axis("off")
        for position, pair in enumerate(pairs[:4]):
            for offset, index in enumerate(pair):
                axis = row_axes[2 * position + offset]
                axis.imshow(Image.open(PROCESSED / rows[index]["path"]))
                axis.set_title(f"{level:.2f} {rows[index]['label']}", fontsize=8)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "near_duplicates.png", dpi=80)

    for level in levels:
        sizes = Counter(find_groups(rows, similarity, level)).values()
        print(f"umbral {level}: {len(sizes)} grupos, el mayor de {max(sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.88)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    RUNS.mkdir(exist_ok=True)
    rows, stats = normalize_images()
    cache = RUNS / "image_embeddings.npy"
    features = np.load(cache) if cache.exists() else None
    if features is None or len(features) != len(rows):
        features = embed(rows, get_device())
        np.save(cache, features)
    similarity = features @ features.T
    if args.inspect:
        save_inspection(rows, similarity)
        return

    groups = find_groups(rows, similarity, args.threshold)
    splits = split_groups(rows, groups)
    for row, group, split in zip(rows, groups, splits, strict=True):
        row.update(group=group, split=split)
    write_csv(MANIFESTS / "images.csv", rows)

    counts = {
        split: dict(Counter(row["label"] for row in rows if row["split"] == split)) for split in RATIOS
    }
    sources = {
        split: dict(Counter(row["source"] for row in rows if row["split"] == split)) for split in RATIOS
    }
    sizes = Counter(groups)
    stats.update(
        total=len(rows),
        threshold=args.threshold,
        groups=len(sizes),
        largest_group=max(sizes.values()),
        images_in_groups=sum(size for size in sizes.values() if size > 1),
        per_split=counts,
        per_source=sources,
    )
    write_json(REPORTS / "data_images.json", stats)
    print(read_json(REPORTS / "data_images.json"))


if __name__ == "__main__":
    main()
