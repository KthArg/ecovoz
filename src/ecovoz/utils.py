"""Semillas, dispositivo y lectura/escritura de archivos."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (CPU y GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def save_half_precision(payload: dict, path: Path) -> None:
    """Guarda un checkpoint con los pesos en float16, para que quepa en el repositorio."""
    state = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in payload["state"].items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "state": state}, path)


def load_checkpoint(path: Path) -> dict:
    """Carga un checkpoint sin ejecutar código (weights_only) y lo devuelve en float32."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state"] = {
        key: value.float() if value.is_floating_point() else value
        for key, value in payload["state"].items()
    }
    return payload
