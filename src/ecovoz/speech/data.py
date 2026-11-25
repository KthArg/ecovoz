"""Audio manifest and batches grouped by duration."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ecovoz.config import MANIFESTS, PROCESSED, SEED
from ecovoz.speech.text import encode
from ecovoz.utils import read_csv

CORPUS_GROUPS = ("all", "openslr", "tux")


def load_rows(split: str, corpus: str = "all") -> list[dict]:
    if corpus not in CORPUS_GROUPS:
        raise ValueError(f"Unknown corpus: {corpus}")
    rows = [row for row in read_csv(MANIFESTS / "audio.csv") if row["split"] == split]
    if corpus == "all":
        return rows
    return [row for row in rows if (row["source"] == "tux") == (corpus == "tux")]


class SpeechDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        audio, _ = sf.read(PROCESSED / self.rows[index]["path"], dtype="int16")
        return audio, encode(self.rows[index]["text"]), index


def collate(items):
    lengths = torch.tensor([len(audio) for audio, _, _ in items])
    batch = np.zeros((len(items), int(lengths.max())), dtype=np.int16)
    for position, (audio, _, _) in enumerate(items):
        batch[position, : len(audio)] = audio
    targets = torch.tensor([label for _, text, _ in items for label in text])
    target_lengths = torch.tensor([len(text) for _, text, _ in items])
    return torch.from_numpy(batch), lengths, targets, target_lengths, [index for _, _, index in items]


class DurationBatches(Sampler):
    """Batches of similar duration, limited by count and by padded seconds."""

    def __init__(
        self, rows: list[dict], batch_size: int = 32, max_seconds: float = 190.0, shuffle: bool = False
    ) -> None:
        self.seconds = [float(row["seconds"]) for row in rows]
        self.batch_size = batch_size
        self.max_seconds = max_seconds
        self.shuffle = shuffle
        self.generator = random.Random(SEED)
        self.batches = self._build()

    def _build(self) -> list[list[int]]:
        jitter = (lambda: self.generator.uniform(-0.3, 0.3)) if self.shuffle else (lambda: 0.0)
        keys = [seconds + jitter() for seconds in self.seconds]
        batches, current, longest = [], [], 0.0
        for index in sorted(range(len(keys)), key=keys.__getitem__):
            longest = max(longest, self.seconds[index])
            full = len(current) >= self.batch_size or longest * (len(current) + 1) > self.max_seconds
            if current and full:
                batches.append(current)
                current, longest = [], self.seconds[index]
            current.append(index)
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            self.batches = self._build()
            self.generator.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def make_loader(rows: list[dict], shuffle: bool = False, workers: int = 2, batch_size: int = 32) -> DataLoader:
    return DataLoader(
        SpeechDataset(rows),
        batch_sampler=DurationBatches(rows, batch_size, shuffle=shuffle),
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
