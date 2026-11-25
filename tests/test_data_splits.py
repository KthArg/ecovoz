"""Checks on the manifests that there is no data leakage between splits."""

from collections import Counter, defaultdict

import pytest

from ecovoz.config import CLASSES, MANIFESTS
from ecovoz.speech.text import normalize
from ecovoz.utils import read_csv

SPLITS = {"train", "val", "test"}


@pytest.fixture(scope="module")
def images():
    return read_csv(MANIFESTS / "images.csv")


@pytest.fixture(scope="module")
def clips():
    return read_csv(MANIFESTS / "audio.csv")


def splits_per(rows, key):
    found = defaultdict(set)
    for row in rows:
        found[row[key]].add(row["split"])
    return found


def test_image_minimum_and_classes(images):
    assert len(images) >= 200
    assert {row["label"] for row in images} == set(CLASSES)
    assert {row["split"] for row in images} == SPLITS


def test_image_split_is_70_15_15_per_class(images):
    per_class = Counter(row["label"] for row in images)
    per_cell = Counter((row["label"], row["split"]) for row in images)
    for label, total in per_class.items():
        assert per_cell[label, "train"] / total == pytest.approx(0.70, abs=0.02)
        assert per_cell[label, "test"] / total == pytest.approx(0.15, abs=0.02)


def test_no_image_is_repeated(images):
    assert len({row["sha256_pixels"] for row in images}) == len(images)


def test_near_duplicate_groups_stay_in_one_split(images):
    assert all(len(found) == 1 for found in splits_per(images, "group").values())


def test_audio_minimum(clips):
    count = Counter(row["split"] for row in clips)
    assert count["train"] >= 100
    assert count["val"] and count["test"]


def test_no_sentence_crosses_splits(clips):
    assert all(len(found) == 1 for found in splits_per(clips, "text").values())


def test_openslr_speakers_stay_in_one_split(clips):
    spoken = [row for row in clips if row["source"] != "tux"]
    assert all(len(found) == 1 for found in splits_per(spoken, "speaker").values())


def test_transcripts_are_normalized(clips):
    assert all(normalize(row["text"]) == row["text"] for row in clips)
