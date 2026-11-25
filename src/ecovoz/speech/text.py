"""Alphabet and transcript normalization."""

from __future__ import annotations

import re
import unicodedata

BLANK = 0
ALPHABET = " abcdefghijklmnopqrstuvwxyzáéíóúüñ"
VOCAB_SIZE = len(ALPHABET) + 1  # plus the CTC blank

_TO_ID = {char: index + 1 for index, char in enumerate(ALPHABET)}
_FOREIGN = str.maketrans("àèìòùâêîôûäëïöçãõ", "aeiouaeiouaeiocao")
_PUNCTUATION = re.compile(r"[^\w\s]|_")


def normalize(text: str) -> str | None:
    """Lowercases and removes punctuation.

    Returns None if digits or letters outside the alphabet remain: we cannot know
    how they were pronounced, so that audio is discarded.
    """
    text = unicodedata.normalize("NFC", text).lower().translate(_FOREIGN)
    text = " ".join(_PUNCTUATION.sub(" ", text).split())
    if not text or any(char not in _TO_ID for char in text):
        return None
    return text


def encode(text: str) -> list[int]:
    return [_TO_ID[char] for char in text]


def decode(ids) -> str:
    return "".join(ALPHABET[index - 1] for index in ids if index != BLANK)


def min_ctc_frames(ids) -> int:
    """Minimum steps to align: CTC requires a blank between repeated characters."""
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
