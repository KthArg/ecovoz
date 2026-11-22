"""Alfabeto y normalización de las transcripciones."""

from __future__ import annotations

import re
import unicodedata

BLANK = 0
ALPHABET = " abcdefghijklmnopqrstuvwxyzáéíóúüñ"
VOCAB_SIZE = len(ALPHABET) + 1  # más el blanco de CTC

_TO_ID = {char: index + 1 for index, char in enumerate(ALPHABET)}
_FOREIGN = str.maketrans("àèìòùâêîôûäëïöçãõ", "aeiouaeiouaeiocao")
_PUNCTUATION = re.compile(r"[^\w\s]|_")


def normalize(text: str) -> str | None:
    """Pasa a minúsculas y quita la puntuación.

    Devuelve None si quedan dígitos o letras fuera del alfabeto: no se sabe cómo
    se pronunciaron, así que ese audio se descarta.
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
    """Pasos mínimos para alinear: CTC exige un blanco entre caracteres repetidos."""
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
