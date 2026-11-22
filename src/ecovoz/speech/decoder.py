"""Decodificación CTC: voraz y beam search por prefijos con modelo de lenguaje de caracteres."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np

from ecovoz.speech.text import ALPHABET, BLANK, decode

_NEG_INF = float("-inf")


def greedy_decode(log_probs: np.ndarray) -> str:
    """Máximo por paso, colapso de repetidos y eliminación de blancos."""
    best = log_probs.argmax(-1)
    keep = np.concatenate([[True], best[1:] != best[:-1]])
    return decode(best[keep].tolist()).strip()


def _add(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b == _NEG_INF else a + math.log1p(math.exp(b - a))


class CharNGram:
    """N-gramas de caracteres con interpolación, contados solo con textos de entrenamiento."""

    def __init__(self, order: int, counts: dict[str, int]) -> None:
        self.order = order
        self.counts = counts
        totals: Counter = Counter()
        for gram, count in counts.items():
            totals[gram[:-1]] += count
        self.totals = totals
        self.score = lru_cache(maxsize=500_000)(self._score)

    @classmethod
    def train(cls, texts: Iterable[str], order: int = 6, min_count: int = 2) -> CharNGram:
        counts: Counter = Counter()
        for text in texts:
            line = f"^{text} "
            for end in range(1, len(line)):
                for size in range(1, min(order, end + 1) + 1):
                    counts[line[end - size + 1 : end + 1]] += 1
        return cls(order, {g: c for g, c in counts.items() if c >= min_count})

    def _score(self, context: str, char: str) -> float:
        probability = 1.0 / len(ALPHABET)
        for size in range(min(self.order - 1, len(context)) + 1):
            prefix = context[len(context) - size :]
            total = self.totals.get(prefix, 0)
            if total:
                # Cuantos más conteos tiene el contexto, menos pesa el orden inferior.
                count = self.counts.get(prefix + char, 0)
                probability = (count + 5.0 * probability) / (total + 5.0)
        return math.log(probability)

    def save(self, path: Path) -> None:
        grams = sorted(self.counts)
        np.savez_compressed(
            path,
            order=self.order,
            grams=np.frombuffer("\n".join(grams).encode("utf-8"), dtype=np.uint8),
            counts=np.array([self.counts[g] for g in grams], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> CharNGram:
        with np.load(path) as data:
            grams = data["grams"].tobytes().decode("utf-8").split("\n")
            return cls(int(data["order"]), dict(zip(grams, data["counts"].tolist(), strict=True)))


def beam_search(
    log_probs: np.ndarray,
    lm: CharNGram | None = None,
    alpha: float = 0.5,
    beta: float = 1.0,
    beam_width: int = 24,
    prune: float = -7.0,
) -> str:
    """Beam search por prefijos (Hannun et al., 2014).

    Cada prefijo guarda la probabilidad de terminar en blanco y la de terminar en
    carácter. `alpha` pondera el modelo de lenguaje y `beta` premia cada carácter.
    """
    # prefijo -> (termina en blanco, termina en carácter, puntaje del lenguaje)
    beams = {"": (0.0, _NEG_INF, 0.0)}
    for step in log_probs:
        candidates = np.flatnonzero(step > prune).tolist() or [int(step.argmax())]
        scores = step.tolist()
        following: dict[str, list[float]] = {}
        for prefix, (blank_end, char_end, bonus) in beams.items():
            either = _add(blank_end, char_end)
            for index in candidates:
                p = scores[index]
                if index == BLANK:
                    entry = following.setdefault(prefix, [_NEG_INF, _NEG_INF, bonus])
                    entry[0] = _add(entry[0], either + p)
                    continue
                char = ALPHABET[index - 1]
                repeated = prefix.endswith(char)
                if repeated:
                    entry = following.setdefault(prefix, [_NEG_INF, _NEG_INF, bonus])
                    entry[1] = _add(entry[1], char_end + p)
                if char == " ":
                    if not prefix:
                        # Un espacio inicial es silencio: cuenta como blanco.
                        entry = following.setdefault(prefix, [_NEG_INF, _NEG_INF, bonus])
                        entry[0] = _add(entry[0], either + p)
                        continue
                    if repeated:
                        continue
                source = blank_end if repeated else either
                if source == _NEG_INF:
                    continue
                extended = prefix + char
                entry = following.get(extended)
                if entry is None:
                    language = 0.0
                    if lm is not None:
                        context = f"^{prefix}"[-(lm.order - 1) :]
                        language = alpha * lm.score(context, char) + beta
                    entry = following[extended] = [_NEG_INF, _NEG_INF, bonus + language]
                entry[1] = _add(entry[1], source + p)
        ranked = sorted(
            following.items(),
            key=lambda item: _add(item[1][0], item[1][1]) + item[1][2],
            reverse=True,
        )
        beams = {prefix: tuple(entry) for prefix, entry in ranked[:beam_width]}
    return next(iter(beams)).strip()
