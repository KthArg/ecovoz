"""WER y CER implementados con distancia de Levenshtein."""

from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
            )
        previous = current
    return previous[-1]


def wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Errores de palabra totales entre palabras de referencia totales."""
    pairs = [(r.split(), h.split()) for r, h in zip(references, hypotheses, strict=True)]
    return sum(edit_distance(r, h) for r, h in pairs) / max(1, sum(len(r) for r, _ in pairs))


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Errores de carácter totales (con espacios) entre caracteres de referencia totales."""
    pairs = list(zip(references, hypotheses, strict=True))
    return sum(edit_distance(r, h) for r, h in pairs) / max(1, sum(len(r) for r, _ in pairs))
