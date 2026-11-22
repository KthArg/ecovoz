"""Evalúa el reconocedor de voz.

1. Compara todos los experimentos sobre la misma validación.
2. Ajusta alfa y beta del beam search en validación.
3. Evalúa la prueba con el modelo final y los valores ya fijados.

Uso: python scripts/evaluate_speech.py
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import torch

from ecovoz.config import MODELS, REPORTS, RUNS, SEED
from ecovoz.speech.data import load_rows, make_loader
from ecovoz.speech.decoder import CharNGram, beam_search, greedy_decode
from ecovoz.speech.features import LogMel
from ecovoz.speech.metrics import cer, edit_distance, wer
from ecovoz.speech.model import DeepSpeech2
from ecovoz.utils import get_device, load_checkpoint, write_csv, write_json

EXPERIMENTS = ("small100", "small100_aug", "tux", "small100_transfer", "multi")
ALPHAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.3)
BETAS = (0.0, 0.5, 1.0, 1.5, 2.0)
BEAM_WIDTH = 24
BOOTSTRAP = 2000


@torch.inference_mode()
def acoustic_outputs(path, rows: list[dict], device: torch.device) -> list[np.ndarray]:
    checkpoint = load_checkpoint(path)
    model = DeepSpeech2(**checkpoint["options"])
    model.load_state_dict(checkpoint["state"])
    model.eval().to(device)
    front = LogMel().to(device)
    outputs: list[np.ndarray | None] = [None] * len(rows)
    for audio, lengths, _, _, indices in make_loader(rows, workers=0):
        features, frames = front(audio.to(device).float() / 32768, lengths.to(device))
        log_probs, steps = model(features, frames)
        for output, length, index in zip(log_probs.cpu().numpy(), steps.tolist(), indices, strict=True):
            outputs[index] = output[:length].copy()
    return outputs


def score(rows: list[dict], hypotheses: list[str]) -> dict:
    references = [row["text"] for row in rows]
    return {
        "n": len(rows),
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "exact": sum(r == h for r, h in zip(references, hypotheses, strict=True)) / len(rows),
    }


def score_groups(rows: list[dict], hypotheses: list[str]) -> dict:
    keys = {
        "all": lambda row: True,
        "openslr": lambda row: row["source"] != "tux",
        "tux": lambda row: row["source"] == "tux",
    }
    for source in sorted({row["source"] for row in rows} - {"tux"}):
        keys[source] = lambda row, name=source: row["source"] == name
    results = {}
    for name, belongs in keys.items():
        chosen = [index for index, row in enumerate(rows) if belongs(row)]
        results[name] = score([rows[i] for i in chosen], [hypotheses[i] for i in chosen])
    return results


def confidence_interval(rows: list[dict], hypotheses: list[str]) -> dict:
    """IC 95 % de WER y CER con bootstrap por enunciado."""
    generator = np.random.default_rng(SEED)
    samples = generator.integers(0, len(rows), (BOOTSTRAP, len(rows)))
    results = {}
    for name, unit in (("wer", str.split), ("cer", list)):
        references = [unit(row["text"]) for row in rows]
        errors = np.array([edit_distance(r, unit(h)) for r, h in zip(references, hypotheses, strict=True)])
        totals = np.array([len(r) for r in references])
        rates = errors[samples].sum(1) / totals[samples].sum(1)
        results[name] = np.percentile(rates, [2.5, 97.5]).tolist()
    return results


def character_confusions(rows: list[dict], hypotheses: list[str]) -> list[dict]:
    counts: Counter = Counter()
    for row, hypothesis in zip(rows, hypotheses, strict=True):
        reference = row["text"]
        for tag, a, b, c, d in SequenceMatcher(None, reference, hypothesis, autojunk=False).get_opcodes():
            if tag != "equal" and b - a <= 1 and d - c <= 1:
                counts[reference[a:b], hypothesis[c:d]] += 1
    return [{"reference": r, "hypothesis": h, "count": n} for (r, h), n in counts.most_common(15)]


def main() -> None:
    device = get_device()
    val, test = load_rows("val"), load_rows("test")

    comparison, val_outputs = {}, []
    for name in EXPERIMENTS:
        outputs = acoustic_outputs(RUNS / "speech" / f"{name}.pt", val, device)
        comparison[name] = score_groups(val, [greedy_decode(output) for output in outputs])
        if name == "multi":
            val_outputs = outputs
        print(name, {key: comparison[name][key] for key in ("openslr", "tux")}, flush=True)

    lm = CharNGram.load(MODELS / "lm_es.npz")
    # La búsqueda usa la mitad de la validación, intercalada, para acotar el tiempo.
    tuning = list(range(0, len(val), 2))
    grid = []
    for alpha in ALPHAS:
        for beta in BETAS:
            found = [beam_search(val_outputs[i], lm, alpha, beta, BEAM_WIDTH) for i in tuning]
            grid.append({"alpha": alpha, "beta": beta, **score([val[i] for i in tuning], found)})
            print(grid[-1], flush=True)
    chosen = min(grid, key=lambda item: (item["wer"], item["cer"]))
    alpha, beta = chosen["alpha"], chosen["beta"]
    write_json(MODELS / "speech.json", {"alpha": alpha, "beta": beta, "beam_width": BEAM_WIDTH})
    val_beam = [beam_search(output, lm, alpha, beta, BEAM_WIDTH) for output in val_outputs]

    outputs = acoustic_outputs(MODELS / "deepspeech2_es.pt", test, device)
    decoded = {
        "greedy": [greedy_decode(output) for output in outputs],
        "beam_no_lm": [beam_search(output, None, beam_width=BEAM_WIDTH) for output in outputs],
        "beam_lm": [beam_search(output, lm, alpha, beta, BEAM_WIDTH) for output in outputs],
    }

    intervals = {}
    for group in ("openslr", "tux"):
        chosen_rows = [i for i, row in enumerate(test) if (row["source"] == "tux") == (group == "tux")]
        hypotheses = [decoded["beam_lm"][i] for i in chosen_rows]
        intervals[group] = confidence_interval([test[i] for i in chosen_rows], hypotheses)

    by_duration = []
    for low, high in ((0, 3), (3, 5), (5, 7), (7, 12.5)):
        inside = [i for i, row in enumerate(test) if low <= float(row["seconds"]) < high]
        if inside:
            result = score([test[i] for i in inside], [decoded["beam_lm"][i] for i in inside])
            by_duration.append({"from": low, "to": high, **result})

    report = {
        "experiments_val_greedy": comparison,
        "grid": grid,
        "chosen": chosen,
        "val_beam_lm": score_groups(val, val_beam),
        "test": {name: score_groups(test, hypotheses) for name, hypotheses in decoded.items()},
        "test_ci95": intervals,
        "by_duration": by_duration,
        "confusions": character_confusions(test, decoded["greedy"]),
    }
    write_json(REPORTS / "speech.json", report)

    predictions = []
    for index, row in enumerate(test):
        entry = {key: row[key] for key in ("path", "source", "speaker", "seconds", "text")}
        entry.update({name: hypotheses[index] for name, hypotheses in decoded.items()})
        entry["wer"] = f"{wer([row['text']], [decoded['beam_lm'][index]]):.4f}"
        predictions.append(entry)
    write_csv(REPORTS / "speech_predictions.csv", predictions)
    print(report["test"])


if __name__ == "__main__":
    main()
