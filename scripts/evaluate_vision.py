"""Evalúa los dos clasificadores finales.

La validación se usa para calibrar la confianza (temperature scaling) y elegir el
modelo por defecto. La prueba se usa solo para reportar.

Uso: python scripts/evaluate_vision.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ecovoz.config import ARCHITECTURES, CLASSES, MODELS, REPORTS, SEED
from ecovoz.utils import get_device, load_checkpoint, write_csv, write_json
from ecovoz.vision.data import WasteDataset
from ecovoz.vision.model import build_model

BOOTSTRAP = 2000


@torch.no_grad()
def collect_logits(model, dataset: WasteDataset, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    outputs = [model(images.to(device)).float().cpu() for images, _ in DataLoader(dataset, batch_size=64)]
    return torch.cat(outputs), np.array(dataset.labels)


def fit_temperature(logits: torch.Tensor, labels: np.ndarray) -> float:
    log_temperature = torch.zeros(1, requires_grad=True)
    target = torch.tensor(labels)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp())


def calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for current in range(bins):
        inside = index == current
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    every = range(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=every, zero_division=0)
    return {
        "n": len(labels),
        "accuracy": float((predicted == labels).mean()),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
        "confusion": confusion_matrix(labels, predicted, labels=every).tolist(),
    }


def main() -> None:
    device = get_device()
    val, test = WasteDataset("val"), WasteDataset("test")
    results, temperatures, test_predictions, test_confidence = {}, {}, {}, {}

    for arch in ARCHITECTURES:
        checkpoint = load_checkpoint(MODELS / f"{arch}.pt")
        model = build_model(arch, len(CLASSES))
        model.load_state_dict(checkpoint["state"])
        model.eval().to(device)

        val_logits, val_labels = collect_logits(model, val, device)
        temperature = fit_temperature(val_logits, val_labels)
        test_logits, labels = collect_logits(model, test, device)
        probabilities = (test_logits / temperature).softmax(1).numpy()
        predicted = probabilities.argmax(1)
        confidence = probabilities.max(1)

        sources = np.array([row["source"] for row in test.rows])
        result = {
            "strategy": checkpoint["strategy"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "file_mb": (MODELS / f"{arch}.pt").stat().st_size / 1e6,
            "temperature": temperature,
            "val": metrics(val_logits.argmax(1).numpy(), val_labels),
            "test": metrics(predicted, labels),
            "test_by_source": {
                name: metrics(predicted[sources == name], labels[sources == name]) for name in sorted(set(sources))
            },
            "ece_before": calibration_error(test_logits.softmax(1).numpy(), labels),
            "ece_after": calibration_error(probabilities, labels),
            "rejection": [],
        }
        for threshold in (0.0, 0.5, 0.7, 0.9):
            kept = confidence >= threshold
            accuracy = float((predicted[kept] == labels[kept]).mean())
            result["rejection"].append({"threshold": threshold, "coverage": float(kept.mean()), "accuracy": accuracy})

        generator = np.random.default_rng(SEED)
        samples = [generator.integers(0, len(labels), len(labels)) for _ in range(BOOTSTRAP)]
        accuracy = [(predicted[sample] == labels[sample]).mean() for sample in samples]
        macro = [f1_score(labels[sample], predicted[sample], average="macro") for sample in samples]
        result["test"]["accuracy_ci95"] = np.percentile(accuracy, [2.5, 97.5]).tolist()
        result["test"]["macro_f1_ci95"] = np.percentile(macro, [2.5, 97.5]).tolist()

        results[arch] = result
        temperatures[arch] = temperature
        test_predictions[arch], test_confidence[arch] = predicted, confidence

    first, second = ARCHITECTURES
    labels = np.array(test.labels)
    first_right, second_right = test_predictions[first] == labels, test_predictions[second] == labels
    only_first = int((first_right & ~second_right).sum())
    only_second = int((~first_right & second_right).sum())
    disagreements = only_first + only_second
    mcnemar = {
        "only_resnet50_right": only_first,
        "only_mobilenet_right": only_second,
        "both_wrong": int((~first_right & ~second_right).sum()),
        "p_value": float(binomtest(only_first, disagreements, 0.5).pvalue) if disagreements else 1.0,
    }

    default = max(ARCHITECTURES, key=lambda arch: results[arch]["val"]["macro_f1"])
    write_json(MODELS / "vision.json", {"default": default, "temperature": temperatures, "classes": list(CLASSES)})
    write_json(REPORTS / "vision.json", {"default": default, "models": results, "mcnemar": mcnemar})

    rows = []
    for index, row in enumerate(test.rows):
        entry = {key: row[key] for key in ("path", "original", "source", "label")}
        for arch in ARCHITECTURES:
            entry[arch] = CLASSES[test_predictions[arch][index]]
            entry[f"{arch}_confidence"] = f"{test_confidence[arch][index]:.4f}"
        rows.append(entry)
    write_csv(REPORTS / "vision_predictions.csv", rows)

    for arch, result in results.items():
        print(arch, result["strategy"], "val", result["val"]["macro_f1"], "test", result["test"])
    print("McNemar", mcnemar)


if __name__ == "__main__":
    main()
