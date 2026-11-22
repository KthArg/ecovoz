"""Genera figuras, tablas LaTeX y el resumen a partir de los JSON de reports/.

Ninguna cifra del paper se escribe a mano: salen de paper/generated/.

Uso: python scripts/make_report.py
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from ecovoz.config import ARCHITECTURES, CLASSES, CLASSES_ES, FIGURES, MANIFESTS, PROCESSED, REPORTS, ROOT
from ecovoz.speech.audio import load_audio
from ecovoz.speech.features import LogMel
from ecovoz.utils import read_csv, read_json

GENERATED = ROOT / "paper" / "generated"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#86b6ef", "#2a78d6", "#0d366b"])
COLUMN, WIDE = 3.45, 7.1  # anchos de una y dos columnas IEEE, en pulgadas

ARCH = {"resnet50": "ResNet-50", "mobilenet_v3_large": "MobileNetV3-L"}
STRATEGY = {
    "head": "Solo cabeza",
    "partial_noaug": "Últimos bloques, sin aumento",
    "partial_aug": "Últimos bloques, con aumento",
    "full_aug": "Red completa, con aumento",
}
EXPERIMENT = {
    "small100": "100 audios",
    "small100_aug": "100 audios + aumento",
    "tux": "Solo TUX (preentr.)",
    "small100_transfer": "100 audios + preentr.",
    "multi": "Modelo final",
}
GROUP = {
    "openslr": "Locutores nuevos",
    "tux": "Locutor TUX",
    "all": "Todo",
    "es_co_female": "Colombia, mujer",
    "es_co_male": "Colombia, hombre",
    "es_pe_female": "Perú, mujer",
    "es_pe_male": "Perú, hombre",
    "es_pr_female": "Puerto Rico, mujer",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#b5b4ad",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": "#e8e7e2",
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
    }
)


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(figure)


def number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def percent(value: float, digits: int = 1) -> str:
    return number(100 * value, digits)


def tex(value: str) -> str:
    for char in "&%_#":
        value = value.replace(char, "\\" + char)
    return value


def write_table(name: str, columns: str, header: list[str], rows: list[list[str]]) -> None:
    end = r" \\"
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(header) + end, r"\midrule"]
    lines += [" & ".join(row) + end for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_figures() -> None:
    images = read_csv(MANIFESTS / "images.csv")
    count = Counter((row["source"], row["label"]) for row in images)
    figure, axis = plt.subplots(figsize=(COLUMN, 2.0))
    x = np.arange(len(CLASSES))
    for offset, source, color, name in ((-0.2, "trashnet", BLUE, "TrashNet"), (0.2, "realwaste", ORANGE, "RealWaste")):
        bars = axis.bar(x + offset, [count[source, label] for label in CLASSES], 0.37, color=color, label=name)
        axis.bar_label(bars, fontsize=5.5, color=MUTED)
    axis.set_xticks(x, [CLASSES_ES[label] for label in CLASSES], rotation=20)
    axis.set_ylabel("Imágenes")
    axis.legend(loc="upper left")
    save(figure, "dataset")

    figure, axes = plt.subplots(2, 6, figsize=(WIDE, 2.5))
    for row_axes, source in zip(axes, ("trashnet", "realwaste"), strict=True):
        for axis, label in zip(row_axes, CLASSES, strict=True):
            row = next(r for r in images if r["source"] == source and r["label"] == label and r["split"] == "train")
            axis.imshow(Image.open(PROCESSED / row["path"]).resize((256, 256)))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
            if source == "trashnet":
                axis.set_title(CLASSES_ES[label])
        row_axes[0].set_ylabel({"trashnet": "TrashNet", "realwaste": "RealWaste"}[source])
    figure.tight_layout(pad=0.3)
    save(figure, "dataset_examples")


def vision_figures(report: dict, runs: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.9), sharex=True)
    for arch, color in zip(ARCHITECTURES, (BLUE, ORANGE), strict=True):
        history = runs[arch, report["models"][arch]["strategy"]]["history"]
        epochs = [item["epoch"] for item in history]
        for axis, key in zip(axes, ("loss", "f1"), strict=True):
            axis.plot(epochs, [item[f"train_{key}"] for item in history], "--", color=color, alpha=0.75, lw=1)
            axis.plot(epochs, [item[f"val_{key}"] for item in history], color=color, label=ARCH[arch])
    for axis, title in zip(axes, ("Pérdida", "F1 macro"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Época")
        axis.grid(axis="y")
    axes[0].legend(fontsize=6)
    figure.text(0.5, -0.04, "Continua: validación. Discontinua: entrenamiento.", ha="center", fontsize=6, color=MUTED)
    figure.tight_layout(pad=0.4)
    save(figure, "vision_curves")

    figure, axes = plt.subplots(1, 2, figsize=(WIDE, 2.9))
    names = [CLASSES_ES[name] for name in CLASSES]
    for axis, arch in zip(axes, ARCHITECTURES, strict=True):
        matrix = np.array(report["models"][arch]["test"]["confusion"])
        share = matrix / matrix.sum(1, keepdims=True)
        axis.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, value, ha="center", va="center", color="white" if share[i, j] > 0.45 else INK)
        axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_xlabel("Predicción")
        axis.set_ylabel("Etiqueta real")
        axis.set_title(ARCH[arch])
    figure.tight_layout()
    save(figure, "confusion_matrices")

    arch = report["default"]
    errors = [row for row in read_csv(REPORTS / "vision_predictions.csv") if row[arch] != row["label"]]
    errors.sort(key=lambda row: -float(row[f"{arch}_confidence"]))
    figure, axes = plt.subplots(2, 4, figsize=(COLUMN, 2.35))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), errors, strict=False):
        axis.imshow(Image.open(PROCESSED / row["path"]).resize((256, 256)))
        share = round(100 * float(row[f"{arch}_confidence"]))
        axis.set_title(f"{CLASSES_ES[row['label']]}\n→ {CLASSES_ES[row[arch]]} {share} %", fontsize=5.3)
    figure.tight_layout(pad=0.2)
    save(figure, "vision_errors")


def speech_figures(report: dict, runs: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(WIDE, 3.3))
    for axis, name in zip(axes[0], ("small100", "multi"), strict=True):
        history = runs[name]["history"]
        epochs = [item["epoch"] for item in history]
        axis.plot(epochs, [item["train_loss"] for item in history], color=BLUE, label="Entrenamiento")
        axis.plot(epochs, [item["val_loss"] for item in history], color=ORANGE, label="Validación")
        axis.set_title(f"Pérdida CTC: {EXPERIMENT[name]}")
        axis.legend()
    styles = (("small100", BLUE), ("small100_aug", ORANGE), ("small100_transfer", AQUA))
    for name, color in styles:
        history = runs[name]["history"]
        axes[1, 0].plot(
            [item["epoch"] for item in history],
            [100 * item["val_cer"] for item in history],
            color=color,
            label=EXPERIMENT[name],
        )
    axes[1, 0].set_title("CER en validación con solo 100 audios")
    history = runs["multi"]["history"]
    for key, color, name in (("val_wer", BLUE, "WER"), ("val_cer", ORANGE, "CER")):
        axes[1, 1].plot([item["epoch"] for item in history], [100 * item[key] for item in history], color=color, label=name)
    axes[1, 1].set_title("Error en validación: modelo final")
    for axis in axes[1]:
        axis.set_ylabel("%")
        axis.set_xlabel("Época")
        axis.legend()
    for axis in axes.ravel():
        axis.grid(axis="y")
    figure.tight_layout(pad=0.6)
    save(figure, "speech_curves")

    names = list(EXPERIMENT)
    figure, axis = plt.subplots(figsize=(COLUMN, 2.3))
    y = np.arange(len(names))
    for offset, key, color in ((-0.2, "wer", BLUE), (0.2, "cer", ORANGE)):
        values = [100 * report["experiments_val_greedy"][name]["openslr"][key] for name in names]
        bars = axis.barh(y + offset, values, 0.37, color=color, label=key.upper())
        axis.bar_label(bars, [number(value) for value in values], fontsize=5.5, padding=2)
    axis.set_yticks(y, [EXPERIMENT[name] for name in names])
    axis.invert_yaxis()
    axis.set_xlim(0, 125)
    axis.set_xlabel("Error en validación, locutores nuevos (%)")
    axis.legend(loc="lower right")
    save(figure, "speech_experiments")

    test = read_csv(REPORTS / "speech_predictions.csv")
    values = {
        "Locutor TUX": [100 * float(row["wer"]) for row in test if row["source"] == "tux"],
        "Locutores nuevos": [100 * float(row["wer"]) for row in test if row["source"] != "tux"],
    }
    figure, axis = plt.subplots(figsize=(COLUMN, 1.8))
    for (name, data), color in zip(values.items(), (BLUE, ORANGE), strict=True):
        ordered = np.sort(data)
        axis.step(ordered, 100 * np.arange(1, len(ordered) + 1) / len(ordered), color=color, where="post", label=name)
    axis.set_xlim(0, 100)
    axis.set_xlabel("WER del enunciado (%)")
    axis.set_ylabel("Enunciados acumulados (%)")
    axis.grid(axis="y")
    axis.legend(loc="lower right")
    save(figure, "speech_distribution")

    row = next(r for r in test if r["source"] != "tux" and float(r["seconds"]) < 4.5)
    audio = torch.from_numpy(load_audio(PROCESSED / row["path"]))[None]
    features = LogMel()(audio, torch.tensor([audio.shape[1]]))[0][0].numpy()
    figure, axis = plt.subplots(figsize=(COLUMN, 1.5))
    extent = (0, audio.shape[1] / 16000, 0, features.shape[0])
    axis.imshow(features, origin="lower", aspect="auto", extent=extent, cmap=SEQUENTIAL, vmin=-1.5, vmax=2.5)
    axis.set_xlabel("Segundos")
    axis.set_ylabel("Banda mel")
    axis.set_title(f"«{row['text']}»", fontsize=6.5)
    save(figure, "spectrogram")


def tables(data_images, data_audio, vision, vision_runs, speech, speech_runs, latency) -> None:
    splits = data_images["per_split"]
    rows = []
    for label in CLASSES:
        values = [splits[split][label] for split in ("train", "val", "test")]
        rows.append([CLASSES_ES[label], *map(str, values), str(sum(values))])
    totals = [sum(splits[split].values()) for split in ("train", "val", "test")]
    rows.append([r"\textbf{Total}", *map(str, totals), str(sum(totals))])
    write_table("tab_images", "lrrrr", ["Clase", "Entren.", "Val.", "Prueba", "Total"], rows)

    rows = []
    for split, name in (("train", "Entrenamiento"), ("val", "Validación"), ("test", "Prueba")):
        info = data_audio["splits"][split]
        tux = info["per_source"]["tux"]
        rows.append([name, str(info["clips"] - tux), str(tux), str(info["speakers"]), number(info["hours"])])
    write_table("tab_audio", "lrrrr", ["Partición", "OpenSLR", "TUX", "Locut.", "Horas"], rows)

    rows = []
    for strategy, name in STRATEGY.items():
        row = [name]
        for arch in ARCHITECTURES:
            value = percent(vision_runs[arch, strategy]["best"]["f1"])
            row.append(rf"\textbf{{{value}}}" if vision["models"][arch]["strategy"] == strategy else value)
        rows.append(row)
    write_table("tab_strategies", "lrr", ["Estrategia", *ARCH.values()], rows)

    models = [vision["models"][arch] for arch in ARCHITECTURES]

    def interval(model, key):
        low, high = model["test"][key]
        return f"[{percent(low)}; {percent(high)}]"

    rows = [
        [r"Accuracy (\%)", *[percent(m["test"]["accuracy"]) for m in models]],
        [r"\quad IC 95\,\%", *[interval(m, "accuracy_ci95") for m in models]],
        [r"F1 macro (\%)", *[percent(m["test"]["macro_f1"]) for m in models]],
        [r"\quad IC 95\,\%", *[interval(m, "macro_f1_ci95") for m in models]],
        [r"Precisión macro (\%)", *[percent(m["test"]["macro_precision"]) for m in models]],
        [r"Recall macro (\%)", *[percent(m["test"]["macro_recall"]) for m in models]],
        [r"Accuracy TrashNet (\%)", *[percent(m["test_by_source"]["trashnet"]["accuracy"]) for m in models]],
        [r"Accuracy RealWaste (\%)", *[percent(m["test_by_source"]["realwaste"]["accuracy"]) for m in models]],
        ["Parámetros (millones)", *[number(m["parameters"] / 1e6) for m in models]],
        ["Archivo, float16 (MB)", *[number(m["file_mb"]) for m in models]],
    ]
    write_table("tab_vision", "lrr", ["Métrica", *ARCH.values()], rows)

    rows = []
    for label in CLASSES:
        row = [CLASSES_ES[label]]
        for model in models:
            row += [percent(model["test"]["per_class"][label][key]) for key in ("precision", "recall", "f1")]
        rows.append([*row, str(models[0]["test"]["per_class"][label]["support"])])
    header = ["Clase", "P (ResNet)", "R (ResNet)", "F1 (ResNet)", "P (MobileNet)", "R (MobileNet)", "F1 (MobileNet)", "$n$"]
    write_table("tab_per_class", "lrrrrrrr", header, rows)

    rows = []
    for name, label in EXPERIMENT.items():
        scores, run = speech["experiments_val_greedy"][name], speech_runs[name]
        cells = [percent(scores[group][key]) for group in ("openslr", "tux") for key in ("wer", "cer")]
        rows.append([tex(label), str(run["train_clips"]), *cells])
    write_table("tab_experiments", "lrrrrr", ["Experimento", "$N$", "WER nuevos", "CER nuevos", "WER TUX", "CER TUX"], rows)

    rows = []
    for decoder, label in (("greedy", "Voraz"), ("beam_no_lm", "Beam"), ("beam_lm", "Beam + LM")):
        scores = speech["test"][decoder]
        rows.append([label, *[percent(scores[group][key]) for group in ("openslr", "tux") for key in ("wer", "cer")]])
    write_table("tab_speech", "lrrrr", ["Decodificador", "WER nuevos", "CER nuevos", "WER TUX", "CER TUX"], rows)

    rows = []
    for key, scores in speech["test"]["beam_lm"].items():
        if key != "all":
            cells = [percent(scores[measure]) for measure in ("wer", "cer", "exact")]
            rows.append([GROUP[key], str(scores["n"]), *cells])
    write_table("tab_speakers", "lrrrr", ["Grupo", "$n$", "WER", "CER", "Exactas"], rows)

    test = [row for row in read_csv(REPORTS / "speech_predictions.csv") if row["source"] != "tux"]
    test.sort(key=lambda row: (float(row["wer"]), -len(row["text"]), row["path"]))
    rows = []
    # Posiciones fijas del ranking, del mejor al peor, para no escoger ejemplos a mano.
    for position in (0, 0.15, 0.4, 0.65, 0.85, 0.98):
        row = test[min(len(test) - 1, int(position * len(test)))]
        rows.append([tex(row["text"]), tex(row["beam_lm"]) or "(vacío)", percent(float(row["wer"]), 0)])
    paragraph = r">{\raggedright\arraybackslash}p{0.44\textwidth}"
    write_table("tab_examples", f"{paragraph}{paragraph}r", ["Referencia", "Transcripción", "WER"], rows)

    rows = []
    for device, result in latency["devices"].items():
        label = "GPU" if device == "cuda" else "CPU"
        for arch, name in ARCH.items():
            image = result["image_ms"][arch]
            rows.append([f"Imagen, {name}", label, f"{number(image['p50'])} / {number(image['p95'])} ms"])
        for arch, name in ARCH.items():
            rows.append([f"Video 720p, {name}", label, f"{number(result['video'][arch]['fps'])} cuadros/s"])
        for seconds in ("5", "10"):
            audio = result["audio_ms_by_length"][seconds]
            value = f"{number(audio['p50'] / 1000, 2)} / {number(audio['p95'] / 1000, 2)} s"
            rows.append([f"Audio de {seconds} s", label, value])
    write_table("tab_latency", "llr", ["Tarea", "Equipo", "Mediana / p95"], rows)


def macros(data_images, data_audio, vision, speech, speech_runs, latency) -> None:
    resnet, mobile = (vision["models"][arch] for arch in ARCHITECTURES)
    test, slow = speech["test"], latency["devices"]["cpu"]
    fast = latency["devices"].get("cuda", slow)
    best = vision["models"][vision["default"]]
    p_value = vision["mcnemar"]["p_value"]
    values = {
        "ImgTotal": str(data_images["total"]),
        "ImgGroups": str(data_images["groups"]),
        "ImgInGroups": str(data_images["images_in_groups"]),
        "AudioTotal": str(sum(part["clips"] for part in data_audio["splits"].values())),
        "AudioTrain": str(data_audio["splits"]["train"]["clips"]),
        "AudioHours": number(sum(part["hours"] for part in data_audio["splits"].values())),
        "AudioLeak": str(data_audio["dropped_to_avoid_leakage"]),
        "ResAcc": percent(resnet["test"]["accuracy"]),
        "ResF": percent(resnet["test"]["macro_f1"]),
        "MobAcc": percent(mobile["test"]["accuracy"]),
        "MobF": percent(mobile["test"]["macro_f1"]),
        "DefaultModel": ARCH[vision["default"]],
        "BestTemp": number(best["temperature"], 2),
        "EceBefore": percent(best["ece_before"]),
        "EceAfter": percent(best["ece_after"]),
        "RejectCoverage": percent(best["rejection"][2]["coverage"]),
        "RejectAcc": percent(best["rejection"][2]["accuracy"]),
        "McP": "< 0,001" if p_value < 0.001 else "= " + number(p_value, 3),
        "McA": str(vision["mcnemar"]["only_resnet50_right"]),
        "McB": str(vision["mcnemar"]["only_mobilenet_right"]),
        "SlrWer": percent(test["beam_lm"]["openslr"]["wer"]),
        "SlrCer": percent(test["beam_lm"]["openslr"]["cer"]),
        "SlrWerGreedy": percent(test["greedy"]["openslr"]["wer"]),
        "SlrWerLow": percent(speech["test_ci95"]["openslr"]["wer"][0]),
        "SlrWerHigh": percent(speech["test_ci95"]["openslr"]["wer"][1]),
        "TuxWer": percent(test["beam_lm"]["tux"]["wer"]),
        "TuxCer": percent(test["beam_lm"]["tux"]["cer"]),
        "Alpha": number(speech["chosen"]["alpha"]),
        "Beta": number(speech["chosen"]["beta"]),
        "SpeechParams": number(speech_runs["multi"]["parameters"] / 1e6),
        "SpeechBestEpoch": str(speech_runs["multi"]["best"]["epoch"]),
        "VideoFps": number(fast["video"][vision["default"]]["fps"]),
        "VideoFpsCpu": number(slow["video"][vision["default"]]["fps"]),
        "VideoAcc": percent(fast["video"][vision["default"]]["frame_accuracy"]),
        "AudioLat": number(fast["audio_ms"]["p95"] / 1000, 2),
        "AudioLatCpu": number(slow["audio_ms"]["p95"] / 1000, 2),
        "HwCpu": tex(latency["hardware"]["cpu"].strip()),
        "HwGpu": tex(latency["hardware"]["gpu"] or "sin GPU"),
    }
    prefixes = ("Scratch", "Aug", "Tux", "Transfer", "Multi")
    for name, key in zip(prefixes, EXPERIMENT, strict=True):
        scores = speech["experiments_val_greedy"][key]["openslr"]
        for key_name, metric in ((f"Exp{name}Wer", "wer"), (f"Exp{name}Cer", "cer")):
            if key_name in values:
                raise ValueError(f"Macro repetida: {key_name}")
            values[key_name] = percent(scores[metric])
    text = "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())
    (GENERATED / "numbers.tex").write_text(text, encoding="utf-8")


def summary(vision, speech, latency) -> None:
    lines = [
        "## Visión (partición de prueba)",
        "",
        "| Modelo | Accuracy | F1 macro |",
        "|---|---:|---:|",
    ]
    for arch in ARCHITECTURES:
        test = vision["models"][arch]["test"]
        lines.append(f"| {ARCH[arch]} | {percent(test['accuracy'])} % | {percent(test['macro_f1'])} % |")
    lines += ["", "## Voz (partición de prueba)", "", "| Grupo | Decodificador | WER | CER |", "|---|---|---:|---:|"]
    for group in ("openslr", "tux"):
        for decoder, name in (("greedy", "voraz"), ("beam_lm", "beam + LM")):
            scores = speech["test"][decoder][group]
            lines.append(f"| {GROUP[group]} | {name} | {percent(scores['wer'])} % | {percent(scores['cer'])} % |")
    lines += [
        "",
        "## Latencia en el equipo de la demostración",
        "",
        "| Equipo | Modelo | Imagen (ms, mediana) | Video (cuadros/s) | Audio (s, p95) |",
        "|---|---|---:|---:|---:|",
    ]
    for device, result in latency["devices"].items():
        for arch in ARCHITECTURES:
            cells = [
                "GPU" if device == "cuda" else "CPU",
                ARCH[arch],
                number(result["image_ms"][arch]["p50"]),
                number(result["video"][arch]["fps"]),
                number(result["audio_ms"]["p95"] / 1000, 2),
            ]
            lines.append(f"| {' | '.join(cells)} |")
    hardware = latency["hardware"]
    lines += ["", f"Equipo: {hardware['cpu'].strip()}; {hardware['gpu']}.", ""]
    (REPORTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    vision_runs = {}
    for path in (REPORTS / "vision_runs").glob("*.json"):
        run = read_json(path)
        vision_runs[run["arch"], run["strategy"]] = run
    speech_runs = {path.stem: read_json(path) for path in (REPORTS / "speech_runs").glob("*.json")}
    data_images, data_audio, vision, speech, latency = (
        read_json(REPORTS / f"{name}.json") for name in ("data_images", "data_audio", "vision", "speech", "latency")
    )
    dataset_figures()
    vision_figures(vision, vision_runs)
    speech_figures(speech, speech_runs)
    tables(data_images, data_audio, vision, vision_runs, speech, speech_runs, latency)
    macros(data_images, data_audio, vision, speech, speech_runs, latency)
    summary(vision, speech, latency)
    print("Figuras, tablas y resumen generados")


if __name__ == "__main__":
    main()
