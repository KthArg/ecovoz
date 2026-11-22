"""Entrena DeepSpeech2 con pérdida CTC y sus experimentos de control.

Experimentos, en orden:
- small100:          100 audios latinoamericanos, desde cero, sin aumento.
- small100_aug:      los mismos 100 audios, con aumento de datos.
- tux:               preentrenamiento con el dataset de la clase 12 (un locutor).
- small100_transfer: los mismos 100 audios, partiendo de `tux`.
- multi:             modelo final: todo el entrenamiento, partiendo de `tux`.

Se guarda la época con menor CER en validación. Este script nunca lee la prueba.

Uso: python scripts/train_speech.py [--experiment E]
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch

from ecovoz.config import MODELS, REPORTS, RUNS, SEED
from ecovoz.speech.data import load_rows, make_loader
from ecovoz.speech.decoder import CharNGram, greedy_decode
from ecovoz.speech.features import LogMel, augment_waveform, spec_augment
from ecovoz.speech.metrics import cer, wer
from ecovoz.speech.model import DeepSpeech2
from ecovoz.utils import get_device, load_checkpoint, save_half_precision, set_seed, write_json

SMALL = {"rnn_units": 256, "rnn_layers": 3, "dropout": 0.1}
LARGE = {"rnn_units": 512, "rnn_layers": 5, "dropout": 0.2}

EXPERIMENTS = {
    "small100": {"model": SMALL, "train": "openslr", "limit": 100, "epochs": 150, "lr": 1e-3},
    "small100_aug": {"model": SMALL, "train": "openslr", "limit": 100, "epochs": 150, "lr": 1e-3, "augment": True},
    "tux": {"model": LARGE, "train": "tux", "select": "tux", "epochs": 35, "lr": 4e-4, "augment": True},
    "small100_transfer": {
        "model": LARGE, "train": "openslr", "limit": 100, "epochs": 60, "lr": 1e-4, "augment": True, "init": "tux",
    },
    "multi": {
        "model": LARGE, "train": "all", "select": "all", "epochs": 50, "lr": 4e-4, "augment": True, "init": "tux",
    },
}


@torch.inference_mode()
def validate(model, front, loader, device) -> dict:
    model.eval()
    rows = loader.dataset.rows
    references, hypotheses, total = [], [], 0.0
    for audio, lengths, targets, target_lengths, indices in loader:
        features, frames = front(audio.to(device).float() / 32768, lengths.to(device))
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            log_probs, steps = model(features, frames)
        total += torch.nn.functional.ctc_loss(
            log_probs.transpose(0, 1), targets.to(device), steps, target_lengths.to(device),
            reduction="none", zero_infinity=True,
        ).div(target_lengths.to(device)).sum().item()
        for output, length, index in zip(log_probs.cpu().numpy(), steps.tolist(), indices, strict=True):
            hypotheses.append(greedy_decode(output[:length]))
            references.append(rows[index]["text"])
    return {
        "loss": total / len(rows),
        "wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses),
        "example": {"reference": references[0], "hypothesis": hypotheses[0]},
    }


def train(name: str, device: torch.device) -> None:
    set_seed(SEED)
    options = EXPERIMENTS[name]
    rows = load_rows("train", options["train"])
    limit = options.get("limit")
    if limit:
        rows = random.Random(SEED).sample(rows, limit)
    selection = options.get("select", "openslr")
    val_rows = load_rows("val", selection)

    small = limit is not None
    train_loader = make_loader(rows, shuffle=True, workers=0 if small else 2, batch_size=10 if small else 32)
    val_loader = make_loader(val_rows, workers=0)

    model = DeepSpeech2(**options["model"]).to(device)
    if options.get("init"):
        model.load_state_dict(load_checkpoint(RUNS / "speech" / f"{options['init']}.pt")["state"])
    front = LogMel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    epochs = options["epochs"]
    total_steps = epochs * len(train_loader)
    warmup = 2 * len(train_loader)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * step / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        # Meseta durante la primera mitad y descenso coseno en la segunda.
        progress = max(0.0, 2 * progress - 1)
        return 0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    augment = options.get("augment", False)
    # Validar cada época no aporta cuando una época son diez lotes.
    every = 5 if small else 1
    history = []
    best = {"cer": float("inf")}
    target = RUNS / "speech" / f"{name}.pt"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = torch.zeros((), device=device), 0
        for audio, lengths, targets, target_lengths, _ in train_loader:
            audio = audio.to(device, non_blocking=True).float() / 32768
            lengths = lengths.to(device)
            if augment:
                audio, lengths = augment_waveform(audio, lengths)
            features, frames = front(audio, lengths)
            if augment:
                features = spec_augment(features, frames)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                log_probs, steps = model(features, frames)
            target_lengths = target_lengths.to(device)
            per_clip = torch.nn.functional.ctc_loss(
                log_probs.transpose(0, 1), targets.to(device), steps, target_lengths,
                reduction="none", zero_infinity=True,
            ) / target_lengths
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(per_clip.mean()).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += per_clip.detach().sum()
            seen += len(per_clip)

        if epoch % every and epoch not in (1, epochs):
            continue
        checked = validate(model, front, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": total.item() / seen,
            "val_loss": checked["loss"],
            "val_wer": checked["wer"],
            "val_cer": checked["cer"],
            "example": checked["example"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(entry)
        print(
            f"{name} {epoch:03d}/{epochs} loss={entry['train_loss']:.3f} val={entry['val_loss']:.3f} "
            f"cer={entry['val_cer']:.4f} wer={entry['val_wer']:.4f} t={entry['seconds']}",
            flush=True,
        )
        if checked["cer"] < best["cer"]:
            best = {"epoch": epoch, "cer": checked["cer"], "wer": checked["wer"]}
            save_half_precision({"options": model.options, "state": model.state_dict()}, target)
        write_json(
            REPORTS / "speech_runs" / f"{name}.json",
            {
                "experiment": name,
                "options": options,
                "train_clips": len(rows),
                "val_clips": len(val_rows),
                "val_corpus": selection,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best": best,
                "history": history,
            },
        )

    if name == "multi":
        MODELS.mkdir(exist_ok=True)
        save_half_precision(load_checkpoint(target), MODELS / "deepspeech2_es.pt")
        CharNGram.train(row["text"] for row in load_rows("train")).save(MODELS / "lm_es.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args()
    device = get_device()
    for name in [args.experiment] if args.experiment else EXPERIMENTS:
        train(name, device)


if __name__ == "__main__":
    main()
