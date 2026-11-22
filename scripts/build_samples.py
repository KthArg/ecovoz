"""Crea los ejemplos de la demostración, el video de prueba y el subconjunto entregable.

- samples/images y samples/audio: ejemplos al azar (semilla fija) de la partición
  de prueba, sin escoger por el resultado del modelo.
- samples/video/montaje_prueba.mp4: montaje propio con fotos de prueba en movimiento
  y la verdad de cada tramo, para medir exactitud por cuadro y velocidad.
- data/subset: 240 imágenes y los 100 audios del experimento `small100`, ya
  preprocesados, con sus etiquetas.

Uso: python scripts/build_samples.py  (necesita ffmpeg para el video)
"""

from __future__ import annotations

import random
import shutil
import subprocess

import cv2
import numpy as np

from ecovoz.config import CLASSES, DATA, MANIFESTS, PROCESSED, SAMPLES, SEED
from ecovoz.utils import read_csv, write_csv, write_json

WIDTH, HEIGHT, FPS = 1280, 720, 25
SECONDS_PER_PHOTO = 2.5
SUBSET_PER_CELL = {"train": 28, "val": 6, "test": 6}


def render_video(chosen: list[dict]) -> None:
    folder = SAMPLES / "video"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "montaje_prueba.mp4"
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
        "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = int(SECONDS_PER_PHOTO * FPS)
    truth = []
    for number, row in enumerate(chosen):
        photo = cv2.imdecode(np.fromfile(PROCESSED / row["path"], np.uint8), cv2.IMREAD_COLOR)
        height, width = photo.shape[:2]
        fill = max(WIDTH / width, HEIGHT / height)
        background = cv2.resize(photo, None, fx=fill * 1.05, fy=fill * 1.05)[:HEIGHT, :WIDTH]
        background = cv2.GaussianBlur(background, (0, 0), 28)
        for step in range(count):
            progress = step / count
            # Cámara simulada: acercamiento, paneo, giro y cambio de luz.
            scale = HEIGHT / height * (1.0 + 0.18 * progress)
            angle = 4 * np.sin(2 * np.pi * progress + number)
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            matrix[0, 2] += WIDTH / 2 - width / 2 + 45 * np.sin(np.pi * progress)
            matrix[1, 2] += HEIGHT / 2 - height / 2
            frame = cv2.warpAffine(
                photo, matrix, (WIDTH, HEIGHT), dst=background.copy(), borderMode=cv2.BORDER_TRANSPARENT
            )
            frame = cv2.convertScaleAbs(frame, alpha=1 + 0.1 * np.sin(2 * np.pi * progress))
            encoder.stdin.write(frame.tobytes())
        truth.append(
            {
                "from_frame": number * count,
                "to_frame": (number + 1) * count,
                "label": row["label"],
                "source": row["source"],
                "original": row["original"],
            }
        )
    encoder.stdin.close()
    if encoder.wait():
        raise SystemExit("ffmpeg no pudo crear el video")
    write_json(folder / "montaje_prueba.json", truth)


def main() -> None:
    generator = random.Random(SEED)
    images = read_csv(MANIFESTS / "images.csv")
    test = [row for row in images if row["split"] == "test"]

    (SAMPLES / "images").mkdir(parents=True, exist_ok=True)
    chosen = []
    for source in ("trashnet", "realwaste"):
        for label in CLASSES:
            row = generator.choice([r for r in test if r["source"] == source and r["label"] == label])
            shutil.copyfile(PROCESSED / row["path"], SAMPLES / "images" / f"{label}_{source}.jpg")
            chosen.append(row)
    render_video(chosen)

    clips = read_csv(MANIFESTS / "audio.csv")
    (SAMPLES / "audio").mkdir(parents=True, exist_ok=True)
    credits = []
    for corpus in sorted({row["source"] for row in clips} - {"tux"}):
        options = [
            row for row in clips
            if row["source"] == corpus and row["split"] == "test" and float(row["seconds"]) <= 7
        ]
        row = generator.choice(options)
        name = f"{corpus}.wav"
        shutil.copyfile(PROCESSED / row["path"], SAMPLES / "audio" / name)
        credits.append({"file": name, "id": row["id"], "text": row["text"], "license": "CC BY-SA 4.0"})
    write_csv(SAMPLES / "audio" / "transcripciones.csv", credits)

    subset = DATA / "subset"
    image_rows = []
    for label in CLASSES:
        for split, count in SUBSET_PER_CELL.items():
            options = [row for row in images if row["label"] == label and row["split"] == split]
            image_rows += generator.sample(options, count)
    # La misma selección que hace train_speech.py para small100.
    speech = [row for row in clips if row["split"] == "train" and row["source"] != "tux"]
    audio_rows = random.Random(SEED).sample(speech, 100)
    for rows, name in ((image_rows, "images.csv"), (audio_rows, "audio.csv")):
        for row in rows:
            target = subset / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROCESSED / row["path"], target)
        write_csv(subset / name, rows)
    print(f"{len(image_rows)} imágenes y {len(audio_rows)} audios en {subset}")


if __name__ == "__main__":
    main()
