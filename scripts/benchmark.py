"""Measures latency and speed on the demo machine, on GPU and on CPU.

- Image: preprocessing plus inference, one image at a time.
- Video: decode, preprocess, infer and draw, frame by frame.
- Audio: resampling, spectrogram, network and beam search.

Usage: python scripts/benchmark.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from ecovoz.config import ARCHITECTURES, PROCESSED, REPORTS, SAMPLE_RATE, SAMPLES, SEED
from ecovoz.inference import Transcriber, VisionClassifier
from ecovoz.speech.audio import load_audio
from ecovoz.speech.data import load_rows
from ecovoz.utils import read_json, write_json
from ecovoz.video import classify_video
from ecovoz.vision.data import WasteDataset


def percentiles(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def hardware() -> dict:
    cpu = platform.processor()
    if os.name == "nt":
        query = "(Get-CimInstance Win32_Processor).Name"
        found = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)
        cpu = found.stdout.strip() or cpu
    return {
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }


def main() -> None:
    generator = np.random.default_rng(SEED)
    rows = WasteDataset("test").rows
    images = []
    for index in generator.choice(len(rows), 200, replace=False):
        images.append(np.array(Image.open(PROCESSED / rows[index]["path"]).convert("RGB")))

    speech = [row for row in load_rows("test", "openslr") if float(row["seconds"]) >= 2]
    picked = generator.choice(len(speech), 60, replace=False)
    clips = [load_audio(PROCESSED / speech[index]["path"]) for index in picked]
    joined = np.concatenate(clips)
    video = SAMPLES / "video" / "test_montage.mp4"
    truth = read_json(video.with_suffix(".json"))

    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    report = {"hardware": hardware(), "devices": {}}
    for device in devices:
        result = {"image_ms": {}, "video": {}}
        for arch in ARCHITECTURES:
            classifier = VisionClassifier(arch, device)
            result["image_ms"][arch] = percentiles([classifier.predict(image).milliseconds for image in images])

            started = time.perf_counter()
            stamps, correct, total = [], 0, 0
            # No smoothing and no waiting: measures what the machine can deliver.
            for frame in classify_video(str(video), classifier, 0.5, 0.0, paced=False):
                stamps.append(time.perf_counter())
                for segment in truth:
                    if segment["from_frame"] <= frame.index < segment["to_frame"]:
                        correct += frame.label == segment["label"]
                        total += 1
            gaps = np.diff([started, *stamps]) * 1000
            result["video"][arch] = {
                "frames": len(stamps),
                "fps": len(stamps) / (stamps[-1] - started),
                "frame_ms": percentiles(gaps.tolist()),
                "frame_accuracy": correct / total,
            }
            del classifier

        transcriber = Transcriber(device)
        result["audio_ms"] = percentiles([transcriber.transcribe(clip, SAMPLE_RATE).milliseconds for clip in clips])
        result["audio_seconds_mean"] = float(np.mean([len(clip) / SAMPLE_RATE for clip in clips]))
        result["audio_ms_by_length"] = {}
        for seconds in (3, 5, 10, 20):
            size = seconds * SAMPLE_RATE
            times = [
                transcriber.transcribe(joined[start : start + size], SAMPLE_RATE).milliseconds
                for start in np.linspace(0, len(joined) - size, 10).astype(int)
            ]
            result["audio_ms_by_length"][str(seconds)] = percentiles(times)
        del transcriber
        report["devices"][device] = result
        print(device, result, flush=True)
    write_json(REPORTS / "latency.json", report)


if __name__ == "__main__":
    main()
