"""Evaluates your own video frame by frame: speed and, if given, accuracy.

Usage:
  python scripts/evaluate_video.py my_video.mp4
  python scripts/evaluate_video.py my_video.mp4 --label plastic
  python scripts/evaluate_video.py my_video.mp4 --segments segments.json

`--label` is for when the whole video shows a single object. `--segments` takes a
list such as [{"from_frame": 0, "to_frame": 90, "label": "glass"}, ...], in the same
format as samples/video/test_montage.json. The classes are: cardboard, glass,
metal, paper, plastic and trash.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ecovoz.config import CLASSES, REPORTS
from ecovoz.inference import VisionClassifier
from ecovoz.utils import write_json
from ecovoz.video import classify_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--arch", default="mobilenet_v3_large", choices=("mobilenet_v3_large", "resnet50"))
    parser.add_argument("--label", choices=CLASSES)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    segments = json.loads(args.segments.read_text(encoding="utf-8")) if args.segments else []
    classifier = VisionClassifier(args.arch, args.device)
    started = time.perf_counter()
    labels, times = [], []
    for frame in classify_video(str(args.video), classifier, momentum=0.0, paced=False):
        labels.append(frame.label)
        times.append(frame.milliseconds)
    seconds = time.perf_counter() - started
    if not labels:
        raise SystemExit("The video has no readable frames")

    def truth(index: int) -> str | None:
        for segment in segments:
            if segment["from_frame"] <= index < segment["to_frame"]:
                return segment["label"]
        return args.label

    expected = [truth(index) for index in range(len(labels))]
    scored = [(got, want) for got, want in zip(labels, expected, strict=True) if want]
    result = {
        "video": args.video.name,
        "model": args.arch,
        "device": str(classifier.device),
        "frames": len(labels),
        "fps": len(labels) / seconds,
        "inference_ms_mean": float(np.mean(times)),
        "predicted": {name: count for name, count in Counter(labels).most_common()},
        "frame_accuracy": sum(got == want for got, want in scored) / len(scored) if scored else None,
        "frames_with_label": len(scored),
    }
    write_json(REPORTS / "own_video" / f"{args.video.stem}_{args.arch}.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
