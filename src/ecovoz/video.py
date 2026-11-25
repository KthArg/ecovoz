"""Frame-by-frame video classification."""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from ecovoz.config import CLASS_LABELS
from ecovoz.inference import VisionClassifier

UNSURE = "not sure"
DISPLAY_WIDTH = 720


@dataclass(frozen=True)
class FrameResult:
    index: int
    image: np.ndarray  # RGB, with the label drawn on it
    label: str  # raw prediction for this frame
    confidence: float
    milliseconds: float  # inference only
    fps: float  # frames processed per second, end to end


def annotate(image: np.ndarray, label: str, confidence: float, min_confidence: float, note: str = "") -> np.ndarray:
    height, width = image.shape[:2]
    if width != DISPLAY_WIDTH:
        image = cv2.resize(image, (DISPLAY_WIDTH, round(height * DISPLAY_WIDTH / width)))
    else:
        image = image.copy()
    sure = confidence >= min_confidence
    text = f"{CLASS_LABELS[label] if sure else UNSURE}  {confidence:.0%}"
    # OpenCV fonts have no accented characters.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    color = (22, 110, 70) if sure else (110, 110, 110)
    cv2.rectangle(image, (0, 0), (image.shape[1], 46), color, -1)
    cv2.putText(image, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    if note:
        position = (image.shape[1] - 290, 30)
        cv2.putText(image, note, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


class Smoother:
    """Exponential moving average of the probabilities, so that the label does not flicker."""

    def __init__(self, momentum: float = 0.6) -> None:
        self.momentum = momentum
        self.state: np.ndarray | None = None

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        if self.state is None or not self.momentum:
            self.state = probabilities
        else:
            self.state = self.momentum * self.state + (1 - self.momentum) * probabilities
        return self.state


def classify_video(
    path: str,
    classifier: VisionClassifier,
    min_confidence: float = 0.5,
    momentum: float = 0.6,
    paced: bool = True,
) -> Iterator[FrameResult]:
    """Classifies every frame. With `paced` it never runs faster than the original video."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the video")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if 1 <= source_fps <= 120 else 25.0
        smoother = Smoother(momentum)
        started = time.perf_counter()
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = classifier.predict(rgb)
            smoothed = smoother.update(result.probabilities)
            fps = (index + 1) / (time.perf_counter() - started)
            label = result.classes[int(smoothed.argmax())]
            image = annotate(rgb, label, float(smoothed.max()), min_confidence, f"{fps:.1f} fps")
            if paced:
                wait = started + index / source_fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            yield FrameResult(index, image, result.label, result.confidence, result.milliseconds, fps)
            index += 1
    finally:
        capture.release()
