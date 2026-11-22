"""Inferencia de imagen y de voz para la aplicación y la medición de latencia."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ecovoz.config import ARCHITECTURES, MODELS, SAMPLE_RATE
from ecovoz.speech.audio import to_mono_16k, trim_silence
from ecovoz.speech.decoder import CharNGram, beam_search, greedy_decode
from ecovoz.speech.features import LogMel
from ecovoz.speech.model import DeepSpeech2
from ecovoz.utils import get_device, load_checkpoint, read_json
from ecovoz.vision.data import eval_transform
from ecovoz.vision.model import build_model

MAX_SEGMENT_SECONDS = 10.0
MIN_AUDIO_SECONDS = 0.3
SILENCE_RMS = 1e-3


@dataclass(frozen=True)
class Classification:
    classes: tuple[str, ...]
    probabilities: np.ndarray
    milliseconds: float

    @property
    def label(self) -> str:
        return self.classes[int(self.probabilities.argmax())]

    @property
    def confidence(self) -> float:
        return float(self.probabilities.max())


@dataclass(frozen=True)
class Transcription:
    text: str
    greedy: str
    milliseconds: float
    seconds: float


class VisionClassifier:
    """Clasifica una imagen RGB (uint8) con probabilidades calibradas en validación."""

    def __init__(self, arch: str, device: str | torch.device = "auto", models_dir: Path = MODELS) -> None:
        if arch not in ARCHITECTURES:
            raise ValueError(f"Arquitectura desconocida: {arch}")
        self.arch = arch
        self.device = get_device(str(device))
        checkpoint = load_checkpoint(models_dir / f"{arch}.pt")
        self.classes = tuple(checkpoint["classes"])
        self.model = build_model(arch, len(self.classes))
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval().to(self.device)
        self.transform = eval_transform()
        settings_path = models_dir / "vision.json"
        settings = read_json(settings_path) if settings_path.exists() else {}
        self.temperature = settings.get("temperature", {}).get(arch, 1.0)
        # La primera pasada inicializa CUDA y tarda mucho más que las siguientes.
        for _ in range(3):
            self.predict(np.zeros((256, 256, 3), dtype=np.uint8))

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Classification:
        started = time.perf_counter()
        tensor = torch.from_numpy(np.array(image[..., :3])).permute(2, 0, 1)
        logits = self.model(self.transform(tensor)[None].to(self.device))[0] / self.temperature
        probabilities = logits.softmax(0).cpu().numpy()
        return Classification(self.classes, probabilities, (time.perf_counter() - started) * 1000)


def split_on_pauses(audio: np.ndarray, max_seconds: float = MAX_SEGMENT_SECONDS) -> list[np.ndarray]:
    """Corta los audios largos en el punto más silencioso cercano al límite."""
    limit = int(max_seconds * SAMPLE_RATE)
    window = int(0.2 * SAMPLE_RATE)
    pieces = []
    while len(audio) > limit:
        start = limit // 2
        energy = np.convolve(audio[start:limit] ** 2, np.ones(window, dtype=np.float32), mode="valid")
        cut = start + int(energy.argmin()) + window // 2
        pieces.append(audio[:cut])
        audio = audio[cut:]
    pieces.append(audio)
    return pieces


class Transcriber:
    """Convierte audio en texto con DeepSpeech2 y beam search con modelo de lenguaje."""

    def __init__(self, device: str | torch.device = "auto", models_dir: Path = MODELS) -> None:
        self.device = get_device(str(device))
        checkpoint = load_checkpoint(models_dir / "deepspeech2_es.pt")
        self.model = DeepSpeech2(**checkpoint["options"])
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval().to(self.device)
        self.model.rnn.flatten_parameters()
        self.front = LogMel().to(self.device)
        self.lm = CharNGram.load(models_dir / "lm_es.npz")
        settings_path = models_dir / "speech.json"
        settings = read_json(settings_path) if settings_path.exists() else {}
        self.alpha = settings.get("alpha", 0.5)
        self.beta = settings.get("beta", 1.0)
        self.beam_width = settings.get("beam_width", 24)
        self.log_probs(np.zeros(SAMPLE_RATE, dtype=np.float32))

    @torch.inference_mode()
    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(audio.astype(np.float32))[None].to(self.device)
        features, frames = self.front(batch, torch.tensor([len(audio)], device=self.device))
        output, steps = self.model(features, frames)
        return output[0, : int(steps[0])].cpu().numpy()

    def transcribe(self, samples: np.ndarray, rate: int, use_beam: bool = True) -> Transcription:
        started = time.perf_counter()
        audio = to_mono_16k(samples, rate)
        seconds = len(audio) / SAMPLE_RATE
        texts, greedy = [], []
        if seconds >= MIN_AUDIO_SECONDS and np.sqrt(np.mean(audio**2)) >= SILENCE_RMS:
            for piece in split_on_pauses(trim_silence(audio)):
                if len(piece) < MIN_AUDIO_SECONDS * SAMPLE_RATE:
                    continue
                output = self.log_probs(piece)
                greedy.append(greedy_decode(output))
                texts.append(
                    beam_search(output, self.lm, self.alpha, self.beta, self.beam_width) if use_beam else greedy[-1]
                )
        return Transcription(
            " ".join(filter(None, texts)),
            " ".join(filter(None, greedy)),
            (time.perf_counter() - started) * 1000,
            seconds,
        )
