"""Lectura de audio y conversión a un formato uniforme: mono, 16 kHz, PCM de 16 bits."""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ecovoz.config import SAMPLE_RATE


def to_mono_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    samples = np.asarray(samples)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / float(np.iinfo(samples.dtype).max + 1)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    if rate != SAMPLE_RATE and samples.size:
        divisor = gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    return samples.astype(np.float32)


def load_audio(source) -> np.ndarray:
    samples, rate = sf.read(source, dtype="float32")
    return to_mono_16k(samples, rate)


def trim_silence(samples: np.ndarray, top_db: float = 40.0, pad: float = 0.15) -> np.ndarray:
    """Recorta el silencio inicial y final con la energía por ventanas de 20 ms."""
    size = int(0.02 * SAMPLE_RATE)
    count = len(samples) // size
    if count < 2:
        return samples
    rms = np.sqrt((samples[: count * size].reshape(count, size) ** 2).mean(axis=1) + 1e-12)
    voiced = np.flatnonzero(rms > rms.max() * 10 ** (-top_db / 20))
    if not voiced.size:
        return samples
    margin = int(pad * SAMPLE_RATE)
    return samples[max(0, voiced[0] * size - margin) : (voiced[-1] + 1) * size + margin]


def to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1, 1) * 32767).round().astype(np.int16)
