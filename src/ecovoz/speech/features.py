"""Log-mel spectrogram and data augmentation, computed in batches on the device."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ecovoz.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def mel_filterbank() -> torch.Tensor:
    top = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, top, N_MELS + 2) / 2595) - 1)
    bins = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    low, center, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins - low) / (center - low)
    falling = (high - bins) / (high - center)
    return torch.tensor(np.maximum(0, np.minimum(rising, falling)), dtype=torch.float32)


class LogMel(nn.Module):
    """Audio (B, N) to per-utterance normalized log-mel (B, MELS, T).

    The mean and deviation use only the real frames, never the batch padding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("filters", mel_filterbank(), persistent=False)

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor):
        audio = torch.cat([audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]], dim=1)  # pre-emphasis
        inside = torch.arange(audio.shape[1], device=audio.device)[None] < lengths[:, None]
        spectrum = torch.stft(
            audio * inside,
            N_FFT,
            HOP_LENGTH,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        features = torch.log(self.filters @ spectrum.abs().pow(2) + 1e-6)
        frames = lengths // HOP_LENGTH + 1
        steps = torch.arange(features.shape[2], device=audio.device)[None]
        mask = (steps < frames[:, None])[:, None]
        count = frames[:, None, None]
        mean = (features * mask).sum(2, keepdim=True) / count
        variance = ((features - mean).pow(2) * mask).sum(2, keepdim=True) / count
        return (features - mean) / (variance + 1e-5).sqrt() * mask, frames


def augment_waveform(audio: torch.Tensor, lengths: torch.Tensor):
    """Speed change and Gaussian noise with an SNR between 10 and 35 dB."""
    if torch.rand(1).item() < 0.5:
        speed = float(torch.empty(1).uniform_(0.88, 1.12))
        size = int(audio.shape[1] / speed)
        audio = nn.functional.interpolate(audio[:, None], size=size, mode="linear")[:, 0]
        lengths = (lengths / speed).long().clamp(max=size)
    noisy = torch.rand(len(audio), 1, device=audio.device) < 0.4
    snr = torch.empty(len(audio), 1, device=audio.device).uniform_(10, 35)
    rms = (audio.pow(2).sum(1, keepdim=True) / lengths[:, None]).sqrt()
    noise = torch.randn_like(audio) * rms * 10 ** (-snr / 20)
    return audio + noise * noisy, lengths


def spec_augment(
    features: torch.Tensor,
    frames: torch.Tensor,
    freq_masks: int = 2,
    max_freq: int = 15,
    time_masks: int = 5,
    max_time: float = 0.05,
) -> torch.Tensor:
    """SpecAugment (Park et al., 2019): masked bands in frequency and time."""
    batch, mels, steps = features.shape
    device = features.device
    mask = torch.zeros_like(features, dtype=torch.bool)
    for _ in range(freq_masks):
        width = torch.randint(0, max_freq + 1, (batch, 1), device=device)
        start = (torch.rand(batch, 1, device=device) * (mels - width)).long()
        axis = torch.arange(mels, device=device)[None]
        mask |= ((axis >= start) & (axis < start + width))[:, :, None]
    for _ in range(time_masks):
        width = (torch.rand(batch, 1, device=device) * max_time * frames[:, None]).long()
        start = (torch.rand(batch, 1, device=device) * (frames[:, None] - width)).long()
        axis = torch.arange(steps, device=device)[None]
        mask |= ((axis >= start) & (axis < start + width))[:, None, :]
    return features.masked_fill(mask, 0.0)
