"""DeepSpeech2: the architecture from class 12, written in PyTorch.

Two 2D convolutions (11x41 and 11x21 in time x frequency, 32 filters), stacked
bidirectional GRUs and a dense layer with log-softmax output for CTC.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ecovoz.config import N_MELS
from ecovoz.speech.text import VOCAB_SIZE


class DeepSpeech2(nn.Module):
    def __init__(self, rnn_units: int = 512, rnn_layers: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.options = {"rnn_units": rnn_units, "rnn_layers": rnn_layers, "dropout": dropout}
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            32 * ((N_MELS + 3) // 4),
            rnn_units,
            rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * rnn_units),
            nn.Linear(2 * rnn_units, rnn_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_units, VOCAB_SIZE),
        )

    @staticmethod
    def output_lengths(frames):
        return (frames + 1) // 2

    def forward(self, features: torch.Tensor, frames: torch.Tensor):
        x = self.conv(features[:, None])
        x = x.flatten(1, 2).transpose(1, 2)
        lengths = self.output_lengths(frames)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=x.shape[1])
        # float32 outside autocast: CTC is unstable in half precision.
        return self.head(x).float().log_softmax(-1), lengths
