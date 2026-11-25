import itertools
import math
from collections import defaultdict

import jiwer
import numpy as np
import pytest
import torch

from ecovoz.inference import split_on_pauses
from ecovoz.speech import text
from ecovoz.speech.audio import to_int16, to_mono_16k, trim_silence
from ecovoz.speech.decoder import CharNGram, beam_search, greedy_decode
from ecovoz.speech.features import LogMel, augment_waveform, spec_augment
from ecovoz.speech.metrics import cer, edit_distance, wer
from ecovoz.speech.model import DeepSpeech2


def test_normalize_keeps_spanish_letters():
    assert text.normalize("¿Qué tal, PINGÜINO ñoño?") == "qué tal pingüino ñoño"


def test_normalize_composes_combining_accents():
    assert text.normalize("acción") == "acción"


@pytest.mark.parametrize("value", ["tengo 3 gatos", "", "...", "日本"])
def test_normalize_rejects_what_cannot_be_transcribed(value):
    assert text.normalize(value) is None


def test_encode_decode_roundtrip():
    assert text.decode(text.encode(text.ALPHABET)) == text.ALPHABET
    assert text.BLANK not in text.encode(text.ALPHABET)
    assert max(text.encode(text.ALPHABET)) == text.VOCAB_SIZE - 1


def test_repeated_characters_need_a_blank():
    assert text.min_ctc_frames(text.encode("calle")) == 6


def test_edit_distance_known_cases():
    assert edit_distance("gato", "gato") == 0
    assert edit_distance("gato", "pato") == 1
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_metrics_match_jiwer():
    references = ["el gato come pescado", "hola mundo", "una sola palabra"]
    hypotheses = ["el pato come", "ola mundo cruel", "una sola palabra"]
    assert wer(references, hypotheses) == pytest.approx(jiwer.wer(references, hypotheses))
    assert cer(references, hypotheses) == pytest.approx(jiwer.cer(references, hypotheses))


def one_hot(indices):
    output = np.full((len(indices), text.VOCAB_SIZE), -20.0)
    output[np.arange(len(indices)), indices] = -1e-4
    return output


def test_decoders_collapse_repeats_and_respect_blanks():
    h, o, ele, a = text.encode("hola")
    path = one_hot([h, h, 0, o, ele, 0, ele, a, a])
    assert greedy_decode(path) == "holla"
    assert beam_search(path) == "holla"


@pytest.mark.parametrize("seed", range(15))
def test_beam_search_finds_the_most_probable_text(seed):
    """Compares against the exhaustive sum over every possible alignment."""
    symbols = [0, *text.encode("ab")]
    probabilities = np.random.default_rng(seed).dirichlet(np.ones(len(symbols)), size=5)
    log_probs = np.full((5, text.VOCAB_SIZE), -60.0)
    log_probs[:, symbols] = np.log(probabilities)

    totals = defaultdict(float)
    for path in itertools.product(range(len(symbols)), repeat=5):
        collapsed = [key for key, _ in itertools.groupby(path)]
        decoded = text.decode([symbols[key] for key in collapsed])
        totals[decoded] += math.prod(probabilities[step, key] for step, key in enumerate(path))
    assert beam_search(log_probs, beam_width=300, prune=-50) == max(totals, key=totals.get)


def test_language_model_breaks_acoustic_ties(tmp_path):
    lm = CharNGram.train(["la casa", "la casa roja", "una casa"] * 3)
    lm.save(tmp_path / "lm.npz")
    restored = CharNGram.load(tmp_path / "lm.npz")
    assert restored.score("la ca", "s") > restored.score("la ca", "z")

    tie = one_hot(text.encode("caza"))
    tie[2, text.encode("zs")] = math.log(0.5)
    assert beam_search(tie, restored, alpha=1.0, beta=0.0) == "casa"


def test_resampling_produces_mono_16k():
    stereo = np.ones((48_000, 2), dtype=np.int16) * 1000
    output = to_mono_16k(stereo, 48_000)
    assert output.shape == (16_000,)
    assert output.dtype == np.float32
    assert to_int16(np.array([2.0, -2.0])).tolist() == [32767, -32767]


def test_trim_silence_removes_both_ends():
    audio = np.zeros(48_000, dtype=np.float32)
    audio[16_000:32_000] = 0.5
    assert 16_000 <= len(trim_silence(audio)) < 23_000


def test_split_on_pauses_preserves_every_sample():
    audio = np.random.default_rng(0).normal(0, 0.1, 16_000 * 27).astype(np.float32)
    pieces = split_on_pauses(audio)
    assert all(len(piece) <= 160_000 for piece in pieces)
    assert np.array_equal(np.concatenate(pieces), audio)


def test_features_do_not_depend_on_padding():
    front = LogMel()
    audio = torch.randn(1, 8000)
    alone, frames = front(audio, torch.tensor([8000]))
    padded, _ = front(torch.cat([audio, torch.zeros(1, 4000)], 1), torch.tensor([8000]))
    count = int(frames[0])
    # The last window touches the edge, so the comparison stops at the frames before it.
    assert torch.allclose(alone[..., : count - 3], padded[..., : count - 3], atol=0.02)


def test_augmentations_keep_shapes_valid():
    torch.manual_seed(0)
    for _ in range(20):
        audio, lengths = augment_waveform(torch.randn(3, 8000), torch.tensor([8000, 7000, 4000]))
        assert int(lengths.max()) <= audio.shape[1]
    features = torch.ones(3, 80, 120)
    assert spec_augment(features, torch.tensor([120, 90, 40])).shape == features.shape


def test_model_output_is_a_distribution_per_step():
    model = DeepSpeech2(rnn_units=16, rnn_layers=1).eval()
    frames = torch.tensor([101, 60])
    output, lengths = model(torch.randn(2, 80, 101), frames)
    assert output.shape == (2, 51, text.VOCAB_SIZE)
    assert lengths.tolist() == [51, 30]
    assert torch.allclose(output.exp().sum(-1), torch.ones(2, 51), atol=1e-4)
