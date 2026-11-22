"""Pruebas de los manejadores de la aplicación con los modelos reales, en CPU."""

import importlib.util

import numpy as np
import pytest
import soundfile as sf

from ecovoz.config import ROOT, SAMPLES


@pytest.fixture(scope="module")
def handlers():
    spec = importlib.util.spec_from_file_location("ecovoz_app", ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.build_interface(module.Services("cpu"))
    return {function.fn.__name__: function.fn for function in demo.fns.values() if function.fn}


def test_image_handler(handlers):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    scores, message = handlers["classify_image"](image, "mobilenet_v3_large", 0.0)
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-3)
    assert "ms" in message


def test_video_handler_yields_every_frame(handlers):
    path = str(SAMPLES / "video" / "montaje_prueba.mp4")
    frames = handlers["play_video"](path, "mobilenet_v3_large", 0.5, 0.6)
    first, second = next(frames), next(frames)
    assert first[0].shape[1] == 720
    assert "**2**" in second[1]


def test_live_handler_keeps_its_state_between_frames(handlers):
    live = handlers["classify_live"]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, first_note, state = live(image, "mobilenet_v3_large", 0.5, None)
    _, second_note, following = live(image, "mobilenet_v3_large", 0.5, state)
    assert "cuadros/s" not in first_note
    assert "cuadros/s" in second_note
    assert following[1] is state[1]
    _, _, changed = live(image, "resnet50", 0.5, following)
    assert changed[1] is not state[1]


def test_audio_handler(handlers):
    samples, rate = sf.read(SAMPLES / "audio" / "es_co_male.wav", dtype="int16")
    text, raw, _ = handlers["transcribe"]((rate, samples), "beam")
    assert "referencia" in text
    assert raw


def test_silence_returns_no_speech(handlers):
    text, _, _ = handlers["transcribe"]((16000, np.zeros(16000, dtype=np.int16)), "beam")
    assert "no se detectó voz" in text
