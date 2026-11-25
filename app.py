"""EcoVoz: image, video and audio in a single interface.

Usage: python app.py [--device auto|cuda|cpu] [--port 7860] [--open]

Everything runs on this computer: the application does not send images or audio to the Internet.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from ecovoz.config import ARCHITECTURES, CLASS_LABELS, FIGURES, MODELS, REPORTS, SAMPLES
from ecovoz.inference import Transcriber, VisionClassifier
from ecovoz.utils import get_device, read_json
from ecovoz.video import Smoother, annotate, classify_video

MODEL_NAMES = {"resnet50": "ResNet-50", "mobilenet_v3_large": "MobileNetV3-Large"}
MAX_AUDIO_SECONDS = 60

BINS = {
    "cardboard": "Paper and cardboard bin: dry and flattened.",
    "paper": "Paper and cardboard bin: dry and grease-free.",
    "glass": "Glass bin: rinsed, without the lid.",
    "metal": "Metals bin: rinsed.",
    "plastic": "Plastics bin: clean and dry.",
    "trash": "Non-recyclable waste.",
}
CSS = """
.gradio-container {max-width: 1120px !important; margin: auto}
footer {display: none !important}
"""


class Services:
    """Loads each model only once."""

    def __init__(self, device: str) -> None:
        self.device = str(get_device(device))
        self.classifier = lru_cache(maxsize=None)(lambda arch: VisionClassifier(arch, self.device))
        # The submission ZIP leaves out ResNet-50 because of the 80 MB limit; the repository has both.
        self.architectures = [arch for arch in ARCHITECTURES if (MODELS / f"{arch}.pt").exists()]
        if not self.architectures:
            raise SystemExit(f"No vision models found in {MODELS}")
        for arch in self.architectures:
            self.classifier(arch)
        self.transcriber = Transcriber(self.device)


def build_interface(services: Services) -> gr.Blocks:
    settings_path = MODELS / "vision.json"
    settings = read_json(settings_path) if settings_path.exists() else {}
    default_arch = settings.get("default")
    if default_arch not in services.architectures:
        default_arch = services.architectures[0]

    def classify_image(image, arch, threshold):
        if image is None:
            return None, ""
        result = services.classifier(arch).predict(image)
        scores = {
            CLASS_LABELS[name]: float(value)
            for name, value in zip(result.classes, result.probabilities, strict=True)
        }
        if result.confidence >= threshold:
            detail = f"### {CLASS_LABELS[result.label].capitalize()}\n{BINS[result.label]}"
        else:
            detail = "### Not sure\nMove the object closer, use a plain background and improve the lighting."
        return scores, f"{detail}\n\nInference: **{result.milliseconds:.1f} ms** ({services.device})"

    def play_video(path, arch, threshold, momentum):
        if not path:
            raise gr.Error("Upload a video or choose an example.")
        times = []
        try:
            for frame in classify_video(path, services.classifier(arch), threshold, momentum):
                times.append(frame.milliseconds)
                status = (
                    f"Frame **{frame.index + 1}** | **{frame.fps:.1f}** fps | "
                    f"mean inference **{np.mean(times):.1f} ms** | required minimum: 5 fps"
                )
                yield frame.image, status
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def classify_live(image, arch, threshold, state):
        if image is None:
            return None, "", None
        now = time.perf_counter()
        if state and state[0] == arch:
            _, smoother, previous = state
        else:
            smoother, previous = Smoother(), None
        result = services.classifier(arch).predict(image)
        smoothed = smoother.update(result.probabilities)
        label = result.classes[int(smoothed.argmax())]
        note = f"Inference **{result.milliseconds:.1f} ms**"
        if previous:
            note += f" | **{1 / max(now - previous, 1e-6):.1f}** fps"
        shown = annotate(image, label, float(smoothed.max()), threshold)
        return shown, note, (arch, smoother, now)

    def transcribe(audio, decoding):
        if audio is None:
            raise gr.Error("Record or upload an audio clip.")
        rate, samples = audio
        if len(samples) > MAX_AUDIO_SECONDS * rate:
            raise gr.Error(f"Use clips of up to {MAX_AUDIO_SECONDS} seconds.")
        result = services.transcriber.transcribe(samples, rate, decoding == "beam")
        text = result.text or "(no speech detected)"
        detail = (
            f"**{result.seconds:.1f} s** of audio, transcribed in **{result.milliseconds / 1000:.2f} s** "
            f"({services.device}). Required maximum: 5 s."
        )
        return text, result.greedy, detail

    model_choices = [(MODEL_NAMES[arch], arch) for arch in services.architectures]

    with gr.Blocks(title="EcoVoz", analytics_enabled=False) as demo:
        gr.Markdown(
            "# EcoVoz\nWaste classification in images and video, and Spanish speech transcription. "
            f"Everything is processed on this computer ({services.device})."
        )
        with gr.Row():
            arch = gr.Dropdown(model_choices, value=default_arch, label="Vision model")
            threshold = gr.Slider(0, 0.95, value=0.5, step=0.05, label="Minimum confidence to answer")

        with gr.Tab("Image"):
            with gr.Row():
                image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="Image", height=360)
                with gr.Column():
                    verdict = gr.Markdown()
                    scores = gr.Label(num_top_classes=6, label="Probabilities")
            examples = sorted((SAMPLES / "images").glob("*.jpg"))
            if examples:
                gr.Examples(
                    [str(path) for path in examples],
                    inputs=image,
                    examples_per_page=12,
                    label="Examples from the test split",
                )
            for trigger in (image.change, arch.change, threshold.release):
                trigger(classify_image, [image, arch, threshold], [scores, verdict], show_progress="hidden")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(sources=["upload"], label="Video", height=300)
                    momentum = gr.Slider(0, 0.9, value=0.6, step=0.1, label="Smoothing between frames")
                    with gr.Row():
                        play = gr.Button("Classify frame by frame", variant="primary")
                        stop = gr.Button("Stop")
                    videos = sorted((SAMPLES / "video").glob("*.mp4"))
                    if videos:
                        gr.Examples([str(path) for path in videos], inputs=video, label="Example videos")
                with gr.Column():
                    screen = gr.Image(label="Result", show_label=False, format="jpeg", height=420, elem_id="video-output")
                    video_status = gr.Markdown()
            playing = play.click(
                play_video, [video, arch, threshold, momentum], [screen, video_status], show_progress="hidden"
            )
            stop.click(None, cancels=[playing])

        with gr.Tab("Live camera"):
            with gr.Row():
                camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera", height=360)
                with gr.Column():
                    live = gr.Image(label="Result", show_label=False, buttons=[], format="jpeg", height=420)
                    live_status = gr.Markdown()
            live_state = gr.State()
            camera.stream(
                classify_live,
                [camera, arch, threshold, live_state],
                [live, live_status, live_state],
                stream_every=0.1,
                time_limit=300,
                concurrency_limit=1,
                show_progress="hidden",
            )

        with gr.Tab("Audio"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(
                        sources=["microphone", "upload"], type="numpy", label="Record or upload audio (Spanish)"
                    )
                    decoding = gr.Radio(
                        [("Beam search with language model", "beam"), ("Greedy", "greedy")],
                        value="beam",
                        label="Decoding",
                    )
                    listen = gr.Button("Transcribe", variant="primary")
                    clips = sorted((SAMPLES / "audio").glob("*.wav"))
                    if clips:
                        gr.Examples([str(path) for path in clips], inputs=audio, label="Examples from the test split")
                with gr.Column():
                    text = gr.Textbox(label="Transcription", lines=4)
                    raw = gr.Textbox(label="Raw CTC output (greedy)", lines=3)
                    audio_status = gr.Markdown()
            for trigger in (listen.click, audio.stop_recording):
                trigger(transcribe, [audio, decoding], [text, raw, audio_status])

        with gr.Tab("Results"):
            summary = REPORTS / "summary.md"
            if summary.exists():
                gr.Markdown(summary.read_text(encoding="utf-8"))
            for name in ("confusion_matrices.png", "vision_curves.png", "speech_curves.png"):
                if (FIGURES / name).exists():
                    gr.Image(str(FIGURES / name), show_label=False, interactive=False)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="opens the browser")
    args = parser.parse_args()
    print("Loading models...", flush=True)
    demo = build_interface(Services(args.device))
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=args.open,
        share=False,
        theme=gr.themes.Soft(primary_hue="green", font=["Segoe UI", "system-ui", "sans-serif"]),
        css=CSS,
        allowed_paths=[str(SAMPLES), str(FIGURES)],
    )


if __name__ == "__main__":
    main()
