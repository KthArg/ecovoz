"""EcoVoz: imagen, video y audio en una sola interfaz.

Uso: python app.py [--device auto|cuda|cpu] [--port 7860] [--open]

Todo corre en esta computadora: la aplicación no envía imágenes ni audio a Internet.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import lru_cache

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np

from ecovoz.config import ARCHITECTURES, CLASSES_ES, FIGURES, MODELS, REPORTS, SAMPLES
from ecovoz.inference import Transcriber, VisionClassifier
from ecovoz.utils import get_device, read_json
from ecovoz.video import Smoother, annotate, classify_video

MODEL_NAMES = {"resnet50": "ResNet-50", "mobilenet_v3_large": "MobileNetV3-Large"}
MAX_AUDIO_SECONDS = 60

BINS = {
    "cardboard": "Papel y cartón: seco y desarmado.",
    "paper": "Papel y cartón: seco y sin grasa.",
    "glass": "Vidrio: enjuagado y sin tapa.",
    "metal": "Metales: enjuagado.",
    "plastic": "Plásticos: limpio y seco.",
    "trash": "Residuos no valorizables.",
}
CSS = """
.gradio-container {max-width: 1120px !important; margin: auto}
footer {display: none !important}
"""


class Services:
    """Carga cada modelo una sola vez."""

    def __init__(self, device: str) -> None:
        self.device = str(get_device(device))
        self.classifier = lru_cache(maxsize=None)(lambda arch: VisionClassifier(arch, self.device))
        for arch in ARCHITECTURES:
            self.classifier(arch)
        self.transcriber = Transcriber(self.device)


def build_interface(services: Services) -> gr.Blocks:
    settings_path = MODELS / "vision.json"
    settings = read_json(settings_path) if settings_path.exists() else {}
    default_arch = settings.get("default", "resnet50")

    def classify_image(image, arch, threshold):
        if image is None:
            return None, ""
        result = services.classifier(arch).predict(image)
        scores = {
            CLASSES_ES[name]: float(value)
            for name, value in zip(result.classes, result.probabilities, strict=True)
        }
        if result.confidence >= threshold:
            detail = f"### {CLASSES_ES[result.label].capitalize()}\n{BINS[result.label]}"
        else:
            detail = "### No estoy seguro\nAcerque el objeto, use un fondo liso y mejore la luz."
        return scores, f"{detail}\n\nInferencia: **{result.milliseconds:.1f} ms** ({services.device})"

    def play_video(path, arch, threshold, momentum):
        if not path:
            raise gr.Error("Cargue un video o elija el de ejemplo.")
        times = []
        try:
            for frame in classify_video(path, services.classifier(arch), threshold, momentum):
                times.append(frame.milliseconds)
                status = (
                    f"Cuadro **{frame.index + 1}** | **{frame.fps:.1f}** cuadros/s | "
                    f"inferencia media **{np.mean(times):.1f} ms** | mínimo exigido: 5 cuadros/s"
                )
                yield frame.image, status
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def classify_live(image, arch, threshold, state):
        if image is None:
            return None, "", None
        now = time.perf_counter()
        smoother, previous = state if state and state[0] == arch else ((arch, Smoother()), None)
        result = services.classifier(arch).predict(image)
        smoothed = smoother[1].update(result.probabilities)
        label = result.classes[int(smoothed.argmax())]
        note = f"Inferencia **{result.milliseconds:.1f} ms**"
        if previous:
            note += f" | **{1 / max(now - previous, 1e-6):.1f}** cuadros/s"
        shown = annotate(image, label, float(smoothed.max()), threshold)
        return shown, note, (smoother, now)

    def transcribe(audio, decoding):
        if audio is None:
            raise gr.Error("Grabe o cargue un audio.")
        rate, samples = audio
        if len(samples) > MAX_AUDIO_SECONDS * rate:
            raise gr.Error(f"Use fragmentos de hasta {MAX_AUDIO_SECONDS} segundos.")
        result = services.transcriber.transcribe(samples, rate, decoding == "beam")
        text = result.text or "(no se detectó voz)"
        detail = (
            f"Audio de **{result.seconds:.1f} s**, transcrito en **{result.milliseconds / 1000:.2f} s** "
            f"({services.device}). Máximo exigido: 5 s."
        )
        return text, result.greedy, detail

    model_choices = [(MODEL_NAMES[arch], arch) for arch in ARCHITECTURES]

    with gr.Blocks(title="EcoVoz", analytics_enabled=False) as demo:
        gr.Markdown(
            "# EcoVoz\nClasificación de residuos en imagen y video, y transcripción de voz en español. "
            f"Todo se procesa en esta computadora ({services.device})."
        )
        with gr.Row():
            arch = gr.Dropdown(model_choices, value=default_arch, label="Modelo de visión")
            threshold = gr.Slider(0, 0.95, value=0.5, step=0.05, label="Confianza mínima para responder")

        with gr.Tab("Imagen"):
            with gr.Row():
                image = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="Imagen", height=360)
                with gr.Column():
                    verdict = gr.Markdown()
                    scores = gr.Label(num_top_classes=6, label="Probabilidades")
            examples = sorted((SAMPLES / "images").glob("*.jpg"))
            if examples:
                gr.Examples([str(path) for path in examples], inputs=image, label="Ejemplos de la partición de prueba")
            for trigger in (image.change, arch.change, threshold.release):
                trigger(classify_image, [image, arch, threshold], [scores, verdict], show_progress="hidden")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(sources=["upload"], label="Video", height=300)
                    momentum = gr.Slider(0, 0.9, value=0.6, step=0.1, label="Suavizado entre cuadros")
                    with gr.Row():
                        play = gr.Button("Clasificar cuadro a cuadro", variant="primary")
                        stop = gr.Button("Detener")
                    videos = sorted((SAMPLES / "video").glob("*.mp4"))
                    if videos:
                        gr.Examples([str(path) for path in videos], inputs=video, label="Video de ejemplo")
                with gr.Column():
                    screen = gr.Image(label="Resultado", format="jpeg", height=420, elem_id="video-output")
                    video_status = gr.Markdown()
            playing = play.click(
                play_video, [video, arch, threshold, momentum], [screen, video_status], show_progress="hidden"
            )
            stop.click(None, cancels=[playing])

        with gr.Tab("Cámara en vivo"):
            with gr.Row():
                camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Cámara", height=360)
                with gr.Column():
                    live = gr.Image(label="Resultado", format="jpeg", height=360)
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
                    audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Grabe o cargue un audio")
                    decoding = gr.Radio(
                        [("Beam search con modelo de lenguaje", "beam"), ("Voraz", "greedy")],
                        value="beam",
                        label="Decodificación",
                    )
                    listen = gr.Button("Transcribir", variant="primary")
                    clips = sorted((SAMPLES / "audio").glob("*.wav"))
                    if clips:
                        gr.Examples([str(path) for path in clips], inputs=audio, label="Ejemplos de la partición de prueba")
                with gr.Column():
                    text = gr.Textbox(label="Transcripción", lines=4)
                    raw = gr.Textbox(label="Salida CTC sin corregir (voraz)", lines=3)
                    audio_status = gr.Markdown()
            for trigger in (listen.click, audio.stop_recording):
                trigger(transcribe, [audio, decoding], [text, raw, audio_status])

        with gr.Tab("Resultados"):
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
    parser.add_argument("--open", action="store_true", help="abre el navegador")
    args = parser.parse_args()
    print("Cargando modelos...", flush=True)
    demo = build_interface(Services(args.device))
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=args.open,
        share=False,
        theme=gr.themes.Soft(primary_hue="green"),
        css=CSS,
        allowed_paths=[str(SAMPLES), str(FIGURES)],
    )


if __name__ == "__main__":
    main()
