"""Paths and constants shared by the data pipeline, training and the application."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DOWNLOADS = DATA / "downloads"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RUNS = ROOT / "runs"
SAMPLES = ROOT / "samples"

SEED = 42

# Vision
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
# Names shown to the user.
CLASS_LABELS = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "non-recyclable",
}
ARCHITECTURES = ("resnet50", "mobilenet_v3_large")
IMAGE_SIZE = 224
STORED_SIDE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Speech
SAMPLE_RATE = 16_000
N_FFT = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
N_MELS = 80
