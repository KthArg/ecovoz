# EcoVoz

Final project for IC-6200 Artificial Intelligence (TEC, second semester 2026): a multimodal deep learning application that

- classifies waste **images** into six classes, with ResNet-50 and MobileNetV3-Large (transfer learning);
- classifies **video** and the live camera, frame by frame;
- transcribes Spanish **audio** with a DeepSpeech2 model trained with **CTC**.

The app interface and the documentation are in English; the speech model recognizes and transcribes **Spanish** speech.

Team: Anthony Jafeth Arias Robleto, Randy Espinoza Downing and Kenneth Jhoan Araya Arguedas.

Results: [reports/summary.md](reports/summary.md). Paper: [paper/EcoVoz_paper.pdf](paper/EcoVoz_paper.pdf). Presentation: [presentation/EcoVoz.pptx](presentation/EcoVoz.pptx).

## Running the application

The trained models are included in the repository: there is no need to download data or train. Python 3.11 or later is required.

```powershell
git clone https://github.com/KthArg/ecovoz.git
cd ecovoz
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python app.py --open
```

It opens <http://127.0.0.1:7860>. On Windows you can also double-click `start.bat`, which does all of the above.

To use an NVIDIA GPU, install PyTorch with CUDA **before** `requirements.txt`:

```powershell
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

The application also runs on CPU only (`--device cpu`). Latencies for both cases are in the results summary.

| Tab | What it does |
|---|---|
| Image | upload, capture or paste an image: class, bin, probabilities and milliseconds |
| Video | play a video classified frame by frame, with frames per second |
| Live camera | classify the webcam feed |
| Audio | record or upload audio: transcription, raw CTC output and latency |
| Results | evaluation metrics and figures |

`samples/` contains images, audio clips, a generated test video and two videos recorded by the team.

## What was trained in this project

| Model | Starting point | What we trained |
|---|---|---|
| ResNet-50 | ImageNet weights (torchvision) | new 6-class head + `layer3` and `layer4` |
| MobileNetV3-Large | ImageNet weights (torchvision) | new 6-class head + the whole network |
| DeepSpeech2 (CTC) | random weights | everything: pretraining on TUX, then the full corpus |
| Language model | nothing | character 6-gram counted from the training texts |

No external speech model is used. The ImageNet weights are the starting point that transfer learning requires.

## Reproducing the work

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\download_data.py      # ~7 GB, requires git
.venv\Scripts\python scripts\prepare_images.py
.venv\Scripts\python scripts\prepare_audio.py
.venv\Scripts\python scripts\train_vision.py       # 8 configurations
.venv\Scripts\python scripts\train_speech.py       # 5 experiments, several hours on GPU
.venv\Scripts\python scripts\evaluate_vision.py
.venv\Scripts\python scripts\evaluate_speech.py
.venv\Scripts\python scripts\build_samples.py      # requires ffmpeg
.venv\Scripts\python scripts\benchmark.py
.venv\Scripts\python scripts\make_report.py
cd paper; tectonic main.tex
```

The seed is 42 in every step. On GPU the results may vary slightly because some CUDA operations are not deterministic.

## Tests

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

They cover text normalization, WER and CER (against `jiwer`), beam search (against the exhaustive sum over alignments), the spectrogram, the model and the absence of leakage between splits.

## Layout

```
app.py                 Gradio application
src/ecovoz/
  config.py            paths and constants
  inference.py         image and speech inference
  video.py             frame-by-frame classification
  vision/              models, data and augmentation
  speech/              text, audio, log-mel, DeepSpeech2, decoders, WER/CER
scripts/               download, preparation, training, evaluation, latency, reports
tests/                 tests
data/README.md         sources, licenses, labeling and preprocessing
data/manifests/        every example with its split
data/subset/           240 images and 100 preprocessed audio clips
models/                trained weights (float16)
reports/               metrics, histories, predictions and figures
paper/                 IEEE paper
presentation/          presentation
docs/DEFENSE_GUIDE.md  explanation of each part and likely questions
```

## Licenses and authorship

- **Data:** TrashNet (MIT), RealWaste (CC BY-NC-SA 4.0), OpenSLR 72/73/74 (CC BY-SA 4.0) and the course's TUX dataset (no declared license; its audio is not redistributed). Details in [data/README.md](data/README.md).
- **Base models:** ResNet-50 and MobileNetV3-Large from torchvision (BSD-3) with ImageNet weights. The speech architecture is based on the [Keras CTC example](https://keras.io/examples/audio/ctc_asr/) (Apache-2.0) covered in class, rewritten in PyTorch.
- **Vision weights:** derived from RealWaste, so they are for non-commercial use only.
- **Generative AI:** the project was developed with the support of Claude (Anthropic) through Claude Code. The team defined the requirements and the course methodology and recorded the own videos; the tool generated the design, code, experiments and analysis, and drafted the paper and the presentation, which the team reviewed, checking the results and requesting corrections. Details in the paper's declaration.
