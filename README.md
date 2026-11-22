# EcoVoz

Proyecto final de IC-6200 Inteligencia Artificial (TEC, II Semestre 2026): una aplicación multimodal de deep learning que

- clasifica **imágenes** de residuos en seis clases, con ResNet-50 y MobileNetV3-Large (transfer learning);
- clasifica **video** y cámara en vivo, cuadro a cuadro;
- transcribe **audio** en español con un DeepSpeech2 entrenado con **CTC**.

Equipo: Anthony Jafeth Arias Robleto, Randy Espinoza Downing y Kenneth Jhoan Araya Arguedas.

Resultados: [reports/summary.md](reports/summary.md). Paper: [paper/EcoVoz_paper.pdf](paper/EcoVoz_paper.pdf). Presentación: [presentation/EcoVoz.pptx](presentation/EcoVoz.pptx).

## Ejecutar la aplicación

Los modelos entrenados vienen en el repositorio: no hace falta descargar datos ni entrenar. Se necesita Python 3.11 o superior.

```powershell
git clone https://github.com/KthArg/ecovoz.git
cd ecovoz
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python app.py --open
```

Abre <http://127.0.0.1:7860>. En Windows también sirve hacer doble clic en `iniciar.bat`, que hace todo lo anterior.

Para usar una GPU NVIDIA, instale PyTorch con CUDA **antes** de `requirements.txt`:

```powershell
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

La aplicación también funciona solo con CPU (`--device cpu`). Las latencias de ambos casos están en el resumen de resultados.

| Pestaña | Qué hace |
|---|---|
| Imagen | cargar, capturar o pegar una imagen: clase, recipiente, probabilidades y milisegundos |
| Video | reproducir un video clasificado cuadro a cuadro, con cuadros por segundo |
| Cámara en vivo | clasificar la cámara web |
| Audio | grabar o cargar audio: transcripción, salida CTC sin corregir y latencia |
| Resultados | métricas y figuras de la evaluación |

En `samples/` hay imágenes, audios y un video de prueba.

## Reproducir el trabajo

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\download_data.py      # ~7 GB, necesita git
.venv\Scripts\python scripts\prepare_images.py
.venv\Scripts\python scripts\prepare_audio.py
.venv\Scripts\python scripts\train_vision.py       # 8 configuraciones
.venv\Scripts\python scripts\train_speech.py       # 5 experimentos, varias horas con GPU
.venv\Scripts\python scripts\evaluate_vision.py
.venv\Scripts\python scripts\evaluate_speech.py
.venv\Scripts\python scripts\build_samples.py      # necesita ffmpeg
.venv\Scripts\python scripts\benchmark.py
.venv\Scripts\python scripts\make_report.py
cd paper; tectonic main.tex
```

La semilla es 42 en todos los pasos. En GPU el resultado puede variar levemente porque algunas operaciones de CUDA no son deterministas.

## Pruebas

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

Cubren la normalización, WER y CER (contra `jiwer`), el beam search (contra la suma exhaustiva de alineamientos), el espectrograma, el modelo y la ausencia de fuga entre particiones.

## Organización

```
app.py                 aplicación Gradio
src/ecovoz/
  config.py            rutas y constantes
  inference.py         inferencia de imagen y voz
  video.py             clasificación cuadro a cuadro
  vision/              modelos, datos y aumento
  speech/              texto, audio, log-mel, DeepSpeech2, decodificadores, WER/CER
scripts/               descarga, preparación, entrenamiento, evaluación, latencia, reportes
tests/                 pruebas
data/README.md         fuentes, licencias, etiquetado y preprocesamiento
data/manifests/        cada ejemplo con su partición
data/subset/           240 imágenes y 100 audios preprocesados
models/                pesos entrenados (float16)
reports/               métricas, historiales, predicciones y figuras
paper/                 artículo IEEE
presentation/          presentación
docs/GUIA_DEFENSA.md   explicación de cada parte y preguntas probables
```

## Licencias y autoría

- **Datos:** TrashNet (MIT), RealWaste (CC BY-NC-SA 4.0), OpenSLR 72/73/74 (CC BY-SA 4.0) y TUX del curso (sin licencia declarada, no se redistribuye). Detalle en [data/README.md](data/README.md).
- **Modelos base:** ResNet-50 y MobileNetV3-Large de torchvision (BSD-3) con pesos de ImageNet. La arquitectura de voz se basa en el [ejemplo CTC de Keras](https://keras.io/examples/audio/ctc_asr/) (Apache-2.0) visto en clase, reescrito en PyTorch.
- **Pesos de visión:** derivan de RealWaste, por lo que son solo para uso no comercial.
- **IA generativa:** el proyecto (diseño, código, experimentos, análisis, paper y presentación) se desarrolló con Claude de Anthropic mediante Claude Code. También se declara en el paper.
