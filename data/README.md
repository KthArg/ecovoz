# Data documentation

`data/manifests/` lists **every** example used, with its source, label and split. The full data can be rebuilt with:

```
python scripts/download_data.py
python scripts/prepare_images.py
python scripts/prepare_audio.py
```

`data/subset/` is the minimal deliverable set, already preprocessed: 240 images (40 per class, with their split) and the 100 transcribed audio clips used in the 100-clip experiment.

## Images

| | TrashNet | RealWaste |
|---|---|---|
| Authors | Yang and Thung, 2016 | Single, Iranmanesh and Raad, 2023 |
| Source | <https://github.com/garythung/trashnet> | <https://archive.ics.uci.edu/dataset/908/realwaste> |
| File | `dataset-resized.zip` | `realwaste.zip` |
| License | MIT | CC BY-NC-SA 4.0, according to the archive's README |
| Capture | objects on white poster board | waste received at a material recovery facility in Wollongong, Australia |
| Labeling | manual, by the authors | manual, by the authors |

**Classes.** Six are used: cardboard, glass, metal, paper, plastic and non-recyclable (trash). RealWaste's `Miscellaneous Trash` maps to `trash`. `Food Organics`, `Textile Trash` and `Vegetation` are excluded because they have no TrashNet counterpart. No image was relabeled.

**Preprocessing.**

1. Decoding to RGB, honoring the EXIF orientation.
2. Removal of exact duplicates by SHA-256 of the pixels. If a duplicate had two labels, both copies were discarded.
3. Bicubic resizing of the short side to 256 px, saved as JPEG at quality 92.
4. At training or inference time: 224 x 224 crop and ImageNet normalization.

**Leak-free 70/15/15 split.** Both datasets contain several photos of the same object. We computed the cosine similarity between ResNet-50 (ImageNet) embeddings and merged photos from the same source with a similarity of 0.88 or more. Each group stays whole in a single split, stratified by class and source. The threshold was chosen by inspecting real pairs (`reports/figures/near_duplicates.png`). It reduces leakage, but it is not guaranteed to catch the same object seen from very different angles.

Counts per class and split are in `reports/data_images.json`.

## Audio

| | TUX | OpenSLR 72, 73 and 74 |
|---|---|---|
| Source | <https://gitlab.com/ejimenezdelgado/tux-100h>, the dataset from class 12 | <https://www.openslr.org/72/>, [73](https://www.openslr.org/73/) and [74](https://www.openslr.org/74/) |
| Content | a single speaker reading books in Spanish | volunteers from Colombia, Peru and Puerto Rico reading sentences |
| Selection | 12,000 of 39,658 clips, at random with seed 42 | all clips (Puerto Rico has female voices only) |
| License | the repository declares no license; it is used as course material and **its audio is not redistributed here** (the manifest keeps only identifiers and transcriptions) | CC BY-SA 4.0, Google LLC (Guevara-Rukoz et al., 2020) |
| Labeling | transcription included in `metadata.csv` | transcription included in `line_index` |

**Preprocessing.**

1. Conversion to mono, 16 kHz, 16-bit PCM WAV.
2. Trimming of leading and trailing silence (40 dB below the peak, with a 150 ms margin).
3. Clips from 0.7 to 12 s with enough frames for CTC alignment are kept.
4. Transcriptions in NFC and lowercase, without punctuation, with a 34-symbol alphabet (space, a-z, accented vowels, ü, ñ).

**Leak-free split.** In OpenSLR the same sentence is read by up to 94 people, so a random split would leak the text. Therefore:

- from each corpus (country and sex), one speaker is held out for validation and another for test;
- held-out sentences are removed from training, no matter who reads them;
- in TUX, which has a single speaker, validation and test use unseen sentences;
- metrics for "new speakers" and "TUX speaker" are reported separately.

Counts are in `reports/data_audio.json`.

## Video

`samples/video/test_montage.mp4` was generated with `scripts/build_samples.py`: twelve photos from the **test split** (six classes times two sources) with simulated camera motion, rotation and lighting changes. Its `.json` file holds the label of each segment, which allows measuring per-frame accuracy. It is a montage, not a camera recording, and it does not replace the image test set.

`samples/video/own_bottle.mp4` and `samples/video/own_notebook.mp4` were recorded by the team with a phone (480x864, 30 frames/s, no audio and no people): a plastic bottle and a notebook, labeled `plastic` and `paper`. They were not used for training or tuning. Their results are in `reports/own_video/` and can be reproduced with `scripts/evaluate_video.py`.

## People and consent

No personal data was collected. The voices come from public corpora, and the OpenSLR speaker identifiers are anonymous. The application processes the camera and microphone locally and sends nothing to the Internet.

## Redistribution

`data/subset/` and `samples/` keep their original licenses. Anything derived from RealWaste, including the vision weights in this repository, is for non-commercial use only.
