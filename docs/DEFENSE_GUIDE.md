# Guide to understanding and defending EcoVoz

The assignment requires every team member to be able to explain every part. The exact figures are in [reports/summary.md](../reports/summary.md) and in the paper.

## 1. The project in one minute

EcoVoz has two models behind one interface:

1. A **CNN with transfer learning** that classifies waste. It works on images and, frame by frame, on video.
2. A **DeepSpeech2 model with CTC loss** that converts Spanish speech to text.

## 2. Where everything is

| Question | File | What to look at |
|---|---|---|
| How are the images split? | `scripts/prepare_images.py` | `find_groups`, `split_groups` |
| How is the audio split? | `scripts/prepare_audio.py` | `assign_splits` |
| Which layers are frozen? | `src/ecovoz/vision/model.py` | `finetune_blocks`, `set_trainable`, `train_mode` |
| What image augmentation is there? | `src/ecovoz/vision/data.py` | `train_transform` |
| How is the vision model trained? | `scripts/train_vision.py` | `STRATEGIES`, `train` |
| How does audio become numbers? | `src/ecovoz/speech/features.py` | `LogMel` |
| What is the speech network? | `src/ecovoz/speech/model.py` | `DeepSpeech2` |
| Where is CTC? | `scripts/train_speech.py` | `ctc_loss` inside `train` |
| How is the output decoded? | `src/ecovoz/speech/decoder.py` | `greedy_decode`, `beam_search`, `CharNGram` |
| How are WER and CER computed? | `src/ecovoz/speech/metrics.py` | `edit_distance` |
| How is video processed? | `src/ecovoz/video.py` | `classify_video` |
| How was latency measured? | `scripts/benchmark.py` | `main` |

## 3. Vision

**Transfer learning.** We start from a network that already knows how to see (ImageNet, 1000 classes), replace the last layer with one that has six outputs, and train briefly.

**Freezing** means not updating a layer's weights. We also keep the BatchNorm statistics of those layers fixed: otherwise they would change even though the weights do not.

**Why two phases.** First only the head, because it is random and its gradients would damage the feature extractor. Then layers are unfrozen with a learning rate lower than the head's.

**The four strategies** are the experiment that justifies fine-tuning. The choice was made on validation, never on test.

**ResNet and MobileNet.** ResNet adds each block's input to its output (residual connection) and is deep. MobileNet splits each convolution into a per-channel one and a 1x1 one, so it has far fewer parameters.

**Metrics.**

- Precision: of what I said was plastic, how much really was.
- Recall: of the plastic that was there, how much I found.
- F1: the harmonic mean of the two.
- Macro F1: the per-class average. It is used because the classes are imbalanced.

**Calibration.** The logits are divided by a temperature learned on validation. It does not change the predicted class; it only makes the percentage more honest. A temperature below 1 means the model was underconfident, which is normal with label smoothing.

**McNemar.** It looks only at the images where one model is right and the other is wrong. With p > 0.05 we cannot claim that one is better.

## 4. Speech

**The path of an audio clip.**

1. Mono at 16,000 samples per second.
2. 25 ms windows every 10 ms, Fourier transform, 80 mel bands and logarithm: an "image" of the sound.
3. Two convolutions halve the time axis: 50 steps per second remain.
4. Five bidirectional GRUs add context from before and after.
5. Each step outputs 35 probabilities: 34 symbols plus the blank.

**Why CTC.** We know what the audio says, but not at which millisecond each letter is spoken. CTC sums the probability of every alignment that yields the correct text.

**Reading rule.** Merge repeats, then remove blanks: `cc_all_lee` becomes `calle` (Spanish for "street"). The blank is not the space: the space is just another character.

**Greedy and beam search.** Greedy takes the best symbol at each step. Beam search keeps the 24 most probable partial texts and sums their paths. The language model counts sequences of up to six characters in the training texts and helps break ties. Alpha is its weight and beta a bonus per character.

**WER and CER.** Substitutions, deletions and insertions divided by the reference length. WER can exceed 100 % if there are many insertions.

**The five experiments.**

| Name | What it shows |
|---|---|
| `small100` | with 100 clips the network memorizes and does not generalize |
| `small100_aug` | how much data augmentation helps |
| `tux` | pretraining on the class dataset, a single voice |
| `small100_transfer` | how much pretraining helps with 100 clips |
| `multi` | the model used in the application |

## 5. Likely questions

**How do you know there is no leakage?**
Images are split by group, speech by speaker and by sentence. `tests/test_data_splits.py` checks this on the manifests. Grouping by similarity reduces leakage between photos of the same object, but it is not perfect, and we say so in the paper.

**Are the models yours, or did you download them?**
- Vision: the only external part is the ImageNet weights, the starting point the assignment asks for ("pretrained models") and the one used in the class 11 transfer learning notebook. The six-class head is new and was trained on our data, as were the unfrozen blocks: `layer3` and `layer4` in ResNet, the whole network in MobileNet.
- Speech: nothing external. DeepSpeech2 starts from random weights and we trained it ourselves with CTC: first on TUX and then on the full corpus. The log confirms it: in epoch 1 the network only emits blanks and the CER is 100 %.
- We also computed the language model and the calibration ourselves.

**Why PyTorch if the class used Keras?**
TensorFlow has not supported the GPU on native Windows since version 2.11. The architecture is the same as in class.

**What did you change from the class notebook?**
Log-mel instead of the STFT magnitude, real lengths in CTC (the notebook uses the padded length), an alphabet with accents and ñ, data augmentation, beam search with a language model, and splitting by speaker and sentence. The notebook uses the English alphabet and drops the accents.

**Is the video your own?**
Yes: we recorded a plastic bottle and a notebook with a phone (`samples/video/own_bottle.mp4` and `samples/video/own_notebook.mp4`). MobileNet got every frame of the bottle right and two thirds of the notebook frames; ResNet, trained without augmentation, got almost everything wrong. There is also a montage generated from test photos, which is useful for measuring because we know which object is in each frame. To evaluate another video: `python scripts/evaluate_video.py video.mp4 --label plastic`.

**Why does it fail with my object or with a different background?**
The data has plain backgrounds and one object per photo. It is a classifier, not a detector: it always looks at the center of the frame and has no "no object" class. That is why the confidence threshold exists.

**Why does it get my voice wrong?**
It was trained on accents from Colombia, Peru and Puerto Rico and on one narrator, always reading. There are no Costa Rican voices and no spontaneous speech. Compare with the "new speakers" WER, not with the TUX one.

**Why float16?**
So the weights fit in the repository. They are converted back to float32 on loading, and the final metrics were measured with those files.

**Did you use generative AI?**
Yes: the project was developed with Claude, and this is declared in the paper and in the README. Do not deny it or minimize it.

## 6. Before the demo

1. Run `start.bat` a few minutes ahead, because loading takes a while.
2. Allow camera and microphone access in the browser and check that the selected microphone is the right one.
3. Objects: only one, centered, on a plain light background, with good lighting.
4. Speech: short Spanish sentences, close to the microphone, with no background music.
5. If something fails, use the examples included in each tab.
6. Plan B without a GPU: `.venv\Scripts\python app.py --device cpu`.

## 7. Suggested presentation split

| Member | Slides | Topic |
|---|---|---|
| Anthony | 1 to 3 and 10 | problem, data and latency |
| Randy | 4 to 6 and 11 | vision, CTC and conclusions |
| Kenneth | 7 to 9 | speech results and demo |

Everyone should be able to answer questions about the other parts. The speaker notes are inside the PPTX, with timings that add up to 10 minutes.
