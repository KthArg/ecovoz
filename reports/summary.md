## Vision (test split)

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ResNet-50 | 91.0 % | 90.6 % |
| MobileNetV3-L | 92.4 % | 92.5 % |

## Speech (test split)

| Group | Decoder | WER | CER |
|---|---|---:|---:|
| Unseen speakers | greedy | 35.1 % | 9.7 % |
| Unseen speakers | beam + LM | 24.6 % | 7.9 % |
| TUX speaker | greedy | 10.8 % | 1.9 % |
| TUX speaker | beam + LM | 8.7 % | 1.8 % |

## Latency on the demonstration machine

| Hardware | Model | Image (ms, median) | Video (fps) | Audio (s, p95) |
|---|---|---:|---:|---:|
| GPU | ResNet-50 | 6.1 | 101.9 | 0.07 |
| GPU | MobileNetV3-L | 5.3 | 106.3 | 0.07 |
| CPU | ResNet-50 | 30.2 | 30.3 | 0.30 |
| CPU | MobileNetV3-L | 8.3 | 84.2 | 0.30 |

Machine: AMD Ryzen 5 7600 6-Core Processor; NVIDIA GeForce RTX 5060.
