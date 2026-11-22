## Visión (partición de prueba)

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| ResNet-50 | 91,0 % | 90,6 % |
| MobileNetV3-L | 92,4 % | 92,5 % |

## Voz (partición de prueba)

| Grupo | Decodificador | WER | CER |
|---|---|---:|---:|
| Locutores nuevos | voraz | 35,1 % | 9,7 % |
| Locutores nuevos | beam + LM | 24,6 % | 7,9 % |
| Locutor TUX | voraz | 10,8 % | 1,9 % |
| Locutor TUX | beam + LM | 8,7 % | 1,8 % |

## Latencia en el equipo de la demostración

| Equipo | Modelo | Imagen (ms, mediana) | Video (cuadros/s) | Audio (s, p95) |
|---|---|---:|---:|---:|
| GPU | ResNet-50 | 6,1 | 101,9 | 0,07 |
| GPU | MobileNetV3-L | 5,3 | 106,3 | 0,07 |
| CPU | ResNet-50 | 30,2 | 30,3 | 0,30 |
| CPU | MobileNetV3-L | 8,3 | 84,2 | 0,30 |

Equipo: AMD Ryzen 5 7600 6-Core Processor; NVIDIA GeForce RTX 5060.
