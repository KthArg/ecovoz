# Guía para entender y defender EcoVoz

El enunciado exige que cada integrante pueda explicar cada parte. Las cifras exactas están en [reports/summary.md](../reports/summary.md) y en el paper.

## 1. El proyecto en un minuto

EcoVoz tiene dos modelos detrás de una interfaz:

1. Una **CNN con transfer learning** que clasifica residuos. Sirve para imágenes y, cuadro a cuadro, para video.
2. Un **DeepSpeech2 con pérdida CTC** que convierte voz en español a texto.

## 2. Dónde está cada cosa

| Pregunta | Archivo | Qué mirar |
|---|---|---|
| ¿Cómo se parten las imágenes? | `scripts/prepare_images.py` | `find_groups`, `split_groups` |
| ¿Cómo se parte el audio? | `scripts/prepare_audio.py` | `assign_splits` |
| ¿Qué capas se congelan? | `src/ecovoz/vision/model.py` | `finetune_blocks`, `set_trainable`, `train_mode` |
| ¿Qué aumento de imágenes hay? | `src/ecovoz/vision/data.py` | `train_transform` |
| ¿Cómo se entrena la visión? | `scripts/train_vision.py` | `STRATEGIES`, `train` |
| ¿Cómo se vuelve número el audio? | `src/ecovoz/speech/features.py` | `LogMel` |
| ¿Cuál es la red de voz? | `src/ecovoz/speech/model.py` | `DeepSpeech2` |
| ¿Dónde está CTC? | `scripts/train_speech.py` | `ctc_loss` dentro de `train` |
| ¿Cómo se decodifica? | `src/ecovoz/speech/decoder.py` | `greedy_decode`, `beam_search`, `CharNGram` |
| ¿Cómo se calculan WER y CER? | `src/ecovoz/speech/metrics.py` | `edit_distance` |
| ¿Cómo se procesa el video? | `src/ecovoz/video.py` | `classify_video` |
| ¿Cómo se midió la latencia? | `scripts/benchmark.py` | `main` |

## 3. Visión

**Transfer learning.** Se parte de una red que ya sabe ver (ImageNet, 1000 clases), se cambia la última capa por una de seis salidas y se entrena poco.

**Congelar** es no actualizar los pesos de una capa. También dejamos quietas las estadísticas de BatchNorm de esas capas: si no, cambiarían aunque los pesos no lo hagan.

**Por qué dos fases.** Primero solo la cabeza, porque es aleatoria y sus gradientes dañarían el extractor. Después se liberan capas con una tasa de aprendizaje menor que la de la cabeza.

**Las cuatro estrategias** son el experimento que justifica el fine-tuning. Se eligió con validación, nunca con prueba.

**ResNet y MobileNet.** ResNet suma la entrada a la salida de cada bloque (conexión residual) y es profunda. MobileNet separa cada convolución en una por canal y una de 1x1, por lo que tiene muchos menos parámetros.

**Métricas.**

- Precisión: de lo que dije que era plástico, cuánto lo era.
- Recall: del plástico que había, cuánto encontré.
- F1: media armónica de las dos.
- F1 macro: promedio por clase. Se usa porque hay desbalance.

**Calibración.** Se dividen los logits entre una temperatura aprendida en validación. No cambia la clase elegida, solo hace más honesto el porcentaje. Una temperatura menor que 1 significa que el modelo era poco confiado, algo normal con suavizado de etiquetas.

**McNemar.** Mira solo las imágenes donde un modelo acierta y el otro no. Con p mayor que 0,05 no se puede afirmar que uno sea mejor.

## 4. Voz

**Recorrido de un audio.**

1. Mono a 16 000 muestras por segundo.
2. Ventanas de 25 ms cada 10 ms, transformada de Fourier, 80 bandas mel y logaritmo: una "imagen" del sonido.
3. Dos convoluciones reducen el tiempo a la mitad: quedan 50 pasos por segundo.
4. Cinco GRU bidireccionales aportan contexto de antes y después.
5. En cada paso salen 35 probabilidades: 34 símbolos más el blanco.

**Por qué CTC.** Sabemos qué dice el audio, pero no en qué milisegundo suena cada letra. CTC suma la probabilidad de todas las alineaciones que dan el texto correcto.

**Regla de lectura.** Unir repetidos y luego quitar blancos: `cc_all_lee` queda `calle`. El blanco no es el espacio: el espacio es un carácter más.

**Voraz y beam search.** El voraz toma lo mejor de cada paso. El beam mantiene los 24 textos parciales más probables y suma sus caminos. El modelo de lenguaje cuenta secuencias de hasta seis caracteres en los textos de entrenamiento y ayuda a desempatar. Alfa es su peso y beta un premio por carácter.

**WER y CER.** Sustituciones, borrados e inserciones entre el largo de la referencia. El WER puede pasar de 100 % si hay muchas inserciones.

**Los cinco experimentos.**

| Nombre | Qué demuestra |
|---|---|
| `small100` | con 100 audios la red memoriza y no generaliza |
| `small100_aug` | cuánto ayuda el aumento de datos |
| `tux` | preentrenamiento con el dataset de clase, de una sola voz |
| `small100_transfer` | cuánto ayuda el preentrenamiento con 100 audios |
| `multi` | el modelo de la aplicación |

## 5. Preguntas probables

**¿Cómo saben que no hay fuga?**
Imágenes por grupo, voz por locutor y por frase. `tests/test_data_splits.py` lo comprueba sobre los manifiestos. La agrupación por similitud reduce la fuga entre fotos del mismo objeto, pero no es perfecta, y lo decimos en el paper.

**¿Los modelos son de ustedes o los bajaron de internet?**
- Visión: lo único externo son los pesos de ImageNet, el punto de partida que pide el enunciado ("modelos preentrenados") y que usa el notebook de transfer learning de la clase 11. La cabeza de seis clases es nueva y se entrenó con nuestros datos, igual que los bloques liberados: `layer3` y `layer4` en ResNet, toda la red en MobileNet.
- Voz: nada externo. DeepSpeech2 empieza con pesos aleatorios y lo entrenamos nosotros con CTC: primero con TUX y después con todo el corpus. Lo confirma el log: en la época 1 la red solo emite blancos y el CER es de 100 %.
- El modelo de lenguaje y la calibración también los calculamos nosotros.

**¿Por qué PyTorch si en clase usaron Keras?**
TensorFlow no usa GPU en Windows nativo desde la versión 2.11. La arquitectura es la misma de clase.

**¿Qué cambiaron respecto al notebook de clase?**
Log-mel en lugar de la magnitud STFT, longitudes reales en CTC (el notebook usa el largo con relleno), alfabeto con tildes y eñe, aumento de datos, beam search con modelo de lenguaje y partición por locutor y frase. En el notebook el alfabeto es el inglés y pierde las tildes.

**¿El video es propio?**
El incluido es un montaje que generamos con fotos de prueba en movimiento, no una grabación. Si el equipo graba uno, se carga en la pestaña Video.

**¿Por qué falla con mi objeto o con otro fondo?**
Los datos tienen fondos lisos y un objeto por foto. Es un clasificador, no un detector: siempre mira el centro del cuadro y no tiene una clase "ningún objeto". Por eso existe el umbral de confianza.

**¿Por qué se equivoca con mi voz?**
Se entrenó con acentos de Colombia, Perú y Puerto Rico y con un narrador, siempre leyendo. No hay voces costarricenses ni habla espontánea. Compare con el WER de "locutores nuevos", no con el de TUX.

**¿Por qué float16?**
Para que los pesos quepan en el repositorio. Al cargar se vuelven a float32 y las métricas finales se midieron con esos archivos.

**¿Usaron IA generativa?**
Sí: el proyecto se desarrolló con Claude y está declarado en el paper y en el README. No lo nieguen ni lo minimicen.

## 6. Antes de la demostración

1. Ejecute `iniciar.bat` unos minutos antes, porque la carga tarda.
2. Permita cámara y micrófono en el navegador y revise que el micrófono elegido sea el correcto.
3. Objetos: uno solo, centrado, fondo liso y claro, buena luz.
4. Voz: frases cortas, cerca del micrófono, sin música de fondo.
5. Si algo falla, use los ejemplos incluidos en cada pestaña.
6. Plan B sin GPU: `.venv\Scripts\python app.py --device cpu`.

## 7. Reparto sugerido de la presentación

| Integrante | Diapositivas | Tema |
|---|---|---|
| Anthony | 1 a 3 y 10 | problema, datos y latencia |
| Randy | 4 a 6 y 11 | visión, CTC y conclusiones |
| Kenneth | 7 a 9 | resultados de voz y demostración |

Cada uno debe poder responder las preguntas de las otras partes. Las notas del orador están dentro del PPTX, con tiempos que suman 10 minutos.
