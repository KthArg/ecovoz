# Documentación de los datos

`data/manifests/` enumera **cada** ejemplo usado, con su origen, etiqueta y partición. Los datos completos se reconstruyen con:

```
python scripts/download_data.py
python scripts/prepare_images.py
python scripts/prepare_audio.py
```

`data/subset/` es el conjunto mínimo entregable, ya preprocesado: 240 imágenes (40 por clase, con su partición) y los 100 audios, con transcripción, del experimento de 100 audios.

## Imágenes

| | TrashNet | RealWaste |
|---|---|---|
| Autores | Yang y Thung, 2016 | Single, Iranmanesh y Raad, 2023 |
| Fuente | <https://github.com/garythung/trashnet> | <https://archive.ics.uci.edu/dataset/908/realwaste> |
| Archivo | `dataset-resized.zip` | `realwaste.zip` |
| Licencia | MIT | CC BY-NC-SA 4.0, según el README del archivo |
| Captura | objetos sobre cartulina blanca | residuos recibidos en un centro de recuperación en Wollongong, Australia |
| Etiquetado | manual, por los autores | manual, por los autores |

**Clases.** Se usan seis: cartón, vidrio, metal, papel, plástico y no reciclable. `Miscellaneous Trash` de RealWaste equivale a `trash`. `Food Organics`, `Textile Trash` y `Vegetation` se excluyen porque no tienen equivalente en TrashNet. No se reetiquetó ninguna imagen.

**Preprocesamiento.**

1. Decodificación a RGB, respetando la orientación EXIF.
2. Eliminación de duplicados exactos por SHA-256 de los píxeles. Si un duplicado tenía dos etiquetas, se descartaron ambas copias.
3. Reescalado bicúbico del lado corto a 256 px y guardado como JPEG de calidad 92.
4. Al entrenar o inferir: recorte de 224 x 224 y normalización de ImageNet.

**Partición 70/15/15 sin fuga.** Ambos conjuntos tienen varias fotos del mismo objeto. Se calculó la similitud coseno entre embeddings de ResNet-50 (ImageNet) y se unieron las fotos de una misma fuente con similitud de 0,88 o más. Cada grupo queda completo en una sola partición, estratificando por clase y fuente. El umbral se eligió revisando pares reales (`reports/figures/near_duplicates.png`). Reduce la fuga, pero no garantiza detectar el mismo objeto visto desde ángulos muy distintos.

Los conteos por clase y partición están en `reports/data_images.json`.

## Audio

| | TUX | OpenSLR 72, 73 y 74 |
|---|---|---|
| Fuente | <https://gitlab.com/ejimenezdelgado/tux-100h>, dataset de la clase 12 | <https://www.openslr.org/72/>, [73](https://www.openslr.org/73/) y [74](https://www.openslr.org/74/) |
| Contenido | un locutor leyendo libros en español | voluntarios de Colombia, Perú y Puerto Rico leyendo frases |
| Selección | 12 000 de 39 658 audios, al azar con semilla 42 | todos los audios (solo hay voces de mujer en Puerto Rico) |
| Licencia | el repositorio no declara licencia; se usa como material del curso y **sus audios no se redistribuyen aquí** (el manifiesto solo conserva identificadores y transcripciones) | CC BY-SA 4.0, Google LLC (Guevara-Rukoz et al., 2020) |
| Etiquetado | transcripción incluida en `metadata.csv` | transcripción incluida en `line_index` |

**Preprocesamiento.**

1. Conversión a mono, 16 kHz, WAV PCM de 16 bits.
2. Recorte del silencio inicial y final (40 dB bajo el máximo, con margen de 150 ms).
3. Se conservan los audios de 0,7 a 12 s con pasos suficientes para alinear con CTC.
4. Transcripciones en NFC y minúsculas, sin puntuación, con alfabeto de 34 símbolos (espacio, a-z, vocales con tilde, ü, ñ).

**Partición sin fuga.** En OpenSLR una misma frase la leen hasta 94 personas, así que una división al azar filtraría el texto. Por eso:

- de cada corpus (país y sexo) se reservan un locutor para validación y otro para prueba;
- las frases reservadas se eliminan del entrenamiento, las lea quien las lea;
- en TUX, que es de un solo locutor, la validación y la prueba usan frases no vistas;
- las métricas de "locutores nuevos" y "locutor TUX" se reportan por separado.

Los conteos están en `reports/data_audio.json`.

## Video

`samples/video/montaje_prueba.mp4` lo generamos con `scripts/build_samples.py`: doce fotos de la **partición de prueba** (seis clases por dos fuentes) con movimiento de cámara, giro y cambios de luz simulados. Su archivo `.json` trae la etiqueta de cada tramo, lo que permite medir la exactitud por cuadro. Es un montaje y no una grabación con cámara. No sustituye al conjunto de imágenes.

## Personas y consentimiento

No se recolectaron datos propios de personas. Las voces vienen de corpus públicos, y los identificadores de locutor de OpenSLR son anónimos. La aplicación procesa la cámara y el micrófono localmente y no envía nada a Internet.

## Redistribución

`data/subset/` y `samples/` conservan las licencias de origen. Lo derivado de RealWaste, incluidos los pesos de visión de este repositorio, es solo para uso no comercial.
