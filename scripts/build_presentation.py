"""Crea la presentación de 10 minutos con las cifras de reports/.

Uso: python scripts/build_presentation.py
"""

from __future__ import annotations

import re

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_THEME_COLOR as COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Inches, Pt

from ecovoz.config import ARCHITECTURES, FIGURES, REPORTS, ROOT, SAMPLES
from ecovoz.utils import read_json

PALETTE = {
    "dk1": "14281D", "lt1": "FFFFFF", "dk2": "14452F", "lt2": "EDF4EF",
    "accent1": "1F6F4A", "accent2": "E0792B", "accent3": "8DBB9B",
    "accent4": "52645A", "accent5": "2A78D6", "accent6": "C9D9CE",
}
DARK, LIGHT = COLOR.TEXT_2, COLOR.BACKGROUND_2
GREEN, ORANGE, MUTED = COLOR.ACCENT_1, COLOR.ACCENT_2, COLOR.ACCENT_4
WHITE, INK = COLOR.BACKGROUND_1, COLOR.TEXT_1
REPOSITORY = "github.com/KthArg/ecovoz"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f} %".replace(".", ",")


def decimal(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def new_presentation():
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    theme = deck.slide_master.part.part_related_by(RT.THEME)
    xml = theme.blob.decode("utf-8")
    for name, value in PALETTE.items():
        replacement = f'<a:{name}><a:srgbClr val="{value}"/></a:{name}>'
        xml = re.sub(rf"<a:{name}>.*?</a:{name}>", replacement, xml, flags=re.S)
    xml = re.sub(r'(<a:majorFont>\s*<a:latin typeface=")[^"]*', r"\g<1>Cambria", xml)
    theme._blob = xml.encode("utf-8")

    for shape in deck.slide_layouts[5].placeholders:
        if shape.placeholder_format.type == 1:  # título
            shape.left, shape.top = Inches(0.6), Inches(0.35)
            shape.width, shape.height = Inches(12.1), Inches(1.1)
    return deck


def add_text(slide, text, x, y, w, h, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, middle=False, title_font=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    if middle:
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(6)
        run = paragraph.add_run()
        run.text = line
        run.font.size, run.font.bold = Pt(size), bold
        run.font.color.theme_color = color
        if title_font:
            run.font.name = "+mj-lt"
    return box


def add_box(slide, x, y, w, h, color=LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.theme_color = color
    box.line.fill.background()
    box.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        box.adjustments[0] = 0.08
    return box


def stat(slide, value, label, x, y, w, h=1.7, color=LIGHT, ink=GREEN, size=46):
    add_box(slide, x, y, w, h, color)
    add_text(slide, value, x + 0.15, y + 0.12, w - 0.3, h * 0.55, size, ink, True, PP_ALIGN.CENTER, True, True)
    label_color = INK if color == LIGHT else WHITE
    add_text(slide, label, x + 0.15, y + h * 0.65, w - 0.3, h * 0.33, 14, label_color, align=PP_ALIGN.CENTER)


def numbered(slide, number, heading, body, x, y, w):
    add_box(slide, x, y, 0.62, 0.62, GREEN, MSO_SHAPE.OVAL)
    add_text(slide, str(number), x, y, 0.62, 0.62, 20, WHITE, True, PP_ALIGN.CENTER, True)
    add_text(slide, heading, x + 0.85, y, w - 0.85, 0.4, 20, DARK, True)
    add_text(slide, body, x + 0.85, y + 0.45, w - 0.85, 0.9, 16)


def picture(slide, path, x, y, w, h):
    """Inserta una imagen dentro del recuadro sin deformarla."""
    with Image.open(path) as image:
        width, height = image.size
    scale = min(w / width, h / height)
    left, top = x + (w - width * scale) / 2, y + (h - height * scale) / 2
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(width * scale), Inches(height * scale))


def bar_chart(slide, categories, series, box, minimum, maximum):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    x, y, w, h = (Inches(value) for value in box)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, data).chart
    chart.font.size = Pt(14)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    axis = chart.value_axis
    axis.minimum_scale, axis.maximum_scale = minimum, maximum
    axis.visible = False
    axis.has_major_gridlines = False
    chart.category_axis.format.line.color.rgb = RGBColor.from_string(PALETTE["accent6"])
    plot = chart.plots[0]
    plot.gap_width, plot.overlap = 70, -8
    for drawn, color in zip(plot.series, (GREEN, ORANGE), strict=False):
        drawn.format.fill.solid()
        drawn.format.fill.fore_color.theme_color = color
        labels = drawn.data_labels
        labels.show_value = True
        labels.number_format = "0.0"
        labels.number_format_is_linked = False
        labels.position = XL_LABEL_POSITION.OUTSIDE_END


class Deck:
    def __init__(self) -> None:
        self.deck = new_presentation()

    def slide(self, title: str, notes: str, dark: bool = False):
        slide = self.deck.slides.add_slide(self.deck.slide_layouts[5])
        heading = slide.shapes.title
        heading.text = title
        paragraph = heading.text_frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT
        paragraph.font.size, paragraph.font.bold = Pt(36), True
        paragraph.font.color.theme_color = WHITE if dark else DARK
        if dark:
            slide.background.fill.solid()
            slide.background.fill.fore_color.theme_color = DARK
        color = LIGHT if dark else MUTED
        add_text(slide, f"EcoVoz  |  {len(self.deck.slides)}", 10.7, 6.98, 2.0, 0.3, 11, color, align=PP_ALIGN.RIGHT)
        slide.notes_slide.notes_text_frame.text = notes
        return slide


def build() -> None:
    images, audio, vision, speech, latency = (
        read_json(REPORTS / f"{name}.json") for name in ("data_images", "data_audio", "vision", "speech", "latency")
    )
    resnet, mobile = (vision["models"][arch] for arch in ARCHITECTURES)
    experiments = speech["experiments_val_greedy"]
    test = speech["test"]
    cpu = latency["devices"]["cpu"]
    gpu = latency["devices"].get("cuda", cpu)
    default = vision["default"]
    deck = Deck()

    slide = deck.slide(
        "EcoVoz",
        "[Anthony, 0:20] Buenas. Presentamos EcoVoz, una aplicación que ve y escucha: clasifica residuos en "
        "imagen y video, y transcribe español con un modelo CTC que entrenamos.",
        dark=True,
    )
    title = slide.shapes.title
    title.top, title.left, title.width, title.height = Inches(1.9), Inches(0.7), Inches(6.2), Inches(1.3)
    title.text_frame.paragraphs[0].font.size = Pt(72)
    add_text(slide, "Aplicación multimodal con deep learning:\nimagen, video y voz", 0.8, 3.3, 5.8, 1.2, 24, LIGHT)
    team = "Anthony Jafeth Arias Robleto\nRandy Espinoza Downing\nKenneth Jhoan Araya Arguedas"
    add_text(slide, team, 0.8, 4.75, 5.8, 1.2, 16, WHITE)
    add_text(slide, "IC-6200 Inteligencia Artificial  |  Prof. Efrén Jiménez Delgado", 0.8, 6.25, 6.2, 0.4, 13, LIGHT)
    for index, path in enumerate(sorted((SAMPLES / "images").glob("*_trashnet.jpg"))):
        picture(slide, path, 7.2 + (index % 2) * 2.8, 0.75 + (index // 2) * 2.05, 2.7, 1.95)

    slide = deck.slide(
        "Tres modalidades en una sola aplicación",
        "[Anthony, 0:50] El problema es decidir en qué recipiente va un residuo. La imagen y el video usan el mismo "
        "clasificador, entrenado con transfer learning. El audio usa una red recurrente con pérdida CTC. Todo corre "
        "en la computadora, sin enviar datos a Internet. Ningún modelo es descargado y usado tal cual: "
        "la visión parte de ImageNet y la ajustamos nosotros, y la red de voz la entrenamos desde cero.",
    )
    modes = (
        ("Imagen", "Cargar, capturar o pegar una foto", "ResNet-50 o MobileNetV3"),
        ("Video", "Cada cuadro, o la cámara en vivo", "El mismo modelo, con suavizado"),
        ("Audio", "Grabar o cargar voz en español", "DeepSpeech2 + CTC"),
    )
    origin = (
        "Visión: partimos de pesos de ImageNet y los ajustamos con nuestros datos (transfer learning).  "
        "Voz: DeepSpeech2 entrenado por nosotros desde pesos aleatorios, sin modelos de voz externos."
    )
    add_text(slide, origin, 0.7, 6.55, 12, 0.4, 13, MUTED)
    for index, (name, what, how) in enumerate(modes):
        x = 0.7 + index * 4.1
        add_box(slide, x, 1.8, 3.7, 4.6)
        add_box(slide, x + 1.15, 2.2, 1.4, 1.4, GREEN, MSO_SHAPE.OVAL)
        add_text(slide, str(index + 1), x + 1.15, 2.2, 1.4, 1.4, 44, WHITE, True, PP_ALIGN.CENTER, True, True)
        add_text(slide, name, x + 0.2, 3.85, 3.3, 0.6, 28, DARK, True, PP_ALIGN.CENTER)
        add_text(slide, what, x + 0.2, 4.6, 3.3, 0.7, 16, align=PP_ALIGN.CENTER)
        add_text(slide, how, x + 0.2, 5.45, 3.3, 0.5, 16, GREEN, True, PP_ALIGN.CENTER)

    slide = deck.slide(
        "Datos separados sin fuga",
        "[Anthony, 1:00] Las imágenes vienen de TrashNet y RealWaste. Hay fotos repetidas del mismo objeto, así que "
        "las agrupamos por similitud y cada grupo va completo a una partición. En voz, una misma frase la leían hasta "
        "94 personas: separamos por locutor y también por frase. Sin eso, las métricas serían mejores pero falsas.",
    )
    total_audio = sum(part["clips"] for part in audio["splits"].values())
    stat(slide, thousands(images["total"]), "imágenes, 6 clases (mínimo: 200)", 0.7, 1.7, 3.9)
    stat(slide, thousands(total_audio), "audios transcritos (mínimo: 100)", 0.7, 3.65, 3.9)
    numbered(
        slide, 1, "Imágenes: por grupo de objeto",
        "Fotos casi idénticas quedan en la misma partición 70/15/15.",
        5.2, 1.8, 7.4,
    )
    numbered(
        slide, 2, "Voz: por locutor y por frase",
        "Los locutores de prueba y sus frases no se entrenan.",
        5.2, 3.2, 7.4,
    )
    numbered(
        slide, 3, "Prueba: solo para reportar",
        "Estrategia, calibración y decodificador se eligen en validación.",
        5.2, 4.6, 7.4,
    )
    sources = "Fuentes: TrashNet (MIT), RealWaste (CC BY-NC-SA 4.0), TUX (curso), OpenSLR 72/73/74 (CC BY-SA 4.0)"
    add_text(slide, sources, 0.7, 6.4, 12, 0.4, 12, MUTED)

    slide = deck.slide(
        "El ajuste fino supera a congelar la red",
        "[Randy, 0:50] Probamos cuatro estrategias por arquitectura. Solo cabeza deja congelado lo aprendido en "
        "ImageNet y es la peor. Luego liberamos los últimos bloques o toda la red, con una tasa de aprendizaje menor "
        "en el extractor. Elegimos con el F1 macro de validación. En ResNet ganó el ajuste parcial sin aumento y en "
        "MobileNet la red completa con aumento.",
    )
    series = []
    for arch, name in zip(ARCHITECTURES, ("ResNet-50", "MobileNetV3"), strict=True):
        scores = [
            100 * read_json(REPORTS / "vision_runs" / f"{arch}_{strategy}.json")["best"]["f1"]
            for strategy in ("head", "partial_noaug", "partial_aug", "full_aug")
        ]
        series.append((name, scores))
    labels = ["Solo cabeza", "Parcial sin aumento", "Parcial con aumento", "Completa con aumento"]
    bar_chart(slide, labels, series, (0.6, 1.6, 12.1, 4.8), 70, 100)
    add_text(slide, "F1 macro en validación (%). El eje empieza en 70.", 0.7, 6.45, 9, 0.4, 12, MUTED)

    slide = deck.slide(
        "La red móvil iguala a la red grande",
        f"[Randy, 1:00] En prueba MobileNet obtuvo {percent(mobile['test']['accuracy'])} y ResNet "
        f"{percent(resnet['test']['accuracy'])}. La prueba de McNemar da p={decimal(vision['mcnemar']['p_value'], 3)}: "
        "la diferencia no es significativa. Elegimos MobileNet porque ganó en validación y pesa cinco veces menos. "
        "El plástico es la clase con más fallos: se confunde con metal y vidrio por el brillo y la transparencia.",
    )
    stat(slide, percent(mobile["test"]["accuracy"]), "MobileNetV3: accuracy en prueba", 0.7, 1.6, 3.7, color=DARK, ink=WHITE)
    stat(slide, percent(resnet["test"]["accuracy"]), "ResNet-50: accuracy en prueba", 0.7, 3.45, 3.7)
    p_value = decimal(vision["mcnemar"]["p_value"], 2)
    verdict = f"McNemar p = {p_value}: diferencia no significativa.\nMobileNet pesa 5 veces menos."
    add_text(slide, verdict, 0.7, 5.4, 3.9, 1.2, 15)
    picture(slide, FIGURES / "confusion_matrices.png", 4.7, 1.6, 8.1, 4.9)

    slide = deck.slide(
        "CTC aprende sin saber cuándo suena cada letra",
        "[Randy, 1:00] La red da una letra o un blanco cada 20 milisegundos. Para leer el texto se unen las "
        "repeticiones y se quitan los blancos. El blanco permite escribir letras dobles, como la ll de calle. "
        "La pérdida suma la probabilidad de todos los alineamientos que producen la frase correcta, con programación "
        "dinámica.",
    )
    symbols = "c c _ a l l _ l e e".split()
    for index, symbol in enumerate(symbols):
        x = 0.9 + index * 1.15
        add_box(slide, x, 1.8, 1.0, 1.0, LIGHT if symbol == "_" else GREEN)
        ink = MUTED if symbol == "_" else WHITE
        shown = "ε" if symbol == "_" else symbol
        add_text(slide, shown, x, 1.8, 1.0, 1.0, 34, ink, True, PP_ALIGN.CENTER, True)
    rule = "Unir repetidos y quitar el blanco ε  →  «calle»"
    add_text(slide, rule, 0.9, 3.05, 11.4, 0.6, 24, DARK, True, PP_ALIGN.CENTER)
    stages = ("Log-mel\n80 bandas", "2 convoluciones\n11x41 y 11x21", "5 BiGRU\n512 unidades", "Softmax + CTC\n35 salidas")
    for index, stage in enumerate(stages):
        x = 0.7 + index * 3.1
        add_box(slide, x, 4.2, 2.6, 1.5)
        add_text(slide, stage, x + 0.1, 4.2, 2.4, 1.5, 17, DARK, True, PP_ALIGN.CENTER, True)
        if index < len(stages) - 1:
            add_box(slide, x + 2.67, 4.75, 0.36, 0.4, GREEN, MSO_SHAPE.RIGHT_ARROW)
    add_text(slide, "DeepSpeech2, la arquitectura de la clase 12, en PyTorch", 0.7, 6.0, 12, 0.4, 14, MUTED)

    slide = deck.slide(
        "Con 100 audios no basta para generalizar",
        "[Kenneth, 1:00] Este es el experimento que pide el enunciado. Con 100 audios la red memoriza: la pérdida "
        "de entrenamiento baja casi a cero y la de validación sube. Medimos cuánto ayudan el aumento de datos y "
        "partir de un modelo preentrenado con el dataset de clase. El modelo final usa todo el corpus. Todas las "
        "barras están medidas con los mismos locutores de validación.",
    )
    names = ("small100", "small100_aug", "small100_transfer", "multi")
    series = [(key.upper(), [100 * experiments[name]["openslr"][key] for name in names]) for key in ("wer", "cer")]
    labels = ["100 audios", "100 + aumento", "100 + preentrenamiento", "Modelo final"]
    bar_chart(slide, labels, series, (0.6, 1.6, 12.1, 4.8), 0, 115)
    caption = "Error en validación con locutores nuevos (%), decodificación voraz. Menos es mejor."
    add_text(slide, caption, 0.7, 6.45, 10, 0.4, 12, MUTED)

    slide = deck.slide(
        "El error depende de quién habla",
        "[Kenneth, 1:00] En prueba separamos dos casos. Con el locutor de TUX medimos frases nuevas. Con OpenSLR "
        "medimos además personas que el modelo nunca oyó, que es lo que pasa en la demostración. WER es mayor que "
        "CER porque una sola letra daña toda la palabra. El beam search con modelo de lenguaje corrige parte de "
        "esos errores.",
    )
    for row, (group, name) in enumerate((("openslr", "Locutores nuevos"), ("tux", "Locutor conocido, frases nuevas"))):
        y = 1.6 + row * 2.45
        scores = test["beam_lm"][group]
        add_text(slide, name, 0.7, y, 6.2, 0.4, 18, DARK, True)
        stat(slide, percent(scores["wer"]), "WER", 0.7, y + 0.5, 2.9, size=40)
        stat(slide, percent(scores["cer"]), "CER", 3.8, y + 0.5, 2.9, size=40)
    picture(slide, FIGURES / "speech_distribution.png", 7.0, 1.6, 5.8, 4.2)
    greedy = percent(test["greedy"]["openslr"]["wer"])
    improvement = f"Con decodificación voraz el WER de locutores nuevos es {greedy}."
    add_text(slide, improvement, 7.1, 5.9, 5.6, 0.7, 13, MUTED)

    slide = deck.slide(
        "Demostración en vivo",
        "[Kenneth, 2:00] Antes de empezar, dejar abierta la aplicación con iniciar.bat. 1) Imagen: un ejemplo y "
        "luego un objeto real. 2) Video: reproducir el video de prueba y señalar los cuadros por segundo. 3) Audio: "
        "un ejemplo y luego grabar una frase corta y clara. Si la cámara o el micrófono fallan, usar los ejemplos "
        "incluidos.",
        dark=True,
    )
    steps = (
        ("Imagen", "un ejemplo y un objeto real"),
        ("Video", "cuadro a cuadro, con cuadros por segundo"),
        ("Audio", "un ejemplo y una frase grabada"),
    )
    for index, (name, what) in enumerate(steps):
        y = 1.9 + index * 1.5
        add_box(slide, 0.9, y, 0.9, 0.9, ORANGE, MSO_SHAPE.OVAL)
        add_text(slide, str(index + 1), 0.9, y, 0.9, 0.9, 28, WHITE, True, PP_ALIGN.CENTER, True)
        add_text(slide, name, 2.1, y, 4.4, 0.5, 26, WHITE, True)
        add_text(slide, what, 2.1, y + 0.52, 4.6, 0.5, 16, LIGHT)
    screenshot = ROOT / "presentation" / "assets" / "app.png"
    if screenshot.exists():
        picture(slide, screenshot, 6.8, 1.5, 6.0, 5.2)

    slide = deck.slide(
        "Cumple tiempo real, incluso sin GPU",
        "[Anthony, 0:30] El enunciado pide 5 cuadros por segundo y menos de 5 segundos por audio. Medimos de punta a "
        "punta en este equipo, con GPU y solo con CPU, y los dos casos cumplen.",
    )
    for row, (result, name) in enumerate(((gpu, latency["hardware"]["gpu"]), (cpu, latency["hardware"]["cpu"].strip()))):
        y = 1.55 + row * 2.5
        add_text(slide, name or "CPU", 0.7, y, 12, 0.4, 16, DARK, True)
        cells = (
            (f"{decimal(result['image_ms'][default]['p50'])} ms", "por imagen"),
            (decimal(result["video"][default]["fps"], 0), "cuadros/s de video (mínimo 5)"),
            (f"{decimal(result['audio_ms']['p95'] / 1000, 2)} s", "por audio, p95 (máximo 5)"),
        )
        for column, (value, label) in enumerate(cells):
            stat(slide, value, label, 0.7 + column * 4.1, y + 0.5, 3.7)

    slide = deck.slide(
        "Lo que aprendimos",
        "[Randy, 0:30] El ajuste fino vale más que una red más grande. En voz lo decisivo son los datos y los "
        "locutores. Y sin separar bien los datos las métricas engañan. Como trabajo futuro: grabar datos propios con "
        "acento costarricense y agregar una clase de ningún objeto. El uso de IA generativa está declarado en el "
        "paper. Gracias, quedamos atentos a preguntas.",
        dark=True,
    )
    lessons = (
        ("Visión", "ajustar el extractor pesa más que agrandar la red"),
        ("Voz", "CTC necesita muchos audios y muchos locutores"),
        ("Método", "sin una partición sin fuga, las métricas engañan"),
        ("Sigue", "datos de Costa Rica y una clase «ningún objeto»"),
    )
    for index, (name, what) in enumerate(lessons):
        y = 1.7 + index * 1.15
        add_box(slide, 0.9, y, 2.3, 0.85, GREEN)
        add_text(slide, name, 0.9, y, 2.3, 0.85, 20, WHITE, True, PP_ALIGN.CENTER, True)
        add_text(slide, what, 3.6, y, 9, 0.85, 20, LIGHT, middle=True)
    footer = f"{REPOSITORY}  |  Uso de IA generativa (Claude) declarado en el paper"
    add_text(slide, footer, 0.9, 6.45, 9.6, 0.4, 13, LIGHT)

    properties = deck.deck.core_properties
    properties.author = properties.last_modified_by = "Equipo EcoVoz"
    properties.title, properties.comments = "EcoVoz", ""
    target = ROOT / "presentation" / "EcoVoz.pptx"
    deck.deck.save(target)
    print(f"{len(deck.deck.slides)} diapositivas en {target}")


if __name__ == "__main__":
    build()
