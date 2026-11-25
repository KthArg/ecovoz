"""Build the 10-minute presentation from the figures in reports/.

Usage: python scripts/build_presentation.py
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
    return f"{100 * value:.{digits}f}%"


def decimal(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def thousands(value: int) -> str:
    return f"{value:,}"


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
        if shape.placeholder_format.type == 1:  # title
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
    """Insert an image inside the box without distorting it."""
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
        "[Anthony, 0:20] Hello. We present EcoVoz, an application that sees and listens: it classifies waste in "
        "images and video, and transcribes Spanish speech with a CTC model we trained ourselves.",
        dark=True,
    )
    title = slide.shapes.title
    title.top, title.left, title.width, title.height = Inches(1.9), Inches(0.7), Inches(6.2), Inches(1.3)
    title.text_frame.paragraphs[0].font.size = Pt(72)
    add_text(slide, "Multimodal deep learning application:\nimage, video and speech", 0.8, 3.3, 5.8, 1.2, 24, LIGHT)
    team = "Anthony Jafeth Arias Robleto\nRandy Espinoza Downing\nKenneth Jhoan Araya Arguedas"
    add_text(slide, team, 0.8, 4.75, 5.8, 1.2, 16, WHITE)
    add_text(slide, "IC-6200 Artificial Intelligence  |  Prof. Efrén Jiménez Delgado", 0.8, 6.25, 6.2, 0.4, 13, LIGHT)
    for index, path in enumerate(sorted((SAMPLES / "images").glob("*_trashnet.jpg"))):
        picture(slide, path, 7.2 + (index % 2) * 2.8, 0.75 + (index // 2) * 2.05, 2.7, 1.95)

    slide = deck.slide(
        "Three modalities in a single application",
        "[Anthony, 0:50] The problem is deciding which bin a piece of waste goes into. Image and video share the same "
        "classifier, trained with transfer learning. Audio uses a recurrent network with CTC loss. Everything runs "
        "on the computer, without sending data to the Internet. No model is downloaded and used as is: "
        "the vision model starts from ImageNet and we fine-tune it, and we trained the speech network from scratch.",
    )
    modes = (
        ("Image", "Upload, capture or paste a photo", "ResNet-50 or MobileNetV3"),
        ("Video", "Every frame, or the live camera", "Same model, with smoothing"),
        ("Audio", "Record or upload Spanish speech", "DeepSpeech2 + CTC"),
    )
    origin = (
        "Vision: we start from ImageNet weights and fine-tune them on our data (transfer learning).  "
        "Speech: DeepSpeech2 trained by us from random weights, with no external speech models."
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
        "Data split without leakage",
        "[Anthony, 1:00] The images come from TrashNet and RealWaste. There are repeated photos of the same object, so "
        "we group them by similarity and each whole group goes to a single split. In speech, the same sentence was read "
        "by up to 94 people: we split by speaker and also by sentence. Without that, the metrics would look better but "
        "they would be false.",
    )
    total_audio = sum(part["clips"] for part in audio["splits"].values())
    stat(slide, thousands(images["total"]), "images, 6 classes (minimum: 200)", 0.7, 1.7, 3.9)
    stat(slide, thousands(total_audio), "transcribed clips (minimum: 100)", 0.7, 3.65, 3.9)
    numbered(
        slide, 1, "Images: by object group",
        "Near-identical photos stay in the same 70/15/15 split.",
        5.2, 1.8, 7.4,
    )
    numbered(
        slide, 2, "Speech: by speaker and by sentence",
        "Test speakers and their sentences are never trained on.",
        5.2, 3.2, 7.4,
    )
    numbered(
        slide, 3, "Test: for reporting only",
        "Strategy, calibration and decoder are chosen on validation.",
        5.2, 4.6, 7.4,
    )
    sources = "Sources: TrashNet (MIT), RealWaste (CC BY-NC-SA 4.0), TUX (course), OpenSLR 72/73/74 (CC BY-SA 4.0)"
    add_text(slide, sources, 0.7, 6.4, 12, 0.4, 12, MUTED)

    slide = deck.slide(
        "Fine-tuning beats freezing the network",
        "[Randy, 0:50] We tried four strategies per architecture. Head only keeps everything learned on ImageNet "
        "frozen, and it is the worst. Then we unfreeze the last blocks or the whole network, with a lower learning rate "
        "in the feature extractor. We choose by validation macro F1. For ResNet, partial fine-tuning without "
        "augmentation won; for MobileNet, the full network with augmentation.",
    )
    series = []
    for arch, name in zip(ARCHITECTURES, ("ResNet-50", "MobileNetV3"), strict=True):
        scores = [
            100 * read_json(REPORTS / "vision_runs" / f"{arch}_{strategy}.json")["best"]["f1"]
            for strategy in ("head", "partial_noaug", "partial_aug", "full_aug")
        ]
        series.append((name, scores))
    labels = ["Head only", "Partial, no augmentation", "Partial + augmentation", "Full + augmentation"]
    bar_chart(slide, labels, series, (0.6, 1.6, 12.1, 4.8), 70, 100)
    add_text(slide, "Validation macro F1 (%). The axis starts at 70.", 0.7, 6.45, 9, 0.4, 12, MUTED)

    slide = deck.slide(
        "Tied on test; MobileNet wins on real video",
        f"[Randy, 1:00] On test, MobileNet got {percent(mobile['test']['accuracy'])} and ResNet "
        f"{percent(resnet['test']['accuracy'])}. McNemar's test gives p={decimal(vision['mcnemar']['p_value'], 3)}: "
        "the difference is not significant. We chose MobileNet because it won on validation and is five times "
        "smaller. And on the videos we recorded with a phone the difference was huge: MobileNet, trained "
        "with data augmentation, got every bottle frame right and two thirds of the notebook frames; ResNet, "
        "without augmentation, got almost everything wrong. Plastic is the class with the most errors: "
        "it gets confused with metal and glass because of shine and transparency.",
    )
    stat(slide, percent(mobile["test"]["accuracy"]), "MobileNetV3: test accuracy", 0.7, 1.6, 3.7, color=DARK, ink=WHITE)
    stat(slide, percent(resnet["test"]["accuracy"]), "ResNet-50: test accuracy", 0.7, 3.45, 3.7)
    p_value = decimal(vision["mcnemar"]["p_value"], 2)
    own = {
        (video, arch): read_json(REPORTS / "own_video" / f"own_{video}_{arch}.json")["frame_accuracy"]
        for video in ("bottle", "notebook")
        for arch in ARCHITECTURES
    }
    verdict = (
        f"McNemar p = {p_value}: no difference on test.\n"
        f"Our own phone video, correct frames:\n"
        f"MobileNet {percent(own['bottle', 'mobilenet_v3_large'])} bottle, "
        f"{percent(own['notebook', 'mobilenet_v3_large'])} notebook\n"
        f"ResNet {percent(own['bottle', 'resnet50'])} bottle, "
        f"{percent(own['notebook', 'resnet50'])} notebook"
    )
    add_text(slide, verdict, 0.7, 5.4, 3.9, 1.2, 15)
    picture(slide, FIGURES / "confusion_matrices.png", 4.7, 1.6, 8.1, 4.9)

    slide = deck.slide(
        "CTC learns without knowing when each letter sounds",
        "[Randy, 1:00] The network outputs a letter or a blank every 20 milliseconds. To read the text, repeated "
        "symbols are merged and blanks are removed. The blank lets us write double letters, like the double l in "
        "calle, Spanish for street. The loss adds up the probability of every alignment that produces the correct "
        "sentence, using dynamic programming.",
    )
    symbols = "c c _ a l l _ l e e".split()
    for index, symbol in enumerate(symbols):
        x = 0.9 + index * 1.15
        add_box(slide, x, 1.8, 1.0, 1.0, LIGHT if symbol == "_" else GREEN)
        ink = MUTED if symbol == "_" else WHITE
        shown = "ε" if symbol == "_" else symbol
        add_text(slide, shown, x, 1.8, 1.0, 1.0, 34, ink, True, PP_ALIGN.CENTER, True)
    rule = 'Merge repeats, drop the blank ε  →  "calle" (street)'
    add_text(slide, rule, 0.9, 3.05, 11.4, 0.6, 24, DARK, True, PP_ALIGN.CENTER)
    stages = ("Log-mel\n80 bands", "2 convolutions\n11x41 and 11x21", "5 BiGRU\n512 units", "Softmax + CTC\n35 outputs")
    for index, stage in enumerate(stages):
        x = 0.7 + index * 3.1
        add_box(slide, x, 4.2, 2.6, 1.5)
        add_text(slide, stage, x + 0.1, 4.2, 2.4, 1.5, 17, DARK, True, PP_ALIGN.CENTER, True)
        if index < len(stages) - 1:
            add_box(slide, x + 2.67, 4.75, 0.36, 0.4, GREEN, MSO_SHAPE.RIGHT_ARROW)
    add_text(slide, "DeepSpeech2, the architecture from lecture 12, in PyTorch", 0.7, 6.0, 12, 0.4, 14, MUTED)

    slide = deck.slide(
        "100 audio clips are not enough to generalize",
        "[Kenneth, 1:00] This is the experiment the assignment asks for. With 100 clips the network memorizes: the "
        "training loss drops almost to zero and the validation loss goes up. We measured how much data augmentation "
        "helps, and how much starting from a model pretrained on the class dataset helps. The final model uses the "
        "whole corpus. All bars are measured on the same validation speakers.",
    )
    names = ("small100", "small100_aug", "small100_transfer", "multi")
    series = [(key.upper(), [100 * experiments[name]["openslr"][key] for name in names]) for key in ("wer", "cer")]
    labels = ["100 clips", "100 + augmentation", "100 + pretraining", "Final model"]
    bar_chart(slide, labels, series, (0.6, 1.6, 12.1, 4.8), 0, 115)
    caption = "Validation error on unseen speakers (%), greedy decoding. Lower is better."
    add_text(slide, caption, 0.7, 6.45, 10, 0.4, 12, MUTED)

    slide = deck.slide(
        "The error depends on who is speaking",
        "[Kenneth, 1:00] On test we separate two cases. With the TUX speaker we measure new sentences. With OpenSLR "
        "we also measure people the model never heard, which is what happens in the demo. WER is higher than "
        "CER because a single wrong letter ruins the whole word. Beam search with a language model fixes part of "
        "those errors.",
    )
    for row, (group, name) in enumerate((("openslr", "Unseen speakers"), ("tux", "Known speaker, new sentences"))):
        y = 1.6 + row * 2.45
        scores = test["beam_lm"][group]
        add_text(slide, name, 0.7, y, 6.2, 0.4, 18, DARK, True)
        stat(slide, percent(scores["wer"]), "WER", 0.7, y + 0.5, 2.9, size=40)
        stat(slide, percent(scores["cer"]), "CER", 3.8, y + 0.5, 2.9, size=40)
    picture(slide, FIGURES / "speech_distribution.png", 7.0, 1.6, 5.8, 4.2)
    greedy = percent(test["greedy"]["openslr"]["wer"])
    improvement = f"With greedy decoding, the WER on unseen speakers is {greedy}."
    add_text(slide, improvement, 7.1, 5.9, 5.6, 0.7, 13, MUTED)

    slide = deck.slide(
        "Live demo",
        "[Kenneth, 2:00] Before starting, have the application already open with start.bat. 1) Image tab: one sample, "
        "then a real object. 2) Video tab: play our own bottle video and point out the frames per second. "
        "3) Audio tab: one sample, then record a short, clear sentence. If the camera or microphone fails, use the "
        "bundled samples.",
        dark=True,
    )
    steps = (
        ("Image", "a sample and a real object"),
        ("Video", "frame by frame, with frames per second"),
        ("Audio", "a sample and a recorded sentence"),
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
        "Real time, even without a GPU",
        "[Anthony, 0:30] The assignment requires 5 frames per second and under 5 seconds per audio clip. We measured "
        "end to end on this machine, with a GPU and with CPU only, and both cases meet the requirements.",
    )
    for row, (result, name) in enumerate(((gpu, latency["hardware"]["gpu"]), (cpu, latency["hardware"]["cpu"].strip()))):
        y = 1.55 + row * 2.5
        add_text(slide, name or "CPU", 0.7, y, 12, 0.4, 16, DARK, True)
        cells = (
            (f"{decimal(result['image_ms'][default]['p50'])} ms", "per image"),
            (decimal(result["video"][default]["fps"], 0), "video frames/s (minimum 5)"),
            (f"{decimal(result['audio_ms']['p95'] / 1000, 2)} s", "per audio clip, p95 (maximum 5)"),
        )
        for column, (value, label) in enumerate(cells):
            stat(slide, value, label, 0.7 + column * 4.1, y + 0.5, 3.7)

    slide = deck.slide(
        "What we learned",
        "[Randy, 0:30] Fine-tuning is worth more than a bigger network. In speech, what matters most is the data and "
        "the speakers. And without a proper data split, metrics mislead. Future work: record our own data with a "
        "Costa Rican accent and add a no-object class. Our use of generative AI is declared in the paper. "
        "Thank you, we are happy to take questions.",
        dark=True,
    )
    lessons = (
        ("Vision", "tuning the extractor beats a bigger network"),
        ("Speech", "CTC needs many clips and many speakers"),
        ("Method", "without a leak-free split, metrics mislead"),
        ("Next", 'Costa Rican data and a "no object" class'),
    )
    for index, (name, what) in enumerate(lessons):
        y = 1.7 + index * 1.15
        add_box(slide, 0.9, y, 2.3, 0.85, GREEN)
        add_text(slide, name, 0.9, y, 2.3, 0.85, 20, WHITE, True, PP_ALIGN.CENTER, True)
        add_text(slide, what, 3.6, y, 9, 0.85, 20, LIGHT, middle=True)
    footer = f"{REPOSITORY}  |  Generative AI use (Claude) declared in the paper"
    add_text(slide, footer, 0.9, 6.45, 9.6, 0.4, 13, LIGHT)

    properties = deck.deck.core_properties
    properties.author = properties.last_modified_by = "EcoVoz Team"
    properties.title, properties.comments = "EcoVoz", ""
    target = ROOT / "presentation" / "EcoVoz.pptx"
    deck.deck.save(target)
    print(f"{len(deck.deck.slides)} slides in {target}")


if __name__ == "__main__":
    build()
