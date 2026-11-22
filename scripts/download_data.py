"""Descarga los datos y los pesos preentrenados desde sus fuentes oficiales.

Uso: python scripts/download_data.py [--only imagenes|voz|pesos]

Los archivos quedan en data/downloads. Se leen sin ejecutarse ni extraerse, y se
comprueba su SHA-256 cuando se conoce. Las licencias están en data/README.md.
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import urllib.request

from ecovoz.config import DOWNLOADS, SEED
from ecovoz.utils import sha256

PYTORCH = "https://download.pytorch.org/models"
OPENSLR = "https://www.openslr.org/resources"

FILES = {
    "imagenes": {
        "dataset-resized.zip": (
            "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
            "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c",
        ),
        "realwaste.zip": (
            "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
            "1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3",
        ),
    },
    "pesos": {
        "resnet50-11ad3fa6.pth": (
            f"{PYTORCH}/resnet50-11ad3fa6.pth",
            "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
        ),
        "mobilenet_v3_large-5c1a4163.pth": (
            f"{PYTORCH}/mobilenet_v3_large-5c1a4163.pth",
            "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997",
        ),
    },
    "voz": {},
}
for number, corpus in (
    (74, "es_pr_female"),
    (72, "es_co_female"),
    (72, "es_co_male"),
    (73, "es_pe_female"),
    (73, "es_pe_male"),
):
    gender = corpus.rsplit("_", 1)[1]
    FILES["voz"][f"{corpus}.zip"] = (f"{OPENSLR}/{number}/{corpus}.zip", None)
    FILES["voz"][f"{corpus}_index.tsv"] = (f"{OPENSLR}/{number}/line_index_{gender}.tsv", None)

TUX_REPO = "https://gitlab.com/ejimenezdelgado/tux-100h.git"
TUX_COMMIT = "87bc547599f101e4144b9e837a0cccd99a407f78"
TUX_CLIPS = 12_000
TUX_BATCH = 500


def fetch(name: str, url: str, expected: str | None) -> None:
    target = DOWNLOADS / name
    if not target.exists():
        print(f"descargando {name}", flush=True)
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    if expected and sha256(target) != expected:
        raise SystemExit(f"SHA-256 incorrecto en {target}; bórrelo y vuelva a descargar")
    print(f"listo {name}", flush=True)


def git(*arguments: str, stdin: str | None = None) -> str:
    command = ["git", "-c", "core.autocrlf=false", "-c", "maintenance.auto=false", *arguments]
    data = stdin.encode() if stdin is not None else None
    return subprocess.run(command, check=True, input=data, stdout=subprocess.PIPE).stdout.decode()


def fetch_tux() -> None:
    """Trae un subconjunto reproducible del dataset de clase (39 658 audios en total).

    El .tar de GitLab se corta en descargas grandes, por eso se usa un clon parcial:
    primero se traen los audios por lotes y después se extraen.
    """
    target = DOWNLOADS / "tux"
    if not target.exists():
        git("clone", "-q", "--filter=blob:none", "--no-checkout", TUX_REPO, str(target))
    repo = ("-C", str(target))
    blobs = {}
    for line in git(*repo, "ls-tree", "-r", TUX_COMMIT).splitlines():
        info, name = line.split("\t", 1)
        blobs[name] = info.split()[2]
    names = sorted(name for name in blobs if re.fullmatch(r"wavs/\d+\.wav", name))
    random.Random(SEED).shuffle(names)
    selected = [name for name in ["metadata.csv", *names[:TUX_CLIPS]] if not (target / name).exists()]
    for start in range(0, len(selected), TUX_BATCH):
        batch = selected[start : start + TUX_BATCH]
        wanted = "".join(f"{blobs[name]}\n" for name in batch)
        for attempt in range(4):
            try:
                git(
                    *repo, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "origin", "--no-tags",
                    "--no-write-fetch-head", "--filter=blob:none", "--stdin", stdin=wanted,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
        paths = "".join(f"{name}\n" for name in batch)
        git(*repo, "restore", f"--source={TUX_COMMIT}", "--worktree", "--pathspec-from-file=-", stdin=paths)
        print(f"tux {start + len(batch)}/{len(selected)}", flush=True)
    print("listo tux")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FILES))
    args = parser.parse_args()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for group in [args.only] if args.only else FILES:
        for name, (url, expected) in FILES[group].items():
            fetch(name, url, expected)
        if group == "voz":
            fetch_tux()


if __name__ == "__main__":
    main()
