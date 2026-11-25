"""Prepares the speech corpus: 16 kHz mono WAV and normalized transcriptions.

Splits:
- OpenSLR 72/73/74: the validation and test speakers do not appear in
  training, and neither do their sentences, even when read by someone else.
- TUX (class 12 dataset, a single speaker): split by sentence.
- No normalized sentence appears in two splits.

Usage: python scripts/prepare_audio.py
"""

from __future__ import annotations

import io
import random
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import soundfile as sf

from ecovoz.config import DOWNLOADS, MANIFESTS, PROCESSED, REPORTS, SAMPLE_RATE, SEED
from ecovoz.speech.audio import load_audio, to_int16, trim_silence
from ecovoz.speech.text import normalize
from ecovoz.utils import write_csv, write_json

CORPORA = ("es_co_female", "es_co_male", "es_pe_female", "es_pe_male", "es_pr_female")
CLIPS_PER_HELD_OUT_SPEAKER = 60
TUX_HELD_OUT = 600
MIN_SECONDS, MAX_SECONDS = 0.7, 12.0


def slr_entries() -> list[dict]:
    entries = []
    for corpus in CORPORA:
        index = (DOWNLOADS / f"{corpus}_index.tsv").read_text(encoding="utf-8")
        for line in index.splitlines():
            clip, text = line.split("\t", 1)
            entries.append(
                {"id": clip, "source": corpus, "speaker": clip.rsplit("_", 1)[0], "raw_text": text.strip()}
            )
    return entries


def tux_entries() -> list[dict]:
    available = {path.stem for path in (DOWNLOADS / "tux" / "wavs").glob("*.wav")}
    entries = []
    for line in (DOWNLOADS / "tux" / "metadata.csv").read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("|")
        if len(parts) == 3 and parts[0] in available:
            entries.append(
                {"id": f"tux_{parts[0]}", "source": "tux", "speaker": "tux", "raw_text": parts[2].strip()}
            )
    return entries


def assign_splits(entries: list[dict]) -> list[dict]:
    """Returns only the examples that can be used without data leakage."""
    generator = random.Random(SEED)
    frequency = Counter(entry["text"] for entry in entries)
    speakers = defaultdict(set)
    for entry in entries:
        if entry["source"] != "tux":
            speakers[entry["source"]].add(entry["speaker"])

    held_out = {}
    for corpus in sorted(speakers):
        order = sorted(speakers[corpus])
        generator.shuffle(order)
        held_out[order[0]] = "test"
        held_out[order[1]] = "val"

    chosen: list[dict] = []
    reserved: dict[str, str] = {}

    def reserve(candidates: list[dict], split: str, limit: int) -> None:
        taken = 0
        for entry in candidates:
            if taken == limit:
                break
            if reserved.setdefault(entry["text"], split) == split:
                chosen.append({**entry, "split": split})
                taken += 1

    for speaker, split in held_out.items():
        own = [entry for entry in entries if entry["speaker"] == speaker]
        generator.shuffle(own)
        # Prefer the least repeated sentences: that way fewer training clips are discarded.
        own.sort(key=lambda entry: frequency[entry["text"]])
        reserve(own, split, CLIPS_PER_HELD_OUT_SPEAKER)

    tux = [entry for entry in entries if entry["source"] == "tux"]
    generator.shuffle(tux)
    reserve(tux[: len(tux) // 2], "test", TUX_HELD_OUT)
    reserve(tux[len(tux) // 2 :], "val", TUX_HELD_OUT)

    for entry in entries:
        if entry["speaker"] not in held_out and entry["text"] not in reserved:
            chosen.append({**entry, "split": "train"})
    return chosen


def convert(entries: list[dict]) -> list[dict]:
    by_source = defaultdict(list)
    for entry in entries:
        by_source[entry["source"]].append(entry)

    kept = []
    for source, group in sorted(by_source.items()):
        bundle, members = None, {}
        if source != "tux":
            bundle = zipfile.ZipFile(DOWNLOADS / f"{source}.zip")
            members = {PurePosixPath(m).stem: m for m in bundle.namelist() if m.endswith(".wav")}

        def process(job: tuple[dict, bytes]) -> dict | None:
            entry, payload = job
            audio = trim_silence(load_audio(io.BytesIO(payload)))
            seconds = len(audio) / SAMPLE_RATE
            if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                return None
            # The network outputs 50 steps per second: if they are not enough, CTC cannot align.
            if len(entry["text"]) * 1.2 > seconds * 50:
                return None
            relative = f"audio/{entry['split']}/{entry['id']}.wav"
            target = PROCESSED / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, to_int16(audio), SAMPLE_RATE, subtype="PCM_16")
            return {"path": relative, **entry, "seconds": f"{seconds:.3f}"}

        with ThreadPoolExecutor(6) as pool:
            # In batches, so the whole .zip files are not loaded into memory.
            for start in range(0, len(group), 256):
                jobs = []
                for entry in group[start : start + 256]:
                    if bundle is None:
                        path = DOWNLOADS / "tux" / "wavs" / f"{entry['id'][4:]}.wav"
                        jobs.append((entry, path.read_bytes()))
                    elif entry["id"] in members:
                        jobs.append((entry, bundle.read(members[entry["id"]])))
                kept += [row for row in pool.map(process, jobs) if row]
        if bundle:
            bundle.close()
        print(f"{source}: done", flush=True)
    return kept


def main() -> None:
    raw = slr_entries() + tux_entries()
    entries = []
    for entry in raw:
        text = normalize(entry["raw_text"])
        if text:
            entries.append({**entry, "text": text})

    selected = assign_splits(entries)
    rows = sorted(convert(selected), key=lambda row: row["path"])
    write_csv(MANIFESTS / "audio.csv", rows)

    summary = {
        "downloaded": len(raw),
        "dropped_by_text": len(raw) - len(entries),
        "dropped_to_avoid_leakage": len(entries) - len(selected),
        "dropped_by_duration": len(selected) - len(rows),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        part = [row for row in rows if row["split"] == split]
        summary["splits"][split] = {
            "clips": len(part),
            "hours": round(sum(float(row["seconds"]) for row in part) / 3600, 3),
            "speakers": len({row["speaker"] for row in part}),
            "per_source": dict(Counter(row["source"] for row in part)),
        }
    write_json(REPORTS / "data_audio.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
