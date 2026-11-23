"""The evaluation folder: which recordings exist, their splits, and their labels.

    <folder>/splits.json          every recording: path, game, split, tracks
    <folder>/labels/<id>.json     one label file per recording
    <folder>/baseline/<id>.json   frozen predictions of the reference detector
    <folder>/runs/latest/<id>.json  predictions of the last evaluation run

Relative recording paths resolve against the folder's parent, so a repo-level
`eval/` can say "GTA-6.mp4" for a video sitting at the repo root. Label files
are kept as plain dicts so fields this code does not know about survive edits.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .. import config
from .metrics import LABELS, SPLITS, Recording, Truth

STATUSES = ("draft", "reviewed")


def default_folder() -> Path:
    """The repo's eval/ when running from source; VVideo-data/eval beside the .exe."""
    if getattr(sys, "frozen", False):
        return config.DATA_DIR / "eval"
    return config.ROOT / "eval"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "recording"


def validate_labels(rec_id: str, data: dict) -> None:
    for m in data.get("moments", []):
        if not 0 <= m["rating"] <= 3:
            raise ValueError(f"{rec_id}: rating must be 0-3, got {m['rating']}")
        if m.get("label") not in (None, *LABELS):
            raise ValueError(f"{rec_id}: unknown label {m.get('label')!r}")
        if m["end"] <= m["start"]:
            raise ValueError(f"{rec_id}: moment ends before it starts: {m}")
    for name in ("reviewed_spans", "non_gameplay"):
        for start, end in data.get(name, []):
            if end <= start:
                raise ValueError(f"{rec_id}: {name} span ends before it starts: {[start, end]}")
    if data.get("status", "draft") not in STATUSES:
        raise ValueError(f"{rec_id}: status must be one of {STATUSES}")


class EvalSet:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    # -- layout -------------------------------------------------------------

    @property
    def splits_file(self) -> Path:
        return self.folder / "splits.json"

    @property
    def baseline_dir(self) -> Path:
        return self.folder / "baseline"

    @property
    def latest_dir(self) -> Path:
        return self.folder / "runs" / "latest"

    def labels_file(self, rec_id: str) -> Path:
        return self.folder / "labels" / f"{rec_id}.json"

    # -- splits -------------------------------------------------------------

    def entries(self) -> dict[str, dict]:
        if not self.splits_file.is_file():
            return {}
        recordings = _read(self.splits_file).get("recordings", {})
        for rec_id, entry in recordings.items():
            if entry.get("split") not in SPLITS:
                raise ValueError(f"{rec_id}: split must be one of {SPLITS}, got {entry.get('split')!r}")
        return recordings

    def games(self) -> list[str]:
        return sorted({e["game"] for e in self.entries().values()})

    def video(self, entry: dict) -> Path:
        path = Path(entry["path"])
        return path if path.is_absolute() else self.folder.parent / path

    def add_recording(self, video: Path, rec_id: str, game: str, split: str, duration: float,
                      voice_track: int | None = None, note: str = "") -> str:
        """Register a recording and start an empty draft label file. Its split never changes afterwards."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        rec_id = slug(rec_id)
        data = _read(self.splits_file) if self.splits_file.is_file() else {"version": 1, "recordings": {}}
        if rec_id in data["recordings"] or self.labels_file(rec_id).exists():
            raise ValueError(f"A recording called '{rec_id}' already exists.")
        video = Path(video).resolve()
        try:
            stored = video.relative_to(self.folder.resolve().parent).as_posix()
        except ValueError:
            stored = str(video)
        entry = {"path": stored, "game": slug(game), "split": split}
        if voice_track is not None:
            entry["voice_track"] = voice_track
        if note:
            entry["note"] = note
        data["recordings"][rec_id] = entry
        _write(self.labels_file(rec_id), {
            "version": 1, "recording": rec_id, "status": "draft", "annotator": "",
            "duration": round(duration, 2), "reviewed_spans": [], "non_gameplay": [], "moments": [],
        })
        _write(self.splits_file, data)
        return rec_id

    # -- labels -------------------------------------------------------------

    def read_labels(self, rec_id: str) -> dict:
        data = _read(self.labels_file(rec_id))
        validate_labels(rec_id, data)
        return data

    def write_labels(self, rec_id: str, data: dict) -> None:
        validate_labels(rec_id, data)
        data["moments"].sort(key=lambda m: m["start"])
        for name in ("reviewed_spans", "non_gameplay"):
            data[name] = merge_spans(data.get(name, []))
        _write(self.labels_file(rec_id), data)

    def recording(self, rec_id: str) -> Recording:
        entry = self.entries()[rec_id]
        data = self.read_labels(rec_id)
        truths = [Truth(m["start"], m["end"], m["rating"], m.get("label"), m.get("label2"), m.get("note", ""))
                  for m in data["moments"]]
        return Recording(
            id=rec_id, game=entry["game"], split=entry["split"], path=str(self.video(entry)),
            status=data.get("status", "draft"), duration=float(data["duration"]),
            reviewed=[tuple(s) for s in data["reviewed_spans"]],
            non_gameplay=[tuple(s) for s in data.get("non_gameplay", [])],
            truths=truths, track=entry.get("track", 0), voice_track=entry.get("voice_track"),
        )

    def select(self, split: str, include_drafts: bool) -> tuple[list[Recording], list[Recording]]:
        """(scored, skipped) for a split name or 'all'."""
        scored, skipped = [], []
        for rec_id, entry in self.entries().items():
            if split != "all" and entry["split"] != split:
                continue
            rec = self.recording(rec_id)
            (scored if rec.status == "reviewed" or include_drafts else skipped).append(rec)
        return scored, skipped


def merge_spans(spans: list) -> list[list[float]]:
    out: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [[round(s, 2), round(e, 2)] for s, e in out]


def subtract_span(spans: list, start: float, end: float) -> list[list[float]]:
    out = []
    for s, e in spans:
        if e <= start or s >= end:
            out.append([s, e])
            continue
        if s < start:
            out.append([s, start])
        if e > end:
            out.append([end, e])
    return merge_spans(out)
