"""Text on screen: big event texts and subtitles (RapidOCR, PP-OCRv6, on DirectML).

One frame every `1 / fps` seconds (0.5 fps by default: every 2 s), resized to
a fixed 960x540 -- a fixed size keeps DirectML from recompiling per frame (51 ms
a frame measured, against 75-150 ms for variable crops). Only the seconds asked
for are read (where someone speaks, and around the action): reading all of a
2-hour recording took 6 minutes. Menu runs are never read.

What comes out:
  events     text matched against the lexicon (GTA 6, Portuguese and English):
             "A polícia está procurando você" (wanted: a chase starts),
             "Fugiram" (the police lost you), "concluído" / "WASTED" ... as
             (time, kind, text). Subtitles are never matched: dialogue is story.
  subtitles  seconds where the game is showing subtitles: speech under them is
             the game's dialogue, not the player (see voice.py).
  lines      every recognised line, for the AI review's dossier.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import video_io

W, H = 960, 540


@dataclass
class ScreenText:
    fps: float
    lines: list[dict] = field(default_factory=list)      # {t, text, score, box: [x0, y0, x1, y1] 0..1, zone}
    events: list[dict] = field(default_factory=list)     # {t, kind, text}

    def subtitle_seconds(self, n: int) -> np.ndarray:
        """1.0 for every second covered by a sample that showed a subtitle (a sample covers 1/fps seconds)."""
        out = np.zeros(n)
        span = max(1, int(round(1 / self.fps)))
        for line in self.lines:
            if line["zone"] == "subtitle":
                t = int(line["t"])
                out[max(0, t - span // 2):min(n, t + span)] = 1.0
        return out

    def near(self, start: float, end: float) -> list[dict]:
        return [line for line in self.lines if start <= line["t"] <= end]

    def to_json(self) -> dict:
        return {"fps": self.fps, "lines": self.lines, "events": self.events}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c) and (c.isalnum() or c == " ")).strip()


def zone(box: list[float]) -> str:
    x0, y0, x1, y1 = box
    cx, cy, height, width = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0, x1 - x0
    if cy >= 0.72 and 0.15 <= cx <= 0.85 and width >= 0.12:
        return "subtitle"
    if height >= 0.06 and 0.15 <= cy <= 0.8:
        return "big"
    return "hud"


def match_events(lines: list[dict], lexicon: dict[str, list[str]]) -> list[dict]:
    """Notifications and big text that match the lexicon. Whole words, accents and case ignored."""
    table = [(kind, normalize(word)) for kind, words in lexicon.items() for word in words]
    found = []
    for line in lines:
        if line["zone"] == "subtitle":
            continue
        padded = f" {normalize(line['text'])} "
        for kind, word in table:
            if f" {word} " in padded:
                found.append({"t": line["t"], "kind": kind, "text": line["text"]})
                break
    return found


def engine(gpu: bool = True):
    from rapidocr import RapidOCR

    params = {"Global.log_level": "error", "Global.use_cls": False}
    if gpu:
        params["EngineConfig.onnxruntime.use_dml"] = True
    return RapidOCR(params=params)


def _read(path: Path, fps: float, cancel: video_io.Canceller, out: queue.Queue) -> None:
    args = [video_io.ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", "0:v:0", "-vf", f"fps={fps:g},scale={W}:{H}:flags=bilinear", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-"]
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   creationflags=video_io._NO_WINDOW)
        with cancel.track(process):
            index = 0
            while True:
                chunk = process.stdout.read(W * H * 3)
                if len(chunk) < W * H * 3:
                    break
                out.put((index / fps, np.frombuffer(chunk, dtype=np.uint8).reshape(H, W, 3)))
                index += 1
            process.wait()
        out.put(None)
    except BaseException as error:  # noqa: BLE001 -- re-raised by the consumer
        out.put(error)


def scan(path: Path, duration: float, cfg: dict, skip: np.ndarray, want: np.ndarray | None = None, *, log,
         progress, cancel: video_io.Canceller) -> ScreenText:
    from .vision import gpu_available

    fps = float(cfg["fps"])
    reader = engine(gpu_available())
    expected = max(1, int(duration * fps))
    share = 1.0 if want is None else float(np.mean(want))
    log(f"   🔤 Reading text on screen ({share:.0%} of the recording, 1 frame every {1 / fps:g}s)…")
    frames: queue.Queue = queue.Queue(maxsize=16)
    threading.Thread(target=_read, args=(path, fps, cancel, frames), daemon=True).start()
    result = ScreenText(fps)
    done = 0
    while True:
        item = frames.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        cancel.check()
        t, image = item
        done += 1
        progress(min(done, expected), expected)
        second = int(t)
        if 0 <= second < len(skip) and skip[second]:
            continue
        if want is not None and not (0 <= second < len(want) and want[second]):
            continue
        found = reader(image)
        for text, score, points in zip(found.txts or (), found.scores or (), found.boxes if found.boxes is not None else ()):
            if float(score) < cfg["min_score"] or len(text.strip()) < 2:
                continue
            pts = np.asarray(points, dtype=float)
            box = [round(float(pts[:, 0].min()) / W, 3), round(float(pts[:, 1].min()) / H, 3),
                   round(float(pts[:, 0].max()) / W, 3), round(float(pts[:, 1].max()) / H, 3)]
            result.lines.append({"t": round(t, 2), "text": text.strip(), "score": round(float(score), 3),
                                 "box": box, "zone": zone(box)})
    result.events = match_events(result.lines, cfg["lexicon"])
    log(f"      {len(result.lines):,} lines of text, {len(result.events)} outcome texts")
    return result


def cache_file(cache: Path, fps: float) -> Path:
    return cache / f"ocr-{fps:g}fps.json"


def load_cached(cache: Path, cfg: dict) -> ScreenText | None:
    file = cache_file(cache, cfg["fps"])
    if not file.is_file():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    text = ScreenText(data["fps"], data["lines"])
    text.events = match_events(text.lines, cfg["lexicon"])  # the lexicon may have changed since
    return text


def save(cache: Path, text: ScreenText) -> None:
    cache_file(cache, text.fps).write_text(json.dumps(text.to_json(), ensure_ascii=False), encoding="utf-8")
