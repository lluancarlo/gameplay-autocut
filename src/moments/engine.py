from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import config, video_io
from ..config import DATA_DIR
from ..pipeline import Span
from . import audio, signals, sounds, speech
from .speech import Word

REASONS = {"loudness": "loud", "speech": "fast speech"}


@dataclass
class Moment:
    start: float
    end: float
    peak: float
    score: int
    reasons: list[str]

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Analysis:
    path: Path
    duration: float
    language: str
    moments: list[Moment]
    curve: list[float]
    cache: Path
    signals: dict[str, list[float]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"path": str(self.path), "duration": self.duration, "language": self.language,
                "moments": [asdict(m) for m in self.moments], "curve": self.curve}


def clock(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _ignore(*_args) -> None:
    pass


def _cache_dir(path: Path, track: int, voice: int) -> Path:
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{track}|{voice}".encode()).hexdigest()
    return DATA_DIR / "cache" / key[:16]


def _sounds(samples, cache: Path, log, span: Span, cancel) -> dict:
    file = cache / "sounds.json"
    if file.is_file():
        data = json.loads(file.read_text(encoding="utf-8"))
        return {"probs": {g: np.array(v) for g, v in data["probs"].items()}, "labels": data["labels"]}
    log("   🔊 Listening for shouts, laughter, gunfire and explosions…")
    result = sounds.tag(samples, log, span.counting(), cancel)
    file.write_text(json.dumps({"probs": {g: v.round(4).tolist() for g, v in result["probs"].items()},
                                "labels": result["labels"]}), encoding="utf-8")
    return result


def _speech(samples, cache: Path, log, span: Span, cancel) -> tuple[str, list[Word]]:
    file = cache / "words.json"
    if file.is_file():
        data = json.loads(file.read_text(encoding="utf-8"))
        return data["language"], [Word(**w) for w in data["words"]]
    log("   📝 Transcribing…")
    language, words = speech.transcribe(samples, log, span.counting(), cancel)
    file.write_text(json.dumps({"language": language, "words": [asdict(w) for w in words]},
                               ensure_ascii=False), encoding="utf-8")
    return language, words


def _peaks(curve: np.ndarray, gap: int, floor: float) -> list[int]:
    half = gap // 2
    neighbourhood = np.lib.stride_tricks.sliding_window_view(np.pad(curve, half, constant_values=-1), 2 * half + 1)
    return [int(i) for i in np.flatnonzero((curve >= neighbourhood.max(axis=1)) & (curve >= floor))]


def _window(curve: np.ndarray, peak: int, cfg: dict) -> tuple[float, float]:
    level = curve[peak] * cfg["extent"]
    a = b = peak
    while a > 0 and curve[a - 1] >= level and peak - a < cfg["max"]:
        a -= 1
    while b < len(curve) - 1 and curve[b + 1] >= level and b - peak < cfg["max"]:
        b += 1
    start, end = a - cfg["setup"], b + 1 + cfg["tail"]
    short = cfg["ideal"] - (end - start)
    if short > 0:
        start, end = start - short * 0.7, end + short * 0.3
    excess = (end - start) - cfg["max"]
    if excess > 0:
        trim_start = min(excess, max(0.0, peak - cfg["setup"] - start))
        start, end = start + trim_start, end - (excess - trim_start)
    start, end = max(0.0, start), min(float(len(curve)), end)
    if end - start < cfg["min"]:
        end = min(float(len(curve)), start + cfg["min"])
        start = max(0.0, end - cfg["min"])
    return start, end


def _snap(start: float, end: float, words: list[Word]) -> tuple[float, float]:
    for word in words:
        if word.start < start < word.end:
            start = word.start
        if word.start < end < word.end:
            end = word.end
    return round(start, 2), round(end, 2)


def _reasons(curves: dict, shares: dict, peak: int, sound_labels: dict, hits: list) -> list[str]:
    lo, hi = max(0, peak - 4), peak + 4
    found = []
    for name in sorted(curves, key=lambda k: -shares[k] * curves[k][lo:hi].max(initial=0)):
        part = curves[name][lo:hi]
        if part.max(initial=0) < 0.5:
            continue
        if name == "keywords":
            found += [f'"{p}"' for t, p in hits if lo <= t < hi and f'"{p}"' not in found]
        elif name in sound_labels:
            label = sound_labels[name][lo + int(part.argmax())].split(",")[0].lower()
            found.append(label)
        else:
            found.append(REASONS[name])
    return list(dict.fromkeys(found))


def find_moments(
    path: Path,
    *,
    track: int = 0,
    voice_track: int | None = None,
    log=_ignore,
    progress=_ignore,
    cancel: video_io.Canceller | None = None,
    fresh: bool = False,
) -> Analysis:
    cfg = config.load("moments")
    cancel = cancel or video_io.Canceller()
    whole = Span(progress)
    voice = track if voice_track is None else voice_track
    cache = _cache_dir(path, track, voice)
    if fresh and cache.is_dir():
        for file in cache.iterdir():
            file.unlink()
    cache.mkdir(parents=True, exist_ok=True)

    log(f"🎮 {path.name}")
    samples = audio.load(path, track, cancel)
    voice_samples = samples if voice == track else audio.load(path, voice, cancel)
    whole.at(0.05)
    n = int(np.ceil(len(samples) / audio.SAMPLE_RATE))
    log(f"   ⏱️  {n // 60}m{n % 60:02d}s of audio")

    tagged = _sounds(samples, cache, log, whole.slice(0.05, 0.25), cancel)
    reaction_tags = tagged
    if voice != track:
        (cache / "voice").mkdir(exist_ok=True)
        reaction_tags = _sounds(voice_samples, cache / "voice", log, whole.slice(0.25, 0.3), cancel)
    language, words = _speech(voice_samples, cache, log, whole.slice(0.3, 0.97), cancel)

    curves = {
        "loudness": signals.loudness(signals.loudness_db(voice_samples, n), cfg["loudness"]),
        "reaction": signals.reaction(reaction_tags["probs"]["reaction"][:n], cfg["reaction"]),
        "action": signals.action(tagged["probs"]["action"][:n], cfg["action"]),
    }
    hits: list = []
    if words:
        phrases = {p: w for lang in cfg["keywords"].values() for p, w in lang.items()}
        curves["speech"] = signals.speech(words, n, cfg["speech"])
        curves["keywords"], hits = signals.keywords(words, n, phrases)

    weights = cfg["weights"]
    total = sum(weights[k] for k in curves)
    shares = {k: weights[k] / total for k in curves}
    combined = sum(shares[k] * curves[k] for k in curves)
    agreeing = sum((curves[k] >= 0.5).astype(float) for k in curves if shares[k] >= 0.08)
    combined = combined * (1 + cfg["agreement_bonus"] * np.maximum(0, agreeing - 1))
    curve = signals.smooth(signals.clip01(combined), cfg["smooth_sigma"])

    labels = {"reaction": reaction_tags["labels"]["reaction"], "action": tagged["labels"]["action"]}
    candidates = []
    for peak in _peaks(curve, cfg["peaks"]["min_gap"], cfg["peaks"]["floor"]):
        start, end = _snap(*_window(curve, peak, cfg["clip"]), words)
        score = round(100 * min(1.0, curve[peak] / cfg["peaks"]["full_score"]))
        candidates.append(Moment(start, end, float(peak), score, _reasons(curves, shares, peak, labels, hits)))

    moments: list[Moment] = []
    for moment in sorted(candidates, key=lambda m: -m.score):
        if all(moment.end <= m.start or moment.start >= m.end for m in moments):
            moments.append(moment)
    whole.done()
    log(f"   ✨ {len(moments)} moments")
    return Analysis(path, len(samples) / audio.SAMPLE_RATE, language, moments,
                    curve.round(4).tolist(), cache, {k: v.round(4).tolist() for k, v in curves.items()})
