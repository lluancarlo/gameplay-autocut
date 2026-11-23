from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import config, video_io
from ..config import DATA_DIR
from ..pipeline import Span
from . import audio, concepts, labels, motion, signals, sounds, speech, vision, visual
from .labels import UNLABELLED, Evidence
from .speech import Word

REASONS = {"loudness": "loud", "speech": "fast speech", "novelty": "sudden change on screen"}
# Sound groups whose curve is "how far above this recording's usual level", like `action`.
RELATIVE_GROUPS = ("vehicle", "siren", "crash")


@dataclass
class Moment:
    start: float
    end: float
    peak: float
    score: int
    reasons: list[str]
    label: str = UNLABELLED
    label2: str = ""
    level: str = ""                 # "low" | "medium" | "high": not a calibrated probability
    label_score: float = 0.0        # how far the best evidence clears its rule's bar (>= 1 qualifies)
    evidence: list[Evidence] = field(default_factory=list)
    explanation: str = ""
    motion: dict | None = None      # camera / local optical-flow summary, for the strongest moments only

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
    vision: dict | None = None   # summary of the visual pass (frames, fps, provider, seconds), None if skipped

    def to_json(self) -> dict:
        return {"path": str(self.path), "duration": self.duration, "language": self.language,
                "moments": [asdict(m) for m in self.moments], "curve": self.curve, "vision": self.vision}


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
    """Every class's probability per second, cached as float16 -- about 4 MB per hour."""
    file = cache / "sounds.npz"
    if file.is_file():
        with np.load(file) as data:
            return {"probs": data["probs"].astype(np.float32), "names": [str(n) for n in data["names"]]}
    log("   🔊 Listening for shouts, gunfire, explosions, engines and sirens…")
    result = sounds.tag(samples, log, span.counting(), cancel)
    np.savez_compressed(file, probs=result["probs"].astype(np.float16), names=np.array(result["names"]))
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


def _label(moment: Moment, groups, curves, hits, phrases, cfg, vis, motion_summary: dict | None = None) -> None:
    rules = cfg["labels"]["rules"]
    moment.evidence = labels.collect(moment.start, moment.end, groups, curves, hits, phrases, cfg["labels"]["present"],
                                     vis, motion_summary, cfg.get("visual", {}).get("motion"))
    moment.label, moment.label2, moment.label_score, moment.level = labels.classify(moment.evidence, rules)
    supporting = tuple(rules.get(moment.label, {}).get("signals", {}))
    moment.explanation = labels.explain(moment.label, moment.label2, moment.evidence, supporting)


def _visual(seen, n: int, cfg: dict, log):
    """Concept, novelty and menu curves from the embeddings, or None (sound-only analysis)."""
    if seen is None or "visual" not in cfg:
        return None
    try:
        bank = concepts.load()
    except (concepts.BankMismatch, OSError, KeyError) as error:
        log(f"   ⚠️  Vision scores skipped: {error}")
        return None
    events = json.loads((config.CONFIG_DIR / "events.json").read_text(encoding="utf-8"))["concepts"]
    describe = {name: entry["prompts"][0] for name, entry in events.items()}
    return visual.build(seen, bank, n, cfg["visual"], describe)


def _onset(start: float, end: float, peak: int, activity: np.ndarray, clip: dict) -> tuple[float, float]:
    """Start where the action starts, not where it peaks.

    The fused score peaks at the loudest point -- usually the reaction, near the
    end of a fight or a chase -- so a window built around the peak opens in the
    middle of the action. Walk back from the peak through the run of seconds
    where something is happening (gunfire, explosions, engines, sirens, crashes,
    the picture, loudness), stepping over pauses up to `gap_s`, and open
    `pre_roll_s` before that run begins. The clip may grow to `max` for it; if it
    must be cut, the end is cut, never the build-up.
    """
    cfg = clip["onset"]
    level, gap = cfg["level"], int(cfg["gap_s"])
    limit = max(0, peak - int(cfg["lookback_s"]))
    t = peak
    while t > limit and (activity[max(limit, t - 1 - gap):t] >= level).any():
        t -= 1
        while t > limit and activity[t] < level:
            t -= 1
    begin = max(0.0, t - cfg["pre_roll_s"])
    if begin >= start:
        return start, end
    end = min(end, begin + clip["max"])
    if end < peak + clip["tail"]:  # never cut off the peak itself
        end = peak + clip["tail"]
        begin = max(begin, end - clip["max"])
    return begin, end


def _runs(flags: np.ndarray, length: int) -> np.ndarray:
    """1.0 where `flags` is part of a run of at least `length` consecutive seconds, else 0.0.

    Real menus and maps stay up for several seconds; the vision model also calls
    the odd dark or upside-down gameplay second a menu, and those come alone.
    """
    out = np.zeros(len(flags))
    t = 0
    while t < len(flags):
        if flags[t]:
            u = t
            while u < len(flags) and flags[u]:
                u += 1
            if u - t >= length:
                out[t:u] = 1.0
            t = u
        else:
            t += 1
    return out


def _avoid_menus(start: float, end: float, peak: int, menu: np.ndarray, clip: dict) -> tuple[float, float]:
    """Never open or close a clip on a menu, map or pause screen.

    Walking back to where the action starts can cross into the map the player
    had open a moment before, and nobody wants a short that opens on a map.
    The clip starts right after the last menu second before the peak and ends
    before the first one after it; if that leaves it too short, it grows on
    whichever side is still gameplay.
    """
    n = len(menu)
    lo, hi = max(0, int(start)), min(n, int(np.ceil(end)))
    before = [t for t in range(lo, min(peak, hi)) if menu[t] >= 0.5]
    after = [t for t in range(peak + 1, hi) if menu[t] >= 0.5]
    if before:
        start = float(before[-1] + 1)
    if after:
        end = float(after[0])
    short = clip["min"] - (end - start)
    if short > 0:
        grow_end = 0
        while grow_end < short and int(end) + grow_end < n and menu[int(end) + grow_end] < 0.5:
            grow_end += 1
        end += grow_end
        grow_start = 0
        while grow_start < short - grow_end and int(start) - grow_start - 1 >= 0                 and menu[int(start) - grow_start - 1] < 0.5:
            grow_start += 1
        start -= grow_start
    return start, end


def _lead_in(moment: Moment, vis, cfg: dict, n: int) -> bool:
    """Start a picture-led moment before what the picture shows begins.

    The concept peaks once the thing is on screen -- someone already in the
    water, a car already wrecked -- so the clip has to start where that run of
    seconds starts, plus a few seconds of lead-in, or it shows only the
    aftermath. True if the bounds moved.
    """
    top = next((e for e in moment.evidence if e.signal.startswith(labels.SEE)
                and e.signal[len(labels.SEE):] in vis.relative), None)
    if top is None or moment.label in (UNLABELLED, "reaction"):
        return False
    rel = vis.relative[top.signal[len(labels.SEE):]]
    lead = cfg["visual"]["lead"]
    t, gap = int(top.at), int(lead["gap_s"])
    # Walk back through the run of seconds showing it, stepping over short dropouts
    # (a murky underwater frame, a cut): `gap` seconds without it end the run.
    while t > 0 and (rel[max(0, t - 1 - gap):t] >= lead["level"]).any():
        t -= 1
        while t > 0 and rel[t] < lead["level"]:
            t -= 1
    start = max(0.0, t - lead["seconds"])
    if start >= moment.start:
        return False
    moment.start = round(start, 2)
    longest = cfg["clip"]["max"]
    if moment.end - moment.start > longest:
        moment.end = round(moment.start + longest, 2)
    return True


def _dedupe(moments: list[Moment], per_second: np.ndarray, cfg: dict) -> list[Moment]:
    """Merge same-label moments that nearly touch, and drop same-label ones that look the same."""
    vcfg, longest = cfg["visual"], cfg["clip"]["max"]
    ordered = sorted(moments, key=lambda m: m.start)
    merged: list[Moment] = []
    for moment in ordered:
        last = merged[-1] if merged else None
        if (last is not None and last.label == moment.label and moment.label != UNLABELLED
                and moment.start - last.end < vcfg["merge_gap_s"] and moment.end - last.start <= longest):
            keep = last if last.score >= moment.score else moment
            merged[-1] = Moment(last.start, moment.end, keep.peak, max(last.score, moment.score),
                                list(dict.fromkeys(last.reasons + moment.reasons)))
            merged[-1].label = moment.label
            continue
        merged.append(moment)

    def look(m: Moment) -> np.ndarray:
        block = per_second[int(m.start):max(int(m.start) + 1, int(m.end))].mean(axis=0)
        norm = np.linalg.norm(block)
        return block / norm if norm else block

    kept: list[Moment] = []
    for moment in sorted(merged, key=lambda m: -m.score):
        vec = look(moment)
        if any(k.label == moment.label and moment.label != UNLABELLED and float(look(k) @ vec) > vcfg["dedupe_cos"]
               for k in kept):
            continue
        kept.append(moment)
    return kept


def _motion(path: Path, moments: list[Moment], groups, curves, hits, phrases, cfg, vis, log, span: Span,
            cancel) -> None:
    """Optical flow on the strongest moments; motion becomes evidence and the labels are redone with it."""
    mcfg = cfg["visual"]["motion"]
    top = sorted(moments, key=lambda m: -m.score)[: int(mcfg["top"])]
    if not top:
        return
    try:
        info = video_io.probe(path)
        aspect = info.width / info.height if info.height else 16 / 9
    except Exception:  # noqa: BLE001
        aspect = 16 / 9
    log(f"   🌊 Measuring motion in the top {len(top)} moments…")
    for index, moment in enumerate(top, start=1):
        cancel.check()
        try:
            measured = motion.window(path, moment.start, moment.end, float(mcfg["fps"]), aspect, cancel).summary()
        except video_io.Cancelled:
            raise
        except Exception as error:  # noqa: BLE001 -- motion is extra evidence, never a reason to fail
            log(f"   ⚠️  Motion skipped for {clock(moment.start)}: {error}")
            continue
        moment.motion = measured
        _label(moment, groups, curves, hits, phrases, cfg, vis, measured)
        span.at(index / len(top))


def _vision(path: Path, duration: float, cache: Path, cfg: dict, log, span: Span, cancel):
    """The SigLIP 2 pass (M2): embeddings of every sampled frame, cached. None if off or unavailable.

    A failure here never costs the user the audio analysis: it is logged and the
    moments are found from the sound alone, as before.
    """
    if not cfg.get("enabled", False):
        return None
    try:
        cached = vision.load_cached(cache, cfg)
        if cached is not None:
            return cached
        result = vision.embed(path, duration, cfg, log=log, progress=span.counting(), cancel=cancel)
        vision.save(cache, result)
        return result
    except video_io.Cancelled:
        raise
    except Exception as error:  # noqa: BLE001 -- vision is an extra; the sound analysis stands on its own
        log(f"   ⚠️  Visual analysis skipped: {error}")
        return None


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
    for name in sorted(shares, key=lambda k: -shares[k] * curves[k][lo:hi].max(initial=0)):
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


def _fit(probs: np.ndarray, names: list[str], n: int) -> tuple[np.ndarray, list[str]]:
    """Trim or pad a per-second series to exactly n seconds (the voice track can differ by one)."""
    probs, names = probs[:n], list(names[:n])
    if len(probs) < n:
        probs = np.pad(probs, (0, n - len(probs)))
        names += [""] * (n - len(names))
    return probs, names


def fuse(curves: dict[str, np.ndarray], cfg: dict,
         suppress: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """The hype curve: weighted curves, a bonus where several agree, smoothed.

    A curve whose weight is 0 (or missing) takes no part at all, so adding a new
    signal at weight 0 leaves the result exactly as it was.
    """
    weights = {k: cfg["weights"].get(k, 0.0) for k in curves}
    used = {k: w for k, w in weights.items() if w > 0}
    total = sum(used.values())
    shares = {k: w / total for k, w in used.items()}
    combined = sum(shares[k] * curves[k] for k in used)
    agreeing = sum((curves[k] >= 0.5).astype(float) for k in used if shares[k] >= 0.08)
    combined = combined * (1 + cfg["agreement_bonus"] * np.maximum(0, agreeing - 1))
    if suppress is not None:
        combined = combined * (1 - np.clip(suppress, 0, 1))
    return signals.smooth(signals.clip01(combined), cfg["smooth_sigma"]), shares


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
    language, words = _speech(voice_samples, cache, log, whole.slice(0.3, 0.7), cancel)
    seen = _vision(path, len(samples) / audio.SAMPLE_RATE, cache, cfg.get("vision", {}), log,
                   whole.slice(0.7, 0.95), cancel)

    members = cfg["sound_groups"]
    groups = {name: sounds.group(reaction_tags if name == "reaction" else tagged, names)
              for name, names in members.items()}
    groups = {name: _fit(probs, names, n) for name, (probs, names) in groups.items()}
    action = np.maximum(groups["gunfire"][0], groups["explosion"][0])
    action_labels = [g if a >= e else x for a, e, g, x in
                     zip(groups["gunfire"][0], groups["explosion"][0], groups["gunfire"][1], groups["explosion"][1])]

    curves = {
        "loudness": signals.loudness(signals.loudness_db(voice_samples, n), cfg["loudness"]),
        "reaction": signals.reaction(groups["reaction"][0], cfg["reaction"]),
        "action": signals.action(action, cfg["action"]),
    }
    for name in RELATIVE_GROUPS:
        curves[name] = signals.action(groups[name][0], cfg["action"])
    hits: list = []
    phrases = {p: w for lang in cfg["keywords"].values() for p, w in lang.items()}
    if words:
        curves["speech"] = signals.speech(words, n, cfg["speech"])
        curves["keywords"], hits = signals.keywords(words, n, phrases)

    vis = _visual(seen, n, cfg, log)
    suppress = None
    if vis is not None:
        curves["vision"] = vis.vision
        curves["novelty"] = vis.novelty * (1 - vis.menu)
        suppress = cfg["visual"]["suppress"] * vis.menu
    curve, shares = fuse(curves, cfg, suppress)
    sound_labels = {"reaction": groups["reaction"][1], "action": action_labels,
                    **{name: groups[name][1] for name in RELATIVE_GROUPS}}
    if vis is not None:
        names = list(vis.relative)
        strongest = np.argmax(np.stack([vis.relative[k] for k in names]), axis=0)
        sound_labels["vision"] = [vis.describe.get(names[i], names[i]) for i in strongest]
    menu_runs = _runs(vis.menu >= cfg["visual"]["menu_run"]["level"], int(cfg["visual"]["menu_run"]["seconds"]))         if vis is not None else np.zeros(n)
    onset_names = [k for k in cfg["clip"]["onset"]["signals"] if k in curves]
    activity = np.max([curves[k] for k in onset_names], axis=0) if onset_names else np.zeros(n)
    candidates = []
    skipped = 0
    for peak in _peaks(curve, cfg["peaks"]["min_gap"], cfg["peaks"]["floor"]):
        if vis is not None and vis.menu[peak] >= cfg["visual"]["skip_menu_peaks"]:
            skipped += 1
            continue  # a menu, map or pause screen, however loud
        start, end = _onset(*_window(curve, peak, cfg["clip"]), peak, activity, cfg["clip"])
        if vis is not None:
            start, end = _avoid_menus(start, end, peak, menu_runs, cfg["clip"])
            if end - start < cfg["clip"]["min"] * 0.8:
                skipped += 1
                continue  # squeezed between menus: too short to be worth a short
        start, end = _snap(start, end, words)
        score = round(100 * min(1.0, curve[peak] / cfg["peaks"]["full_score"]))
        moment = Moment(start, end, float(peak), score, _reasons(curves, shares, peak, sound_labels, hits))
        _label(moment, groups, curves, hits, phrases, cfg, vis)
        candidates.append(moment)

    moments: list[Moment] = []
    for moment in sorted(candidates, key=lambda m: -m.score):
        if all(moment.end <= m.start or moment.start >= m.end for m in moments):
            moments.append(moment)
    if vis is not None:
        moments = _dedupe(moments, seen.per_second(n), cfg)
        for moment in moments:
            _label(moment, groups, curves, hits, phrases, cfg, vis)
            if _lead_in(moment, vis, cfg, len(curve)):
                _label(moment, groups, curves, hits, phrases, cfg, vis)
        _motion(path, moments, groups, curves, hits, phrases, cfg, vis, log, whole.slice(0.95, 0.995), cancel)
    for moment in moments:
        if moment.label == "vehicle" and labels.escape(curves, moment.end, cfg["labels"]["escape"]):
            moment.label2 = "escape"
    moments.sort(key=lambda m: -m.score)
    whole.done()
    log(f"   ✨ {len(moments)} moments")
    shown = {k: v.round(4).tolist() for k, v in curves.items()}
    summary = None
    if seen is not None:
        summary = {"frames": seen.frames, "fps": seen.fps, "provider": seen.provider, "seconds": round(seen.seconds, 1)}
    if vis is not None:
        shown["menu"] = vis.menu.round(3).tolist()
        summary["menu_seconds"] = int((vis.menu >= 0.5).sum())
        summary["menu_skipped"] = skipped
    return Analysis(path, len(samples) / audio.SAMPLE_RATE, language, moments,
                    curve.round(4).tolist(), cache, shown, summary)
