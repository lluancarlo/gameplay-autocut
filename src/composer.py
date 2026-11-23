"""Render one short: optional teaser, the clip (minus its dead seconds), captions, framing, clean audio.

    [teaser: 1.5 s of the best moment] + [piece] + [piece] ...  ->  1080x1920, captions burned in

Every piece is its own seek into the recording, reframed the same way, then
all are concatenated. One ffmpeg call per short.

Framing modes:
  bars     the clip centred at `zoom` with blurred copies above and below (shorts.py).
  follow   a full-height 9:16 window that glides to where the action is (motion.focus_track).

Audio:
  - every track is mixed, so a microphone on its own track is in the short;
  - when the player's voice has its own track, the game ducks under it and the
    voice gets a light denoise;
  - the result is normalised to the loudness the platforms play at (-14 LUFS),
    so no short is too quiet or clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import captions, config, shorts, video_io

TEASER_S = 1.5
FADE_S = 0.02    # a few milliseconds of fade at every cut: no click where two pieces meet


@dataclass
class Short:
    start: float
    end: float
    teaser_at: float | None = None                       # centre of the teaser, seconds in the recording
    captions: Path | None = None                          # .ass file, timed on the short's own timeline
    focus: list[tuple[float, float]] = field(default_factory=list)   # follow mode: (seconds from start, x 0..1)
    segments: list[tuple[float, float]] = field(default_factory=list)  # what is kept of [start, end]; empty = all
    voice_track: int | None = None                        # the player's microphone track, when separate

    @property
    def teaser(self) -> float:
        return TEASER_S if self.teaser_at is not None else 0.0

    @property
    def pieces(self) -> list[tuple[float, float]]:
        return self.segments or [(self.start, self.end)]

    @property
    def length(self) -> float:
        return self.teaser + sum(b - a for a, b in self.pieces)


def audio_settings() -> dict:
    return config.load("shorts").get("audio", {})


def trim_settings() -> dict:
    return config.load("shorts").get("trim", {})


def keep_segments(start: float, end: float, curve: list[float], talking: list[tuple[float, float]],
                  protect: list[float], cfg: dict) -> list[tuple[float, float]]:
    """The parts of [start, end] worth keeping: runs of dead seconds inside the clip are cut out.

    A second is dead when the moment curve sits below `level` of the clip's own
    peak and nobody is talking (the player, or game dialogue on screen). Only
    runs of at least `min_s` count; `keep_s` of each side stays, so the cut is a
    jump, not a jolt. The opening and closing seconds, and the seconds around
    `protect` (the peak, the teaser), are never cut, and the clip never drops
    below `min_length_s`: the longest runs go first.
    """
    first, last = int(np.ceil(start)), int(np.floor(end))
    if last - first < 4 or not curve:
        return [(start, end)]
    values = np.array([curve[s] if 0 <= s < len(curve) else 0.0 for s in range(first, last)])
    if values.max() <= 0:
        return [(start, end)]
    dead = values < cfg["level"] * values.max()
    guard = cfg["edge_s"]
    for s in range(first, last):
        i = s - first
        if s < start + guard or s + 1 > end - guard:
            dead[i] = False
        elif any(a - 0.5 < s + 1 and s < b + 0.5 for a, b in talking):
            dead[i] = False
        elif any(abs(s + 0.5 - p) <= cfg["protect_s"] for p in protect):
            dead[i] = False
    runs, i = [], 0
    while i < len(dead):
        if dead[i]:
            j = i
            while j < len(dead) and dead[j]:
                j += 1
            if j - i >= cfg["min_s"]:
                runs.append((first + i + cfg["keep_s"], first + j - cfg["keep_s"]))
            i = j
        else:
            i += 1
    budget = (end - start) - cfg["min_length_s"]
    cuts = []
    for a, b in sorted(runs, key=lambda r: r[0] - r[1]):   # longest first
        if b - a <= budget:
            cuts.append((a, b))
            budget -= b - a
    if not cuts:
        return [(start, end)]
    kept, at = [], start
    for a, b in sorted(cuts):
        kept.append((round(at, 2), round(a, 2)))
        at = b
    kept.append((round(at, 2), round(end, 2)))
    return kept


def _lerp_expr(keys: list[tuple[float, float]]) -> str:
    """ffmpeg expression of t, piecewise linear through (t, value) keys."""
    if len(keys) == 1:
        return f"{keys[0][1]:.1f}"
    expr = f"{keys[-1][1]:.1f}"
    for (t0, v0), (t1, v1) in reversed(list(zip(keys[:-1], keys[1:]))):
        slope = (v1 - v0) / max(t1 - t0, 1e-3)
        expr = f"if(lt(t,{t1:.2f}),{v0:.1f}+({slope:.3f})*(t-{t0:.2f}),{expr})"
    return expr


def follow_graph(source: str, out: str, info: video_io.VideoInfo, settings: shorts.Settings,
                 keys: list[tuple[float, float]]) -> str:
    """Full-height 9:16 crop whose horizontal centre follows `keys` (x as a 0..1 fraction)."""
    width = min(info.width, int(round(info.height * settings.canvas_w / settings.canvas_h / 2)) * 2)
    half = width / 2
    pixels = [(t, min(max(x * info.width - half, 0.0), info.width - width)) for t, x in (keys or [(0.0, 0.5)])]
    return (f"[{source}]crop=w={width}:h={info.height}:x='{_lerp_expr(pixels)}':y=0,"
            f"scale={settings.canvas_w}:{settings.canvas_h},setsar=1[{out}]")


def _focus_at(focus: list[tuple[float, float]], seconds: float) -> float:
    return min(focus, key=lambda k: abs(k[0] - seconds))[1] if focus else 0.5


def _focus_piece(focus: list[tuple[float, float]], offset: float, length: float) -> list[tuple[float, float]]:
    """The focus keys of one piece, on that piece's own clock (each input starts at t=0)."""
    keys = [(t - offset, x) for t, x in focus if offset <= t <= offset + length]
    if not keys or keys[0][0] > 0:
        keys.insert(0, (0.0, _focus_at(focus, offset)))
    return keys


def _audio_graph(i: int, tracks: int, voice: int | None, cfg: dict) -> str:
    """One piece's audio, as [a{i}]: tracks mixed, the game ducked under a separate voice track."""
    if tracks == 0:
        return ""
    if voice is not None and 0 <= voice < tracks and tracks > 1 and cfg.get("duck", True):
        game = [k for k in range(tracks) if k != voice]
        parts = []
        if len(game) > 1:
            parts.append("".join(f"[{i}:a:{k}]" for k in game) + f"amix=inputs={len(game)}:normalize=0[g{i}]")
        else:
            parts.append(f"[{i}:a:{game[0]}]anull[g{i}]")
        denoise = f"afftdn=nf={cfg['denoise_nf']}," if cfg.get("denoise_nf") else ""
        parts.append(f"[{i}:a:{voice}]{denoise}asplit=2[vk{i}][vs{i}]")
        parts.append(f"[g{i}][vs{i}]sidechaincompress=threshold={cfg['duck_threshold']}:ratio={cfg['duck_ratio']}"
                     f":attack=20:release={cfg['duck_release_ms']}[gd{i}]")
        parts.append(f"[gd{i}][vk{i}]amix=inputs=2:normalize=0[a{i}]")
        return ";".join(parts)
    if tracks > 1:
        return "".join(f"[{i}:a:{k}]" for k in range(tracks)) + f"amix=inputs={tracks}:normalize=0[a{i}]"
    return f"[{i}:a:0]anull[a{i}]"


def render(source: Path, dest: Path, short: Short, settings: shorts.Settings, info: video_io.VideoInfo, *,
           mode: str = "bars", audio_tracks: int = 1, progress=None, cancel: video_io.Canceller | None = None) -> str:
    """Encode `short` to `dest`. Returns the video encoder used."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    codec, options = video_io._encoder()
    acfg = audio_settings()
    pieces = []                       # (seek, length, focus keys)
    if short.teaser_at is not None:
        pieces.append((max(0.0, short.teaser_at - TEASER_S / 2), TEASER_S,
                       [(0.0, _focus_at(short.focus, short.teaser_at - short.start))]))
    for a, b in short.pieces:
        pieces.append((a, b - a, _focus_piece(short.focus, a - short.start, b - a)))

    args = [video_io.ffmpeg_exe(), "-y", "-loglevel", "error"]
    for start, length, _keys in pieces:
        args += ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source)]

    graph, layout = [], shorts.plan(info, settings)
    for i, (_start, length, keys) in enumerate(pieces):
        if mode == "follow":
            graph.append(follow_graph(f"{i}:v", f"v{i}", info, settings, keys))
        else:
            graph.append(shorts.filtergraph(layout, settings, f"{i}:v", f"v{i}", tag=str(i))[0])
        if audio_tracks:
            graph.append(_audio_graph(i, audio_tracks, short.voice_track, acfg))
            if len(pieces) > 1:
                graph.append(f"[a{i}]afade=t=in:d={FADE_S},afade=t=out:st={max(0.0, length - FADE_S):.3f}"
                             f":d={FADE_S}[af{i}]")
            else:
                graph.append(f"[a{i}]anull[af{i}]")
    streams = "".join(f"[v{i}]" + (f"[af{i}]" if audio_tracks else "") for i in range(len(pieces)))
    graph.append(f"{streams}concat=n={len(pieces)}:v=1:a={1 if audio_tracks else 0}[vc]" +
                 ("[ac]" if audio_tracks else ""))
    video_out, audio_out = "vc", "ac"
    if short.captions is not None:
        graph.append(f"[vc]{captions.filter_arg(short.captions)}[vs]")
        video_out = "vs"
    if audio_tracks and acfg.get("loudness_lufs") is not None:
        graph.append(f"[ac]loudnorm=I={acfg['loudness_lufs']}:TP={acfg['true_peak_db']}:LRA=11,"
                     f"aresample=48000[an]")
        audio_out = "an"

    args += ["-filter_complex", ";".join(graph), "-map", f"[{video_out}]"]
    if audio_tracks:
        args += ["-map", f"[{audio_out}]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    args += ["-c:v", codec, *options, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    if progress is not None:
        video_io._run_reporting(args, short.length, progress, cancel)
    else:
        video_io._run(args, cancel)
    return codec
