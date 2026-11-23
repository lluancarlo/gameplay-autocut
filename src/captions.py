"""Burned-in captions for a short, as an ASS subtitle file rendered by libass.

Two kinds, in two places on the 1080x1920 canvas:

  voice   the player's words, big, in the bottom bar, a few words at a time,
          the word being said in yellow (karaoke, from Whisper's word timings).
  game    the game's dialogue as GTA 6 subtitles it on screen (read by OCR), in
          the top bar, so it stays readable once the frame is shrunk to fit.

Times are relative to the short's own timeline: `offset` shifts them when a
teaser plays first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FONTS_DIR = "C:/Windows/Fonts"   # libass needs a fonts folder; Arial Black ships with Windows
HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Voice,Arial Black,76,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,{voice_margin},1
Style: Game,Arial,52,&H00F0F0F0,&H00FFFFFF,&H00000000,&H96000000,1,0,0,0,100,100,0,0,3,4,0,8,80,80,{game_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
YELLOW = "{\\c&H00FFFF&}"
WHITE = "{\\c&HFFFFFF&}"


@dataclass
class Line:
    start: float
    end: float
    text: str


def _time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _clean(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def chunks(words: list, most: int = 4, gap: float = 0.8) -> list[list]:
    """A few words per caption: at most `most`, and a pause longer than `gap` starts a new one."""
    out: list[list] = []
    for word in words:
        if out and len(out[-1]) < most and word.start - out[-1][-1].end <= gap:
            out[-1].append(word)
        else:
            out.append([word])
    return out


def _place(a: float, b: float, pieces: list[tuple[float, float]], offset: float) -> list[tuple[float, float]]:
    """Where the recording's [a, b] lands on the short's timeline: one span per kept piece it overlaps."""
    out, at = [], offset
    for pa, pb in pieces:
        lo, hi = max(a, pa), min(b, pb)
        if hi - lo > 0.05:
            out.append((at + lo - pa, at + hi - pa))
        at += pb - pa
    return out


def voice_events(words: list, pieces: list[tuple[float, float]], offset: float) -> list[str]:
    """One event per word: its chunk shown, the word itself in yellow, upper case."""
    start, end = pieces[0][0], pieces[-1][1]
    inside = [w for w in words if start <= w.start < end]
    events = []
    for group in chunks(inside):
        texts = [_clean(w.text).upper() for w in group]
        for i, word in enumerate(group):
            show_from = word.start
            show_to = group[i + 1].start if i + 1 < len(group) else max(word.end, word.start + 0.4)
            text = " ".join((YELLOW + t + WHITE) if j == i else t for j, t in enumerate(texts))
            for a, b in _place(show_from, show_to, pieces, offset):
                events.append(f"Dialogue: 0,{_time(a)},{_time(b)},Voice,,0,0,0,,{text}")
    return events


def game_lines(subtitles: list[dict], hold: float = 2.0) -> list[Line]:
    """OCR subtitle samples (one every couple of seconds) merged into timed lines."""
    lines: list[Line] = []
    for sample in sorted(subtitles, key=lambda s: s["t"]):
        text = _clean(sample["text"])
        if lines and lines[-1].text == text and sample["t"] - lines[-1].end <= hold + 0.5:
            lines[-1].end = sample["t"] + hold / 2
        elif lines and sample["t"] - lines[-1].end <= 0.1 and len(lines[-1].text) + len(text) < 120 \
                and lines[-1].end - lines[-1].start < 0.5:
            lines[-1].text += " " + text
        else:
            lines.append(Line(sample["t"] - hold / 2, sample["t"] + hold / 2, text))
    return lines


def game_events(lines: list[Line], pieces: list[tuple[float, float]], offset: float) -> list[str]:
    events = []
    for line in lines:
        for a, b in _place(line.start, line.end, pieces, offset):
            events.append(f"Dialogue: 0,{_time(a)},{_time(b)},Game,,0,0,0,,{line.text}")
    return events


def write(path: Path, start: float, end: float, *, voice_words: list | None = None,
          subtitles: list[dict] | None = None, offset: float = 0.0, follow: bool = False,
          pieces: list[tuple[float, float]] | None = None) -> Path | None:
    """The .ass for one short, or None when there is nothing to show. `pieces` are the parts of
    [start, end] the short keeps (dead seconds cut out); times follow them."""
    pieces = pieces or [(start, end)]
    events = voice_events(voice_words or [], pieces, offset)
    events += game_events(game_lines(subtitles or []), pieces, offset)
    if not events:
        return None
    # With blurred bars the captions sit in the bars; following the action there are no bars,
    # so they move closer to the edges.
    margins = {"voice_margin": 300 if follow else 260, "game_margin": 160 if follow else 300}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(**margins) + "\n".join(events) + "\n", encoding="utf-8")
    return path


def filter_arg(path: Path) -> str:
    """The libass filter for `path`, escaped for a filter_complex on Windows."""
    def esc(p: str) -> str:
        return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"ass='{esc(str(path))}':fontsdir='{esc(FONTS_DIR)}'"
