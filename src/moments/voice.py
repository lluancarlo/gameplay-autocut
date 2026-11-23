"""Is the player talking, and where? Detected, never asked.

Two situations:

  several audio tracks   OBS put the microphone on its own track. Each extra
                         track is sampled (a dozen 20 s chunks spread over the
                         recording) through the same VAD Whisper uses; the one
                         with the most speech is the voice, if it has any.
  one mixed track        the player and the game share it. In GTA 6 the game's
                         own dialogue is subtitled, so speech with a subtitle on
                         screen is the game's; speech without one is the player
                         (or unsubtitled street chatter). If enough of the
                         speech is unsubtitled, there is commentary.

Only the player's words feed the speech-rate and trigger-phrase signals: an
NPC swearing in a cutscene is story, not a reaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import video_io
from .audio import SAMPLE_RATE
from .speech import Word


@dataclass
class Voice:
    mode: str                       # "track" | "mixed" | "none" | "unknown"
    track: int | None = None        # the voice track, when mode == "track"
    player_words: list[Word] = field(default_factory=list)
    game_words: list[Word] = field(default_factory=list)
    detail: str = ""

    def summary(self) -> dict:
        return {"mode": self.mode, "track": self.track, "detail": self.detail,
                "player_words": len(self.player_words), "game_words": len(self.game_words)}


def speech_ratio(path, track: int, duration: float, cancel: video_io.Canceller, chunks: int = 12,
                 seconds: int = 20) -> float:
    """Share of sampled audio that the VAD calls speech, on one track."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    total = speech = 0.0
    for i in range(chunks):
        cancel.check()
        start = max(0.0, (i + 0.5) * duration / chunks - seconds / 2)
        args = [video_io.ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", str(seconds),
                "-i", str(path), "-map", f"0:a:{track}", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
        code, out, _err = video_io._capture(args, cancel, text=False)
        if code != 0 or not out:
            continue
        samples = np.frombuffer(out, dtype=np.float32)
        total += len(samples) / SAMPLE_RATE
        for segment in get_speech_timestamps(samples, VadOptions()):
            speech += (segment["end"] - segment["start"]) / SAMPLE_RATE
    return speech / total if total else 0.0


def pick_track(path, tracks: int, duration: float, cfg: dict, log, cancel: video_io.Canceller) -> int | None:
    """The extra audio track that carries speech, if any (track 0 is the game)."""
    if tracks < 2:
        return None
    ratios = {t: speech_ratio(path, t, duration, cancel) for t in range(1, tracks)}
    best = max(ratios, key=ratios.get)
    log("   🎙️  Speech per extra track: " + ", ".join(f"track {t + 1} {r:.0%}" for t, r in ratios.items()))
    return best if ratios[best] >= cfg["track_min_speech"] else None


def attribute(words: list[Word], subtitles: np.ndarray, cfg: dict) -> Voice:
    """Split one mixed track's words into the game's and the player's.

    By language first: GTA 6's dialogue is English, so speech in another language
    is the player. Subtitles are the fallback when the languages cannot tell.
    """
    if not words:
        return Voice("none", detail="no speech")
    game_language = cfg.get("game_language", "")
    if game_language and all(w.lang for w in words):
        player = [w for w in words if w.lang != game_language]
        game = [w for w in words if w.lang == game_language]
        seconds = len({int(w.start) for w in player})
        if player:
            return Voice("mixed", player_words=player, game_words=game,
                         detail=f"you talk for ~{max(1, seconds)} s in total (speech not in {game_language})")
        return Voice("none", game_words=game, detail=f"no speech outside the game's language ({game_language})")
    if subtitles.sum() < cfg["min_subtitle_s"]:
        return Voice("unknown", player_words=list(words),
                     detail="no subtitles to tell the game's dialogue from yours")
    n, pad = len(subtitles), int(cfg["subtitle_pad_s"])
    player, game = [], []
    for word in words:
        t = int(word.start)
        (game if subtitles[max(0, t - pad):min(n, t + pad + 1)].any() else player).append(word)
    ratio = len(player) / len(words)
    seconds = len({int(w.start) for w in player})
    if ratio >= cfg["commentary_ratio"] and seconds >= cfg["commentary_min_s"]:
        return Voice("mixed", player_words=player, game_words=game,
                     detail=f"{ratio:.0%} of the speech is yours (not subtitled)")
    return Voice("none", game_words=list(words), detail=f"only {ratio:.0%} unsubtitled speech: game dialogue")
