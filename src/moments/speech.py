from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .. import video_io
from . import models
from .audio import SAMPLE_RATE

LANGUAGES = ("en", "it", "pt")


@dataclass
class Word:
    start: float
    end: float
    text: str
    lang: str = ""   # the language of its passage, when it differs from the recording's main one or was checked


def _load_model(device: str, log):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from faster_whisper import WhisperModel

    name, compute, size = ("large-v3-turbo", "float16", "1.6 GB") if device == "cuda" else ("small", "int8", "0.5 GB")
    root = models.models_dir() / "whisper"
    if not any(root.glob(f"*{name}*")):
        log(f"   ⬇️  Downloading Whisper {name} ({size}, once)…")
    log(f"   🗣️  Whisper {name} on {device.upper()}")
    return WhisperModel(name, device=device, compute_type=compute, download_root=str(root))


def _language(model, samples: np.ndarray) -> str:
    _lang, _prob, scores = model.detect_language(samples, vad_filter=True, language_detection_segments=4)
    allowed = {code: prob for code, prob in scores if code in LANGUAGES}
    return max(allowed, key=allowed.get)


def transcribe(samples: np.ndarray, log, progress, cancel: video_io.Canceller) -> tuple[str, list[Word]]:
    try:
        if not models.ensure_cuda(log):
            raise RuntimeError("no NVIDIA GPU")
        models.enable_cuda()
        model = _load_model("cuda", log)
        language = _language(model, samples)
    except (RuntimeError, OSError, ValueError) as error:
        log(f"   ⚠️  GPU unavailable ({error}); using the CPU")
        model = _load_model("cpu", log)
        language = _language(model, samples)
    log(f"   🌐 Language: {language}")

    total = len(samples) / SAMPLE_RATE
    segments, _info = model.transcribe(samples, language=language, word_timestamps=True, vad_filter=True,
                                       beam_size=5, condition_on_previous_text=False)
    words, passages = [], []
    for segment in segments:
        cancel.check()
        words += [Word(w.start, w.end, w.word.strip(), language) for w in segment.words or [] if w.word.strip()]
        passages.append((segment.start, segment.end))
        progress(int(segment.end), int(total) + int(total) // 5)
    words = _second_language(model, samples, language, words, passages, log, cancel)
    progress(int(total), int(total))
    return language, words


def _passages(spans: list[tuple[float, float]], gap: float = 2.0, longest: float = 20.0) -> list[tuple[float, float]]:
    """Whisper's segments joined into passages of at most `longest` seconds."""
    out: list[list[float]] = []
    for start, end in spans:
        if out and start - out[-1][1] <= gap and end - out[-1][0] <= longest:
            out[-1][1] = end
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def _second_language(model, samples, language: str, words: list[Word], spans, log, cancel) -> list[Word]:
    """Find passages spoken in another language and transcribe them in it.

    A recording is transcribed in one language, but a Brazilian streamer over an
    English-speaking game is two: the player's Portuguese would come out as
    nonsense English. Each passage gets its own language check; passages that
    are clearly another allowed language are transcribed again in it, and every
    word keeps its passage's language -- which is also how the player's voice is
    told apart from the game's (voice.py).
    """
    redone = 0
    for start, end in _passages(spans):
        cancel.check()
        a, b = int(start * SAMPLE_RATE), int(max(end, start + 2.0) * SAMPLE_RATE)
        chunk = samples[a:b]
        if len(chunk) < SAMPLE_RATE:
            continue
        _lang, _prob, scores = model.detect_language(chunk, language_detection_segments=1)
        allowed = {code: prob for code, prob in scores if code in LANGUAGES}
        other = max(allowed, key=allowed.get)
        if other == language or allowed[other] < 0.7:
            continue
        segments, _info = model.transcribe(chunk, language=other, word_timestamps=True, vad_filter=False,
                                           beam_size=5, condition_on_previous_text=False)
        fresh = [Word(start + w.start, start + w.end, w.word.strip(), other)
                 for s in segments for w in s.words or [] if w.word.strip()]
        words = [w for w in words if not start <= w.start < end] + fresh
        redone += 1
    if redone:
        log(f"   🌐 {redone} passages in another language transcribed again")
    return sorted(words, key=lambda w: w.start)
