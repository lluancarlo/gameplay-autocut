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
    words = []
    for segment in segments:
        cancel.check()
        words += [Word(w.start, w.end, w.word.strip()) for w in segment.words or [] if w.word.strip()]
        progress(int(segment.end), int(total))
    progress(int(total), int(total))
    return language, words
