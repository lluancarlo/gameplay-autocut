from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import video_io

SAMPLE_RATE = 16000


def track_count(path: Path) -> int:
    _code, _out, err = video_io._capture([video_io.ffmpeg_exe(), "-hide_banner", "-i", str(path)])
    return len(re.findall(r"Stream #\d+:\d+.*?: Audio:", err))


def load(path: Path, track: int = 0, cancel: video_io.Canceller | None = None) -> np.ndarray:
    args = [
        video_io.ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(path),
        "-map", f"0:a:{track}", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    code, out, err = video_io._capture(args, cancel, text=False)
    if code != 0:
        raise RuntimeError(f"Could not read audio track {track}: {err.decode(errors='replace')[-300:]}")
    return np.frombuffer(out, dtype=np.float32)
