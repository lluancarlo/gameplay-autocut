"""Video IO.

Probing goes through OpenCV; rendering goes through the ffmpeg binary bundled
by `imageio-ffmpeg`. Note that imageio-ffmpeg ships *ffmpeg only* -- there is
no ffprobe -- which is why probing uses OpenCV.
"""

from __future__ import annotations

import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg


def ffmpeg_exe() -> str:
    """Path to the pip-installed ffmpeg binary. Never relies on system PATH."""
    return imageio_ffmpeg.get_ffmpeg_exe()


# The windowed build has no console of its own, so Windows hands every child
# process a brand new one: without this, each ffmpeg call flashes up a terminal
# in front of the window, and a run makes one per clip. Zero everywhere else --
# creationflags exists on POSIX but must be 0 there.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Cancelled(RuntimeError):
    """A run was stopped on purpose. Not a failure to report as one."""


class Canceller:
    """The stop button for one run, and the ffmpeg processes it owns.

    Killing the child is the whole point. ffmpeg does the encoding in its own
    process and holds the GPU while it does; dropping the Python thread that
    started it leaves it running to completion on its own -- which is why
    closing the window used to leave an encode chewing on the card.

    One of these per run rather than one global flag, so a cancel only ever
    reaches the run it was pressed for.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def cancel(self) -> None:
        """Trip the flag and kill anything already running. Safe from any thread."""
        self._stop.set()
        with self._lock:
            running = list(self._processes)
        for process in running:
            try:
                process.kill()
            except OSError:
                pass  # already gone

    def check(self) -> None:
        """Raise if cancelled. Call between steps that cannot be interrupted."""
        if self._stop.is_set():
            raise Cancelled("Cancelled.")

    @contextmanager
    def track(self, process: subprocess.Popen):
        """Kill `process` if this run is cancelled while it is in flight."""
        with self._lock:
            self._processes.add(process)
        try:
            # Cancelled between spawning and registering: kill it now, or it
            # would run on unnoticed.
            if self._stop.is_set():
                try:
                    process.kill()
                except OSError:
                    pass
            yield process
        finally:
            with self._lock:
                self._processes.discard(process)


# Stands in when a caller does not care about cancellation. Never tripped.
_UNCANCELLED = Canceller()


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    duration = frames / fps if fps > 0 else 0.0
    return VideoInfo(path, width, height, fps, duration)


def _capture(
    args: list[str], cancel: Canceller | None = None, text: bool = True
) -> tuple[int, str, str]:
    """Run to completion under `cancel`, returning (code, stdout, stderr)."""
    cancel = cancel or _UNCANCELLED
    cancel.check()
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        errors="replace" if text else None,
        creationflags=_NO_WINDOW,
    )
    with cancel.track(process):
        out, err = process.communicate()
    cancel.check()  # a kill shows up as a non-zero exit; say why it really failed
    return process.returncode, out or "", err or ""


def _run(args: list[str], cancel: Canceller | None = None) -> None:
    code, _out, err = _capture(args, cancel)
    if code != 0:
        tail = "\n".join(err.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({code}):\n{tail}")


def _run_reporting(
    args: list[str],
    total: float,
    progress: Callable[[int, int], None],
    cancel: Canceller | None = None,
) -> None:
    """Run ffmpeg, reporting seconds encoded out of `total` as it goes.

    `-progress pipe:1` makes ffmpeg write machine-readable key=value lines to
    stdout as it encodes, which is the only honest way to fill a bar for a
    single long render -- there is nothing else to count.

    stderr is drained by a thread rather than read after the fact: it is where
    any error lands, and a pipe nobody is reading will eventually fill and
    block the encode.
    """
    cancel = cancel or _UNCANCELLED
    cancel.check()
    whole = max(1, int(round(total)))
    # Global options, so they go straight after the binary. Appended to the end
    # they would land past the output filename, where ffmpeg reads them as the
    # start of another output.
    process = subprocess.Popen(
        [args[0], "-progress", "pipe:1", "-nostats", *args[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        creationflags=_NO_WINDOW,
    )
    errors: list[str] = []
    with cancel.track(process):
        drain = threading.Thread(
            target=errors.extend, args=(process.stderr,), daemon=True
        )
        drain.start()

        progress(0, whole)
        for line in process.stdout:
            key, _, value = line.partition("=")
            # Microseconds, and "N/A" until the first frame is out.
            if key.strip() == "out_time_us" and value.strip().isdigit():
                progress(min(int(value) // 1_000_000, whole), whole)

        process.wait()
        drain.join(timeout=1.0)

    cancel.check()
    if process.returncode != 0:
        tail = "\n".join("".join(errors).strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed ({process.returncode}):\n{tail}")
    progress(whole, whole)


@lru_cache(maxsize=1)
def _encoder() -> tuple[str, tuple[str, ...]]:
    """Video encoder for the render: NVENC when the GPU takes it, else x264.

    Probed with a real tiny encode rather than by listing encoders --
    h264_nvenc is compiled into the binary regardless of whether an NVIDIA
    card is present, and only an encode attempt tells the truth.
    """
    probe = subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=black:size=256x144:rate=30:duration=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
    )
    if probe.returncode == 0:
        return "h264_nvenc", ("-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0")
    return "libx264", ("-preset", "medium", "-crf", "18")


def render(
    path: str | Path,
    dest: str | Path,
    filter_complex: str,
    *,
    out_label: str = "out",
    start: float | None = None,
    duration: float | None = None,
    total: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Canceller | None = None,
) -> str:
    """Re-encode `path` through `filter_complex` to `dest`.

    The video comes from a filtergraph rather than straight off the source, and
    `out_label` names the graph's video output pad. Audio is passed through the
    encoder untouched, and optional (`0:a?`) so a silent source still renders.

    Pass `progress` with the `total` seconds being rendered to be told how far
    along the encode is, in seconds out of that total.

    Returns the name of the video encoder used.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    codec, options = _encoder()

    args = [ffmpeg_exe(), "-y", "-loglevel", "error"]
    if start is not None:
        args += ["-ss", f"{max(0.0, start):.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{max(0.1, duration):.3f}"]
    args += [
        "-filter_complex", filter_complex,
        "-map", f"[{out_label}]",
        "-map", "0:a?",
        "-c:v", codec, *options,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dest),
    ]
    if progress is not None and total:
        _run_reporting(args, total, progress, cancel)
    else:
        _run(args, cancel)
    return codec


def render_still(
    path: str | Path,
    dest: str | Path,
    filter_complex: str,
    timestamp: float,
    *,
    out_label: str = "out",
    cancel: Canceller | None = None,
) -> Path:
    """One frame at `timestamp`, through `filter_complex`, written as a PNG.

    Checking a reframe by eye costs one frame instead of a whole encode.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-ss", f"{max(0.0, timestamp):.3f}",
            "-i", str(path),
            "-filter_complex", filter_complex,
            "-map", f"[{out_label}]",
            "-frames:v", "1",
            "-update", "1",
            str(dest),
        ],
        cancel,
    )
    return dest
