"""The job Gameplay AutoCut does, with its terminal output factored out.

`make-short.py` drives these from a console; the GUI drives the same functions
from a worker thread. Everything a caller might want to show the user arrives
through the `log` and `progress` callbacks rather than `print`, so neither front
end owns the pipeline.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import shorts, video_io

Log = Callable[[str], None]
Progress = Callable[[int, int], None]

# The bar is always this many steps, start to finish, whatever a run contains.
# Reporting real counts instead -- seconds encoded, then videos -- moves the
# scale under the caller several times per run and makes the bar jump back.
STEPS = 1000


def _ignore(*_args) -> None:
    """Default callback: say nothing."""


def _count(number: int, noun: str, plural: str | None = None) -> str:
    """'1 video' / '3 videos'. The log is read by a person, not parsed."""
    if number == 1:
        return f"{number} {noun}"
    return f"{number} {plural or noun + 's'}"


class Span:
    """A slice of the one progress bar, filled by whatever is working now.

    A run is carved up before any work starts: each video gets an equal share,
    and the render fills that share as ffmpeg reports the seconds it has
    encoded. So two videos means the first finishes at 50%, and nothing ever
    goes backwards -- one video's share ending where the next begins is what
    makes the bar advance steadily rather than restarting per file.
    """

    def __init__(self, report: Progress, low: float = 0.0, high: float = 1.0) -> None:
        self._report = report
        self._low = low
        self._high = high

    def slice(self, low: float, high: float) -> Span:
        """A sub-span, addressed in this span's own 0..1 coordinates."""
        width = self._high - self._low
        return Span(self._report, self._low + width * low, self._low + width * high)

    def split(self, count: int) -> list[Span]:
        """`count` equal sub-spans, for a list of things of unknown cost."""
        if count <= 0:
            return []
        return [self.slice(i / count, (i + 1) / count) for i in range(count)]

    def at(self, fraction: float) -> None:
        position = self._low + (self._high - self._low) * min(max(fraction, 0.0), 1.0)
        self._report(int(position * STEPS), STEPS)

    def done(self) -> None:
        self.at(1.0)

    def counting(self) -> Progress:
        """Adapter for the code that reports its own (done, total)."""

        def report(done: int, total: int) -> None:
            if total > 0:
                self.at(done / total)

        return report


# What counts as a video when a folder is handed to one of the batch runners.
# Same list the window's file dialog used to filter on.
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts"}
)


def sources(path: Path) -> list[Path]:
    """The videos `path` names: itself if a file, its contents if a folder.

    Not recursive. A folder of recordings is the unit people actually have --
    nested trees would quietly pull in whatever else is under there, and the
    outputs all land in one place regardless.
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise RuntimeError(f"No such file or folder: {path}")
    found = sorted(
        entry
        for entry in path.iterdir()
        if entry.is_file() and entry.suffix.lower() in VIDEO_SUFFIXES
    )
    if not found:
        raise RuntimeError(
            f"No videos in {path}. Looked for "
            f"{', '.join(sorted(VIDEO_SUFFIXES))}."
        )
    return found


@dataclass
class BatchResult:
    """What a run over a folder produced, and what it could not."""

    videos: list[Path]
    written: list[Path]
    failed: list[tuple[Path, str]]

    def summary(self) -> str:
        text = f"{_count(len(self.written), 'file')} from {_count(len(self.videos), 'video')}"
        if self.failed:
            text += f", {len(self.failed)} skipped"
        return text


def _batch(
    path: Path,
    log: Log,
    progress: Progress,
    run: Callable[[Path, Span], list[Path]],
    cancel: video_io.Canceller | None = None,
) -> BatchResult:
    """Run `run` over every video `path` names, surviving individual failures.

    One unreadable file, or one clip that will not render, must not throw
    away the work already done on the others -- so failures are collected and
    reported at the end. It only raises when nothing at all came out, or when
    the run was cancelled, which is not a failure to tidy up and carry on from.
    """
    cancel = cancel or video_io.Canceller()
    videos = sources(path)
    whole = Span(progress)
    shares = whole.split(len(videos))

    log(f"📁 {path}")
    log(f"   {_count(len(videos), 'video')} to process")

    result = BatchResult(videos=videos, written=[], failed=[])
    for index, (video, share) in enumerate(zip(videos, shares), start=1):
        cancel.check()
        log("")
        log(f"━━ [{index}/{len(videos)}] {video.name}")
        try:
            result.written.extend(run(video, share))
        except video_io.Cancelled:
            raise
        except Exception as error:  # noqa: BLE001 -- one bad video is not fatal
            result.failed.append((video, str(error)))
            log(f"   ⚠️  Skipped: {error}")
        share.done()  # a video that failed still used up its share of the bar

    log("")
    log("─" * 44)
    if result.written:
        log(f"🎉 Done — {result.summary()}")
    for video, why in result.failed:
        log(f"   ⚠️  {video.name}: {why}")

    if result.failed and not result.written:
        raise RuntimeError(
            "Nothing could be produced:\n"
            + "\n".join(f"  {video.name}: {why}" for video, why in result.failed)
        )
    return result


def _short_plan(
    source: Path,
    settings: shorts.Settings | None,
    start: float | None,
    duration: float | None,
    log: Log,
) -> tuple[video_io.VideoInfo, shorts.Layout, str, str, float]:
    """Everything both short jobs need: geometry, filtergraph, seconds kept."""
    settings = settings or shorts.Settings.load()
    info = video_io.probe(source)
    layout = shorts.plan(info, settings)
    graph, label = shorts.filtergraph(layout, settings)

    log(f"   🎬 {info.width}×{info.height} · {info.fps:.0f}fps · {info.duration:.1f}s")
    for line in layout.describe().splitlines():
        log(f"   {line.strip()}")
    if info.width < info.height:
        log("   ⚠️  Source is already portrait — the sides will not be cropped")

    kept = info.duration - (start or 0.0)
    if duration is not None:
        kept = min(kept, duration)
    if kept <= 0:
        raise RuntimeError(
            f"Nothing left to render: the clip is {info.duration:.1f}s long."
        )
    if start is not None or duration is not None:
        log(f"   ✂️  Trimmed {start or 0:.1f}s → {(start or 0) + kept:.1f}s")

    return info, layout, graph, label, kept


def _short_dest(source: Path, dest_dir: Path, suffix: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{source.stem}{suffix}"
    if dest.resolve() == source.resolve():
        raise RuntimeError(f"Output would overwrite the source: {dest}")
    return dest


def make_short(
    source: Path,
    dest_dir: Path,
    *,
    settings: shorts.Settings | None = None,
    start: float | None = None,
    duration: float | None = None,
    log: Log = _ignore,
    progress: Progress = _ignore,
    span: Span | None = None,
    cancel: video_io.Canceller | None = None,
) -> Path:
    """Reframe `source` as a 1080x1920 short in `dest_dir`.

    The whole Make shorts tab for one clip. Progress comes from ffmpeg's own
    count of seconds encoded, so it tracks the render rather than guessing.
    """
    cancel = cancel or video_io.Canceller()
    span = span if span is not None else Span(progress)
    cancel.check()

    _info, layout, graph, label, kept = _short_plan(
        source, settings, start, duration, log
    )
    dest = _short_dest(source, dest_dir, ".mp4")

    log(f"   🎞️  Rendering {kept:.1f}s to {layout.canvas_w}×{layout.canvas_h}…")
    codec = video_io.render(
        source, dest, graph, out_label=label, start=start, duration=duration,
        total=kept, progress=span.counting(), cancel=cancel,
    )
    span.done()
    log(f"      encoded with {codec}")
    log(f"      ✅ {dest.name}")
    return dest


def make_shorts_batch(
    path: Path,
    dest_dir: Path,
    *,
    settings: shorts.Settings | None = None,
    start: float | None = None,
    duration: float | None = None,
    log: Log = _ignore,
    progress: Progress = _ignore,
    cancel: video_io.Canceller | None = None,
) -> BatchResult:
    """Run `make_short` over one video or every video in a folder.

    The whole Make shorts tab in one call. Each short keeps its source's name,
    which is already unique within a folder, so nothing extra is needed to tell
    them apart.
    """
    return _batch(
        path,
        log,
        progress,
        lambda video, share: [
            make_short(
                video, dest_dir, settings=settings, start=start,
                duration=duration, log=log, span=share, cancel=cancel,
            )
        ],
        cancel,
    )


def preview_short(
    source: Path,
    dest_dir: Path,
    *,
    settings: shorts.Settings | None = None,
    at: float | None = None,
    start: float | None = None,
    duration: float | None = None,
    log: Log = _ignore,
    cancel: video_io.Canceller | None = None,
) -> Path:
    """Composite one frame to a PNG instead of encoding the whole clip.

    Checking the framing against a reference still costs one frame here
    rather than a full render, which is what makes zoom worth tuning by eye.
    Defaults to the middle of whatever range would be rendered.
    """
    _info, layout, graph, label, kept = _short_plan(
        source, settings, start, duration, log
    )
    dest = _short_dest(source, dest_dir, ".preview.png")

    timestamp = (start or 0.0) + kept / 2 if at is None else at
    log(f"   🖼️  Preview frame at {timestamp:.1f}s "
        f"({layout.canvas_w}×{layout.canvas_h})…")
    video_io.render_still(
        source, dest, graph, timestamp, out_label=label, cancel=cancel
    )
    log(f"      ✅ {dest.name}")
    return dest


def clip_shorts(
    source: Path,
    clips: list[tuple[float, float]],
    dest_dir: Path,
    *,
    settings: shorts.Settings | None = None,
    log: Log = _ignore,
    progress: Progress = _ignore,
    cancel: video_io.Canceller | None = None,
) -> list[Path]:
    """Cut each (start, end) out of `source` and reframe it as a vertical short, in one encode.

    Re-encoding anyway means the cut is exact -- no keyframe snapping -- and the
    bar is shared out by clip length, so a 30 s clip takes three times the 10 s one.
    """
    cancel = cancel or video_io.Canceller()
    settings = settings or shorts.Settings.load()
    info = video_io.probe(source)
    layout = shorts.plan(info, settings)
    graph, label = shorts.filtergraph(layout, settings)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total = sum(end - start for start, end in clips) or 1.0
    span = Span(progress)
    done = 0.0
    written = []
    log(f"🎬 {_count(len(clips), 'short')} from {source.name} · {layout.canvas_w}×{layout.canvas_h}")
    for index, (start, end) in enumerate(clips, start=1):
        cancel.check()
        length = end - start
        share = span.slice(done / total, (done + length) / total)
        dest = dest_dir / moment_name(source, start)
        if dest.resolve() == source.resolve():
            raise RuntimeError(f"Output would overwrite the source: {dest}")
        log(f"   🎞️  {index}/{len(clips)} · {clock(start)} → {clock(end)} ({length:.0f}s)")
        video_io.render(source, dest, graph, out_label=label, start=start, duration=length,
                        total=length, progress=share.counting(), cancel=cancel)
        share.done()
        done += length
        written.append(dest)
        log(f"      ✅ {dest.name}")
    span.done()
    return written


def preview_clip(
    source: Path,
    at: float,
    dest: Path,
    *,
    settings: shorts.Settings | None = None,
    cancel: video_io.Canceller | None = None,
) -> Path:
    """One frame of `source` at `at` seconds, composited exactly as the short will be."""
    settings = settings or shorts.Settings.load()
    layout = shorts.plan(video_io.probe(source), settings)
    graph, label = shorts.filtergraph(layout, settings)
    return video_io.render_still(source, dest, graph, at, out_label=label, cancel=cancel)


def clock(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def keyframe_before(keyframes: list[float], seconds: float) -> float:
    """Where a copy that should start at `seconds` really starts."""
    index = bisect.bisect_right(keyframes, seconds + 1e-3) - 1
    return keyframes[index] if index >= 0 else 0.0


def moment_name(source: Path, start: float) -> str:
    seconds = int(start)
    return f"{source.stem}_{seconds // 60:02d}m{seconds % 60:02d}s.mp4"


def cut_moments(
    source: Path,
    ranges: list[tuple[float, float]],
    keyframes: list[float],
    dest_dir: Path,
    *,
    log: Log = _ignore,
    progress: Progress = _ignore,
    cancel: video_io.Canceller | None = None,
) -> list[Path]:
    """Copy each (start, end) out of `source` as an .mp4, starting at the keyframe before it."""
    cancel = cancel or video_io.Canceller()
    span = Span(progress)
    log(f"✂️  {_count(len(ranges), 'moment')} from {source.name}")
    written = []
    for index, (start, end) in enumerate(ranges, start=1):
        cancel.check()
        dest = dest_dir / moment_name(source, start)
        video_io.cut(source, dest, keyframe_before(keyframes, start), end, cancel)
        written.append(dest)
        log(f"   ✅ {dest.name}")
        span.at(index / len(ranges))
    return written
