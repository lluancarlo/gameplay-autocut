"""Landscape highlight -> 1080x1920 Shorts video, blurred bars top and bottom.

    python make-short.py "clips\\highlight.mp4"
    python make-short.py clip.mp4 --preview        # one PNG frame, encode nothing
    python make-short.py clip.mp4 --start 12 --duration 45
    python make-short.py clip.mp4 --zoom 0.6 --offset-y -80

Replaces the DaVinci Resolve setup this was done by hand with: the clip on a
portrait timeline at zoom 0.5, plus a Blanking Fill at 75% background opacity
and no crop. The clip is scaled to half the frame height and centred; the
blanking left above and below it is filled with a blurred, darkened copy of the
same frame.

Zoom is measured the way Resolve measures it, with 1.0 meaning FILL the
portrait frame -- so the foreground is always `1920 * zoom` tall and the bars
are always `1920 * (1 - zoom) / 2`, whatever the source aspect. How much of the
source's WIDTH survives the crop does depend on it: a 3440x1440 clip keeps its
central ~47% at zoom 0.5.

--preview writes a single composited frame as a PNG instead of encoding, which
is the cheap way to check the framing before committing to a render.

Every tunable lives in config/shorts.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config, shorts, video_io


def export_root(base: str | None = None) -> Path:
    """export/shorts/ -- one flat folder.

    A short is named after the clip it came from, which is already unique
    within a folder, so there is nothing for a date folder to disambiguate.
    """
    return Path(base) if base else config.ROOT / "export" / "shorts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make-short.py",
        description="Reframe a landscape highlight as a 1080x1920 Shorts video "
        "with blur-filled bars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to the highlight clip.")
    parser.add_argument(
        "--zoom",
        type=float,
        help="Foreground size, where 1.0 fills the portrait frame height and "
        "crops the sides (default: config/shorts.json, 0.5).",
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        help="Shift the foreground off centre by this many pixels, positive "
        "down (default: 0).",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        help="Blurred background opacity, 0-1 (default: 0.75).",
    )
    parser.add_argument(
        "--blur",
        type=float,
        help="Background blur sigma, before the upscale that multiplies it "
        "(default: 8, i.e. ~64 at full resolution).",
    )
    parser.add_argument("--start", type=float, help="Trim: start at this second.")
    parser.add_argument(
        "--duration", type=float, help="Trim: keep this many seconds."
    )
    parser.add_argument(
        "--preview",
        type=float,
        nargs="?",
        const=-1.0,
        metavar="SECONDS",
        help="Write one composited frame as a PNG and stop, instead of "
        "encoding. Defaults to the middle of the clip.",
    )
    parser.add_argument(
        "--out",
        help="Output file, or a directory to write into (default: export/shorts/).",
    )
    return parser


def resolve_dest(source: Path, out: str | None) -> Path:
    """Where the rendered short lands -- named after the clip it came from."""
    if out:
        candidate = Path(out)
        # A path that exists as a directory, or that looks like one, is a
        # destination folder; anything else is the file itself.
        if candidate.is_dir() or not candidate.suffix:
            return candidate / f"{source.stem}.mp4"
        return candidate
    return export_root() / f"{source.stem}.mp4"


def main(argv: list[str] | None = None) -> int:
    # The pipeline's progress lines carry emoji. A Windows console handles them,
    # but `> log.txt` redirects to the locale encoding, where they raise.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    args = build_parser().parse_args(argv)

    source = Path(args.video)
    if not source.exists():
        raise SystemExit(f"No such file: {source}")

    settings = shorts.Settings.load(
        zoom=args.zoom,
        offset_y=args.offset_y,
        opacity=args.opacity,
        blur_sigma=args.blur,
    )
    if settings.zoom <= 0:
        raise SystemExit(f"--zoom must be positive, got {settings.zoom}")

    info = video_io.probe(source)
    print(f"\n{source.name}")
    print(f"  {info.width}x{info.height} @ {info.fps:.2f}fps, {info.duration:.1f}s")

    layout = shorts.plan(info, settings)
    graph, label = shorts.filtergraph(layout, settings)
    print(f"\n{layout.describe()}")

    if info.width < info.height:
        print("\n  ! source is already portrait -- the sides will not be cropped")

    start = args.start
    duration = args.duration
    kept = info.duration - (start or 0.0)
    if duration is not None:
        kept = min(kept, duration)
    if kept <= 0:
        raise SystemExit(
            f"Nothing left to render: the clip is {info.duration:.1f}s long."
        )
    if start is not None or duration is not None:
        print(f"  trimmed    {start or 0:.1f}s -> {(start or 0) + kept:.1f}s")

    dest = resolve_dest(source, args.out)
    if dest.resolve() == source.resolve():
        raise SystemExit(f"Output would overwrite the source: {dest}")

    if args.preview is not None:
        at = args.preview
        if at < 0:
            at = (start or 0.0) + kept / 2
        still = dest.with_suffix(".preview.png")
        print(f"\npreview frame at {at:.1f}s...")
        video_io.render_still(source, still, graph, at, out_label=label)
        print(f"  -> {still}")
        print("\n--preview: nothing encoded.")
        return 0

    print(f"\nrendering {kept:.1f}s to {layout.canvas_w}x{layout.canvas_h}...")
    codec = video_io.render(
        source, dest, graph, out_label=label, start=start, duration=duration
    )
    print(f"  encoded with {codec}")
    print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
