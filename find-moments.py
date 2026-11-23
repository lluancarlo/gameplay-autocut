"""Suggest the best moments of a gameplay recording from its audio and speech.

    python find-moments.py "D:\\OBS\\session.mkv"
    python find-moments.py session.mkv --min-score 50 --json moments.json
    python find-moments.py session.mkv --voice-track 1

Every tunable lives in config/moments.json. Models download on first use into VVideo-data/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.moments import clock, find_moments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="find-moments.py", description=__doc__.splitlines()[0])
    parser.add_argument("video", help="Gameplay recording.")
    parser.add_argument("--min-score", type=int, default=0, help="Hide moments below this score (0-100).")
    parser.add_argument("--json", help="Also write the moments and the hype curve to this file.")
    parser.add_argument("--track", type=int, default=0, help="Audio track with the game (default: 0).")
    parser.add_argument("--voice-track", type=int, help="Audio track with your microphone, if separate.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the cached analysis.")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    args = build_parser().parse_args(argv)
    source = Path(args.video)
    if not source.exists():
        raise SystemExit(f"No such file: {source}")

    analysis = find_moments(source, track=args.track, voice_track=args.voice_track,
                            fresh=args.fresh, log=print)
    shown = [m for m in analysis.moments if m.score >= args.min_score]
    print("\n  #  score  start     end       length  label                 why")
    for index, moment in enumerate(shown, start=1):
        label = moment.label + (f" ({moment.level})" if moment.level else "")
        print(f"{index:>3}  {moment.score:>5}  {clock(moment.start):<8}  {clock(moment.end):<8}  "
              f"{moment.length:>5.0f}s  {label:<20}  {moment.explanation}")
    if args.json:
        Path(args.json).write_text(json.dumps(analysis.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
