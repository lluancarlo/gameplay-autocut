"""Score the Moments detector against the hand-labelled recordings in eval/.

Developer tooling: none of this is in the app or the .exe.

    python tools/evaluate.py                                # tune split, reviewed labels only
    python tools/evaluate.py --split all --include-drafts   # everything, drafts too
    python tools/evaluate.py --rescore-baseline             # score eval/baseline without running models
    python tools/evaluate.py --freeze                       # after a run: make it the new baseline
    python tools/evaluate.py --gui                          # window: label recordings, run, compare

Methodology: docs/ai-highlights/DATASET_RESEARCH.md, sections 4-5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import runner  # noqa: E402
from src.evaluation.dataset import EvalSet, default_folder  # noqa: E402
from src.evaluation.metrics import SPLITS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(prog="evaluate.py", description=__doc__.splitlines()[0])
    parser.add_argument("--folder", type=Path, default=default_folder(), help="Evaluation folder (default: eval/).")
    parser.add_argument("--split", default="tune", choices=(*SPLITS, "all"))
    parser.add_argument("--include-drafts", action="store_true", help="Also score recordings with draft labels.")
    parser.add_argument("--rescore-baseline", action="store_true", help="Score the saved baseline, run nothing.")
    parser.add_argument("--no-compare", action="store_true", help="Do not compare with the baseline.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the detector's analysis cache.")
    parser.add_argument("--title", default="latest", help="Name shown in the report.")
    parser.add_argument("--report", type=Path, help="Also write the Markdown report here.")
    parser.add_argument("--freeze", action="store_true", help="Copy the latest predictions into baseline/ and exit.")
    parser.add_argument("--gui", action="store_true", help="Open the labelling/evaluation window.")
    args = parser.parse_args(argv)

    if args.gui:
        from src.evaluation.window import run
        return run()
    es = EvalSet(args.folder)
    if args.freeze:
        print("Baseline frozen for: " + ", ".join(runner.freeze_baseline(es)))
        return 0
    try:
        text, _ = runner.evaluate(es, args.split, args.include_drafts, run_new=not args.rescore_baseline,
                                  against_baseline=not args.no_compare, log=print, fresh=args.fresh, title=args.title)
    except ValueError as error:
        print(error)
        return 1
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
