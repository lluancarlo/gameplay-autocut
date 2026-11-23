"""Run the detector over an evaluation set, score it, and write the report.

Shared by the Evaluation tab and tools/evaluate.py. The report is Markdown; times
in it are links of the form `vvideo://seek/<recording>/<seconds>` so the window
can jump to them (in a plain Markdown viewer they are just text links).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .. import config, pipeline, video_io
from . import metrics as m
from .dataset import EvalSet

METRICS = ("p@5", "p@10", "p@20", "ndcg@5", "ndcg@10", "ndcg@20", "recall", "recall_strict", "non_gameplay_fp_per_h")
DESCRIBE = {
    "p@5": "share of the top 5 that are highlights",
    "p@10": "share of the top 10 that are highlights",
    "p@20": "share of the top 20 that are highlights",
    "ndcg@5": "ranking quality of the top 5 (uses the 0–3 ratings)",
    "ndcg@10": "ranking quality of the top 10",
    "ndcg@20": "ranking quality of the top 20",
    "recall": "share of highlights found (IoU ≥ 0.3 or peak inside)",
    "recall_strict": "share of highlights found with IoU ≥ 0.5",
    "non_gameplay_fp_per_h": "suggestions inside menus/map/pause, per hour",
}
SEEK = "vvideo://seek"


def _ignore(*_args) -> None:
    pass


def code_version() -> str:
    if getattr(sys, "frozen", False):
        return "frozen build"
    try:
        def run(*args):
            return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10).stdout.strip()
        commit = run("rev-parse", "--short", "HEAD")
        if commit:
            return commit + ("+dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def run_detector(rec: m.Recording, save: Path, *, log=_ignore, progress=_ignore,
                 cancel: video_io.Canceller | None = None, fresh: bool = False) -> list[m.Pred]:
    from ..moments import find_moments

    source = Path(rec.path)
    if not source.is_file():
        raise FileNotFoundError(f"{rec.id}: recording not found at {source}")
    analysis = find_moments(source, track=rec.track, voice_track=rec.voice_track, fresh=fresh,
                            log=log, progress=progress, cancel=cancel)
    data = {
        "recording": rec.id,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "code": code_version(),
        "language": analysis.language,
        "moments": [
            {"start": x.start, "end": x.end, "peak": x.peak, "score": x.score, "reasons": x.reasons,
             **({"label": x.label} if hasattr(x, "label") else {}),
             **({"confidence": x.confidence} if getattr(x, "confidence", None) is not None else {})}
            for x in analysis.moments
        ],
    }
    save.mkdir(parents=True, exist_ok=True)
    (save / f"{rec.id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return m.preds_from_json(data)


def load_preds(directory: Path, rec_id: str) -> list[m.Pred] | None:
    file = directory / f"{rec_id}.json"
    if not file.is_file():
        return None
    return m.preds_from_json(json.loads(file.read_text(encoding="utf-8")))


def score(recs: list[m.Recording], preds: dict[str, list[m.Pred]]) -> dict:
    return {
        "relevance": {r.id: m.relevance(r, preds[r.id]) for r in recs},
        "classification": {r.id: m.classification(r, preds[r.id]) for r in recs},
    }


def evaluate(es: EvalSet, split: str, include_drafts: bool, *, run_new: bool = True, against_baseline: bool = True,
             log=_ignore, progress=_ignore, cancel: video_io.Canceller | None = None,
             fresh: bool = False, title: str = "latest") -> tuple[str, Path]:
    """Score a split. Returns the Markdown report and the folder the predictions are in."""
    recs, skipped = es.select(split, include_drafts)
    for rec in skipped:
        log(f"skipped {rec.id}: labels are '{rec.status}', not 'reviewed'")
    if not recs:
        raise ValueError(f"No recordings to score in split '{split}'"
                         + (" (all have draft labels — tick 'Include draft labels')." if skipped else "."))
    cancel = cancel or video_io.Canceller()
    span = pipeline.Span(progress)
    preds: dict[str, list[m.Pred]] = {}
    source_dir = es.latest_dir
    for rec, part in zip(recs, span.split(len(recs))):
        cancel.check()
        if run_new:
            log(f"🎮 {rec.id}")
            preds[rec.id] = run_detector(rec, es.latest_dir, log=log, progress=part.counting(),
                                         cancel=cancel, fresh=fresh)
        else:
            source_dir = es.baseline_dir
            found = load_preds(source_dir, rec.id)
            if found is None:
                raise ValueError(f"No saved predictions for {rec.id} in {source_dir}")
            preds[rec.id] = found
        part.done()
    result = score(recs, preds)
    against = None
    if against_baseline and run_new:
        base = {r.id: load_preds(es.baseline_dir, r.id) for r in recs}
        if all(v is not None for v in base.values()):
            against = score(recs, base)
        elif any(v is not None for v in base.values()):
            log("baseline comparison skipped: not every recording has baseline predictions")
    text = report(recs, skipped, preds, result, split=split, include_drafts=include_drafts,
                  title=title, source=source_dir, against=against)
    out = es.folder / "runs" / "latest-report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return text, source_dir


def freeze_baseline(es: EvalSet, rec_ids: list[str] | None = None) -> list[str]:
    """Copy the latest predictions into baseline/ (all of them, or the given recordings)."""
    files = sorted(es.latest_dir.glob("*.json"))
    if rec_ids is not None:
        files = [f for f in files if f.stem in rec_ids]
    if not files:
        raise ValueError("Run an evaluation first: there are no latest predictions to freeze.")
    es.baseline_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, es.baseline_dir / file.name)
    report_file = es.folder / "runs" / "latest-report.md"
    if report_file.is_file():
        shutil.copy2(report_file, es.baseline_dir / "report.md")
    return [f.stem for f in files]


# --------------------------------------------------------------------------- report


def fmt(v, pct: bool = True) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.2f}"


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _seek(rec_id: str, start: float, end: float) -> str:
    return f"[{clock(start)}–{clock(end)}]({SEEK}/{rec_id}/{start:.2f})"


def _relevance_rows(recs, result, against) -> list[str]:
    games = sorted({r.game for r in recs})
    head = "| Metric | All | 95 % CI |" + "".join(f" {g} |" for g in games)
    sep = "|---|---|---|" + "---|" * len(games)
    if against:
        head += " Baseline | Δ |"
        sep += "---|---|"
    rows = [head, sep]
    for key in METRICS:
        pct = key != "non_gameplay_fp_per_h"
        values = [result["relevance"][r.id][key] for r in recs]
        ci = m.bootstrap_ci(values)
        cells = [fmt(m.mean(values), pct), f"{fmt(ci[0], pct)} – {fmt(ci[1], pct)}" if ci else "n/a"]
        cells += [fmt(m.mean([result["relevance"][r.id][key] for r in recs if r.game == g]), pct) for g in games]
        if against:
            base, cur = m.mean([against["relevance"][r.id][key] for r in recs]), m.mean(values)
            delta = None if base is None or cur is None else cur - base
            sign = "+" if delta is not None and delta > 0 else ""
            cells += [fmt(base, pct), sign + fmt(delta, pct) if delta is not None else "–"]
        rows.append(f"| {key} | " + " | ".join(cells) + " |")
    rows += ["", "CI = 95 % bootstrap interval over recordings; n/a with fewer than 2 recordings.", ""]
    rows += [f"- **{k}**: {v}" for k, v in DESCRIBE.items()]
    return rows


def _classification_rows(recs, result) -> list[str]:
    pairs = [p for r in recs for p in result["classification"][r.id]["pairs"]]
    conf = [c for r in recs for c in result["classification"][r.id]["confidences"]]
    rows = [f"Matched moments with a labelled ground truth: **{len(pairs)}**. "
            f"Left unlabelled by the detector: **{fmt(m.abstention(pairs))}**.", ""]
    table = m.label_table(pairs)
    if any(t["n_pred"] for t in table.values()):
        rows += ["| Label | Precision | Recall | F1 | n predicted | n true |", "|---|---|---|---|---|---|"]
        for label, t in table.items():
            flag = " (n < 10)" if t["n_true"] < 10 else ""
            rows.append(f"| {label}{flag} | {fmt(t['precision'])} | {fmt(t['recall'])} | {fmt(t['f1'])} "
                        f"| {t['n_pred']} | {t['n_true']} |")
    else:
        rows.append("The detector does not label moments yet, so there is nothing to score here.")
    rows += ["", f"Calibration: ECE {fmt(m.ece(conf), False)}, Brier {fmt(m.brier(conf), False)}"
             + (" — the detector emits no confidences yet." if not conf else f" over {len(conf)} pairs.")]
    return rows


def _detail_rows(rec: m.Recording, preds: list[m.Pred]) -> list[str]:
    ps, ts = m.in_scope(rec, preds)
    ranked = sorted(ps, key=lambda p: -p.score)
    pairs = m.match(ranked, ts)
    rows = [f"### {rec.id} — {rec.game}, {rec.split}, labels {rec.status}", "",
            "| # | Score | Suggested | Matches label | In menu | Reasons |", "|---|---|---|---|---|---|"]
    for i, p in enumerate(ranked):
        t = ts[pairs[i]] if i in pairs else None
        got = (f"rating {t.rating}, {t.label or '–'}" + (f" — {t.note}" if t.note else "")) if t else "—"
        rows.append(f"| {i + 1} | {p.score:.0f} | {_seek(rec.id, p.start, p.end)} | {got} "
                    f"| {'yes' if m.inside(p.peak, rec.non_gameplay) else ''} | {', '.join(p.reasons)} |")
    missed = [t for ti, t in enumerate(ts) if t.rating >= m.RELEVANT and ti not in pairs.values()]
    if missed:
        rows += ["", "**Missed highlights** (rating ≥ 2, nothing matched):", ""]
        rows += [f"- {_seek(rec.id, t.start, t.end)} rating {t.rating}, {t.label or '–'} — {t.note}" for t in missed]
    return rows


def report(recs, skipped, preds, result, *, split, include_drafts, title, source, against=None) -> str:
    lines = [
        f"# Evaluation — {title}", "",
        f"- Date: {dt.datetime.now().isoformat(timespec='minutes')}",
        f"- Code: `{code_version()}`",
        f"- Split: `{split}`" + (" — **draft labels included, not valid for decisions**" if include_drafts else ""),
        f"- Predictions: `{source}`" + ("" if against is None else " — compared with the baseline"),
    ]
    if skipped:
        lines.append(f"- Skipped (draft labels): {', '.join(r.id for r in skipped)}")
    lines += ["", "## Recordings", "",
              "| Recording | Game | Split | Labels | Reviewed | Highlights (rating ≥ 2) |", "|---|---|---|---|---|---|"]
    for r in recs:
        minutes = sum(e - s for s, e in r.reviewed) / 60
        lines.append(f"| {r.id} | {r.game} | {r.split} | {r.status} | {minutes:.1f} min "
                     f"| {result['relevance'][r.id]['n_relevant']} |")
    lines += ["", "## Highlight relevance", ""] + _relevance_rows(recs, result, against)
    lines += ["", "## Event classification", ""] + _classification_rows(recs, result)
    lines += ["", "## Per recording", ""]
    for r in recs:
        lines += _detail_rows(r, preds[r.id]) + [""]
    return "\n".join(lines)
