"""Metrics for scoring the Moments detector against hand-labelled recordings.

The methodology is documented in docs/ai-highlights/DATASET_RESEARCH.md
(sections 4 and 5); this module is the code form of it, with no model, video or
file dependency so it can be unit tested.

Two questions, kept apart:
  relevance       -- did the detector find the moments worth clipping, and
                     rank them first? (precision@k, nDCG@k, temporal recall)
  classification  -- for moments it found, is the event label right?
                     (per-label precision/recall on matched pairs only)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

RELEVANT = 2  # a ground-truth moment rated >= this is a highlight
IOU_MATCH = 0.3
IOU_STRICT = 0.5
UNLABELLED = "unlabelled"
LABELS = ("combat", "vehicle", "destruction", "achievement", "failure", "reaction", "unusual")
SPLITS = ("train", "tune", "holdout")

Span = tuple[float, float]


# --------------------------------------------------------------------------- data


@dataclass
class Truth:
    start: float
    end: float
    rating: int
    label: str | None = None
    label2: str | None = None
    note: str = ""


@dataclass
class Pred:
    start: float
    end: float
    peak: float
    score: float
    label: str = UNLABELLED
    confidence: float | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class Recording:
    id: str
    game: str
    split: str
    path: str
    status: str
    duration: float
    reviewed: list[Span]
    non_gameplay: list[Span]
    truths: list[Truth]
    track: int = 0
    voice_track: int | None = None


def preds_from_json(data: dict) -> list[Pred]:
    return [
        Pred(m["start"], m["end"], m["peak"], m["score"], m.get("label", UNLABELLED),
             m.get("confidence"), m.get("reasons", []))
        for m in data["moments"]
    ]


# --------------------------------------------------------------------------- geometry


def iou(a: Span, b: Span) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def inside(t: float, spans: list[Span]) -> bool:
    return any(s <= t <= e for s, e in spans)


def covered(span: Span, spans: list[Span]) -> float:
    """Fraction of `span` lying inside `spans` (assumed non-overlapping)."""
    length = span[1] - span[0]
    if length <= 0:
        return 0.0
    return sum(max(0.0, min(span[1], e) - max(span[0], s)) for s, e in spans) / length


def hits(p: Pred, t: Truth, iou_min: float, peak_counts: bool) -> bool:
    return iou((p.start, p.end), (t.start, t.end)) >= iou_min or (
        peak_counts and t.start <= p.peak <= t.end
    )


def match(preds: list[Pred], truths: list[Truth], iou_min: float = IOU_MATCH,
          peak_counts: bool = True) -> dict[int, int]:
    """One-to-one, greedy by prediction score: each prediction takes the free truth it overlaps most."""
    pairs: dict[int, int] = {}
    taken: set[int] = set()
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i].score):
        p = preds[pi]
        options = [ti for ti, t in enumerate(truths) if ti not in taken and hits(p, t, iou_min, peak_counts)]
        if options:
            best = max(options, key=lambda ti: iou((p.start, p.end), (truths[ti].start, truths[ti].end)))
            pairs[pi] = best
            taken.add(best)
    return pairs


def in_scope(rec: Recording, preds: list[Pred]) -> tuple[list[Pred], list[Truth]]:
    """Only what was reviewed counts: predictions by peak, truths by >= half their length."""
    ps = [p for p in preds if inside(p.peak, rec.reviewed)]
    ts = [t for t in rec.truths if covered((t.start, t.end), rec.reviewed) >= 0.5]
    return ps, ts


# --------------------------------------------------------------------------- relevance


def _dcg(gains: list[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def relevance(rec: Recording, preds: list[Pred], ks=(5, 10, 20)) -> dict:
    preds, truths = in_scope(rec, preds)
    ranked = sorted(preds, key=lambda p: -p.score)
    pairs = match(ranked, truths)
    rating = {pi: truths[ti].rating for pi, ti in pairs.items()}

    # Binary view: rating-1 matches are neither hits nor misses, so they leave the list.
    binary = [rating.get(pi, 0) >= RELEVANT for pi in range(len(ranked)) if rating.get(pi) != 1]
    graded = [float(rating.get(pi, 0)) for pi in range(len(ranked))]
    ideal = sorted((float(t.rating) for t in truths), reverse=True)

    out: dict = {"n_pred": len(ranked), "n_relevant": sum(t.rating >= RELEVANT for t in truths)}
    for k in ks:
        out[f"p@{k}"] = sum(binary[:k]) / k if out["n_relevant"] else None
        idcg = _dcg(ideal[:k])
        out[f"ndcg@{k}"] = _dcg(graded[:k]) / idcg if idcg > 0 else None
    relevant = [t for t in truths if t.rating >= RELEVANT]
    for name, thr, peak in (("recall", IOU_MATCH, True), ("recall_strict", IOU_STRICT, False)):
        found = sum(any(hits(p, t, thr, peak) for p in ranked) for t in relevant)
        out[name] = found / len(relevant) if relevant else None
    hours = sum(e - s for s, e in rec.reviewed) / 3600
    out["non_gameplay_fp_per_h"] = (
        sum(inside(p.peak, rec.non_gameplay) for p in ranked) / hours if hours else None
    )
    return out


# --------------------------------------------------------------------------- classification


def classification(rec: Recording, preds: list[Pred]) -> dict:
    """Label quality on matched pairs only, so a localisation miss is not counted twice."""
    preds, truths = in_scope(rec, preds)
    pairs = [(preds[pi], truths[ti]) for pi, ti in match(preds, truths).items() if truths[ti].label]
    return {
        "pairs": [(p.label, t.label) for p, t in pairs],
        "confidences": [(p.confidence, p.label == t.label) for p, t in pairs if p.confidence is not None],
    }


def label_table(pairs: list[tuple[str, str]]) -> dict[str, dict]:
    table = {}
    for label in LABELS:
        tp = sum(p == label and t == label for p, t in pairs)
        n_pred = sum(p == label for p, _ in pairs)
        n_true = sum(t == label for _, t in pairs)
        if not (n_pred or n_true):
            continue
        prec = tp / n_pred if n_pred else None
        rec = tp / n_true if n_true else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if n_pred and n_true else None)
        table[label] = {"precision": prec, "recall": rec, "f1": f1, "n_pred": n_pred, "n_true": n_true}
    return table


def abstention(pairs: list[tuple[str, str]]) -> float | None:
    return sum(p == UNLABELLED for p, _ in pairs) / len(pairs) if pairs else None


# --------------------------------------------------------------------------- calibration


def ece(items: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not items:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(c, ok) for c, ok in items if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if group:
            conf = sum(c for c, _ in group) / len(group)
            acc = sum(ok for _, ok in group) / len(group)
            total += len(group) / len(items) * abs(conf - acc)
    return total


def brier(items: list[tuple[float, bool]]) -> float | None:
    return sum((c - ok) ** 2 for c, ok in items) / len(items) if items else None


# --------------------------------------------------------------------------- aggregation


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(values: list[float | None], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95 % interval of the mean, resampling recordings (never seconds within one)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]
