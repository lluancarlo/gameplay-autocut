"""What kind of moment it is, and why: event labels and their evidence.

Evidence comes from the sound (CED groups, loudness), the speech (rate, trigger
phrases), the picture (SigLIP 2 concept curves and semantic novelty, M3) and,
for the strongest moments, physical motion (optical flow, M3). A label needs
its evidence to hold for a minimum number of seconds inside the moment; when
nothing qualifies the moment stays `unlabelled` rather than guessing.

A label carried by the picture alone is capped at "low": image-text similarity
is not event recognition, so it needs another modality to agree before it is
reported with more confidence. Levels are not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNLABELLED = "unlabelled"
ICONS = {"combat": "🔫", "vehicle": "🚗", "destruction": "💥", "reaction": "😮", "unusual": "🌀", UNLABELLED: "✨"}
SEE = "see:"


@dataclass
class Evidence:
    modality: str      # "audio" | "speech" | "vision" | "motion"
    signal: str        # a sound group ("gunfire"), "loudness", "speech", "keyword", "see:<concept>", "novelty", "motion"
    value: float       # strength, 0..1
    at: float          # second where it peaked
    detail: str = ""   # the AudioSet class, the phrase, or what the picture looks like
    seconds: int = 0   # seconds of the moment it was present for


def _short(name: str) -> str:
    """'Gunshot, gunfire' -> 'gunshot'; 'Police car (siren)' -> 'police car siren'."""
    return name.split(",")[0].replace("(", "").replace(")", "").strip().lower()


def _span(start: float, end: float, n: int) -> tuple[int, int]:
    return max(0, int(start)), min(n, int(np.ceil(end))) if n else int(np.ceil(end))


def collect(start: float, end: float, groups: dict[str, tuple[np.ndarray, list[str]]],
            curves: dict[str, np.ndarray], hits: list[tuple[float, str]], weights: dict[str, float],
            present: float, visual=None, motion: dict | None = None, motion_cfg: dict | None = None) -> list[Evidence]:
    """Everything that speaks for the moment [start, end), strongest first."""
    n = len(next(iter(curves.values()))) if curves else 0
    lo, hi = _span(start, end, n)
    found: list[Evidence] = []
    for name, (probs, classes) in groups.items():
        part = probs[lo:hi]
        if len(part) == 0 or part.max() < present / 2:
            continue
        peak = int(part.argmax())
        found.append(Evidence("audio", name, round(float(part.max()), 3), float(lo + peak),
                              classes[lo + peak], int((part >= present).sum())))
    for name, modality in (("loudness", "audio"), ("speech", "speech")):
        if name in curves:
            part = curves[name][lo:hi]
            if len(part) and part.max() >= 0.5:
                found.append(Evidence(modality, name, round(float(part.max()), 3), float(lo + int(part.argmax())),
                                      seconds=int((part >= 0.5).sum())))
    seen = set()
    for t, phrase_ in hits:
        if lo <= t < hi and phrase_ not in seen:
            seen.add(phrase_)
            found.append(Evidence("speech", "keyword", float(weights.get(phrase_, 0.5)), float(t), phrase_, 1))
    if visual is not None:
        for name, rel in visual.relative.items():
            part = rel[lo:hi]
            if len(part) and part.max() >= 0.25:
                found.append(Evidence("vision", SEE + name, round(float(part.max()), 3),
                                      float(lo + int(part.argmax())), visual.describe.get(name, name),
                                      int((part >= 0.5).sum())))
        part = visual.novelty[lo:hi]
        if len(part) and part.max() >= 0.5:
            found.append(Evidence("vision", "novelty", round(float(part.max()), 3), float(lo + int(part.argmax())),
                                  seconds=int((part >= 0.5).sum())))
    if motion:
        mcfg = motion_cfg or {"floor": 2.0, "span": 6.0}
        value = float(np.clip((motion["local"] - mcfg["floor"]) / mcfg["span"], 0, 1))
        if value >= 0.5:
            found.append(Evidence("motion", "motion", round(value, 3), float(lo), f"{motion['local']:.1f} px/frame",
                                  max(1, int(hi - lo))))
    return sorted(found, key=lambda e: -e.value)


def classify(evidence: list[Evidence], rules: dict) -> tuple[str, str, float, str]:
    """(label, detail, strength, level).

    Each rule maps signals to [min_value, min_seconds]. Strength is how far the
    best qualifying evidence clears its own bar (>= 1). Picture-only labels are
    capped at "low" unless sound, speech or motion agree.
    """
    best = (UNLABELLED, "", 0.0, False)
    # Events first: a shout is usually a reaction *to* something, so "reaction"
    # is only the label when no event qualifies.
    ordered = [k for k in rules if k != "reaction"] + [k for k in rules if k == "reaction"]
    for label in ordered:
        rule = rules[label]
        if label == "reaction" and best[0] != UNLABELLED:
            break
        qualifying = []
        for e in evidence:
            bar = rule["signals"].get(e.signal)
            if bar and e.value >= bar[0] and e.seconds >= bar[1]:
                qualifying.append((e.value / bar[0], e))
        if not qualifying:
            continue
        strength, top = max(qualifying, key=lambda x: x[0])
        if strength > best[2]:
            if top.signal.startswith(SEE):
                detail = top.signal[len(SEE):]
            elif top.signal == "keyword":
                detail = ""  # the phrase is listed right after
            else:
                detail = _short(top.detail)
            signals = {e.signal for _s, e in qualifying}
            if detail == label:
                detail = ""  # "Vehicle (vehicle)" says nothing
            if label == "vehicle" and "siren" in signals:
                detail = "siren"
            elif label == "vehicle" and SEE + "police" in signals:
                detail = "police"
            seen_only = all(e.modality == "vision" for _s, e in qualifying) and not any(
                e.modality != "vision" and e.value >= 0.5 for e in evidence)
            best = (label, detail, strength, seen_only)
    label, detail, strength, seen_only = best
    if label == UNLABELLED:
        return label, "", 0.0, ""
    level = "high" if strength >= 2 else "medium" if strength >= 1.4 else "low"
    if seen_only:
        level = "low"
    return label, detail, round(strength, 3), level


def phrase(e: Evidence) -> str:
    if e.signal == "keyword":
        return f'"{e.detail}"'
    if e.signal == "loudness":
        return "much louder than usual"
    if e.signal == "speech":
        return "fast speech"
    if e.signal.startswith(SEE):
        return f"looks like {e.detail}"
    if e.signal == "novelty":
        return "sudden change on screen"
    if e.signal == "motion":
        return "fast movement in the scene"
    name = _short(e.detail) or e.signal
    return f"{name} for {e.seconds}s" if e.seconds > 1 else name


def explain(label: str, detail: str, evidence: list[Evidence], supporting: tuple[str, ...] = (),
            limit: int = 3) -> str:
    """One line a person can read: what it is, then its evidence -- what supports the label first."""
    head = "No clear event" if label == UNLABELLED else label.title() + (f" ({detail})" if detail else "")
    ordered = sorted(evidence, key=lambda e: (e.signal not in supporting, -e.value))
    parts = list(dict.fromkeys(phrase(e) for e in ordered))[:limit]
    return head + (" — " + ", ".join(parts) if parts else "")


def escape(curves: dict[str, np.ndarray], end: float, cfg: dict) -> bool:
    """Temporal rule `pursuit_then_clear`: a siren-backed vehicle moment followed by calm.

    Upgrades vehicle -> escape when, in the `calm_s` seconds after the moment,
    siren and action stay below `calm_below`. Off unless enabled in config:
    it needs reviewed tuning data to prove its precision first.
    """
    if not cfg.get("enabled", False):
        return False
    lo = int(np.ceil(end))
    hi = lo + int(cfg["calm_s"])
    after = [curves[k][lo:hi] for k in ("siren", "action") if k in curves]
    if not after or any(len(a) < cfg["calm_s"] for a in after):
        return False
    return all(a.max() < cfg["calm_below"] for a in after)
