"""Per-second visual curves from the SigLIP 2 embeddings (milestone M3).

Three kinds of signal, kept apart on purpose:

  concept curves   how much more a second looks like "a car chase" (etc.) than
                   this recording usually does -- the SigLIP logit minus its
                   rolling median, so a driving game does not make every drive
                   a highlight. Image-text similarity, not event recognition:
                   they are evidence, never labels on their own.
  novelty          how different the screen is from the seconds before it --
                   semantic change (new place, flash, death screen). Says
                   nothing about movement; motion.py measures that.
  non-gameplay     menus, maps, pause and loading screens, used to push the
                   score down rather than to find anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import concepts, signals
from .vision import Embeddings


@dataclass
class Visual:
    logits: dict[str, np.ndarray]     # concept -> per-second max logit over the frames of that second
    relative: dict[str, np.ndarray]   # event concept -> 0..1, above the recording's own usual level
    novelty: np.ndarray               # 0..1
    menu: np.ndarray                  # 0..1, how sure the second is non-gameplay
    vision: np.ndarray                # 0..1, the strongest event concept per second (what the score uses)
    describe: dict[str, str]          # concept -> a phrase for explanations ("a person jumping into water")


def per_second_logits(emb: Embeddings, bank: concepts.Bank, n: int) -> dict[str, np.ndarray]:
    """Max over the frames of each second: a 0.3 s explosion should not be averaged away."""
    z = bank.scale * (emb.vectors.astype(np.float32) @ bank.vectors.T) + bank.bias
    out = np.full((n, len(bank.names)), np.nan, dtype=np.float32)
    seconds = np.minimum(emb.times.astype(int), n - 1)
    for column in range(len(bank.names)):
        np.fmax.at(out[:, column], seconds, z[:, column])
    filled = {}
    for column, name in enumerate(bank.names):
        series = out[:, column]
        fallback = np.nanmedian(series) if np.isfinite(series).any() else -20.0
        filled[name] = np.where(np.isfinite(series), series, fallback)
    return filled


def relative(z: np.ndarray, cfg: dict) -> np.ndarray:
    """Above the recording's usual level AND actually present.

    Relative alone makes any car look like "police cars chasing a car" in a game
    where cars are rare; the presence gate (an absolute logit floor) stops that.
    """
    above = signals.clip01((z - signals.rolling_median(z, cfg["baseline_s"]) - cfg["floor"]) / cfg["span"])
    present = signals.clip01((z - cfg["present_floor"]) / cfg["present_span"])
    return above * present


def novelty(per_second: np.ndarray, cfg: dict) -> np.ndarray:
    """1 - cosine to the mean of the previous `window_s` seconds, relative to the recording's usual change."""
    n = len(per_second)
    window = int(cfg["window_s"])
    raw = np.zeros(n)
    csum = np.cumsum(np.vstack([np.zeros((1, per_second.shape[1])), per_second]), axis=0)
    for t in range(1, n):
        lo = max(0, t - window)
        context = (csum[t] - csum[lo]) / (t - lo)
        norm = np.linalg.norm(context) * np.linalg.norm(per_second[t])
        raw[t] = 0.0 if norm == 0 else 1.0 - float(per_second[t] @ context) / norm
    base = signals.rolling_median(raw, cfg["baseline_s"])
    return signals.clip01((raw - base - cfg["floor"]) / cfg["span"])


def build(emb: Embeddings, bank: concepts.Bank, n: int, cfg: dict, describe: dict[str, str]) -> Visual:
    logits = per_second_logits(emb, bank, n)
    events = [name for name in bank.names if bank.kinds.get(name) == "event"]
    rel = {name: relative(logits[name], cfg["concepts"]) for name in events}
    menu_names = [name for name in bank.names if bank.kinds.get(name) == "suppress"]
    neutral = [name for name in bank.names if bank.kinds.get(name) == "neutral"]
    if menu_names:
        margin = np.max([logits[m] for m in menu_names], axis=0)
        if neutral:
            margin = margin - np.max([logits[o] for o in neutral], axis=0)
        menu = signals.clip01((margin - cfg["menu"]["floor"]) / cfg["menu"]["span"])
    else:
        menu = np.zeros(n)
    nov = novelty(emb.per_second(n), cfg["novelty"])
    strongest = np.max([rel[name] for name in events], axis=0) if events else np.zeros(n)
    return Visual(logits, rel, nov, menu, strongest * (1 - menu), describe)
