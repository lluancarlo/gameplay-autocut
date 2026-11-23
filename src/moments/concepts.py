"""Score image embeddings against the prompt bank (config/events.npz).

A concept's score for a frame is SigLIP 2's sigmoid probability that the frame
matches the concept's averaged prompt embedding. It is image-text similarity,
not event recognition: a parked car can score like a chase. The moments
engine (M3) uses these as features relative to the recording's own baseline,
never as labels on their own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from . import vision


@dataclass
class Bank:
    names: list[str]          # concept names, in order
    kinds: dict[str, str]     # concept -> "event" | "suppress" | "neutral"
    vectors: np.ndarray       # float32 [concepts, 768], ensembled and L2-normalised
    scale: float
    bias: float
    meta: dict


class BankMismatch(RuntimeError):
    """The prompt bank was built for a different image model or a different events.json."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder: Path | None = None) -> Bank:
    folder = folder or config.CONFIG_DIR
    with np.load(folder / "events.npz") as data:
        meta = json.loads(str(data["meta"]))
        if meta["image_encoder"]["sha256"] != vision.MODEL["sha256"]:
            raise BankMismatch("config/events.npz was built for a different vision model; rebuild it with "
                               "tools/build_prompt_bank.py")
        events = folder / "events.json"
        if events.is_file() and meta["events_json_sha256"] != _sha256(events):
            raise BankMismatch("config/events.json changed since config/events.npz was built; rebuild it with "
                               "tools/build_prompt_bank.py")
        embeddings, owners, kinds = data["embeddings"], [str(c) for c in data["concepts"]], data["kinds"]
        names = list(dict.fromkeys(owners))
        vectors = np.stack([embeddings[[o == n for o in owners]].mean(axis=0) for n in names])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return Bank(names, {o: str(k) for o, k in zip(owners, kinds)}, vectors.astype(np.float32),
                    float(data["logit_scale"]), float(data["logit_bias"]), meta)


def scores(bank: Bank, vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Concept -> probability per row of `vectors` (frames or seconds)."""
    logits = bank.scale * (vectors.astype(np.float32) @ bank.vectors.T) + bank.bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {name: probs[:, i] for i, name in enumerate(bank.names)}
