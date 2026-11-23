"""Sound events per second, from CED-mini (an AudioSet tagger, 527 classes).

`tag` keeps every class's probability for every second, so which classes count
as "gunfire" or "vehicle" is decided later from config/moments.json and can
change without listening to the recording again.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import video_io
from . import models
from .audio import SAMPLE_RATE

WINDOW_S = 2


def class_names(folder) -> list[str]:
    """The model's classes in index order, from its class_labels_indices.csv."""
    import csv

    with open(folder / "class_labels_indices.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return [row[2] for row in rows[1:]]


def tag(samples: np.ndarray, log, progress, cancel: video_io.Canceller) -> dict:
    """{'probs': float32 [seconds, classes], 'names': class names}: each class's probability, per second."""
    import sherpa_onnx

    folder = models.ced_dir(log)
    names = class_names(folder)
    column = {name: i for i, name in enumerate(names)}
    workers = max(1, min(6, (os.cpu_count() or 2) // 2))
    taggers = [
        sherpa_onnx.AudioTagging(sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=str(folder / "model.int8.onnx"), num_threads=1, provider="cpu"),
            labels=str(folder / "class_labels_indices.csv"), top_k=len(names)))
        for _ in range(workers)
    ]
    n = int(np.ceil(len(samples) / SAMPLE_RATE))
    probs = np.zeros((n, len(names)), dtype=np.float32)
    done = [0]

    def run(worker: int) -> None:
        tagger = taggers[worker]
        for second in range(worker, n, workers):
            cancel.check()
            window = samples[second * SAMPLE_RATE:(second + WINDOW_S) * SAMPLE_RATE]
            if len(window) < SAMPLE_RATE // 2:
                continue
            stream = tagger.create_stream()
            stream.accept_waveform(SAMPLE_RATE, window)
            row = probs[second]
            for event in tagger.compute(stream):
                index = column.get(event.name)
                if index is not None:
                    row[index] = event.prob
            done[0] += 1
            if worker == 0:
                progress(done[0], n)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, range(workers)))
    progress(n, n)
    return {"probs": probs, "names": names}


def group(tags: dict, members: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per second: the highest probability among `members`, and the class that gave it."""
    names = tags["names"]
    columns = [i for i, name in enumerate(names) if name in set(members)]
    n = tags["probs"].shape[0]
    if not columns:
        return np.zeros(n), [""] * n
    block = tags["probs"][:, columns]
    best = block.argmax(axis=1)
    values = block[np.arange(n), best].astype(float)
    labels = [names[columns[b]] if v > 0 else "" for b, v in zip(best, values)]
    return values, labels
