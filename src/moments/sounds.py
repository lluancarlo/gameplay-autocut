from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import video_io
from . import models
from .audio import SAMPLE_RATE

GROUPS = {
    "reaction": {"Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Screaming", "Laughter",
                 "Giggle", "Belly laugh", "Chuckle, chortle", "Cheering"},
    "action": {"Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire",
               "Cap gun", "Burst, pop", "Boom"},
}
WINDOW_S = 2


def tag(samples: np.ndarray, log, progress, cancel: video_io.Canceller) -> dict:
    """Per second: the highest probability of each group, and the class that gave it."""
    import sherpa_onnx

    folder = models.ced_dir(log)
    workers = max(1, min(6, (os.cpu_count() or 2) // 2))
    taggers = [
        sherpa_onnx.AudioTagging(sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=str(folder / "model.int8.onnx"), num_threads=1, provider="cpu"),
            labels=str(folder / "class_labels_indices.csv"), top_k=50))
        for _ in range(workers)
    ]
    n = int(np.ceil(len(samples) / SAMPLE_RATE))
    probs = {group: np.zeros(n) for group in GROUPS}
    labels = {group: [""] * n for group in GROUPS}
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
            for event in tagger.compute(stream):
                for group, names in GROUPS.items():
                    if event.name in names and event.prob > probs[group][second]:
                        probs[group][second] = event.prob
                        labels[group][second] = event.name
            done[0] += 1
            if worker == 0:
                progress(done[0], n)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, range(workers)))
    progress(n, n)
    return {"probs": probs, "labels": labels}
