from __future__ import annotations

import re
import unicodedata
import warnings

import numpy as np

from .audio import SAMPLE_RATE
from .speech import Word


def clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    half = max(1, window // 2)
    view = np.lib.stride_tricks.sliding_window_view(np.pad(values, half, mode="edge"), 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(view, axis=1)[: len(values)]


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel / kernel.sum(), mode="valid")


def loudness_db(samples: np.ndarray, n: int) -> np.ndarray:
    half = SAMPLE_RATE // 2
    frames = np.zeros(n * 2 * half, dtype=np.float32)
    frames[: min(len(samples), len(frames))] = samples[: len(frames)]
    rms = np.sqrt(np.mean(frames.reshape(-1, half) ** 2, axis=1))
    return (20 * np.log10(np.maximum(rms, 1e-5))).reshape(n, 2).max(axis=1)


def loudness(db: np.ndarray, cfg: dict) -> np.ndarray:
    return clip01((db - rolling_median(db, cfg["baseline_s"]) - cfg["floor_db"]) / cfg["span_db"])


def reaction(prob: np.ndarray, cfg: dict) -> np.ndarray:
    return clip01(prob / cfg["full"])


def action(prob: np.ndarray, cfg: dict) -> np.ndarray:
    return clip01((prob - rolling_median(prob, cfg["baseline_s"])) / cfg["span"])


def words_per_second(words: list[Word], n: int) -> np.ndarray:
    rate = np.zeros(n)
    for word in words:
        rate[min(n - 1, int(word.start))] += 1
    return rate


def speech(words: list[Word], n: int, cfg: dict) -> np.ndarray:
    rate = np.convolve(words_per_second(words, n), np.ones(3) / 3, mode="same")
    talking = np.where(rate > 0, rate, np.nan)
    base = np.nan_to_num(rolling_median(talking, cfg["baseline_s"]), nan=cfg["burst_rate"] / 2)
    excited = clip01((rate - base) / cfg["span"])
    raw = words_per_second(words, n)
    burst = np.zeros(n)
    for t in range(2, n - 1):
        if raw[t - 2:t].sum() == 0 and raw[t:t + 2].mean() >= cfg["burst_rate"]:
            burst[t] = 1.0
    return np.maximum(excited, burst)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"(\w)\1+", r"\1", text).strip()


def keywords(words: list[Word], n: int, phrases: dict[str, float]) -> tuple[np.ndarray, list[tuple[float, str]]]:
    table = {tuple(normalize(p).split()): (p, w) for p, w in phrases.items()}
    tokens = [normalize(w.text) for w in words]
    curve, hits = np.zeros(n), []
    for i in range(len(tokens)):
        for key, (phrase, weight) in table.items():
            if key and tuple(tokens[i:i + len(key)]) == key:
                t = int(words[i].start)
                curve[max(0, t - 1):t + 2] = np.maximum(curve[max(0, t - 1):t + 2], weight)
                hits.append((words[i].start, phrase))
    return curve, hits
