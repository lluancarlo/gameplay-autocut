"""What the recording looks like: SigLIP 2 image embeddings, frame by frame.

Milestone M2. Frames are sampled at `vision.fps` (10 by default) through one
ffmpeg pipe, converted to RGB at full size and only then resized to 224x224 --
the same order as the reference preprocessing, which keeps the embeddings
within cos 0.999 of it -- and run through the SigLIP 2 base image encoder in
ONNX Runtime, on the GPU through DirectML when there is one.

The result is one L2-normalised 768-d vector per frame, cached beside the
audio analysis. Nothing here decides anything yet: concepts.py scores the
vectors against the prompt bank, and the moments engine uses those scores
from milestone M3 on.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import video_io
from . import models

SIZE = 224
FRAME_BYTES = SIZE * SIZE * 3
DIM = 768

# google/siglip2-base-patch16-224 (Apache-2.0), exported to ONNX by onnx-community. Pinned to one revision
# and checked by SHA-256, like every other download.
MODEL = {
    "name": "siglip2-base-patch16-224",
    "source": "google/siglip2-base-patch16-224",
    "repo": "onnx-community/siglip2-base-patch16-224-ONNX",
    "revision": "ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
    "file": "onnx/vision_model_fp16.onnx",
    "sha256": "a1959f7bd3993a607e48839f6d01e25b876fe76afda301b028b78eef68aabd95",
    "size_mb": 186,
    "licence": "Apache-2.0",
}
ENCODER_ID = MODEL["sha256"][:12]


@dataclass
class Embeddings:
    vectors: np.ndarray   # float16 [frames, 768], L2-normalised
    times: np.ndarray     # float32 [frames], seconds
    fps: float
    provider: str         # "DirectML" | "CPU"
    seconds: float        # how long the pass took

    @property
    def frames(self) -> int:
        return len(self.times)

    def per_second(self, n: int) -> np.ndarray:
        """Mean vector of each second, re-normalised -> float32 [n, 768]. Empty seconds stay zero."""
        out = np.zeros((n, self.vectors.shape[1]), dtype=np.float32)
        counts = np.zeros(n)
        index = np.minimum(self.times.astype(int), n - 1)
        np.add.at(out, index, self.vectors.astype(np.float32))
        np.add.at(counts, index, 1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=out, where=norms > 0)


def model_path(log) -> Path:
    folder = models.models_dir() / MODEL["name"]
    path = folder / Path(MODEL["file"]).name
    if not path.is_file():
        log(f"   ⬇️  Downloading the vision model ({MODEL['size_mb']} MB, once)…")
        url = f"https://huggingface.co/{MODEL['repo']}/resolve/{MODEL['revision']}/{MODEL['file']}"
        part = models._download(url, MODEL["sha256"], folder / "download.part")
        part.replace(path)
    return path


def session(path: Path, prefer_gpu: bool = True):
    """(InferenceSession, provider name). DirectML first; CPU if the GPU cannot take it."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if prefer_gpu and "DmlExecutionProvider" in available:
        options = ort.SessionOptions()
        # DirectML's documented requirements: no memory pattern, sequential execution.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            sess = ort.InferenceSession(str(path), options, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
            if sess.get_providers()[0] == "DmlExecutionProvider":
                return sess, "DirectML"
        except Exception:  # noqa: BLE001 -- no usable GPU: fall through to the CPU
            pass
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), "CPU"


def to_pixels(frames: np.ndarray) -> np.ndarray:
    """uint8 [b, 224, 224, 3] -> float32 [b, 3, 224, 224] in [-1, 1] (rescale 1/255, mean = std = 0.5)."""
    return (frames.astype(np.float32) * (2.0 / 255.0) - 1.0).transpose(0, 3, 1, 2)


def video_codec(path: Path) -> str:
    """The first video stream's codec name as ffmpeg reports it ("h264", "hevc", "av1"...), or ""."""
    import re

    _code, _out, err = video_io._capture([video_io.ffmpeg_exe(), "-hide_banner", "-i", str(path)])
    match = re.search(r"Stream #\d+:\d+.*?: Video: (\w+)", err)
    return match.group(1).lower() if match else ""


def nvidia_present() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def decoder(path: Path, cfg: dict) -> str:
    """'nvdec' or 'cpu' -- measured, not assumed (docs/ai-highlights/PERFORMANCE_PLAN.md, M2b).

    On the target machine the CPU decodes H.264 faster than NVDEC, but HEVC and
    especially AV1 are much faster on NVDEC (AV1 3440x1440p60: 0.8x realtime on
    the CPU, 5.6x on NVDEC).
    """
    mode = cfg.get("decode", "auto")
    if mode in ("cpu", "nvdec"):
        return mode
    return "nvdec" if video_codec(path) in cfg.get("nvdec_codecs", ()) and nvidia_present() else "cpu"


def read_frames(path: Path, fps: float, cancel: video_io.Canceller, out: queue.Queue, batch: int,
                hwaccel: bool = False) -> None:
    """Producer: push uint8 batches of frames into `out`, then None. Errors are pushed as the exception.

    With `hwaccel` the decode runs on NVDEC and frames come back to system memory; ffmpeg falls back to
    software decoding by itself if the GPU cannot take the stream.
    """
    args = [
        video_io.ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        *(["-hwaccel", "cuda"] if hwaccel else []), "-i", str(path),
        "-map", "0:v:0", "-vf", f"fps={fps:g},format=rgb24,scale={SIZE}:{SIZE}:flags=bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=video_io._NO_WINDOW)
        with cancel.track(process):
            pending: list[np.ndarray] = []
            while True:
                chunk = process.stdout.read(FRAME_BYTES)
                if len(chunk) < FRAME_BYTES:
                    break
                pending.append(np.frombuffer(chunk, dtype=np.uint8).reshape(SIZE, SIZE, 3))
                if len(pending) == batch:
                    out.put(np.stack(pending))
                    pending = []
            if pending:
                out.put(np.stack(pending))
            err = process.stderr.read().decode(errors="replace")
            code = process.wait()
        cancel.check()
        if code != 0:
            raise RuntimeError(f"Could not read the video frames: {err.strip()[-300:]}")
        out.put(None)
    except BaseException as error:  # noqa: BLE001 -- handed to the consumer, which re-raises
        out.put(error)


def embed(path: Path, duration: float, cfg: dict, *, log, progress, cancel: video_io.Canceller) -> Embeddings:
    """Sample the video, encode every frame. GPU at cfg['fps']; CPU falls back to cfg['cpu_fps']."""
    sess, provider = session(model_path(log))
    fps = float(cfg["fps"] if provider != "CPU" else cfg["cpu_fps"])
    decode = decoder(path, cfg)
    expected = max(1, int(duration * fps))
    if provider == "CPU":
        log(f"   ⚠️  No GPU for the vision model: {fps:g} frames/s on the CPU instead of {cfg['fps']:g}")
    log(f"   👁️  Looking at the video: {fps:g} frames/s on {provider}, decoding on {decode.upper()} "
        f"(~{expected:,} frames)…")
    name = sess.get_inputs()[0].name
    batches: queue.Queue = queue.Queue(maxsize=8)
    producer = threading.Thread(target=read_frames, args=(path, fps, cancel, batches, int(cfg["batch"]),
                                                          decode == "nvdec"), daemon=True)
    started = time.perf_counter()
    producer.start()
    vectors: list[np.ndarray] = []
    done = 0
    while True:
        item = batches.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        cancel.check()
        out = sess.run(["pooler_output"], {name: to_pixels(item)})[0]
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        vectors.append(out.astype(np.float16))
        done += len(item)
        progress(min(done, expected), expected)
    producer.join()
    elapsed = time.perf_counter() - started
    stacked = np.concatenate(vectors) if vectors else np.zeros((0, DIM), dtype=np.float16)
    times = (np.arange(len(stacked)) / fps).astype(np.float32)
    log(f"      {len(stacked):,} frames in {elapsed:.0f}s ({len(stacked) / max(elapsed, 1e-6):.0f}/s)")
    return Embeddings(stacked, times, fps, provider, elapsed)


def cache_file(cache: Path, fps: float) -> Path:
    return cache / f"vision-{ENCODER_ID}-{fps:g}fps.npz"


def gpu_available() -> bool:
    import onnxruntime as ort

    return "DmlExecutionProvider" in ort.get_available_providers()


def load_cached(cache: Path, cfg: dict) -> Embeddings | None:
    """A previous pass at the rate this machine would run now (a CPU-rate pass is redone once a GPU appears)."""
    for fps in ((cfg["fps"],) if gpu_available() else (cfg["fps"], cfg["cpu_fps"])):
        file = cache_file(cache, fps)
        if file.is_file():
            with np.load(file) as data:
                return Embeddings(data["vectors"], data["times"], float(data["fps"]), str(data["provider"]),
                                  float(data["seconds"]))
    return None


def save(cache: Path, result: Embeddings) -> None:
    np.savez(cache_file(cache, result.fps), vectors=result.vectors, times=result.times, fps=result.fps,
             provider=result.provider, seconds=result.seconds)
