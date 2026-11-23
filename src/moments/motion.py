"""Physical motion inside candidate moments: camera vs. things moving in the scene (milestone M3).

Only the moments the score already found are looked at, so the cost does not
grow with the recording. Each window is decoded small and grey, dense optical
flow (OpenCV DIS) runs between consecutive frames, and:

  camera   a similarity transform fitted to the flow with RANSAC -- in first-
           and third-person games the player's camera moves all the time, and
           raw flow would mostly measure that.
  local    what the camera does not explain: the residual flow (vehicles,
           debris, characters), 90th percentile over the frame.
  HUD      pixels that barely change across the window (minimap, health bar,
           kill-feed frame) are masked out of both.

Camera motion is reported but never labels anything on its own; local motion
supports vehicle / destruction labels when sound or vision agree.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from .. import video_io

WIDTH = 320


@dataclass
class Motion:
    camera: np.ndarray   # px per frame at 320 px width, per sampled frame pair
    local: np.ndarray
    fps: float

    def summary(self) -> dict:
        if len(self.local) == 0:
            return {"camera": 0.0, "local": 0.0, "local_peak": 0.0}
        return {"camera": round(float(np.median(self.camera)), 3), "local": round(float(np.median(self.local)), 3),
                "local_peak": round(float(np.percentile(self.local, 90)), 3)}


def frames(path, start: float, end: float, fps: float, height: int,
           cancel: video_io.Canceller) -> np.ndarray:
    """Grey uint8 frames [count, height, 320] of [start, end) at `fps`."""
    args = [video_io.ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", str(path), "-t", f"{max(0.1, end - start):.3f}",
            "-map", "0:v:0", "-vf", f"fps={fps:g},scale={WIDTH}:{height}:flags=area,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=video_io._NO_WINDOW)
    with cancel.track(process):
        out, err = process.communicate()
    cancel.check()
    if process.returncode != 0:
        raise RuntimeError(f"Could not read frames for motion: {err.decode(errors='replace')[-200:]}")
    size = WIDTH * height
    count = len(out) // size
    return np.frombuffer(out[:count * size], dtype=np.uint8).reshape(count, height, WIDTH)


def static_mask(stack: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """True where the picture moves: pixels whose brightness barely changes over the window are HUD/overlay."""
    if len(stack) < 3:
        return np.ones(stack.shape[1:], dtype=bool)
    return stack.astype(np.float32).std(axis=0) >= threshold


def measure(stack: np.ndarray, fps: float, step: int = 8) -> Motion:
    import cv2

    if len(stack) < 2:
        return Motion(np.zeros(0), np.zeros(0), fps)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    moving = static_mask(stack)
    h, w = stack.shape[1:]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    keep = moving[ys, xs]
    ys, xs = ys[keep], xs[keep]
    camera, local = [], []
    for a, b in zip(stack[:-1], stack[1:]):
        flow = dis.calc(a, b, None)
        if len(xs) < 12:
            camera.append(0.0)
            local.append(0.0)
            continue
        src = np.stack([xs, ys], axis=1).astype(np.float32)
        vec = flow[ys, xs]
        dst = src + vec
        model, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if model is None:
            predicted = np.zeros_like(vec) + np.median(vec, axis=0)
        else:
            predicted = src @ model[:, :2].T + model[:, 2] - src
        residual = np.linalg.norm(vec - predicted, axis=1)
        camera.append(float(np.median(np.linalg.norm(predicted, axis=1))))
        local.append(float(np.percentile(residual, 90)))
    return Motion(np.array(camera), np.array(local), fps)


def window(path, start: float, end: float, fps: float, aspect: float, cancel: video_io.Canceller) -> Motion:
    height = max(2, int(round(WIDTH / max(aspect, 0.1) / 2)) * 2)
    return measure(frames(path, start, end, fps, height, cancel), fps)
