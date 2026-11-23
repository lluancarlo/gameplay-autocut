"""`GameplayAutoCut.exe --self-test`: check that the bundle can run every model it needs.

The window has no console, so the result goes to GameplayAutoCut-data/self-test.json
(and the exit code is non-zero if anything failed). Used to validate a build on
a clean machine: no GPU, a GPU with only the driver, and the dev machine.
Downloads nothing; a model that is not downloaded yet is reported as such.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback

from . import config


def _check(report: dict, name: str, fn) -> None:
    started = time.perf_counter()
    try:
        report[name] = {"ok": True, **(fn() or {})}
    except Exception as error:  # noqa: BLE001 -- the point is to report it
        report[name] = {"ok": False, "error": f"{type(error).__name__}: {error}",
                        "trace": traceback.format_exc().splitlines()[-3:]}
    report[name]["ms"] = round(1000 * (time.perf_counter() - started))


def run() -> int:
    import numpy as np

    report: dict = {"python": sys.version.split()[0], "frozen": bool(getattr(sys, "frozen", False)),
                    "platform": platform.platform()}

    def ffmpeg():
        from . import video_io
        return {"exe": video_io.ffmpeg_exe(), "encoder": video_io._encoder()[0]}

    def onnxruntime():
        import onnxruntime as ort
        return {"version": ort.__version__, "providers": ort.get_available_providers()}

    def vision():
        from .moments import vision as v
        path = v.models.models_dir() / v.MODEL["name"] / "vision_model_fp16.onnx"
        if not path.is_file():
            return {"model": "not downloaded yet (first analysis downloads it)"}
        sess, provider = v.session(path)
        frames = np.random.default_rng(0).integers(0, 255, (4, v.SIZE, v.SIZE, 3), dtype=np.uint8)
        out = sess.run(["pooler_output"], {sess.get_inputs()[0].name: v.to_pixels(frames)})[0]
        return {"provider": provider, "output": list(out.shape)}

    def prompt_bank():
        from .moments import concepts
        bank = concepts.load()
        return {"concepts": bank.names}

    def sound_model():
        import sherpa_onnx
        from .moments import models
        return {"sherpa_onnx": getattr(sherpa_onnx, "__version__", "?"),
                "ced_downloaded": (models.models_dir() / "ced-mini" / "model.int8.onnx").is_file()}

    def whisper_runtime():
        import ctranslate2
        return {"ctranslate2": ctranslate2.__version__, "cuda_devices": ctranslate2.get_cuda_device_count()}

    def screen_text():
        import cv2
        from .moments import ocr, vision as v
        image = np.full((540, 960, 3), 30, dtype=np.uint8)
        cv2.putText(image, "WASTED", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        found = ocr.engine(v.gpu_available())(image)
        return {"read": list(found.txts or [])}

    def ai_review():
        import subprocess
        from .moments import judge
        folder = judge._folder()
        exe = folder / "llama" / "llama-server.exe"
        files = {name: (folder / name).is_file() for name in judge.model_for(config.load("moments")["judge"])["files"]}
        if not exe.is_file():
            return {"runtime": "not downloaded yet (first AI review downloads it)", "model_files": files}
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"runtime": (out.stdout + out.stderr).strip().splitlines()[0], "model_files": files}

    for name, fn in (("ffmpeg", ffmpeg), ("onnxruntime", onnxruntime), ("vision", vision),
                     ("prompt_bank", prompt_bank), ("sound_model", sound_model), ("whisper_runtime", whisper_runtime),
                     ("screen_text", screen_text), ("ai_review", ai_review)):
        _check(report, name, fn)
    out = config.DATA_DIR / "self-test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0 if all(v.get("ok") for v in report.values() if isinstance(v, dict)) else 1
