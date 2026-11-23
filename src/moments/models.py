from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ..config import DATA_DIR

CED_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/"
           "sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2")
CED_SHA256 = "bec8bfa0af2c20ec3a9e7c6dd6c92d0fb6e96d6b25370ebe7a48e7283a60a02c"
CED_FILES = ("model.int8.onnx", "class_labels_indices.csv")

PYPI = "https://files.pythonhosted.org/packages/"
CUDA_WHEELS = (
    (PYPI + "20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
     "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
     "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661", 553),
    (PYPI + "c7/bd/1d6709d2223e99f04079f16005f8ae6e11a14ee16b5eab6833686cf72fe1/"
     "nvidia_cudnn_cu12-9.28.1.4-py3-none-win_amd64.whl",
     "3467aafa9b260e6adf5fbbfc0e21d180133305b42032aa1160be2d3142bb8b09", 696),
    (PYPI + "52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/"
     "nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
     "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349", 76),
)

_dll_handles: list = []


def models_dir() -> Path:
    return DATA_DIR / "models"


def cuda_dir() -> Path:
    return DATA_DIR / "cuda"


def _download(url: str, sha256: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as fh:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            fh.write(chunk)
    if digest.hexdigest() != sha256:
        dest.unlink()
        raise RuntimeError(f"The download of {url.rsplit('/', 1)[-1]} is damaged. Try again.")
    return dest


def ced_dir(log) -> Path:
    folder = models_dir() / "ced-mini"
    if all((folder / name).is_file() for name in CED_FILES):
        return folder
    log("   ⬇️  Downloading the sound model (45 MB)…")
    part = _download(CED_URL, CED_SHA256, folder / "download.part")
    with tarfile.open(part, "r:bz2") as tar:
        for member in tar.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if name in CED_FILES and member.isfile():
                (folder / name).write_bytes(tar.extractfile(member).read())
    part.unlink()
    return folder


def ensure_cuda(log) -> bool:
    """True when Whisper can try the GPU. The frozen build downloads the CUDA libraries once."""
    import ctranslate2

    if ctranslate2.get_cuda_device_count() == 0:
        return False
    if not getattr(sys, "frozen", False) or (cuda_dir() / "ready").is_file():
        return True
    total = sum(size for _url, _sha, size in CUDA_WHEELS)
    log(f"   ⬇️  Downloading the GPU libraries for Whisper ({total / 1000:.1f} GB, once)…")
    for url, sha256, _size in CUDA_WHEELS:
        wheel = _download(url, sha256, cuda_dir() / "download.part")
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith(".dll") and "/bin/" in name:
                    (cuda_dir() / Path(name).name).write_bytes(archive.read(name))
        wheel.unlink()
    (cuda_dir() / "ready").touch()
    return True


def enable_cuda() -> None:
    if os.name != "nt" or _dll_handles:
        return
    dirs = [cuda_dir()] if cuda_dir().is_dir() else []
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        dirs += [d for base in spec.submodule_search_locations for d in Path(base).glob("*/bin") if d.is_dir()]
    _dll_handles.extend(os.add_dll_directory(str(d)) for d in dirs)
    os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
