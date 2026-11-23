"""Config loading. Every tunable in this project lives in config/*.json."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


def _root() -> Path:
    """Where config/ sits.

    Running from source that is the project folder. Inside the PyInstaller
    build it is the bundle's extraction directory, since config/ ships as
    bundled data rather than beside the .exe -- __file__ points into the
    frozen archive and its parent is meaningless.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return Path(__file__).resolve().parent.parent


# The file-system name: the .exe, its settings file, its data folder, the release zip.
FILE_NAME = "GameplayAutoCut"
LEGACY_NAME = "VVideo"   # the project's name before Gameplay AutoCut


def _home() -> Path:
    """Beside the .exe when frozen, the project folder when running from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """GameplayAutoCut-data/ beside the .exe (or the project); delete it and everything is gone.

    A folder left by a build from before the rename (VVideo-data/, models and
    caches -- several GB) is renamed rather than downloaded again.
    """
    home = _home()
    folder, legacy = home / f"{FILE_NAME}-data", home / f"{LEGACY_NAME}-data"
    if not folder.exists() and legacy.is_dir():
        try:
            legacy.rename(folder)
        except OSError:
            return legacy  # in use or not ours to move: keep using it as it is
    return folder


ROOT = _root()
CONFIG_DIR = ROOT / "config"
DATA_DIR = _data_root()


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load config/<name>.json."""
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
