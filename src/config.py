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


def _data_root() -> Path:
    """VVideo-data/ beside the .exe (or the project), so deleting the folder removes everything."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "VVideo-data"
    return Path(__file__).resolve().parent.parent / "VVideo-data"


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
