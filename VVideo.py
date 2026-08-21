"""VVideo -- the window. Also the entry point PyInstaller freezes into the .exe.

    python VVideo.py          # run from source
    pyinstaller VVideo.spec   # build dist/VVideo.exe

One page over the shorts pipeline: a landscape highlight becomes a 1080x1920
portrait video. The same job is available as a CLI -- see make-short.py.
"""

from __future__ import annotations

import multiprocessing
import sys

from src import gui

if __name__ == "__main__":
    # Harmless from source, required in the frozen build: without it any
    # accidental child process re-runs the whole app instead of the child.
    multiprocessing.freeze_support()
    sys.exit(gui.run())
