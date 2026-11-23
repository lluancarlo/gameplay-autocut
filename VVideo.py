"""VVideo -- the window. Also the entry point PyInstaller freezes into the .exe.

    python VVideo.py               # run from source
    pyinstaller VVideo.spec        # build publish/VVideo.exe
    VVideo.exe --self-test         # check the bundle; writes VVideo-data/self-test.json

A long gameplay recording in, vertical shorts of its best moments out.
"""

from __future__ import annotations

import multiprocessing
import sys

from src import gui

if __name__ == "__main__":
    # Harmless from source, required in the frozen build: without it any
    # accidental child process re-runs the whole app instead of the child.
    multiprocessing.freeze_support()
    if "--self-test" in sys.argv:
        from src import selftest

        sys.exit(selftest.run())
    sys.exit(gui.run())
