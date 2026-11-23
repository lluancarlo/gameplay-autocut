# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for VVideo.

    pyinstaller VVideo.spec        -> publish\\VVideo.exe

Two things have to be carried in by hand, because neither is an ordinary
import PyInstaller can follow:

  config/       every tunable. src/config.py reads it from sys._MEIPASS when
                frozen, which is where this puts it.
  imageio_ffmpeg  the bundled ffmpeg.exe -- a data file inside the wheel, not a
                  linked library, so nothing in the import graph mentions it.

Built as a folder (VVideo.exe plus its libraries), so it starts instantly.
Whisper, the sound model and the CUDA libraries are not bundled: they download
on first use into VVideo-data/ beside the .exe.
"""

import os

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

CONF["distpath"] = SPECPATH


# The .ico is the file's own icon, baked into the executable below. The .png is
# carried along as data because Qt sets the *window* icon separately -- that is
# what the taskbar and Alt-Tab show while the app is running.
datas = [("config", "config"), ("vvideo-icon.png", ".")]
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("faster_whisper")
# onnxruntime-directml: onnxruntime.dll and DirectML.dll for the vision model (M2).
binaries = (collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("sherpa_onnx")
            + collect_dynamic_libs("onnxruntime"))

a = Analysis(
    ["VVideo.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=["sherpa_onnx"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Nothing here is imported, and each one costs tens of megabytes.
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "nvidia",
        "tkinter",
        "matplotlib",
        "pandas",
        "scipy",
        "IPython",
        "pytest",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VVideo",
    # Multi-size, built from vvideo-icon.png: Windows reaches for a different
    # size for the taskbar, Alt-Tab, Explorer and the title bar.
    icon="vvideo-icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window: this is the GUI. Anything that would have gone to
    # stdout is unreachable, which is why the window logs to a pane of its own
    # and shows tracebacks in it.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="publish")

if os.path.isdir(DISTPATH) and not os.listdir(DISTPATH):
    os.rmdir(DISTPATH)
