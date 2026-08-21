# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for VVideo.exe.

    pyinstaller VVideo.spec        -> dist\\VVideo.exe

Two things have to be carried in by hand, because neither is an ordinary
import PyInstaller can follow:

  config/       every tunable. src/config.py reads it from sys._MEIPASS when
                frozen, which is where this puts it.
  imageio_ffmpeg  the bundled ffmpeg.exe -- a data file inside the wheel, not a
                  linked library, so nothing in the import graph mentions it.

Built one-file, so the result is a single portable .exe. The cost is startup:
everything above unpacks to a temp folder on each launch. Swapping to a folder
build (instant startup, VVideo.exe plus its DLLs beside it) means moving
a.binaries/a.datas out of EXE and into a COLLECT -- see the note at the bottom.
"""

from PyInstaller.utils.hooks import collect_data_files


# The .ico is the file's own icon, baked into the executable below. The .png is
# carried along as data because Qt sets the *window* icon separately -- that is
# what the taskbar and Alt-Tab show while the app is running.
datas = [("config", "config"), ("vvideo-icon.png", ".")]
datas += collect_data_files("imageio_ffmpeg")

a = Analysis(
    ["VVideo.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Nothing here is imported, and each one costs tens of megabytes.
        "torch",
        "torchvision",
        "torchaudio",
        "onnxruntime",
        "transformers",
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
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VVideo",
    # Multi-size, built from vvideo-icon.png: Windows reaches for a different
    # size for the taskbar, Alt-Tab, Explorer and the title bar.
    icon="vvideo-icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
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

# For a folder build instead, delete a.binaries and a.datas from the EXE(...)
# call above and add:
#
#   coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="VVideo")
