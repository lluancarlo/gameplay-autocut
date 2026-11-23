"""Gameplay AutoCut's window: a long recording in, vertical shorts of its best moments out.

This module holds what the whole app shares -- where files go, the settings
file, the background Worker, the zoom bar -- and the main window, which walks
through the four steps in steps.py. Work runs on a QThread; touching widgets
from it would crash Qt, so everything it has to say comes back as signals
(`message`, `progress`, `finished`) that Qt delivers on the UI thread.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, QStandardPaths, QTimer, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from . import config, sysmon, video_io

APP_NAME = "Gameplay AutoCut"
TAGLINE = "A totally offline solution to transform gameplay into viral clips"

# Beside the project root when running from source, and unpacked next to
# config/ in the frozen build -- config._root() already knows which is which.
# This is the *window* icon: the .exe's own icon is set in GameplayAutoCut.spec.
ICON_PATH = config.ROOT / "gameplay-autocut-logo.png"


@lru_cache(maxsize=1)
def videos_dir() -> Path:
    """The user's Videos folder.

    Not %USERPROFILE%\\Videos, which is only where it usually is: Windows keeps
    the real location in the FOLDERID_Videos known folder, and that moves when
    someone relocates it -- to a second drive, or to OneDrive, which takes it
    over on a lot of machines. Qt's MoviesLocation reads that same known folder
    (and the XDG videos dir elsewhere), so it follows. Home/Videos is only the
    fallback for when Qt has nothing to report.
    """
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.MoviesLocation
    )
    return Path(location) if location else Path.home() / "Videos"


def default_source() -> Path:
    """Where the window looks for clips until told otherwise."""
    return videos_dir()


def default_output() -> Path:
    """Where shorts land unless the user picks somewhere else.

    A subfolder rather than Videos/ itself, and deliberately: a run reads every
    video in the source folder, so pointing the output at the folder it just
    read would feed the next run its own shorts. Nothing creates it until
    something is actually written there.
    """
    return videos_dir() / APP_NAME


def _drop_legacy_registry_key() -> None:
    """Remove HKCU\\Software\\VVideo (the old name) once nothing is left in it.

    QSettings empties the values but leaves the keys themselves standing, and
    an empty key is still a key -- exactly the trace this build is trying not
    to leave. Deepest first: a key with subkeys under it will not delete.
    """
    if sys.platform != "win32":
        return
    import winreg

    legacy = config.LEGACY_NAME
    for path in (rf"Software\{legacy}\{legacy}", rf"Software\{legacy}"):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
        except OSError:
            pass  # already gone, or not empty -- either way, nothing to do


def adopt_legacy_settings() -> None:
    """Move an older build's settings out of the registry and delete them.

    Builds before this one used QSettings' Windows default, which is a registry
    key under HKCU\\Software. Shipping the ini alone would leave that key
    sitting on every machine that ever ran one of them, so the first launch of
    this build carries the values across and takes the key with it. On a
    machine that never ran an older build this finds nothing and does nothing.
    """
    legacy = QSettings(config.LEGACY_NAME, config.LEGACY_NAME)
    keys = legacy.allKeys()
    if keys:
        target = QSettings(str(settings_file()), QSettings.Format.IniFormat)
        # Only if the ini has nothing to say yet -- a file next to the .exe is
        # this build's own state and outranks whatever the registry remembers.
        if not target.allKeys():
            for key in keys:
                # Only groups a tab still reads. Older builds had pages this
                # one does not, and copying their leftovers across would keep
                # the ini alive forever holding values nothing ever looks at.
                if key.startswith("short/"):
                    target.setValue(key, legacy.value(key))
            target.sync()
        legacy.clear()
        legacy.sync()
    _drop_legacy_registry_key()


def same(value: object, default: object) -> bool:
    """Is this control still on its default?

    Floats go through a slider's integer steps and a spin box's decimals on the
    way out, so a value nobody touched can come back a hair off the number it
    started as. Comparing exactly would write a settings file for a difference
    of 1e-16.
    """
    if isinstance(value, float) and isinstance(default, (int, float)):
        return abs(value - float(default)) < 1e-9
    return value == default


def settings_file() -> Path:
    """Where preferences live: GameplayAutoCut.ini beside the executable.

    QSettings would otherwise default to the registry on Windows -- a key left
    behind on a machine that only ever ran a portable .exe. An ini file next to
    the .exe keeps the whole app in one place: delete it and nothing remains.
    Written only when a value actually differs from its default, and removed
    again when none do. A VVideo.ini from before the rename is adopted.

    sys.executable, not config.ROOT: in the frozen build ROOT is the temporary
    unpack directory, which is gone the moment the process ends.
    """
    home = config._home()
    file, legacy = home / f"{config.FILE_NAME}.ini", home / f"{config.LEGACY_NAME}.ini"
    if not file.exists() and legacy.is_file():
        try:
            legacy.rename(file)
        except OSError:
            return legacy
    return file


# How often the strip along the bottom re-reads CPU, GPU and memory. Fast
# enough to show an encode starting and stopping; the reads themselves are a
# few kernel calls, so the cost of doing it twice a second is not measurable.
STATUS_INTERVAL_MS = 500

# Each gauge keeps this width whether it reads 0% or 100%, so the three of them
# stay put instead of shuffling sideways every time a digit is gained or lost.
GAUGE_WIDTH = 96


@dataclass
class Outcome:
    """What a finished job hands back to the window.

    `folder` is what the "Open output folder" button opens, and `preview` --
    when the job produced a still rather than a video -- is shown right away,
    since the whole point of a preview is looking at it.
    """

    summary: str
    folder: Path | None = None
    preview: Path | None = None


class Worker(QObject):
    """Runs one job off the UI thread and reports back by signal."""

    message = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool, object)
    cancelled = Signal()

    def __init__(self, job: Callable[..., Outcome]) -> None:
        super().__init__()
        self._job = job

    def run(self) -> None:
        try:
            outcome = self._job(self.message.emit, self.progress.emit)
        except video_io.Cancelled:
            # Asked for, so not an error: no traceback, no dialog.
            self.message.emit("")
            self.message.emit("🛑 Cancelled.")
            self.cancelled.emit()
            return
        except Exception as error:  # noqa: BLE001 -- the UI is the error handler
            # The traceback is the only diagnostic a windowed .exe can offer;
            # there is no console for it to fall out of.
            self.message.emit("")
            self.message.emit(traceback.format_exc().rstrip())
            self.finished.emit(False, str(error) or error.__class__.__name__)
            return
        self.finished.emit(True, outcome)


class ZoomBar(QWidget):
    """Zoom as a drag bar, with the factor it is sitting on beside it.

    Qt sliders are integer-only and zoom is not, so the slider counts in the
    fractions below and everything outside this class keeps talking in the
    float the pipeline actually takes. The readout is not decoration: dragging a bare bar
    tells you nothing about the number you landed on.
    """

    valueChanged = Signal(float)

    # Thousandths rather than the hundredths the spin box used to step in. It
    # is finer than anyone drags, and that is the point: the pixel boxes below
    # can only land on a size the bar can represent, and at 1/100 a typed 1920
    # snapped to 1912.
    STEPS_PER_UNIT = 1000

    def __init__(self, *, low: float, high: float, value: float, step: float) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(self._raw(low), self._raw(high))
        self.slider.setSingleStep(self._raw(step))
        self.slider.setPageStep(self._raw(step) * 2)
        self.slider.setValue(self._raw(value))
        self.slider.setMinimumWidth(150)

        self._readout = QLabel()
        # Fixed width, so the bar does not twitch sideways as the number under
        # it goes from 0.5 to 1.25 mid-drag.
        self._readout.setFixedWidth(40)
        self._readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        row.addWidget(self.slider, stretch=1)
        row.addWidget(self._readout)

        self.slider.valueChanged.connect(self._relay)
        self._show()

    @classmethod
    def _raw(cls, zoom: float) -> int:
        return round(float(zoom) * cls.STEPS_PER_UNIT)

    def value(self) -> float:
        return self.slider.value() / self.STEPS_PER_UNIT

    def setValue(self, zoom: float) -> None:  # noqa: N802 -- matches QSpinBox
        self.slider.setValue(self._raw(zoom))

    def _relay(self, _raw: int) -> None:
        self._show()
        self.valueChanged.emit(self.value())

    def _show(self) -> None:
        # %g, not a fixed 2dp: the bar is finer than 2dp now, and a readout
        # that rounded would disagree with the pixels underneath it.
        self._readout.setText(f"{self.value():g}")



class MainWindow(QMainWindow):
    """Four steps, one after the other: video -> find clips -> pick clips -> make shorts.

    The header shows where the user is but cannot be clicked: each step hands
    over to the next one itself, and only offers Back where going back makes
    sense. The steps live in steps.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)

        from .steps import AnalyseStep, PickStep, Prefs, SelectStep, ShortsStep, StepHeader

        self.prefs = Prefs()
        self.header = StepHeader(("Video", "Find clips", "Pick clips", "Make shorts"))
        self.select_step = SelectStep(self.prefs)
        self.analyse_step = AnalyseStep()
        self.pick_step = PickStep(self.prefs)
        self.shorts_step = ShortsStep(self.prefs)
        self.steps = (self.select_step, self.analyse_step, self.pick_step, self.shorts_step)
        self.stack = QStackedWidget()
        for step in self.steps:
            self.stack.addWidget(step)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 6)
        layout.addWidget(self.header)
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.select_step.chosen.connect(self._analyse)
        self.analyse_step.finished.connect(self._pick)
        self.analyse_step.abandoned.connect(lambda: self._go(0))
        self.pick_step.back.connect(lambda: self._go(0))
        self.pick_step.proceed.connect(self._make_shorts)
        self.shorts_step.back.connect(lambda: self._go(2))
        self.shorts_step.restart.connect(self._restart)

        self._build_status_bar()
        self._go(0)
        self.resize(1180, 780)
        self.setMinimumSize(960, 640)

    def _go(self, index: int) -> None:
        for step in self.steps:
            if step is not self.steps[index]:
                step.leave()
        self.stack.setCurrentIndex(index)
        self.header.set_current(index)

    def _analyse(self, source, voice) -> None:
        self._go(1)
        self.analyse_step.start(source, voice)

    def _pick(self, analysis) -> None:
        self.pick_step.show_analysis(analysis)
        self._go(2)

    def _make_shorts(self, clips) -> None:
        self.shorts_step.prepare(self.pick_step.analysis, clips)
        self._go(3)

    def _restart(self) -> None:
        self.select_step.reset()
        self._go(0)

    # -- the strip along the bottom ---------------------------------------

    def _build_status_bar(self) -> None:
        """CPU, GPU and memory, under the log.

        It belongs to the window rather than to the page: one timer and one set
        of counters for the whole app, sitting below the page -- which is below
        the log either way.
        """
        self.monitor = sysmon.Monitor()

        bar = QStatusBar()
        # No grip: the window is resizable from any edge anyway, and it would
        # sit in the corner competing with the numbers.
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)

        self.gauges: dict[str, QLabel] = {}
        for key, caption in (("cpu", "CPU"), ("gpu", "GPU"), ("memory", "Memory")):
            label = QLabel(f"{caption} --")
            label.setMinimumWidth(GAUGE_WIDTH)
            self.gauges[key] = label
            bar.addWidget(label)

        gpu = self.monitor.gpu_name
        self.gauges["gpu"].setToolTip(gpu or "No NVIDIA GPU found")

        self._ticker = QTimer(self)
        self._ticker.setInterval(STATUS_INTERVAL_MS)
        self._ticker.timeout.connect(self._refresh_status)
        self._ticker.start()
        self._refresh_status()

    def _refresh_status(self) -> None:
        reading = self.monitor.read()
        gpu = self.monitor.gpu_name

        self._set_gauge("cpu", "CPU", reading.cpu, "")
        self._set_gauge(
            "gpu",
            "GPU",
            reading.gpu,
            # The card's name is the useful half of this: it is what says
            # whether the encode is landing on the GPU it was meant to.
            "\n".join(
                part
                for part in (gpu, sysmon.gigabytes(reading.vram_used, reading.vram_total))
                if part
            )
            or "No NVIDIA GPU found",
        )
        self._set_gauge(
            "memory",
            "Memory",
            reading.memory,
            sysmon.gigabytes(reading.memory_used, reading.memory_total),
        )

    def _set_gauge(self, key: str, caption: str, value: float | None, tip: str) -> None:
        """One gauge. A dash rather than a zero when the counter is unreadable.

        No GPU, or a platform without the Win32 counters, has to look like the
        absence of a number -- "GPU 0%" is a reading, and a wrong one.
        """
        label = self.gauges[key]
        label.setText(f"{caption} --" if value is None else f"{caption} {value:.0f}%")
        if tip:
            label.setToolTip(tip)

    def closeEvent(self, event) -> None:
        busy = [step for step in self.steps if step.busy()]
        if busy:
            # Killing a running ffmpeg or a half-written mp4 on a stray click is
            # worth one confirmation.
            answer = QMessageBox.question(
                self, APP_NAME, f"{APP_NAME} is still working. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        # Quitting has to take the work with it: ffmpeg is a separate process
        # and would otherwise run its encode to the end with no window to show.
        for step in self.steps:
            step.stop_for_close()
        self.prefs.save()
        self._stop_monitoring()
        event.accept()

    def _stop_monitoring(self) -> None:
        """Let go of NVML before the window does, on the way out.

        Stopping the timer first matters: a tick that lands between the
        shutdown and the process actually ending would be reading through a
        handle that is no longer good for anything.
        """
        self._ticker.stop()
        self.monitor.close()


def run() -> int:
    QApplication.setApplicationName(APP_NAME)
    QApplication.setOrganizationName(APP_NAME)
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    # Before the window, which reads its settings while it is being built.
    adopt_legacy_settings()
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window.show()
    return app.exec()
