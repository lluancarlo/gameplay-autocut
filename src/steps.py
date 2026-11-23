"""The four steps of the window: video -> find clips -> pick clips -> make shorts.

Each step is a page of the main window's stack. None of them can be reached by
clicking: a step says it is done with a signal and the window moves on (see
gui.MainWindow). Long work -- the analysis, the renders, the preview frame --
runs on a Worker thread through JobRunner, so the window never freezes.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from . import pipeline, shorts, video_io
from .gui import APP_NAME, Worker, ZoomBar, default_output, default_source, same, settings_file
from .moments import Analysis, clock, find_moments
from .moments.audio import track_count
from .moments.labels import ICONS, UNLABELLED

ACCENT = QColor("#24a148")
CONTEXT_S = 15
THUMB = QSize(240, 135)
PREVIEW = QSize(270, 480)


def _title(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setPointSize(font.pointSize() + 6)
    font.setBold(True)
    label.setFont(font)
    return label


def _primary(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(38)
    button.setMinimumWidth(180)
    font = button.font()
    font.setBold(True)
    button.setFont(font)
    button.setStyleSheet(
        "QPushButton { background: #24a148; color: white; border-radius: 4px; padding: 4px 16px; }"
        "QPushButton:disabled { background: #8d8d8d; color: #e0e0e0; }"
        "QPushButton:hover:!disabled { background: #198038; }")
    return button


def _plain(text: str) -> str:
    """Without emoji: on a card, Qt measures the line without the emoji font and elides what follows."""
    return " ".join("".join(c for c in text if ord(c) < 0x2600 and c not in "️‍").split())


def _elastic(label: QLabel) -> QLabel:
    """Long paths in a label must not set the window's minimum width."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)
    return label


# --------------------------------------------------------------------------- shared


class Prefs:
    """What the window remembers, in Gameplay AutoCut.ini -- only values that differ from their default.

    Leave everything at its default and there is no file at all; delete the file
    and the app is back to new.
    """

    GROUPS = ("app", "framing", "short", "moments")  # the last two are older builds' groups

    def __init__(self) -> None:
        self.settings = QSettings(str(settings_file()), QSettings.Format.IniFormat)
        self.values: dict[str, object] = {}
        self.defaults: dict[str, object] = {}
        framing = Framing.defaults()
        self.register("app/source_dir", str(default_source()), legacy="moments/source_dir")
        self.register("app/dest", str(default_output()))
        self.register("app/min_score", 30, int)
        for key, legacy, value, cast in (
            ("zoom", "short/zoom", framing.zoom, float), ("offset_y", "short/offset_y", framing.offset_y, int),
            ("opacity", "short/opacity", framing.opacity, float), ("blur", "short/blur", framing.blur_sigma, float),
        ):
            self.register(f"framing/{key}", value, cast, legacy)

    def register(self, key: str, default, cast=str, legacy: str | None = None) -> None:
        self.defaults[key] = default
        raw = self.settings.value(key)
        if raw is None and legacy:
            raw = self.settings.value(legacy)
        try:
            self.values[key] = default if raw is None else cast(raw)
        except (TypeError, ValueError):
            self.values[key] = default

    def get(self, key: str):
        return self.values[key]

    def set(self, key: str, value) -> None:
        self.values[key] = value

    def save(self) -> None:
        for group in self.GROUPS:
            self.settings.remove(group)
        for key, value in self.values.items():
            if not same(value, self.defaults[key]):
                self.settings.setValue(key, value)
        self.settings.sync()
        if not self.settings.allKeys():
            settings_file().unlink(missing_ok=True)


class Framing:
    @staticmethod
    def defaults() -> shorts.Settings:
        try:
            return shorts.Settings.load()
        except Exception:  # noqa: BLE001 -- a missing config must not stop the window
            return shorts.Settings(canvas_w=1080, canvas_h=1920, zoom=0.5, offset_y=0,
                                   opacity=0.75, blur_sigma=8.0, blur_downscale=8, crop=(0, 0, 0, 0))

    @staticmethod
    def from_prefs(prefs: Prefs) -> shorts.Settings:
        return shorts.Settings.load(zoom=prefs.get("framing/zoom"), offset_y=prefs.get("framing/offset_y"),
                                    opacity=prefs.get("framing/opacity"), blur_sigma=prefs.get("framing/blur"))


class JobRunner(QObject):
    """One background job at a time, reported back on the UI thread."""

    message = Signal(str)
    progress = Signal(int, int)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self.cancel: video_io.Canceller | None = None
        self._done = self._failed = self._cancelled = None

    def busy(self) -> bool:
        return self._thread is not None

    def start(self, job, cancel: video_io.Canceller, on_done, on_failed=None, on_cancelled=None) -> bool:
        if self._thread is not None:
            return False
        self.cancel = cancel
        self._done, self._failed, self._cancelled = on_done, on_failed, on_cancelled
        self._thread = QThread(self)
        self._worker = Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self.message)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._finished)
        self._worker.cancelled.connect(self._was_cancelled)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self.cancel is not None:
            self.cancel.cancel()

    def stop_for_close(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)

    def _retire(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        self._thread = self._worker = self.cancel = None

    def _finished(self, ok: bool, payload: object) -> None:
        done, failed = self._done, self._failed
        self._retire()
        if ok:
            done(payload)
        elif failed is not None:
            failed(str(payload))

    def _was_cancelled(self) -> None:
        cancelled = self._cancelled
        self._retire()
        if cancelled is not None:
            cancelled()


class Step(QWidget):
    """A page of the window. Subclasses override what applies to them."""

    def leave(self) -> None:
        """The window moved to another step."""

    def busy(self) -> bool:
        return False

    def stop_for_close(self) -> None:
        pass


class StepHeader(QWidget):
    """Where the user is in the four steps. Read-only on purpose: the steps move themselves."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 8)
        row.setSpacing(6)
        self.labels: list[QLabel] = []
        self.names = names
        for index, _name in enumerate(names):
            if index:
                arrow = QLabel("›")
                arrow.setStyleSheet("color: palette(mid); font-size: 16px;")
                row.addWidget(arrow)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(30)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.labels.append(label)
            row.addWidget(label)
        self.set_current(0)

    def set_current(self, current: int) -> None:
        for index, (label, name) in enumerate(zip(self.labels, self.names)):
            if index < current:
                label.setText(f"✓  {name}")
                label.setStyleSheet("color: #24a148; border-bottom: 3px solid #24a148; font-weight: bold;")
            elif index == current:
                label.setText(f"{index + 1}  {name}")
                label.setStyleSheet("color: palette(text); border-bottom: 3px solid #24a148; font-weight: bold;"
                                    "background: rgba(36,161,72,40);")
            else:
                label.setText(f"{index + 1}  {name}")
                label.setStyleSheet("color: palette(mid); border-bottom: 3px solid palette(midlight);")


# --------------------------------------------------------------------------- 1. video


class SelectStep(Step):
    """Step 1: choose the long recording."""

    chosen = Signal(object, object)  # Path, voice track or None

    def __init__(self, prefs: Prefs) -> None:
        super().__init__()
        self.prefs = prefs
        self.source: Path | None = None
        self.setAcceptDrops(True)

        outer = QVBoxLayout(self)
        outer.addStretch(1)
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        box.setMaximumWidth(720)
        inner = QVBoxLayout(box)
        inner.setContentsMargins(32, 28, 32, 28)
        inner.setSpacing(12)
        inner.addWidget(_title("Choose a gameplay recording"))
        intro = QLabel("Gameplay AutoCut listens to the whole recording, finds its best moments, and turns the ones you "
                       "pick into vertical 1080×1920 clips for Shorts, Reels and TikTok.")
        intro.setWordWrap(True)
        inner.addWidget(intro)

        drop = QLabel("Drop a video here\nor")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop.setMinimumHeight(90)
        drop.setStyleSheet("border: 2px dashed palette(mid); border-radius: 8px; color: palette(mid);")
        inner.addWidget(drop)
        browse = QPushButton("Choose video...")
        browse.setMinimumHeight(34)
        browse.clicked.connect(self._browse)
        inner.addWidget(browse)

        self.file_label = _elastic(QLabel())
        font = self.file_label.font()
        font.setBold(True)
        self.file_label.setFont(font)
        inner.addWidget(self.file_label)
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: palette(mid);")
        inner.addWidget(self.info_label)

        voice_note = QLabel("Your voice is found automatically — on its own audio track, or mixed with the "
                            "game (Gameplay AutoCut tells your Portuguese from the game's English).")
        voice_note.setWordWrap(True)
        voice_note.setStyleSheet("color: palette(mid);")
        inner.addWidget(voice_note)

        go = QHBoxLayout()
        go.addStretch(1)
        self.next_button = _primary("Find clips  ▶")
        self.next_button.clicked.connect(self._next)
        go.addWidget(self.next_button)
        inner.addLayout(go)

        centre = QHBoxLayout()
        centre.addStretch(1)
        centre.addWidget(box, 3)
        centre.addStretch(1)
        outer.addLayout(centre)
        outer.addStretch(2)
        self.reset()

    def reset(self) -> None:
        self.source = None
        self.file_label.setText("No video chosen yet.")
        self.info_label.setText("")
        self.next_button.setEnabled(False)

    def _browse(self) -> None:
        suffixes = " ".join(f"*{s}" for s in sorted(pipeline.VIDEO_SUFFIXES))
        start = str(self.source.parent) if self.source else str(self.prefs.get("app/source_dir"))
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a recording", start, f"Videos ({suffixes})")
        if chosen:
            self.use(Path(chosen))

    def use(self, path: Path) -> None:
        if path.suffix.lower() not in pipeline.VIDEO_SUFFIXES or not path.is_file():
            QMessageBox.warning(self, APP_NAME, f"Not a video Gameplay AutoCut can read:\n{path}")
            return
        try:
            info = video_io.probe(path)
            tracks = track_count(path)
        except Exception as error:  # noqa: BLE001 -- shown to the user, not fatal
            QMessageBox.warning(self, APP_NAME, f"Could not read {path.name}:\n{error}")
            return
        if tracks == 0:
            QMessageBox.warning(self, APP_NAME, f"{path.name} has no audio — Gameplay AutoCut finds moments by their sound.")
            return
        self.source = path
        self.prefs.set("app/source_dir", str(path.parent))
        self.file_label.setText(path.name)
        audio = "1 audio track" if tracks == 1 else f"{tracks} audio tracks"
        self.info_label.setText(f"{info.width}×{info.height} · {info.fps:.0f} fps · {clock(info.duration)} · {audio}")
        self.next_button.setEnabled(True)

    def _next(self) -> None:
        if self.source is not None:
            self.chosen.emit(self.source, None)  # None: the voice track is detected

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.use(Path(urls[0].toLocalFile()))


# --------------------------------------------------------------------------- 2. find clips


class AnalyseStep(Step):
    """Step 2: find the moments. Moves on by itself when done; Cancel goes back to step 1."""

    finished = Signal(object)  # Analysis
    abandoned = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.runner = JobRunner(self)
        self.runner.message.connect(self._message)
        self.runner.progress.connect(self._progress)

        outer = QVBoxLayout(self)
        outer.addStretch(1)
        box = QWidget()
        box.setMaximumWidth(720)
        inner = QVBoxLayout(box)
        inner.setSpacing(12)
        inner.addWidget(_title("Finding the best moments"))
        self.file_label = _elastic(QLabel())
        inner.addWidget(self.file_label)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setMinimumHeight(22)
        inner.addWidget(self.bar)
        self.status = _elastic(QLabel())
        inner.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setFixedHeight(150)
        inner.addWidget(self.log)
        hint = QLabel("Gameplay AutoCut listens to the whole recording, looks at it 10 times a second, reads the text on "
                      "screen, and then an AI (offline, on your GPU) watches the best moments and judges them. The "
                      "first time it downloads its models (about 5 GB, plus 1.3 GB of GPU libraries on NVIDIA "
                      "cards). A recording analysed before opens instantly.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        inner.addWidget(hint)
        row = QHBoxLayout()
        row.addStretch(1)
        self.enough_button = QPushButton("Use what we have  ▶")
        self.enough_button.setMinimumHeight(32)
        self.enough_button.setToolTip("Stop the AI review now and show the moments: reviewed ones use the AI's "
                                      "judgement, the rest the detectors' score.")
        self.enough_button.clicked.connect(self._enough)
        self.enough_button.setVisible(False)
        row.addWidget(self.enough_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(32)
        self.cancel_button.clicked.connect(self._cancel)
        row.addWidget(self.cancel_button)
        inner.addLayout(row)
        centre = QHBoxLayout()
        centre.addStretch(1)
        centre.addWidget(box, 3)
        centre.addStretch(1)
        outer.addLayout(centre)
        outer.addStretch(2)

    def start(self, source: Path, voice: int | None) -> None:
        self.file_label.setText(source.name)
        self.log.clear()
        self.status.setText("Starting…")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.cancel_button.setEnabled(True)
        self.enough_button.setVisible(False)
        self.enough_button.setEnabled(True)
        cancel = video_io.Canceller()
        self.skip = threading.Event()
        skip = self.skip

        def job(log, progress):
            span = pipeline.Span(progress)
            analysis = find_moments(source, voice_track=voice, log=log, review_skip=skip,
                                    progress=span.slice(0, 0.95).counting(), cancel=cancel)
            thumbs = span.slice(0.95, 1)
            if analysis.moments:
                log("   🖼️  Thumbnails…")
            for index, moment in enumerate(analysis.moments, start=1):
                thumb = analysis.cache / "thumbs" / f"{int(moment.peak)}.jpg"
                if not thumb.is_file():
                    video_io.thumbnail(source, thumb, moment.peak, THUMB.width(), cancel)
                thumbs.at(index / len(analysis.moments))
            return analysis

        self.runner.start(job, cancel, self._done, self._failed, self.abandoned.emit)

    def _message(self, line: str) -> None:
        if line.strip():
            self.log.appendPlainText(line.rstrip())
            self.status.setText(line.strip().splitlines()[-1])
            if "AI review" in line and "stopped" not in line:
                self.enough_button.setVisible(True)
            if "Thumbnails" in line:
                self.enough_button.setVisible(False)

    def _enough(self) -> None:
        self.enough_button.setEnabled(False)
        self.status.setText("Finishing the review in progress, then showing the moments…")
        self.skip.set()

    def _progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)

    def _done(self, analysis: Analysis) -> None:
        if not analysis.moments:
            QMessageBox.information(self, APP_NAME, "No moments stood out in this recording. Try another one.")
            self.abandoned.emit()
            return
        self.finished.emit(analysis)

    def _failed(self, error: str) -> None:
        QMessageBox.critical(self, APP_NAME, f"The analysis failed:\n{error}")
        self.abandoned.emit()

    def _cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.status.setText("Stopping…")
        self.runner.stop()

    def busy(self) -> bool:
        return self.runner.busy()

    def stop_for_close(self) -> None:
        self.runner.stop_for_close()


# --------------------------------------------------------------------------- 3. pick clips


class HypeCurve(QWidget):
    clicked = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(70)
        self.curve: list[float] = []
        self.spans: list[tuple[float, float, bool, bool]] = []
        self.menu: list[float] = []

    def show_curve(self, curve: list[float], spans: list[tuple[float, float, bool, bool]],
                   menu: list[float] | None = None) -> None:
        self.curve, self.spans = curve, spans
        if menu is not None:
            self.menu = menu
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if not self.curve:
            return
        width, height, count = self.width(), self.height() - 4, len(self.curve)
        grey = QColor(128, 128, 128, 70)
        for second, value in enumerate(self.menu[:count]):
            if value >= 0.5:  # menus, maps, pause screens: never offered as moments
                painter.fillRect(int(second / count * width), 0, max(1, int(width / count) + 1), self.height(), grey)
        for start, end, current, picked in self.spans:
            colour = QColor(ACCENT)
            colour.setAlpha(150 if current else 95 if picked else 35)
            painter.fillRect(int(start / count * width), 0, max(2, int((end - start) / count * width)),
                             self.height(), colour)
        top = max(self.curve) or 1.0
        line = QPolygonF([QPointF(i / count * width, 2 + height - v / top * height) for i, v in enumerate(self.curve)])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.palette().text().color(), 1.2))
        painter.drawPolyline(line)

    def mousePressEvent(self, event) -> None:
        if self.curve:
            self.clicked.emit(event.position().x() / self.width() * len(self.curve))


class RangeBar(QWidget):
    """The clip on a strip of context. Clicking or dragging moves the playhead inside the clip.

    The bounds are not draggable: they change only through the Start/End ±1s
    buttons, so a click meant to replay a bit can never shorten the clip.
    """

    seek = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click or drag to play from that point. Use the Start / End buttons to trim.")
        self.low, self.high, self.start, self.end, self.playhead = 0.0, 1.0, 0.0, 1.0, 0.0
        self._dragging = False

    def set_range(self, low: float, high: float, start: float, end: float) -> None:
        self.low, self.high, self.start, self.end = low, high, start, end
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        self.update()

    def _x(self, seconds: float) -> float:
        return 8 + (seconds - self.low) / max(0.001, self.high - self.low) * (self.width() - 16)

    def _seconds(self, x: float) -> float:
        return self.low + (x - 8) / max(1, self.width() - 16) * (self.high - self.low)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        mid = self.height() // 2
        painter.fillRect(8, mid - 4, self.width() - 16, 8, self.palette().mid())
        painter.fillRect(int(self._x(self.start)), mid - 6, int(self._x(self.end) - self._x(self.start)), 12, ACCENT)
        painter.setPen(QPen(self.palette().text().color(), 2))
        for seconds in (self.start, self.end):
            painter.drawLine(int(self._x(seconds)), 2, int(self._x(seconds)), self.height() - 2)
        painter.setPen(QPen(QColor("#e8b100"), 2))
        painter.drawLine(int(self._x(self.playhead)), 4, int(self._x(self.playhead)), self.height() - 4)

    def mousePressEvent(self, event) -> None:
        self._dragging = True
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging:
            return
        # Inside the clip only: outside it the loop would jump straight back anyway.
        seconds = min(max(self._seconds(event.position().x()), self.start), max(self.start, self.end - 0.2))
        self.playhead = seconds
        self.update()
        self.seek.emit(seconds)

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = False


class PickStep(Step):
    """Step 3: the moments as cards. Tick the ones to keep, trim them, go on."""

    back = Signal()
    proceed = Signal(object)  # list[(start, end, peak)]

    def __init__(self, prefs: Prefs) -> None:
        super().__init__()
        self.prefs = prefs
        self.analysis: Analysis | None = None
        self.ranges: list[tuple[float, float]] = []
        self.current: int | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self.summary = _elastic(QLabel())
        top.addWidget(self.summary, 1)
        top.addWidget(QLabel("Minimum score"))
        self.min_score = QSlider(Qt.Orientation.Horizontal)
        self.min_score.setRange(0, 100)
        self.min_score.setValue(int(prefs.get("app/min_score")))
        self.min_score.setFixedWidth(220)
        self.min_score.valueChanged.connect(self._filter)
        top.addWidget(self.min_score)
        self.shown_label = QLabel()
        self.shown_label.setMinimumWidth(110)
        top.addWidget(self.shown_label)
        top.addWidget(QLabel("Type"))
        self.type_combo = QComboBox()
        self.type_combo.setMinimumWidth(170)
        self.type_combo.setToolTip("Show only one kind of moment. The type comes from what was heard: "
                                   "gunfire, engines and sirens, crashes, shouting or trigger phrases.")
        self.type_combo.currentIndexChanged.connect(lambda _i: self._filter())
        top.addWidget(self.type_combo)
        layout.addLayout(top)

        self.hype = HypeCurve()
        self.hype.setToolTip("How strong each second is, from sound and picture. Click a shaded moment to "
                             "play it. Grey: menus, maps and pause screens, left out.")
        self.hype.clicked.connect(self._jump_to)
        layout.addWidget(self.hype)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        cards_box = QWidget()
        cards_layout = QVBoxLayout(cards_box)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards = QListWidget()
        self.cards.setViewMode(QListWidget.ViewMode.IconMode)
        self.cards.setIconSize(THUMB)
        self.cards.setGridSize(QSize(THUMB.width() + 16, THUMB.height() + 80))
        self.cards.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.cards.setMovement(QListWidget.Movement.Static)
        self.cards.setWordWrap(True)
        self.cards.setMinimumSize(280, 260)
        self.cards.currentItemChanged.connect(lambda item, _old: self._select(item))
        self.cards.itemChanged.connect(lambda _item: self._count_picked())
        cards_layout.addWidget(self.cards)
        picks = QHBoxLayout()
        for text, state in (("Tick all shown", True), ("Untick all", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, s=state: self._tick_all(s))
            picks.addWidget(button)
        picks.addStretch(1)
        cards_layout.addLayout(picks)
        splitter.addWidget(cards_box)

        player_box = QWidget()
        player = QVBoxLayout(player_box)
        player.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        self.video.setMinimumSize(420, 236)
        player.addWidget(self.video, stretch=1)
        self.why_label = _elastic(QLabel())
        self.why_label.setWordWrap(True)
        self.why_label.setTextFormat(Qt.TextFormat.RichText)
        player.addWidget(self.why_label)
        self.range_bar = RangeBar()
        self.range_bar.seek.connect(lambda seconds: self.player.setPosition(int(seconds * 1000)))
        player.addWidget(self.range_bar)
        nudges = QHBoxLayout()
        for text, edge, delta in (("-1s", "start", -1), ("+1s", "start", 1), (None, None, 0),
                                  ("-1s", "end", -1), ("+1s", "end", 1)):
            if text is None:
                self.range_label = QLabel()
                self.range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                nudges.addWidget(self.range_label, stretch=1)
                continue
            button = QPushButton(f"{edge.title()} {text}")
            button.clicked.connect(lambda _c=False, e=edge, d=delta: self._nudge(e, d))
            nudges.addWidget(button)
        player.addLayout(nudges)
        splitter.addWidget(player_box)
        splitter.setSizes([560, 520])
        layout.addWidget(splitter, stretch=1)

        bottom = QHBoxLayout()
        back = QPushButton("◀  Choose another video")
        back.setMinimumHeight(34)
        back.clicked.connect(self.back.emit)
        bottom.addWidget(back)
        bottom.addStretch(1)
        self.next_button = _primary("Make shorts  ▶")
        self.next_button.clicked.connect(self._next)
        bottom.addWidget(self.next_button)
        layout.addLayout(bottom)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.loop_timer = QTimer(self)
        self.loop_timer.setInterval(40)
        self.loop_timer.timeout.connect(self._keep_looping)

    def show_analysis(self, analysis: Analysis) -> None:
        self.analysis = analysis
        self.ranges = [(m.start, m.end) for m in analysis.moments]
        self.current = None
        self.player.setSource(QUrl.fromLocalFile(str(analysis.path)))
        self.cards.blockSignals(True)
        self.cards.clear()
        for index, moment in enumerate(analysis.moments):
            thumb = analysis.cache / "thumbs" / f"{int(moment.peak)}.jpg"
            item = QListWidgetItem(QIcon(QPixmap(str(thumb))), "")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setToolTip(moment.explanation or ", ".join(moment.reasons))
            self.cards.addItem(item)
            self._label(item)
        self.cards.blockSignals(False)
        self._fill_types()
        seen = analysis.vision
        looked = (f" · {seen['frames']:,} frames seen ({seen['fps']:g}/s, {seen['provider']})" if seen
                  else " · sound only")
        if seen and seen.get("menu_seconds"):
            looked += f" · {clock(seen['menu_seconds'])} of menus and maps left out (grey)"
        if analysis.voice:
            looked += f" · voice: {analysis.voice['detail']}"
        if analysis.review:
            r = analysis.review
            looked += f" · AI reviewed {r['reviewed']} of {r['planned']}" + (" (stopped early)" if r["skipped"] else "")
        self.summary.setText(f"<b>{analysis.path.name}</b> · {len(analysis.moments)} moments in "
                             f"{clock(analysis.duration)}{looked} · tick the ones you want as shorts")
        plain = self.summary.text().replace("<b>", "").replace("</b>", "")
        self.summary.setToolTip(plain.replace(" · ", chr(10)))
        self.range_label.setText("Click a moment to play it.")
        self.why_label.setText("")
        self._filter()

    def _fill_types(self) -> None:
        counts: dict[str, int] = {}
        for moment in self.analysis.moments:
            counts[moment.label] = counts.get(moment.label, 0) + 1
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        self.type_combo.addItem(f"All types ({len(self.analysis.moments)})", None)
        for label in sorted(counts, key=lambda k: (k == UNLABELLED, -counts[k])):
            self.type_combo.addItem(f"{ICONS.get(label, '')} {label} ({counts[label]})", label)
        self.type_combo.blockSignals(False)

    def _label(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        moment = self.analysis.moments[index]
        start, end = self.ranges[index]
        # No emoji on the card itself: in icon mode Qt measures the line without the emoji font's
        # height and elides everything after it. The filter and the line under the player keep them.
        kind = moment.label.upper() + (f" · {moment.label2}" if moment.label2 else "")
        if moment.ai:
            kind = f"AI {moment.ai['score']}/10 · {_plain(moment.ai['title'])[:34]}"
        item.setText(f"#{index + 1} · score {moment.score}\n{kind}\n{clock(start)} · {end - start:.0f}s")

    def _filter(self) -> None:
        self.prefs.set("app/min_score", self.min_score.value())
        if self.analysis is None:
            return
        limit, kind = self.min_score.value(), self.type_combo.currentData()
        shown = 0
        for row in range(self.cards.count()):
            item = self.cards.item(row)
            moment = self.analysis.moments[item.data(Qt.ItemDataRole.UserRole)]
            hidden = moment.score < limit or (kind is not None and moment.label != kind)
            item.setHidden(hidden)
            shown += not hidden
        self.shown_label.setText(f"{limit} · {shown} of {self.cards.count()}")
        self._draw_curve()
        self._count_picked()

    def _draw_curve(self) -> None:
        if self.analysis is None:
            return
        limit, picked = self.min_score.value(), set(self.picked_indices())
        spans = [(start, end, index == self.current, index in picked)
                 for index, (start, end) in enumerate(self.ranges)
                 if self.analysis.moments[index].score >= limit or index in picked]
        self.hype.show_curve(self.analysis.curve, spans, self.analysis.signals.get("menu", []))

    def _jump_to(self, seconds: float) -> None:
        for row in range(self.cards.count()):
            item = self.cards.item(row)
            start, end = self.ranges[item.data(Qt.ItemDataRole.UserRole)]
            if not item.isHidden() and start <= seconds <= end:
                self.cards.setCurrentItem(item)
                return

    def _select(self, item: QListWidgetItem | None) -> None:
        if item is None or self.analysis is None:
            return
        self.current = item.data(Qt.ItemDataRole.UserRole)
        self._update_range()
        self.player.setPosition(int(self.ranges[self.current][0] * 1000))
        self.player.play()
        self.loop_timer.start()
        self._draw_curve()

    def _update_range(self) -> None:
        start, end = self.ranges[self.current]
        low = max(0.0, start - CONTEXT_S)
        high = min(self.analysis.duration, end + CONTEXT_S)
        self.range_bar.set_range(low, high, start, end)
        self.range_label.setText(f"{clock(start)} → {clock(end)} · {end - start:.1f}s")
        moment = self.analysis.moments[self.current]
        level = f" <span style='color:#8d8d8d'>· {moment.level} confidence</span>" if moment.level else ""
        text = f"<b>{ICONS.get(moment.label, '')} #{self.current + 1}</b> {moment.explanation}{level}"
        if moment.ai:
            text = (f"<b>{ICONS.get(moment.label, '')} {moment.ai['title']}</b> — AI {moment.ai['score']}/10<br>"
                    f"{moment.ai['description']}<br><span style='color:#8d8d8d'>{moment.explanation}</span>")
        self.why_label.setText(text)
        item = self.cards.currentItem()
        if item is not None:
            self._label(item)

    def _range_changed(self, start: float, end: float) -> None:
        if self.current is None:
            return
        self.ranges[self.current] = (round(start, 2), round(end, 2))
        self._update_range()
        self._draw_curve()

    def _nudge(self, edge: str, delta: float) -> None:
        if self.current is None:
            return
        start, end = self.ranges[self.current]
        if edge == "start":
            start = min(max(0.0, start + delta), end - 1)
        else:
            end = max(min(self.analysis.duration, end + delta), start + 1)
        self._range_changed(start, end)

    def _keep_looping(self) -> None:
        if self.current is None:
            return
        position = self.player.position() / 1000
        self.range_bar.set_playhead(position)
        start, end = self.ranges[self.current]
        if position >= end or position < start - 0.5:
            self.player.setPosition(int(start * 1000))

    def picked_indices(self) -> list[int]:
        return [self.cards.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.cards.count())
                if self.cards.item(row).checkState() == Qt.CheckState.Checked]

    def _tick_all(self, state: bool) -> None:
        check = Qt.CheckState.Checked if state else Qt.CheckState.Unchecked
        self.cards.blockSignals(True)
        for row in range(self.cards.count()):
            item = self.cards.item(row)
            if not state or not item.isHidden():
                item.setCheckState(check)
        self.cards.blockSignals(False)
        self._count_picked()

    def _count_picked(self) -> None:
        count = len(self.picked_indices())
        self.next_button.setText(f"Make {count} short{'s' if count != 1 else ''}  ▶" if count else "Make shorts  ▶")
        self.next_button.setEnabled(count > 0)
        self._draw_curve()

    def _next(self) -> None:
        picked = sorted(self.picked_indices(), key=lambda i: self.ranges[i][0])
        clips = [(*self.ranges[i], self.analysis.moments[i].peak, self.analysis.moments[i].label,
                  self.analysis.moments[i].ai or {}) for i in picked]
        if clips:
            self.proceed.emit(clips)

    def leave(self) -> None:
        self.loop_timer.stop()
        self.player.pause()

    def stop_for_close(self) -> None:
        self.loop_timer.stop()
        self.player.stop()


# --------------------------------------------------------------------------- 4. make shorts


class FramingDialog(QDialog):
    """Zoom, position, background and blur. Every change re-renders the preview beside it."""

    changed = Signal()

    def __init__(self, parent: QWidget, prefs: Prefs) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adjust framing")
        self.prefs = prefs
        self._before = {k: prefs.get(k) for k in ("framing/zoom", "framing/offset_y", "framing/opacity", "framing/blur")}
        form = QFormLayout(self)
        intro = QLabel("The preview in the main window follows every change.")
        intro.setStyleSheet("color: palette(mid);")
        form.addRow(intro)
        self.zoom = ZoomBar(low=0.05, high=2.0, value=prefs.get("framing/zoom"), step=0.05)
        self.zoom.setToolTip("1.0 fills the portrait frame and crops the sides; 0.5 leaves a blurred bar above and below.")
        form.addRow("Zoom", self.zoom)
        self.offset = QSpinBox()
        self.offset.setRange(-1920, 1920)
        self.offset.setSingleStep(10)
        self.offset.setSuffix(" px")
        self.offset.setValue(prefs.get("framing/offset_y"))
        self.offset.setToolTip("Move the clip up (negative) or down (positive).")
        form.addRow("Vertical position", self.offset)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(prefs.get("framing/opacity"))
        self.opacity.setToolTip("Brightness of the blurred fill above and below, 0 (black) to 1.")
        form.addRow("Background", self.opacity)
        self.blur = QDoubleSpinBox()
        self.blur.setRange(0.0, 64.0)
        self.blur.setSingleStep(1.0)
        self.blur.setDecimals(1)
        self.blur.setValue(prefs.get("framing/blur"))
        form.addRow("Blur", self.blur)
        for widget in (self.zoom, self.offset, self.opacity, self.blur):
            widget.valueChanged.connect(self._apply)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        defaults = buttons.addButton("Defaults", QDialogButtonBox.ButtonRole.ResetRole)
        defaults.clicked.connect(self._defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.setMinimumWidth(420)

    def _apply(self, *_ignored) -> None:
        self.prefs.set("framing/zoom", self.zoom.value())
        self.prefs.set("framing/offset_y", self.offset.value())
        self.prefs.set("framing/opacity", round(self.opacity.value(), 2))
        self.prefs.set("framing/blur", round(self.blur.value(), 1))
        self.changed.emit()

    def _defaults(self) -> None:
        d = Framing.defaults()
        for widget, value in ((self.zoom, d.zoom), (self.offset, d.offset_y), (self.opacity, d.opacity),
                              (self.blur, d.blur_sigma)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._apply()

    def reject(self) -> None:
        for key, value in self._before.items():
            self.prefs.set(key, value)
        self.changed.emit()
        super().reject()


class ShortsStep(Step):
    """Step 4: preview the framing, pick where the files go, render the shorts."""

    back = Signal()
    restart = Signal()

    def __init__(self, prefs: Prefs) -> None:
        super().__init__()
        self.prefs = prefs
        self.analysis: Analysis | None = None
        self.clips: list[tuple[float, float, float, str, dict]] = []
        self.written: list[Path] = []
        self.render_runner = JobRunner(self)
        self.render_runner.message.connect(self._message)
        self.render_runner.progress.connect(self._progress)
        self.preview_runner = JobRunner(self)
        self._preview_again = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._render_preview)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        self.preview = QLabel("Preview")
        self.preview.setFixedSize(PREVIEW)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: black; color: #8d8d8d;")
        left.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview of"))
        self.preview_combo = QComboBox()
        self.preview_combo.currentIndexChanged.connect(lambda _i: self._schedule_preview())
        preview_row.addWidget(self.preview_combo, 1)
        left.addLayout(preview_row)
        self.framing_label = QLabel()
        self.framing_label.setStyleSheet("color: palette(mid);")
        left.addWidget(self.framing_label)
        self.adjust_button = QPushButton("Adjust framing...")
        self.adjust_button.clicked.connect(self._adjust)
        left.addWidget(self.adjust_button)
        left.addStretch(1)
        layout.addLayout(left)

        right = QVBoxLayout()
        right.addWidget(_title("Make the shorts"))
        self.summary = _elastic(QLabel())
        right.addWidget(self.summary)
        self.clip_list = QListWidget()
        right.addWidget(self.clip_list, 1)
        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Save to"))
        self.dest_edit = QLineEdit(str(prefs.get("app/dest")))
        self.dest_edit.editingFinished.connect(lambda: prefs.set("app/dest", self.dest_edit.text().strip()))
        dest_row.addWidget(self.dest_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_dest)
        dest_row.addWidget(browse)
        right.addLayout(dest_row)
        self.bar = QProgressBar()
        self.bar.setMinimumHeight(20)
        right.addWidget(self.bar)
        self.status = _elastic(QLabel())
        right.addWidget(self.status)

        buttons = QHBoxLayout()
        self.back_button = QPushButton("◀  Back to clips")
        self.back_button.setMinimumHeight(34)
        self.back_button.clicked.connect(self.back.emit)
        buttons.addWidget(self.back_button)
        buttons.addStretch(1)
        self.open_button = QPushButton("Open folder")
        self.open_button.setMinimumHeight(34)
        self.open_button.clicked.connect(self._open_dest)
        buttons.addWidget(self.open_button)
        self.new_button = QPushButton("New video")
        self.new_button.setMinimumHeight(34)
        self.new_button.clicked.connect(self.restart.emit)
        buttons.addWidget(self.new_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(34)
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_button)
        self.go_button = _primary("Generate shorts")
        self.go_button.clicked.connect(self._generate)
        buttons.addWidget(self.go_button)
        right.addLayout(buttons)
        layout.addLayout(right, 1)

    def prepare(self, analysis: Analysis, clips: list[tuple[float, float, float, str, dict]]) -> None:
        self.analysis, self.clips, self.written = analysis, clips, []
        total = sum(c[1] - c[0] for c in clips)
        self.summary.setText(f"{len(clips)} clip{'s' if len(clips) != 1 else ''} from <b>{analysis.path.name}</b> · "
                             f"{total:.0f}s in all · 1080×1920 MP4")
        self.clip_list.clear()
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        for number, (start, end, _peak, label, ai) in enumerate(clips, start=1):
            name = pipeline.moment_name(analysis.path, start)
            what = ai.get("title") or label
            self.clip_list.addItem(f"{number}.  {ICONS.get(label, '')} {what[:48]}   {clock(start)} → {clock(end)}  "
                                   f"({end - start:.0f}s)   →  {name}")
            self.preview_combo.addItem(f"clip {number} · {clock(start)}", number - 1)
        self.preview_combo.blockSignals(False)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.status.setText("Check the framing on the left, then press Generate.")
        self.go_button.setText(f"Generate {len(clips)} short{'s' if len(clips) != 1 else ''}")
        self._set_mode("ready")
        self._show_framing()
        self._schedule_preview()

    def _set_mode(self, mode: str) -> None:
        """ready -> rendering -> done. Each mode shows only the buttons that make sense in it."""
        self.go_button.setVisible(mode == "ready")
        self.cancel_button.setVisible(mode == "rendering")
        self.cancel_button.setEnabled(mode == "rendering")
        self.open_button.setVisible(mode == "done")
        self.new_button.setVisible(mode == "done")
        self.back_button.setEnabled(mode != "rendering")
        self.adjust_button.setEnabled(mode == "ready")
        self.dest_edit.setEnabled(mode == "ready")

    def _show_framing(self) -> None:
        self.framing_label.setText(
            f"zoom {self.prefs.get('framing/zoom'):g} · position {self.prefs.get('framing/offset_y')} px · "
            f"background {self.prefs.get('framing/opacity'):.0%} · blur {self.prefs.get('framing/blur'):g}")

    # -- preview

    def _schedule_preview(self) -> None:
        if self.analysis is not None:
            self._preview_timer.start()

    def _render_preview(self) -> None:
        if self.analysis is None or self.preview_combo.currentIndex() < 0:
            return
        if self.preview_runner.busy():
            self._preview_again = True
            return
        source = self.analysis.path
        peak = self.clips[self.preview_combo.currentData()][2]
        dest = self.analysis.cache / "preview.png"
        try:
            settings = Framing.from_prefs(self.prefs)
        except Exception as error:  # noqa: BLE001
            self.preview.setText(f"Framing not valid:\n{error}")
            return
        cancel = video_io.Canceller()
        self.preview.setText("Rendering preview…")

        def job(_log, _progress):
            return pipeline.preview_clip(source, peak, dest, settings=settings, cancel=cancel)

        self.preview_runner.start(job, cancel, self._preview_ready, self._preview_failed)

    def _preview_ready(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        self.preview.setPixmap(pixmap.scaled(PREVIEW, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
        self._again()

    def _preview_failed(self, error: str) -> None:
        self.preview.setText(f"No preview:\n{error.splitlines()[-1] if error else ''}")
        self._again()

    def _again(self) -> None:
        if self._preview_again:
            self._preview_again = False
            self._schedule_preview()

    def _adjust(self) -> None:
        dialog = FramingDialog(self, self.prefs)
        dialog.changed.connect(self._show_framing)
        dialog.changed.connect(self._schedule_preview)
        dialog.exec()
        self._show_framing()
        self._schedule_preview()

    # -- render

    def _browse_dest(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Save shorts to", self.dest_edit.text().strip())
        if chosen:
            self.dest_edit.setText(str(Path(chosen)))
            self.prefs.set("app/dest", str(Path(chosen)))

    def _dest(self) -> Path:
        return Path(self.dest_edit.text().strip() or str(default_output()))

    def _generate(self) -> None:
        if self.analysis is None:
            return
        source, clips, dest = self.analysis.path, [(c[0], c[1]) for c in self.clips], self._dest()
        self.prefs.set("app/dest", str(dest))
        try:
            settings = Framing.from_prefs(self.prefs)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"The framing is not valid:\n{error}")
            return
        cancel = video_io.Canceller()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self._set_mode("rendering")

        def job(log, progress):
            return pipeline.clip_shorts(source, clips, dest, settings=settings, log=log, progress=progress,
                                        cancel=cancel)

        self.render_runner.start(job, cancel, self._rendered, self._render_failed, self._render_cancelled)

    def _message(self, line: str) -> None:
        if line.strip():
            self.status.setText(line.strip().splitlines()[-1])

    def _progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)

    def _rendered(self, written: list[Path]) -> None:
        self.written = written
        self._write_titles(written)
        self.bar.setValue(self.bar.maximum())
        self.status.setText(f"✅ {len(written)} short{'s' if len(written) != 1 else ''} saved to {self._dest()}")
        for row, path in enumerate(written):
            item = self.clip_list.item(row)
            if item is not None:
                item.setText("✅  " + item.text())
        self._set_mode("done")

    def _write_titles(self, written: list[Path]) -> None:
        """Titles and descriptions beside the shorts, ready to paste when posting."""
        if self.analysis is None or not written:
            return
        lines = []
        for path, clip in zip(written, self.clips):
            ai = clip[4] if len(clip) > 4 else {}
            lines.append(path.name)
            if ai.get("title"):
                lines.append(f"  Title: {ai['title']}")
            if ai.get("description"):
                lines.append(f"  Description: {ai['description']}")
            lines.append("")
        target = written[0].parent / f"{self.analysis.path.stem}_titles.txt"
        try:
            target.write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            pass

    def _render_failed(self, error: str) -> None:
        QMessageBox.critical(self, APP_NAME, f"The shorts could not be made:\n{error}")
        self.status.setText("Failed — see the message. Nothing else was changed.")
        self._set_mode("ready")

    def _render_cancelled(self) -> None:
        self.status.setText("🛑 Cancelled. Shorts finished before the cancel are kept.")
        self.bar.setValue(0)
        self._set_mode("ready")

    def _cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.status.setText("Stopping…")
        self.render_runner.stop()

    def _open_dest(self) -> None:
        dest = self._dest()
        if dest.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest)))

    def busy(self) -> bool:
        return self.render_runner.busy()

    def stop_for_close(self) -> None:
        self.render_runner.stop_for_close()
        self.preview_runner.stop_for_close()

