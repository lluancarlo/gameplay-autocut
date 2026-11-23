"""Evaluation window (developer tool): label recordings and score the Moments detector.

Not part of the app. Open it with `python tools/evaluate.py --gui`.

Two pages over one evaluation folder (see src/evaluation/dataset.py):
  Label    -- watch a recording, mark moments (start, end, rating 0-3, label),
              menu/non-gameplay stretches and what has been reviewed. Every
              change is saved straight to the label file.
  Results  -- run the detector on a split, score it, compare with the frozen
              baseline, freeze a new baseline. Times in the report jump to the
              Label page.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPainter, QPen, QShortcut, QValidator
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from .. import pipeline, video_io
from ..gui import APP_NAME, Worker, default_source, settings_file
from ..moments.audio import track_count
from . import runner
from .dataset import EvalSet, default_folder, merge_spans, slug, subtract_span
from .metrics import LABELS, SPLITS

RATING_COLOURS = {0: "#8d8d8d", 1: "#4589ff", 2: "#f1c21b", 3: "#da1e28"}
RATING_NAMES = {0: "0 · not a highlight", 1: "1 · mildly interesting", 2: "2 · good", 3: "3 · must clip"}
MENU = QColor(120, 120, 120, 110)
REVIEWED = QColor("#24a148")
SUGGESTION = QColor("#8a3ffc")
PLAYHEAD = QColor("#e8b100")


def clock(seconds: float, tenths: bool = False) -> str:
    seconds = max(0.0, seconds)
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes):02d}:{rest:04.1f}" if tenths else f"{int(minutes):02d}:{int(rest):02d}"


def parse_clock(text: str) -> float | None:
    text = text.strip()
    match = re.fullmatch(r"(?:(\d+):)?(\d+(?:[.,]\d*)?)", text)
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    return minutes * 60 + float(match.group(2).replace(",", "."))


def _elastic(label: QLabel) -> None:
    """Long paths in a label must not set the window's minimum width."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(0)


class TimeBox(QDoubleSpinBox):
    """Seconds, shown and typed as mm:ss.s (plain seconds are accepted too)."""

    def __init__(self) -> None:
        super().__init__()
        self.setDecimals(1)
        self.setRange(0.0, 24 * 3600.0)
        self.setSingleStep(0.5)
        self.setKeyboardTracking(False)

    def textFromValue(self, value: float) -> str:
        return clock(value, tenths=True)

    def valueFromText(self, text: str) -> float:
        return parse_clock(text) or 0.0

    def validate(self, text: str, pos: int):
        state = QValidator.State.Acceptable if parse_clock(text) is not None else QValidator.State.Intermediate
        return state, text, pos


class Timeline(QWidget):
    """The whole recording: labelled moments by rating, menus, reviewed time, suggestions, playhead."""

    seek = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(78)
        self.setMouseTracking(True)
        self.duration = 1.0
        self.moments: list[dict] = []
        self.menus: list = []
        self.reviewed: list = []
        self.suggestions: list[tuple[float, float]] = []
        self.selected: dict | None = None
        self.marks: tuple[float, float] | None = None
        self.playhead = 0.0
        self._dragging = False

    def show_labels(self, duration: float, labels: dict, selected: dict | None,
                    suggestions: list[tuple[float, float]], marks: tuple[float, float] | None) -> None:
        self.duration = max(1.0, duration)
        self.moments = labels.get("moments", [])
        self.menus = labels.get("non_gameplay", [])
        self.reviewed = labels.get("reviewed_spans", [])
        self.selected, self.suggestions, self.marks = selected, suggestions, marks
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        self.update()

    def _x(self, seconds: float) -> float:
        return 6 + seconds / self.duration * (self.width() - 12)

    def _seconds(self, x: float) -> float:
        return min(max((x - 6) / max(1, self.width() - 12) * self.duration, 0.0), self.duration)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        w = self.width()
        p.fillRect(self.rect(), self.palette().base())
        lane_s, lane_m, lane_r = (2, 12), (16, 34), (52, 6)  # (top, height) of suggestions, moments, reviewed

        def bar(start, end, top, height, colour):
            x0, x1 = self._x(start), self._x(end)
            p.fillRect(int(x0), top, max(2, int(x1 - x0)), height, colour)

        p.fillRect(6, lane_m[0], w - 12, lane_m[1], self.palette().alternateBase())
        for start, end in self.menus:
            bar(start, end, lane_m[0], lane_m[1], MENU)
        for start, end in self.suggestions:
            bar(start, end, *lane_s, SUGGESTION)
        if self.marks:
            faded = QColor(PLAYHEAD)
            faded.setAlpha(70)
            bar(min(self.marks), max(self.marks), lane_m[0], lane_m[1], faded)
        for moment in self.moments:
            colour = QColor(RATING_COLOURS[moment["rating"]])
            bar(moment["start"], moment["end"], lane_m[0] + 4, lane_m[1] - 8, colour)
            if moment is self.selected:
                p.setPen(QPen(self.palette().text().color(), 2))
                x0, x1 = self._x(moment["start"]), self._x(moment["end"])
                p.drawRect(int(x0) - 1, lane_m[0] + 2, max(3, int(x1 - x0)) + 2, lane_m[1] - 4)
        p.fillRect(6, lane_r[0], w - 12, lane_r[1], self.palette().mid())
        for start, end in self.reviewed:
            bar(start, end, *lane_r, REVIEWED)

        p.setPen(self.palette().text().color())
        step = next(s for s in (30, 60, 120, 300, 600, 900, 1800, 3600) if self.duration / s <= 12)
        t = 0
        while t <= self.duration:
            x = int(self._x(t))
            p.drawLine(x, 60, x, 64)
            p.drawText(x + 2, 75, clock(t))
            t += step
        p.setPen(QPen(PLAYHEAD, 2))
        x = int(self._x(self.playhead))
        p.drawLine(x, 0, x, 58)

    def mousePressEvent(self, event) -> None:
        self._dragging = True
        self.seek.emit(self._seconds(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        seconds = self._seconds(event.position().x())
        self.setToolTip(clock(seconds))
        if self._dragging:
            self.seek.emit(seconds)

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = False


class AddRecordingDialog(QDialog):
    def __init__(self, parent, games: list[str], start_dir: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add recording")
        self.start_dir = start_dir
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.video = QLineEdit()
        self.video.editingFinished.connect(self._video_changed)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(self.video, 1)
        row.addWidget(browse)
        form.addRow("Video", row)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.game = QComboBox()
        self.game.setEditable(True)
        self.game.addItems(games or ["the-finals", "wardogs", "gta6"])
        form.addRow("Game", self.game)
        self.split = QComboBox()
        for split, text in (("tune", "tune — used while developing"), ("holdout", "holdout — final test, never tuned on"),
                            ("train", "train — examples for learning")):
            self.split.addItem(text, split)
        form.addRow("Split", self.split)
        hint = QLabel("The split cannot be changed later. Use raw, uncut sessions; one video goes in one split only.")
        hint.setWordWrap(True)
        form.addRow("", hint)
        self.voice = QComboBox()
        form.addRow("Voice track", self.voice)
        self._tracks(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.resize(560, self.sizeHint().height())

    def _browse(self) -> None:
        suffixes = " ".join(f"*{s}" for s in sorted(pipeline.VIDEO_SUFFIXES))
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a recording", self.start_dir, f"Videos ({suffixes})")
        if chosen:
            self.video.setText(str(Path(chosen)))
            self._video_changed()

    def _video_changed(self) -> None:
        path = Path(self.video.text().strip())
        if path.is_file():
            if not self.name.text().strip():
                self.name.setText(slug(path.stem))
            self._tracks(track_count(path))

    def _tracks(self, count: int) -> None:
        self.voice.clear()
        self.voice.addItem("Same as game (one mixed track)", None)
        for track in range(1, count):
            self.voice.addItem(f"Track {track + 1}", track)
        self.voice.setEnabled(count > 1)

    def _accept(self) -> None:
        if not Path(self.video.text().strip()).is_file():
            QMessageBox.warning(self, APP_NAME, "Choose a video file.")
            return
        if not self.name.text().strip() or not self.game.currentText().strip():
            QMessageBox.warning(self, APP_NAME, "Give the recording a name and a game.")
            return
        self.accept()


class EvaluationTab(QWidget):
    key = "evaluation"

    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings(str(settings_file()), QSettings.Format.IniFormat)
        self.es = EvalSet(default_folder())
        self.rec_id: str | None = None
        self.labels: dict | None = None
        self.selected: dict | None = None
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._cancel: video_io.Canceller | None = None
        self._on_done = None
        self._build_ui()
        self.restore()

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Evaluation folder"))
        self.folder_edit = QLineEdit()
        self.folder_edit.editingFinished.connect(lambda: self._use_folder(Path(self.folder_edit.text().strip())))
        top.addWidget(self.folder_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_folder)
        top.addWidget(browse)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(lambda: self._open(self.es.folder))
        top.addWidget(open_folder)
        layout.addLayout(top)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("Recording"))
        self.rec_combo = QComboBox()
        self.rec_combo.setMinimumWidth(220)
        self.rec_combo.currentIndexChanged.connect(lambda _i: self._load_recording(self.rec_combo.currentData()))
        rec_row.addWidget(self.rec_combo)
        self.rec_info = QLabel()
        rec_row.addWidget(self.rec_info, 1)
        add = QPushButton("Add recording...")
        add.clicked.connect(self._add_recording)
        rec_row.addWidget(add)
        layout.addLayout(rec_row)

        self.pages = QTabWidget()
        self.pages.addTab(self._build_label_page(), "Label")
        self.pages.addTab(self._build_results_page(), "Results")
        layout.addWidget(self.pages, 1)

        bottom = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        bottom.addWidget(self.progress_bar, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_job)
        bottom.addWidget(self.cancel_button)
        layout.addLayout(bottom)
        self.status = QLabel("")
        layout.addWidget(self.status)
        for label in (self.status, self.rec_info):
            _elastic(label)

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        self.label_page = page
        split = QSplitter(Qt.Orientation.Horizontal, page)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        self.video.setMinimumSize(480, 270)
        lv.addWidget(self.video, 1)
        self.timeline = Timeline()
        self.timeline.seek.connect(self._seek)
        lv.addWidget(self.timeline)
        legend = QLabel(
            "Moments by rating: <span style='color:#8d8d8d'>■</span> 0 "
            "<span style='color:#4589ff'>■</span> 1 <span style='color:#f1c21b'>■</span> 2 "
            "<span style='color:#da1e28'>■</span> 3 · <span style='color:#787878'>■</span> menu / non-gameplay · "
            "<span style='color:#24a148'>■</span> reviewed · <span style='color:#8a3ffc'>■</span> detector suggestion")
        legend.setWordWrap(True)
        lv.addWidget(legend)

        transport = QHBoxLayout()
        for text, delta in (("◀◀ 5s (J)", -5.0), ("◀ 1s", -1.0)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, d=delta: self._step(d))
            transport.addWidget(b)
        self.play_button = QPushButton("Play (Space)")
        self.play_button.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_button)
        for text, delta in (("1s ▶", 1.0), ("5s ▶▶ (L)", 5.0)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, d=delta: self._step(d))
            transport.addWidget(b)
        transport.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for rate in (1.0, 1.5, 2.0):
            self.speed.addItem(f"{rate:g}×", rate)
        self.speed.currentIndexChanged.connect(lambda _i: self.player.setPlaybackRate(self.speed.currentData()))
        transport.addWidget(self.speed)
        self.time_label = QLabel("00:00.0 / 00:00")
        self.time_label.setMinimumWidth(130)
        transport.addWidget(self.time_label)
        transport.addStretch(1)
        self.suggest_box = QCheckBox("Show detector suggestions")
        self.suggest_box.setToolTip("Off by default: label what happened, not what the detector found.")
        self.suggest_box.toggled.connect(lambda _on: self._refresh())
        transport.addWidget(self.suggest_box)
        lv.addLayout(transport)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        edit = QGroupBox("Moment")
        grid = QGridLayout(edit)
        self.start_box, self.end_box = TimeBox(), TimeBox()
        self.start_box.valueChanged.connect(lambda _v: self._refresh_timeline())
        self.end_box.valueChanged.connect(lambda _v: self._refresh_timeline())
        grid.addWidget(QLabel("Start"), 0, 0)
        grid.addWidget(self.start_box, 0, 1)
        set_in = QPushButton("= playhead (I)")
        set_in.clicked.connect(lambda: self._mark("start"))
        grid.addWidget(set_in, 0, 2)
        grid.addWidget(QLabel("End"), 1, 0)
        grid.addWidget(self.end_box, 1, 1)
        set_out = QPushButton("= playhead (O)")
        set_out.clicked.connect(lambda: self._mark("end"))
        grid.addWidget(set_out, 1, 2)
        grid.addWidget(QLabel("Rating"), 2, 0)
        self.rating = QComboBox()
        for value, text in RATING_NAMES.items():
            self.rating.addItem(text, value)
        self.rating.setCurrentIndex(2)
        grid.addWidget(self.rating, 2, 1, 1, 2)
        grid.addWidget(QLabel("Label"), 3, 0)
        self.label_combo = QComboBox()
        self.label_combo.addItem("(none)", None)
        for label in LABELS:
            self.label_combo.addItem(label, label)
        grid.addWidget(self.label_combo, 3, 1, 1, 2)
        grid.addWidget(QLabel("Detail"), 4, 0)
        self.label2 = QLineEdit()
        self.label2.setPlaceholderText("optional: chase, escape, victory, stunt, funny...")
        grid.addWidget(self.label2, 4, 1, 1, 2)
        grid.addWidget(QLabel("Note"), 5, 0)
        self.note = QLineEdit()
        grid.addWidget(self.note, 5, 1, 1, 2)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add moment (Ctrl+Enter)")
        self.add_button.clicked.connect(self._add_moment)
        self.update_button = QPushButton("Update selected")
        self.update_button.clicked.connect(self._update_moment)
        self.delete_button = QPushButton("Delete selected")
        self.delete_button.clicked.connect(self._delete_moment)
        for b in (self.add_button, self.update_button, self.delete_button):
            buttons.addWidget(b)
        grid.addLayout(buttons, 6, 0, 1, 3)
        rv.addWidget(edit)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Rating", "Label", "Detail", "Note"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._table_selected)
        rv.addWidget(self.table, 1)

        spans = QGroupBox("Menus and review")
        sg = QGridLayout(spans)
        menu_add = QPushButton("Start–End is menu / non-gameplay")
        menu_add.clicked.connect(lambda: self._edit_spans("non_gameplay", add=True))
        menu_del = QPushButton("Remove menu span at playhead")
        menu_del.clicked.connect(self._remove_menu_at_playhead)
        rev_add = QPushButton("Start–End reviewed")
        rev_add.clicked.connect(lambda: self._edit_spans("reviewed_spans", add=True))
        rev_all = QPushButton("Whole video reviewed")
        rev_all.clicked.connect(self._review_all)
        rev_del = QPushButton("Start–End not reviewed")
        rev_del.clicked.connect(lambda: self._edit_spans("reviewed_spans", add=False))
        sg.addWidget(menu_add, 0, 0)
        sg.addWidget(menu_del, 0, 1)
        sg.addWidget(rev_add, 1, 0)
        sg.addWidget(rev_del, 1, 1)
        sg.addWidget(rev_all, 2, 0)
        self.coverage = QLabel()
        sg.addWidget(self.coverage, 2, 1)
        sg.addWidget(QLabel("Annotator"), 3, 0)
        self.annotator = QLineEdit()
        self.annotator.editingFinished.connect(self._annotator_changed)
        sg.addWidget(self.annotator, 3, 1)
        sg.addWidget(QLabel("Labels status"), 4, 0)
        self.status_combo = QComboBox()
        self.status_combo.addItem("draft — not counted in decisions", "draft")
        self.status_combo.addItem("reviewed — watched end to end with sound", "reviewed")
        self.status_combo.activated.connect(self._status_changed)
        sg.addWidget(self.status_combo, 4, 1)
        rv.addWidget(spans)
        self.saved_label = QLabel()
        _elastic(self.saved_label)
        self.coverage.setWordWrap(True)
        rv.addWidget(self.saved_label)
        split.addWidget(right)
        split.setSizes([640, 460])

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._position_changed)
        self.player.playbackStateChanged.connect(self._play_state_changed)

        for keys, action in (("Space", self._toggle_play), ("I", lambda: self._mark("start")),
                             ("O", lambda: self._mark("end")), ("J", lambda: self._step(-5)),
                             ("L", lambda: self._step(5)), ("Ctrl+Return", self._add_moment),
                             ("Ctrl+Enter", self._add_moment)):
            shortcut = QShortcut(QKeySequence(keys), page)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Split"))
        self.split_combo = QComboBox()
        for split in (*SPLITS, "all"):
            self.split_combo.addItem(split, split)
        self.split_combo.setCurrentIndex(1)
        row.addWidget(self.split_combo)
        self.drafts_box = QCheckBox("Include draft labels")
        self.drafts_box.setToolTip("Draft labels give indicative numbers only; decisions need reviewed labels.")
        row.addWidget(self.drafts_box)
        self.compare_box = QCheckBox("Compare with baseline")
        self.compare_box.setChecked(True)
        row.addWidget(self.compare_box)
        self.fresh_box = QCheckBox("Re-analyse (ignore cache)")
        row.addWidget(self.fresh_box)
        row.addStretch(1)
        v.addLayout(row)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run evaluation")
        self.run_button.setMinimumHeight(32)
        self.run_button.clicked.connect(lambda: self._evaluate(run_new=True))
        actions.addWidget(self.run_button)
        self.rescore_button = QPushButton("Score saved baseline")
        self.rescore_button.setToolTip("Score the frozen baseline predictions against the current labels. Runs no model.")
        self.rescore_button.clicked.connect(lambda: self._evaluate(run_new=False))
        actions.addWidget(self.rescore_button)
        self.freeze_button = QPushButton("Freeze last run as baseline...")
        self.freeze_button.clicked.connect(self._freeze)
        actions.addWidget(self.freeze_button)
        open_report = QPushButton("Open report file")
        open_report.clicked.connect(lambda: self._open(self.es.folder / "runs" / "latest-report.md"))
        actions.addWidget(open_report)
        actions.addStretch(1)
        v.addLayout(actions)

        self.report_view = QTextBrowser()
        self.report_view.setOpenLinks(False)
        self.report_view.anchorClicked.connect(self._report_link)
        self.report_view.setPlaceholderText("Run an evaluation to see the report. Times in it open the recording on the Label page.")
        v.addWidget(self.report_view, 1)
        return page

    # ------------------------------------------------------------------ folder and recordings

    def _browse_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Evaluation folder", str(self.es.folder))
        if chosen:
            self._use_folder(Path(chosen))

    def _use_folder(self, folder: Path, select: str | None = None) -> None:
        self.es = EvalSet(folder)
        self.folder_edit.setText(str(folder))
        self._fill_recordings(select)

    def _fill_recordings(self, select: str | None = None) -> None:
        try:
            entries = self.es.entries()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, APP_NAME, f"Could not read {self.es.splits_file}:\n{error}")
            entries = {}
        self.rec_combo.blockSignals(True)
        self.rec_combo.clear()
        for rec_id, entry in entries.items():
            self.rec_combo.addItem(f"{rec_id}  ({entry['game']}, {entry['split']})", rec_id)
        index = self.rec_combo.findData(select) if select else 0
        self.rec_combo.setCurrentIndex(max(0, index))
        self.rec_combo.blockSignals(False)
        if not entries:
            self.rec_info.setText("No recordings yet — use Add recording..." if self.es.folder.is_dir()
                                  else "This folder does not exist yet; Add recording... creates it.")
            self._load_recording(None)
        else:
            self._load_recording(self.rec_combo.currentData())

    def _add_recording(self) -> None:
        dialog = AddRecordingDialog(self, self.es.games() if self.es.splits_file.is_file() else [],
                                    self._video_dir or str(default_source()))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        video = Path(dialog.video.text().strip())
        try:
            duration = video_io.probe(video).duration
            rec_id = self.es.add_recording(video, dialog.name.text().strip(), dialog.game.currentText().strip(),
                                           dialog.split.currentData(), duration, dialog.voice.currentData())
        except (ValueError, OSError, RuntimeError) as error:
            QMessageBox.warning(self, APP_NAME, str(error))
            return
        self._video_dir = str(video.parent)
        self._fill_recordings(rec_id)
        self.pages.setCurrentIndex(0)

    def _load_recording(self, rec_id: str | None) -> None:
        self.player.stop()
        self.rec_id, self.labels, self.selected = rec_id, None, None
        if rec_id is None:
            self.player.setSource(QUrl())
            self._refresh()
            return
        entry = self.es.entries()[rec_id]
        try:
            self.labels = self.es.read_labels(rec_id)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, APP_NAME, f"Could not read the labels of {rec_id}:\n{error}")
            self._refresh()
            return
        video = self.es.video(entry)
        voice = "mixed audio" if entry.get("voice_track") is None else f"voice on track {entry['voice_track'] + 1}"
        missing = "" if video.is_file() else " · ⚠ video not found"
        self.rec_info.setText(f"{entry['game']} · split {entry['split']} · {voice} · {video.name}{missing}")
        self.player.setSource(QUrl.fromLocalFile(str(video)) if video.is_file() else QUrl())
        self.annotator.setText(self.labels.get("annotator", ""))
        self.status_combo.setCurrentIndex(self.status_combo.findData(self.labels.get("status", "draft")))
        self.start_box.setMaximum(self.labels["duration"])
        self.end_box.setMaximum(self.labels["duration"])
        self.saved_label.setText(f"Labels: {self.es.labels_file(rec_id)}")
        self._refresh()

    # ------------------------------------------------------------------ player

    def _seek(self, seconds: float) -> None:
        self.player.setPosition(int(seconds * 1000))
        self.timeline.set_playhead(seconds)

    def _step(self, delta: float) -> None:
        self._seek(max(0.0, self.player.position() / 1000 + delta))

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _play_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("Pause (Space)" if playing else "Play (Space)")

    def _position_changed(self, ms: int) -> None:
        seconds = ms / 1000
        self.timeline.set_playhead(seconds)
        total = self.labels["duration"] if self.labels else 0
        self.time_label.setText(f"{clock(seconds, True)} / {clock(total)}")

    def _now(self) -> float:
        return self.player.position() / 1000

    def _mark(self, edge: str) -> None:
        (self.start_box if edge == "start" else self.end_box).setValue(self._now())

    # ------------------------------------------------------------------ labels

    def _form(self) -> dict | None:
        start, end = self.start_box.value(), self.end_box.value()
        if end <= start:
            QMessageBox.warning(self, APP_NAME, "End must be after Start. Use I and O at the playhead to set them.")
            return None
        moment = {"start": round(start, 1), "end": round(end, 1), "rating": self.rating.currentData()}
        if self.label_combo.currentData():
            moment["label"] = self.label_combo.currentData()
        if self.label2.text().strip():
            moment["label2"] = self.label2.text().strip()
        if self.note.text().strip():
            moment["note"] = self.note.text().strip()
        return moment

    def _add_moment(self) -> None:
        if self.labels is None:
            return
        moment = self._form()
        if moment is None:
            return
        self.labels["moments"].append(moment)
        self.selected = moment
        self._save()

    def _update_moment(self) -> None:
        if self.labels is None or self.selected is None:
            QMessageBox.information(self, APP_NAME, "Select a moment in the list first.")
            return
        moment = self._form()
        if moment is None:
            return
        self.selected.clear()
        self.selected.update(moment)
        self._save()

    def _delete_moment(self) -> None:
        if self.labels is None or self.selected is None:
            return
        self.labels["moments"] = [m for m in self.labels["moments"] if m is not self.selected]
        self.selected = None
        self._save()

    def _table_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.labels else []
        if not rows:
            return
        moment = self.labels["moments"][rows[0].row()]
        if moment is self.selected:
            return
        self.selected = moment
        self.start_box.setValue(moment["start"])
        self.end_box.setValue(moment["end"])
        self.rating.setCurrentIndex(self.rating.findData(moment["rating"]))
        self.label_combo.setCurrentIndex(max(0, self.label_combo.findData(moment.get("label"))))
        self.label2.setText(moment.get("label2", ""))
        self.note.setText(moment.get("note", ""))
        self._seek(moment["start"])
        self._refresh_timeline()

    def _edit_spans(self, name: str, add: bool) -> None:
        if self.labels is None:
            return
        start, end = self.start_box.value(), self.end_box.value()
        if end <= start:
            QMessageBox.warning(self, APP_NAME, "Set Start and End first (I and O at the playhead).")
            return
        spans = self.labels.get(name, [])
        self.labels[name] = merge_spans(spans + [[start, end]]) if add else subtract_span(spans, start, end)
        self._save()

    def _remove_menu_at_playhead(self) -> None:
        if self.labels is None:
            return
        now = self._now()
        self.labels["non_gameplay"] = [s for s in self.labels.get("non_gameplay", []) if not s[0] <= now <= s[1]]
        self._save()

    def _review_all(self) -> None:
        if self.labels is not None:
            self.labels["reviewed_spans"] = [[0.0, self.labels["duration"]]]
            self._save()

    def _annotator_changed(self) -> None:
        if self.labels is not None and self.labels.get("annotator", "") != self.annotator.text().strip():
            self.labels["annotator"] = self.annotator.text().strip()
            self._save()

    def _status_changed(self, _index: int) -> None:
        if self.labels is None:
            return
        status = self.status_combo.currentData()
        if status == "reviewed":
            reviewed = sum(e - s for s, e in self.labels["reviewed_spans"])
            if reviewed < 0.99 * self.labels["duration"]:
                answer = QMessageBox.question(
                    self, APP_NAME,
                    f"Only {clock(reviewed)} of {clock(self.labels['duration'])} is marked reviewed. "
                    "Metrics count only reviewed time.\n\nMark these labels as reviewed anyway?")
                if answer != QMessageBox.StandardButton.Yes:
                    self.status_combo.setCurrentIndex(self.status_combo.findData(self.labels.get("status", "draft")))
                    return
        self.labels["status"] = status
        self._save()

    def _save(self) -> None:
        try:
            self.es.write_labels(self.rec_id, self.labels)
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, APP_NAME, f"Not saved:\n{error}")
            self.labels = self.es.read_labels(self.rec_id)
            self.selected = None
        else:
            self.saved_label.setText(f"Saved {dt.datetime.now():%H:%M:%S} · {self.es.labels_file(self.rec_id)}")
        self._refresh()

    # ------------------------------------------------------------------ drawing

    def _suggestions(self) -> list[tuple[float, float]]:
        if not (self.suggest_box.isChecked() and self.rec_id):
            return []
        for folder in (self.es.latest_dir, self.es.baseline_dir):
            preds = runner.load_preds(folder, self.rec_id)
            if preds is not None:
                return [(p.start, p.end) for p in preds]
        self.status.setText("No detector suggestions yet for this recording: run an evaluation first.")
        return []

    def _refresh_timeline(self) -> None:
        if self.labels is None:
            self.timeline.show_labels(1.0, {}, None, [], None)
            return
        start, end = self.start_box.value(), self.end_box.value()
        marks = (start, end) if end > start else None
        self.timeline.show_labels(self.labels["duration"], self.labels, self.selected, self._suggestions(), marks)

    def _refresh(self) -> None:
        has = self.labels is not None
        for widget in (self.add_button, self.update_button, self.delete_button, self.annotator, self.status_combo):
            widget.setEnabled(has)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if has:
            for row, moment in enumerate(self.labels["moments"]):
                self.table.insertRow(row)
                cells = (clock(moment["start"], True), clock(moment["end"], True), str(moment["rating"]),
                         moment.get("label") or "", moment.get("label2", ""), moment.get("note", ""))
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col == 2:
                        item.setBackground(QColor(RATING_COLOURS[moment["rating"]]))
                    self.table.setItem(row, col, item)
                if moment is self.selected:
                    self.table.selectRow(row)
            reviewed = sum(e - s for s, e in self.labels["reviewed_spans"])
            self.coverage.setText(f"Reviewed {clock(reviewed)} of {clock(self.labels['duration'])} · "
                                  f"{len(self.labels['non_gameplay'])} menu spans · "
                                  f"{len(self.labels['moments'])} moments")
        else:
            self.coverage.setText("")
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.blockSignals(False)
        self._refresh_timeline()

    # ------------------------------------------------------------------ results

    def _evaluate(self, run_new: bool) -> None:
        split = self.split_combo.currentData()
        if split in ("holdout", "all") and run_new:
            answer = QMessageBox.question(
                self, APP_NAME,
                "The holdout split is for milestone gates only — looking at it while tuning makes it useless as a "
                "final test.\n\nScore it anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        es, drafts, compare, fresh = self.es, self.drafts_box.isChecked(), self.compare_box.isChecked(), \
            self.fresh_box.isChecked()
        cancel = video_io.Canceller()
        title = "detector run" if run_new else "saved baseline"

        def job(log, progress):
            return runner.evaluate(es, split, drafts, run_new=run_new, against_baseline=compare, log=log,
                                   progress=progress, cancel=cancel, fresh=fresh, title=title)[0]

        self.player.pause()
        self._run(job, cancel, self._evaluated)

    def _evaluated(self, text: str) -> None:
        self.report_view.setMarkdown(text)
        self.status.setText(f"Report saved to {self.es.folder / 'runs' / 'latest-report.md'}")
        self._refresh_timeline()

    def _freeze(self) -> None:
        files = sorted(p.stem for p in self.es.latest_dir.glob("*.json"))
        if not files:
            QMessageBox.information(self, APP_NAME, "Run an evaluation first: there is nothing to freeze.")
            return
        answer = QMessageBox.question(
            self, APP_NAME,
            "Make the last run the new baseline for:\n  " + "\n  ".join(files) +
            "\n\nThe current baseline for these recordings is replaced. Every later change is compared against it.")
        if answer == QMessageBox.StandardButton.Yes:
            frozen = runner.freeze_baseline(self.es)
            self.status.setText(f"Baseline frozen for {', '.join(frozen)}")

    def _report_link(self, url: QUrl) -> None:
        text = url.toString()
        match = re.match(r"vvideo://seek/([^/]+)/([\d.]+)", text)
        if not match:
            QDesktopServices.openUrl(url)
            return
        rec_id, seconds = match.group(1), float(match.group(2))
        if rec_id != self.rec_id:
            index = self.rec_combo.findData(rec_id)
            if index < 0:
                return
            self.rec_combo.setCurrentIndex(index)
        self.pages.setCurrentIndex(0)
        self._seek(seconds)
        self.player.play()

    # ------------------------------------------------------------------ jobs

    def _run(self, job, cancel: video_io.Canceller, on_done) -> None:
        if self._thread is not None:
            return
        self._cancel, self._on_done = cancel, on_done
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(lambda line: line.strip() and self.status.setText(line.strip().splitlines()[-1]))
        self._worker.progress.connect(self._advance)
        self._worker.finished.connect(self._finished)
        self._worker.cancelled.connect(self._retire)
        self._thread.start()

    def _advance(self, done: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)

    def _finished(self, ok: bool, payload: object) -> None:
        self._retire()
        self.progress_bar.setValue(0)
        if ok:
            self._on_done(payload)
        else:
            QMessageBox.critical(self, APP_NAME, str(payload))

    def _retire(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        self._thread = self._worker = self._cancel = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.run_button, self.rescore_button, self.freeze_button):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _cancel_job(self) -> None:
        if self._cancel is not None:
            self.cancel_button.setEnabled(False)
            self._cancel.cancel()

    def _open(self, target: Path) -> None:
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QMessageBox.information(self, APP_NAME, f"Nothing there yet:\n{target}")

    # ------------------------------------------------------------------ window hooks

    def closing(self) -> bool:
        return self._thread is None

    def stop_for_close(self) -> None:
        self.player.stop()
        if self._cancel is not None:
            self._cancel.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)

    def save(self) -> None:
        self.settings.remove(self.key)
        if self.es.folder != default_folder():
            self.settings.setValue(f"{self.key}/folder", str(self.es.folder))
        if self.rec_id:
            self.settings.setValue(f"{self.key}/recording", self.rec_id)
        if self._video_dir:
            self.settings.setValue(f"{self.key}/video_dir", self._video_dir)
        self.settings.sync()
        if not self.settings.allKeys():
            settings_file().unlink(missing_ok=True)

    def restore(self) -> None:
        self._video_dir = self.settings.value(f"{self.key}/video_dir") or ""
        folder = self.settings.value(f"{self.key}/folder")
        self._use_folder(Path(folder) if folder else default_folder(),
                         select=self.settings.value(f"{self.key}/recording"))


def run() -> int:
    """Open the evaluation window on its own."""
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = QMainWindow()
    window.setWindowTitle(f"{APP_NAME} — evaluation (developer tool)")
    tab = EvaluationTab()
    window.setCentralWidget(tab)
    window.resize(1400, 860)

    def close(event) -> None:
        tab.stop_for_close()
        tab.save()
        event.accept()

    window.closeEvent = close
    window.show()
    return app.exec()
