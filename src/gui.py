"""VVideo's window: one page over the shorts pipeline.

Pick a folder of clips, pick an output folder, set the framing, press the
button. The work runs on a QThread; touching widgets from it would crash Qt, so
everything it has to say comes back as signals (`message`, `progress`,
`finished`) that Qt delivers on the UI thread.

Every control the CLI exposes is here, because the window is the only front end
for anyone running the .exe -- there is no console behind it to fall back to.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QObject, QRectF, QSettings, QStandardPaths, QThread, QTimer, QUrl, Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import config, pipeline, shorts, sysmon, video_io

APP_NAME = "VVideo"

# Beside the project root when running from source, and unpacked next to
# config/ in the frozen build -- config._root() already knows which is which.
# This is the *window* icon: the .exe's own icon is set in VVideo.spec.
ICON_PATH = config.ROOT / "vvideo-icon.png"


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
    """Remove HKCU\\Software\\VVideo once nothing is left in it.

    QSettings empties the values but leaves the keys themselves standing, and
    an empty key is still a key -- exactly the trace this build is trying not
    to leave. Deepest first: a key with subkeys under it will not delete.
    """
    if sys.platform != "win32":
        return
    import winreg

    for path in (rf"Software\{APP_NAME}\{APP_NAME}", rf"Software\{APP_NAME}"):
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
    legacy = QSettings(APP_NAME, APP_NAME)
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
                if key.startswith(f"{ShortTab.key}/"):
                    target.setValue(key, legacy.value(key))
            target.sync()
        legacy.clear()
        legacy.sync()
    _drop_legacy_registry_key()


def _same(value: object, default: object) -> bool:
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
    """Where preferences live: VVideo.ini beside the executable.

    QSettings would otherwise default to HKCU\\Software\\VVideo on Windows --
    a registry key left behind on a machine that only ever ran a portable .exe.
    An ini file next to the .exe keeps the whole app in one place: delete it
    and nothing of VVideo remains. Written only when a value actually differs
    from its default, and removed again when none do, so the common case leaves
    no file at all. See JobTab.save.

    sys.executable, not config.ROOT: in the frozen build ROOT is the temporary
    unpack directory, which is gone the moment the process ends.
    """
    base = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else config.ROOT
    )
    return base / f"{APP_NAME}.ini"


# Option grids run two [label][control][reset] pairs per row, separated by an
# empty gutter column and followed by a stretching one that takes the leftover
# width so nothing else has to.
PAIR_STRIDE = 4
GUTTER_COLUMN = 3
SLACK_COLUMN = 7
OPTION_COLUMNS = 8
RESET_COLUMNS = (2, 6)

# The reset button's column keeps this width whether or not the button is
# showing, so a control appearing or disappearing never shifts the row.
RESET_WIDTH = 24

# How much of the log is on screen. Enough to read the last few lines without
# the box dominating the window -- the whole run is scrollable regardless.
LOG_LINES = 4

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


class ResetButton(QToolButton):
    """Puts one input back to its default, and only shows up when it is off it.

    The default is whatever the control was built with -- config/shorts.json
    for the framing numbers, the Videos folder for the two paths, unchecked for a
    checkbox. It is captured during construction, before `restore` puts the
    last session's values over the top, so "default" keeps meaning the shipped
    value rather than whatever happened to be used last time.

    It reads the default back off the control after setting it rather than
    trusting the number passed in: a spin box rounds to its own decimals, and
    a default it cannot represent exactly would leave the button showing
    permanently on a value the user never touched.
    """

    def __init__(self, default, read, write, changed, describe: str) -> None:
        super().__init__()
        self._default = default
        self._read = read
        self._write = write

        self.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.setIconSize(self.iconSize() * 0.7)
        self.setAutoRaise(True)
        self.setFixedWidth(RESET_WIDTH)
        # Not a tab stop: it comes and goes, and a focus order that changes as
        # you edit is worse than one button being mouse-only.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip(f"Back to the default ({describe})")

        self.clicked.connect(lambda: self._write(self._default))
        changed.connect(self.refresh)
        self.refresh()

    def refresh(self, *_ignored) -> None:
        self.setVisible(self._read() != self._default)


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


class ScaledSize(QWidget):
    """The zoom said in pixels -- the size the source is scaled to on canvas.

    Two boxes for one degree of freedom: the aspect is the source's, so
    whichever one is typed in settles the other. Neither holds state. Both
    write back a zoom and then wait to be told what that zoom worked out to,
    which is how a width that is not reachable -- an odd number, or one that
    falls between two slider steps -- comes back corrected instead of sticking.
    """

    zoomEdited = Signal(float)

    def __init__(self, canvas_h: int, source: tuple[int, int]) -> None:
        super().__init__()
        self._canvas_h = canvas_h
        self._source = source
        self._echo = False  # set while writing our own boxes, so they stay quiet

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.width_spin = self._box("Width of the scaled clip, in pixels.")
        self.height_spin = self._box(
            "Height of the scaled clip, in pixels. The canvas is "
            f"{canvas_h}px tall, so that is where zoom 1.0 lands."
        )
        row.addWidget(self.width_spin)
        row.addWidget(QLabel("×"))
        row.addWidget(self.height_spin)
        row.addStretch(1)

        self.width_spin.valueChanged.connect(self._from_width)
        self.height_spin.valueChanged.connect(self._from_height)

    def _box(self, tip: str) -> QSpinBox:
        spin = QSpinBox()
        # Wider than any canvas either dimension can reach at the slider's top
        # end, on any source aspect worth feeding it.
        spin.setRange(2, 99999)
        spin.setSingleStep(2)  # yuv420p wants even numbers; odd ones round anyway
        spin.setSuffix(" px")
        spin.setMinimumWidth(84)
        spin.setToolTip(tip)
        # Without this every keystroke of "1080" is its own value -- 1, then
        # 10, then 108 -- each one moving the zoom and rewriting the other box
        # out from under the cursor.
        spin.setKeyboardTracking(False)
        return spin

    def set_source(self, source: tuple[int, int]) -> None:
        self._source = source

    def show_zoom(self, zoom: float) -> None:
        """Display what `zoom` comes to against the current source."""
        source_w, source_h = self._source
        scale_w, scale_h = shorts.scaled_size(source_w, source_h, self._canvas_h, zoom)
        self._echo = True
        try:
            self.width_spin.setValue(scale_w)
            self.height_spin.setValue(scale_h)
        finally:
            self._echo = False

    def _from_width(self, value: int) -> None:
        if self._echo:
            return
        source_w, source_h = self._source
        self.zoomEdited.emit(
            shorts.zoom_for_width(value, source_w, source_h, self._canvas_h)
        )

    def _from_height(self, value: int) -> None:
        if self._echo:
            return
        self.zoomEdited.emit(shorts.zoom_for_height(value, self._canvas_h))


class PreviewDialog(QDialog):
    """A rendered still, scaled to fit on screen.

    Portrait 1080x1920 does not fit on most displays at full size, so it is
    shown scaled and the real file stays on disk for a proper look.
    """

    def __init__(self, image: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} preview")
        self._image = image

        layout = QVBoxLayout(self)
        canvas = QLabel()
        canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(image))
        if pixmap.isNull():
            canvas.setText(f"Could not display\n{image}")
        else:
            canvas.setPixmap(
                pixmap.scaled(
                    420,
                    760,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        layout.addWidget(canvas)

        caption = QLabel(str(image))
        caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        caption.setStyleSheet("color: palette(mid);")
        layout.addWidget(caption)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        open_button = buttons.addButton(
            "Open file", QDialogButtonBox.ButtonRole.ActionRole
        )
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._image)))
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class FramingPreview(QWidget):
    """The framing, drawn to scale: what the crop keeps and what comes out.

    Two diagrams, because the interesting thing is the relationship between
    them. On top, the source frame at its real aspect with the region that
    survives the centre-crop picked out; below, the 1080x1920 canvas with that
    region placed on it and the blur-filled bars above and below.

    It is geometry, not a render -- `shorts.plan` resolves exactly the same
    Layout the encode will use, so the boxes are true to the pixel even though
    nothing has been decoded. That is the point: it answers "how much am I
    throwing away" before a frame is read, which the real preview cannot do
    without paying ffmpeg for a frame.
    """

    # Sized so the portrait diagram is still readable. Below this the two
    # boxes stop saying anything useful -- and the height is what the window's
    # own floor ends up being set from, since this is the tallest thing on the
    # page: 1080x1920 drawn small enough to fit a short panel is a sliver.
    MIN_WIDTH = 260
    MIN_HEIGHT = 520

    # The source box's ceiling, as a fraction of the panel. Deliberately mean:
    # a landscape frame says what it needs to in a short strip, and every pixel
    # it does not take goes to the portrait diagram, which needs them.
    SOURCE_SHARE = 0.22

    BACKDROP = QColor("#000000")
    FRAME_FILL = QColor("#141414")  # the source, before anything is kept
    FRAME_EDGE = QColor("#4a4a4a")
    KEPT_FILL = QColor(36, 161, 72, 70)  # the progress bar's green, at 27%
    KEPT_EDGE = QColor("#24a148")
    CAPTION = QColor("#e6e6e6")
    DETAIL = QColor("#8f8f8f")

    def __init__(self) -> None:
        super().__init__()
        self._layout: shorts.Layout | None = None
        self._settings: shorts.Settings | None = None
        self._source_label = ""
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )

    # -- what to draw ----------------------------------------------------

    def show_framing(
        self,
        settings: shorts.Settings,
        width: int,
        height: int,
        *,
        label: str,
    ) -> None:
        """Redraw for `settings` against a source `width` x `height`.

        `label` says where those dimensions came from, since a source that has
        not been picked yet is drawn against an assumed resolution and the
        diagram would otherwise claim more than it knows.
        """
        self._source_label = label
        try:
            info = video_io.VideoInfo(
                path=Path(), width=width, height=height, fps=0.0, duration=0.0
            )
            self._layout = shorts.plan(info, settings)
            self._settings = settings
        except (ValueError, ZeroDivisionError):
            # Out-of-range option values are the user mid-edit, not an error
            # worth a dialog. Keep the last good drawing until they land.
            pass
        self.update()

    def kept_in_source(self) -> tuple[float, float, float, float]:
        """The kept region in source pixels, as (x, y, w, h).

        The foreground is scaled, then inset by the configured crop, then
        centre-cropped to the canvas -- so the survivor is a box centred on the
        inset region, mapped back through the scale factor. Dividing by it is
        what puts the answer in the source's own coordinates, which is the only
        frame of reference the top diagram can honestly be drawn in.
        """
        layout = self._layout
        assert layout is not None
        scale = layout.scale_h / layout.source_h if layout.source_h else 1.0
        left, _right, top, _bottom = layout.crop
        inset_w = min(layout.scale_w, max(2, layout.scale_w - left - _right))
        inset_h = min(layout.scale_h, max(2, layout.scale_h - top - _bottom))
        centre_x = left + inset_w / 2
        centre_y = top + inset_h / 2
        return (
            (centre_x - layout.visible_w / 2) / scale,
            (centre_y - layout.visible_h / 2) / scale,
            layout.visible_w / scale,
            layout.visible_h / scale,
        )

    # -- drawing ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.BACKDROP)

        margin = 12
        left = margin
        width = max(1, self.width() - 2 * margin)
        if self._layout is None:
            self._text(painter, left, margin, width, "Framing", self.CAPTION)
            return

        cursor = margin
        cursor = self._draw_source(painter, left, cursor, width)
        cursor += 12
        self._draw_canvas(painter, left, cursor, width)

    def _text(
        self, painter: QPainter, x: int, y: int, width: int, text: str, colour: QColor
    ) -> int:
        """One line, returning the y below it."""
        painter.setPen(colour)
        metrics = painter.fontMetrics()
        painter.drawText(
            QRectF(x, y, width, metrics.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        return y + metrics.height()

    def _box(
        self, painter: QPainter, rect: QRectF, fill: QColor, edge: QColor, width: float
    ) -> None:
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(edge, width))
        painter.drawRect(rect)

    def _draw_source(self, painter: QPainter, x: int, y: int, width: int) -> int:
        """The source frame with the kept region over it. Returns the y below."""
        layout = self._layout
        assert layout is not None

        y = self._text(
            painter, x, y, width,
            f"Source · {layout.source_w}×{layout.source_h}",
            self.CAPTION,
        )
        y = self._text(painter, x, y, width, self._source_label, self.DETAIL)
        y += 6

        # Capped rather than given the width outright: see SOURCE_SHARE.
        box_w = float(width)
        box_h = box_w * layout.source_h / layout.source_w
        cap = self.height() * self.SOURCE_SHARE
        if box_h > cap:
            box_h = cap
            box_w = box_h * layout.source_w / layout.source_h
        frame = QRectF(x + (width - box_w) / 2, y, box_w, box_h)
        self._box(painter, frame, self.FRAME_FILL, self.FRAME_EDGE, 1.0)

        scale = box_w / layout.source_w
        kx, ky, kw, kh = self.kept_in_source()
        kept = QRectF(
            frame.left() + kx * scale,
            frame.top() + ky * scale,
            kw * scale,
            kh * scale,
        ).intersected(frame)
        self._box(painter, kept, self.KEPT_FILL, self.KEPT_EDGE, 2.0)

        y = int(frame.bottom()) + 6
        return self._text(
            painter, x, y, width,
            f"keeps {layout.width_kept:.0%} of the width"
            f" · {int(kw)}×{int(kh)} px",
            self.DETAIL,
        )

    def _draw_canvas(self, painter: QPainter, x: int, y: int, width: int) -> None:
        """The output canvas: bars, foreground, and where the offset puts it."""
        layout = self._layout
        assert layout is not None

        y = self._text(
            painter, x, y, width,
            f"Short · {layout.canvas_w}×{layout.canvas_h}",
            self.CAPTION,
        )
        y += 6

        # Two lines of detail are reserved below, so the canvas never grows
        # into them and the whole panel stays inside its height.
        detail = painter.fontMetrics().height() * 2 + 8
        available = max(40.0, self.height() - y - detail - 12)
        box_h = available
        box_w = box_h * layout.canvas_w / layout.canvas_h
        if box_w > width:
            box_w = float(width)
            box_h = box_w * layout.canvas_h / layout.canvas_w
        canvas = QRectF(x + (width - box_w) / 2, y, box_w, box_h)

        scale = box_h / layout.canvas_h
        # The bars are the blurred copy of the frame darkened toward black, so
        # they are drawn as what they are: a grey whose lightness follows the
        # opacity setting rather than a flat placeholder.
        shade = int(20 + 70 * max(0.0, min(1.0, self._settings.opacity)))
        self._box(painter, canvas, QColor(shade, shade, shade), self.FRAME_EDGE, 1.0)

        front = QRectF(
            canvas.left() + (layout.canvas_w - layout.visible_w) / 2 * scale,
            canvas.top()
            + ((layout.canvas_h - layout.visible_h) / 2 + layout.offset_y) * scale,
            layout.visible_w * scale,
            layout.visible_h * scale,
        )
        # Clipped, not just drawn: an offset bigger than the bar pushes the
        # foreground off the canvas, and the diagram has to lose the same
        # sliver the encode does rather than drawing outside the frame.
        painter.save()
        painter.setClipRect(canvas)
        self._box(painter, front, self.KEPT_FILL, self.KEPT_EDGE, 2.0)
        painter.restore()

        y = int(canvas.bottom()) + 6
        bars = (
            f"{layout.bar_h}px bars, blur-filled"
            if layout.bar_h > 0
            else "no bars at this zoom — the fill never shows"
        )
        y = self._text(painter, x, y, width, bars, self.DETAIL)
        self._text(
            painter, x, y, width,
            f"background {self._settings.opacity:.0%}"
            f" · blur σ{self._settings.blur_sigma:g}",
            self.DETAIL,
        )


class JobTab(QWidget):
    """A video input, an output folder, options, a button, a log.

    Subclasses supply `build_job`, which returns the callable the worker runs,
    and may add their own controls in `add_options`.
    """

    key = "job"  # QSettings namespace for this page's remembered values
    button_text = "Run"
    input_label = "Video"

    def __init__(self) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._cancel: video_io.Canceller | None = None
        self._last_folder: Path | None = None
        # Where the file dialog opens: last used folder, the Videos folder
        # until there is one.
        self._source_hint = str(default_source())
        # An ini beside the .exe rather than QSettings' registry default --
        # see settings_file().
        self.settings = QSettings(
            str(settings_file()), QSettings.Format.IniFormat
        )
        # Every reset button on this tab, so restore() can re-check them all
        # once at the end instead of relying on each control having signalled.
        self.resets: list[ResetButton] = []
        self._build_ui()
        self.restore()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        # The page is two columns: the controls, and whatever the job wants to
        # show alongside them. The side panel is built last, once every control
        # it watches exists.
        page = QHBoxLayout(self)
        page.setSpacing(12)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        page.addLayout(layout, stretch=1)

        box = QGroupBox()
        form = QVBoxLayout(box)
        # Both fields start on the user's Videos folder: it is where Windows
        # puts recordings, so it is a better guess than an empty box, and
        # restore() replaces either one the moment there is a remembered value.
        self.source_edit = self._file_row(
            form, self.input_label, "Choose a folder of videos...",
            self._browse_source, default=str(default_source()),
        )
        self.source_edit.setText(str(default_source()))
        self.dest_edit = self._file_row(
            form, "Save to", str(default_output()), self._browse_dest,
            default=str(default_output()),
        )
        self.dest_edit.setText(str(default_output()))
        layout.addWidget(box)

        options = QGroupBox(self.options_title)
        self.options_layout = QGridLayout(options)
        self.add_options(self.options_layout)
        # Two label/control pairs per row, laid out as [label][control] gutter
        # [label][control] slack. Every column but the last sizes to its
        # contents, so each caption sits against the control it names instead
        # of drifting away from it as the window widens; the gutter keeps the
        # two pairs from reading as one run of four columns.
        self.options_layout.setColumnMinimumWidth(GUTTER_COLUMN, 40)
        self.options_layout.setColumnStretch(SLACK_COLUMN, 1)
        for column in RESET_COLUMNS:
            self.options_layout.setColumnMinimumWidth(column, RESET_WIDTH)
        # A tab with no knobs should not show an empty box.
        options.setVisible(self.options_layout.count() > 0)
        layout.addWidget(options)

        buttons = QHBoxLayout()
        self.run_button = QPushButton(self.button_text)
        self.run_button.setMinimumHeight(34)
        self.run_button.clicked.connect(self.start)
        buttons.addWidget(self.run_button, stretch=1)

        self.secondary_button: QPushButton | None = None
        if self.secondary_text:
            self.secondary_button = QPushButton(self.secondary_text)
            self.secondary_button.setMinimumHeight(34)
            self.secondary_button.clicked.connect(self.start_secondary)
            buttons.addWidget(self.secondary_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumHeight(34)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.cancel_button)

        self.open_button = QPushButton("Open output folder")
        self.open_button.setMinimumHeight(34)
        self.open_button.clicked.connect(self._open_output)
        buttons.addWidget(self.open_button)

        # Everything from here down is anchored to the bottom of the window:
        # the slack of a taller window opens up here, between the options and
        # the buttons, rather than under the log. Put below the log instead it
        # leaves the log stranded mid-window when the window grows, and pushes
        # it off the bottom when the window narrows.
        layout.addStretch(1)
        layout.addLayout(buttons)

        # Sits directly on top of the log, so progress and the lines explaining
        # it read as one block. Styled rather than left to the theme: on the
        # dark palette the default groove is barely distinguishable from the
        # window behind it.
        #
        # On screen at all times, empty when idle, rather than appearing for
        # the duration of a job. It is what keeps the page's height constant,
        # which is what lets the window sit exactly as tall as its content --
        # a bar that came and went would need 24px that are no longer there.
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: white; border: 1px solid #3c3c3c;"
            " border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #24a148; border-radius: 2px; }"
        )
        self._reset_progress()
        layout.addWidget(self.progress_bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Output appears here.")
        # Long paths should scroll, not reflow into an unreadable wall.
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.log.setFixedHeight(self._log_height())
        layout.addWidget(self.log)

        panel = self.build_side_panel()
        if panel is not None:
            page.addWidget(panel)

    def build_side_panel(self) -> QWidget | None:
        """The right-hand column, or None for a job with nothing to show."""
        return None

    def _log_height(self) -> int:
        """Exactly LOG_LINES of text, plus the chrome around them.

        The horizontal scrollbar is pinned on and counted in. Wrapping is off,
        so a long output path makes it appear on its own; left to come and go
        it would steal a line from a box this short exactly when the run starts
        producing the paths worth reading.
        """
        return (
            LOG_LINES * self.log.fontMetrics().lineSpacing()
            + 2 * int(self.log.document().documentMargin())
            + 2 * self.log.frameWidth()
            + self.log.horizontalScrollBar().sizeHint().height()
        )

    def _file_row(
        self, form: QVBoxLayout, label: str, placeholder: str, handler,
        *, default: str | None = None,
    ) -> QLineEdit:
        row = QHBoxLayout()
        caption = QLabel(label)
        caption.setMinimumWidth(60)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        button = QPushButton("Browse...")
        button.clicked.connect(handler)
        row.addWidget(caption)
        row.addWidget(edit, stretch=1)
        if default is not None:
            # Between the field and Browse. The field is the only stretching
            # thing in the row, so it absorbs the width when the button hides
            # and Browse stays put either way.
            row.addWidget(
                self._reset(
                    default, edit.text, edit.setText, edit.textChanged, default
                )
            )
        row.addWidget(button)
        form.addLayout(row)
        return edit

    # -- options ---------------------------------------------------------

    options_title = "Options"
    secondary_text = ""

    def add_options(self, grid: QGridLayout) -> None:
        """Add this job's controls. Default: none."""

    def _reset(self, default, read, write, changed, describe) -> ResetButton:
        button = ResetButton(default, read, write, changed, str(describe))
        self.resets.append(button)
        return button

    def _captioned(
        self,
        grid: QGridLayout,
        row: int,
        column: int,
        label: str,
        control: QWidget,
        *,
        tip: str = "",
    ) -> QWidget:
        """Drop a control into a pair's [label][control] cells."""
        caption = QLabel(label)
        if tip:
            caption.setToolTip(tip)
            control.setToolTip(tip)
        base = column * PAIR_STRIDE
        grid.addWidget(caption, row, base)
        grid.addWidget(control, row, base + 1)
        return control

    def _spin(
        self,
        grid: QGridLayout,
        row: int,
        column: int,
        label: str,
        *,
        low: float,
        high: float,
        value: float,
        step: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        special: str = "",
        tip: str = "",
    ) -> QSpinBox | QDoubleSpinBox:
        spin: QSpinBox | QDoubleSpinBox
        if decimals:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
        else:
            spin = QSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        if special:
            # Shown instead of the number at the minimum -- how "unset" is
            # spelled on a spin box that has to hold a number regardless.
            spin.setSpecialValueText(special)
        spin.setMinimumWidth(96)
        self._captioned(grid, row, column, label, spin, tip=tip)

        # Read back rather than reuse `value`: the spin box has already rounded
        # it to its own decimals, and that rounded number is what comparisons
        # against the live value have to be made against.
        default = spin.value()
        described = special if special and default == spin.minimum() else f"{default:g}"

        grid.addWidget(
            self._reset(default, spin.value, spin.setValue, spin.valueChanged, described),
            row,
            column * PAIR_STRIDE + 2,
        )
        return spin

    # -- browsing --------------------------------------------------------

    def _pick_folder(self, title: str, start: str) -> str:
        """A folder picker that also lists the files in each folder.

        Windows' native picker only ever shows folders, so there is no way to
        tell from it whether the folder about to be chosen is the one holding
        the recordings -- which is the only thing worth checking before
        pressing OK. Qt's own dialog lists the files too, greyed out because a
        folder is still what gets picked, but it has to be asked for by name:
        the native dialog is the default and quietly ignores ShowDirsOnly.

        The filter is set to everything rather than to the video extensions
        pipeline.sources accepts. A filter that hid files would be back to not
        being able to see what is in the folder, which is the whole point --
        and an .mkv listed but not picked up is worth seeing, not hiding.
        """
        dialog = QFileDialog(self, title, start)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilter("All files (*)")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return ""
        chosen = dialog.selectedFiles()
        return chosen[0] if chosen else ""

    def _browse_source(self) -> None:
        start = self.source_edit.text().strip() or self._source_hint
        chosen = self._pick_folder(
            f"Select {self.input_label.lower()} folder", start
        )
        if chosen:
            self.source_edit.setText(str(Path(chosen)))

    def _browse_dest(self) -> None:
        start = self.dest_edit.text().strip() or str(default_output())
        # The same dialog as the source, not the native one: two folder pickers
        # in one window that look nothing alike is worse than either.
        chosen = self._pick_folder("Select output folder", start)
        if chosen:
            self.dest_edit.setText(str(Path(chosen)))

    def _open_output(self) -> None:
        target = self._last_folder or Path(
            self.dest_edit.text().strip() or str(default_output())
        )
        if not target.is_dir():
            QMessageBox.information(
                self, APP_NAME, f"Nothing there yet:\n{target}"
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    # -- persistence -----------------------------------------------------

    def remembered(self) -> dict[str, object]:
        """Option values worth carrying to the next launch. Default: none."""
        return {}

    def option_defaults(self) -> dict[str, object]:
        """What `remembered` reads back when nothing has been touched.

        save() writes only the difference between the two, so a launch that
        changed nothing leaves nothing on disk.
        """
        return {}

    def restore_option(self, name: str, spin, cast) -> None:
        """Put a remembered number back on a spin box, if it still parses."""
        stored = self.settings.value(f"{self.key}/{name}")
        if stored is None:
            return
        try:
            spin.setValue(cast(stored))
        except (TypeError, ValueError):
            pass  # a hand-edited or stale settings file is not worth a crash

    def save(self) -> None:
        """Write the settings that differ from their defaults, and only those.

        Everything at its default is left out, and a session that changed
        nothing removes the file rather than writing one full of values the
        next launch would have picked anyway. That is what keeps a fresh
        machine clean: run the .exe, close it, nothing is left behind.
        """
        values: dict[str, object] = {}

        source = self.source_edit.text().strip()
        if source:
            folder = Path(source)
            # A single video is a legitimate thing to have picked; what is
            # worth remembering is the folder it sat in. Anything else is kept
            # exactly as typed -- a folder that does not exist right now (an
            # external drive, unplugged) is still the folder they chose, and
            # walking up to its parent would quietly lose it.
            if folder.is_file():
                folder = folder.parent
            if folder != default_source():
                values["source_dir"] = str(folder)

        dest = self.dest_edit.text().strip()
        if dest and Path(dest) != default_output():
            values["dest"] = dest

        defaults = self.option_defaults()
        for name, value in self.remembered().items():
            if not _same(value, defaults.get(name)):
                values[name] = value

        # Rewritten wholesale: a value put back to its default has to leave the
        # file, not sit there as a stale override.
        self.settings.remove(self.key)
        for name, value in values.items():
            self.settings.setValue(f"{self.key}/{name}", value)
        self.settings.sync()
        if not self.settings.allKeys():
            try:
                settings_file().unlink(missing_ok=True)
            except OSError:
                pass  # a file we cannot remove is not worth failing the exit over

    def restore(self) -> None:
        dest = self.settings.value(f"{self.key}/dest")
        if dest:
            self.dest_edit.setText(str(dest))
        folder = self.settings.value(f"{self.key}/source_dir")
        if folder:
            # A remembered folder replaces the default outright -- someone who
            # picked a folder last time is far likelier to want it again than
            # to want Videos/ back.
            self.source_edit.setText(str(folder))
        self._source_hint = self.source_edit.text().strip() or str(default_source())
        self._refresh_resets()

    def _refresh_resets(self) -> None:
        """Re-check every reset button. Call after restoring values in bulk.

        Setting a control emits its change signal and the button follows on its
        own, but only when the value actually changed -- so a restore that puts
        a control back to what it already held would otherwise leave the button
        showing whatever it was showing before.
        """
        for button in self.resets:
            button.refresh()

    # -- running ---------------------------------------------------------

    def build_job(self, source: Path, dest: Path) -> Callable[..., Outcome]:
        raise NotImplementedError

    def build_secondary_job(self, source: Path, dest: Path) -> Callable[..., Outcome]:
        raise NotImplementedError

    def start(self) -> None:
        self._launch(self.build_job)

    def start_secondary(self) -> None:
        self._launch(self.build_secondary_job)

    def _launch(self, factory) -> None:
        if self._thread is not None:
            return  # already running; the buttons are disabled anyway

        raw_source = self.source_edit.text().strip()
        if not raw_source:
            QMessageBox.warning(self, APP_NAME, "Choose a folder first.")
            return
        source = Path(raw_source)
        # Browse offers folders, but a single video typed or pasted in still
        # works -- pipeline.sources takes either.
        if not source.exists():
            QMessageBox.warning(self, APP_NAME, f"No such file or folder:\n{source}")
            return

        raw_dest = self.dest_edit.text().strip() or str(default_output())
        dest = Path(raw_dest)

        # One per run, handed to the pipeline so Cancel can reach the ffmpeg
        # it is currently running.
        self._cancel = video_io.Canceller()
        try:
            job = factory(source, dest)
        except Exception as error:  # noqa: BLE001 -- bad option values land here
            QMessageBox.critical(self, APP_NAME, str(error))
            return

        self.save()
        self.log.clear()
        self._set_busy(True)
        # Starts empty and fills. Not the indeterminate range: both jobs report
        # real progress, and a bar sweeping back and forth says less than one
        # sitting at 0 for the second or two before the first report arrives.
        self._reset_progress()

        self._thread = QThread(self)
        self._worker = Worker(job)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._append)
        self._worker.progress.connect(self._advance)
        self._worker.finished.connect(self._done)
        self._worker.cancelled.connect(self._stopped)
        self._thread.start()

    def cancel(self) -> None:
        """Stop the running job, killing whatever ffmpeg it has in flight."""
        if self._cancel is None:
            return
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Stopping...")
        self._append("")
        self._append("🛑 Cancelling — stopping ffmpeg…")
        self._cancel.cancel()

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.run_button.setText("Working..." if busy else self.button_text)
        if self.secondary_button is not None:
            self.secondary_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.cancel_button.setText("Cancel")

    def _reset_progress(self) -> None:
        """Empty and still. Not the indeterminate range, which would animate."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def _append(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _advance(self, done: int, total: int) -> None:
        if total <= 0:
            return
        # The pipeline reports one fixed scale for a whole run, so the range is
        # set once and the bar only ever moves forward.
        if self.progress_bar.maximum() != total:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _retire_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._cancel = None
        self._set_busy(False)

    def _stopped(self) -> None:
        """The job acknowledged the cancel. Leave the bar where it stopped."""
        self._retire_thread()

    def _done(self, ok: bool, payload: object) -> None:
        self._retire_thread()
        self._reset_progress()

        if not ok:
            QMessageBox.critical(self, APP_NAME, str(payload))
            return

        outcome = payload if isinstance(payload, Outcome) else Outcome(str(payload))
        self._last_folder = outcome.folder
        if outcome.preview is not None and outcome.preview.exists():
            PreviewDialog(outcome.preview, self).exec()

    def closing(self) -> bool:
        """True when it is safe to close the window."""
        return self._thread is None

    def stop_for_close(self) -> None:
        """Kill the job and wait for it, so nothing outlives the window.

        ffmpeg runs in its own process: without this the window disappears and
        the encode carries on holding the GPU until it finishes on its own.
        """
        if self._cancel is not None:
            self._cancel.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)


class ShortTab(JobTab):
    key = "short"
    button_text = "Make shorts"
    secondary_text = "Preview frame"
    input_label = "Highlights"
    options_title = "Framing"

    # What the diagram is drawn against until a folder is picked and the real
    # thing can be read off the first video in it. 1080p rather than the 21:9
    # the author records at: it is the aspect most sources actually are, and
    # the panel says in as many words that it is an assumption.
    ASSUMED_SOURCE = (1920, 1080)

    def add_options(self, grid: QGridLayout) -> None:
        defaults = self._defaults()

        zoom_tip = (
            "1.0 fills the portrait frame height and crops the sides; "
            "0.5 leaves a blurred bar of equal height above and below."
        )
        self.zoom_bar = ZoomBar(
            low=0.05, high=2.0, value=defaults.zoom, step=0.05
        )
        self._captioned(grid, 0, 0, "Zoom", self.zoom_bar, tip=zoom_tip)
        grid.addWidget(
            self._reset(
                self.zoom_bar.value(),
                self.zoom_bar.value,
                self.zoom_bar.setValue,
                self.zoom_bar.valueChanged,
                f"{self.zoom_bar.value():g}",
            ),
            0,
            2,
        )

        # Directly under the bar, because it is the same number: the bar is the
        # one worth dragging, the pixels are the one worth typing when the size
        # has to match something. No reset of its own -- the bar's covers both.
        self.size_fields = ScaledSize(defaults.canvas_h, self.ASSUMED_SOURCE)
        self._captioned(
            grid, 1, 0, "Size", self.size_fields,
            tip="What the clip is scaled to at this zoom, before cropping. "
                "Type either number and the other follows the source's aspect.",
        )

        self.offset_spin = self._spin(
            grid, 0, 1, "Vertical offset",
            low=-1920, high=1920, value=defaults.offset_y, step=10, suffix=" px",
            tip="Shift the clip off centre, positive down.",
        )
        self.opacity_spin = self._spin(
            grid, 2, 0, "Background",
            low=0.0, high=1.0, value=defaults.opacity, step=0.05, decimals=2,
            tip="Opacity of the blurred fill behind the bars, 0 (black) to 1.",
        )
        self.blur_spin = self._spin(
            grid, 2, 1, "Blur",
            low=0.0, high=64.0, value=defaults.blur_sigma, step=1.0, decimals=1,
            tip="Blur sigma before the upscale that multiplies it -- 8 here is "
                "about 64 at full resolution.",
        )
        self.start_spin = self._spin(
            grid, 3, 0, "Trim from",
            low=0.0, high=86400.0, value=0.0, step=1.0, decimals=1, suffix=" s",
            special="start",
            tip="Skip this many seconds of the clip.",
        )
        self.duration_spin = self._spin(
            grid, 3, 1, "Length",
            low=0.0, high=86400.0, value=0.0, step=1.0, decimals=1, suffix=" s",
            special="whole clip",
            tip="How many seconds to keep. Leave at 'whole clip' for all of it.",
        )

    def build_side_panel(self) -> QWidget | None:
        """The framing diagram, wired to every control that moves it."""
        self.preview = FramingPreview()
        for control in (
            self.zoom_bar, self.offset_spin, self.opacity_spin, self.blur_spin
        ):
            control.valueChanged.connect(self._refresh_preview)
        self.zoom_bar.valueChanged.connect(self._show_scaled_size)
        self.size_fields.zoomEdited.connect(self._zoom_typed)

        # Reading the source costs a file open, so it waits for typing to stop
        # rather than firing on every keystroke of a pasted path.
        self._probe_timer = QTimer(self)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(400)
        self._probe_timer.timeout.connect(self._probe_source)
        self.source_edit.textChanged.connect(self._probe_timer.start)

        self._source_size = self.ASSUMED_SOURCE
        self._source_note = "assumed — pick a folder to read the real size"
        self._probed: str | None = None
        self._show_scaled_size()
        self._refresh_preview()
        return self.preview

    def _show_scaled_size(self, *_ignored) -> None:
        self.size_fields.show_zoom(self.zoom_bar.value())

    def _zoom_typed(self, zoom: float) -> None:
        """A pixel size was typed. Move the bar, then say what it settled on.

        The bar rounds to its own step, and when that rounding lands back on
        the value it already held it emits nothing -- leaving the boxes showing
        the number that was typed rather than the one in force. Hence the
        unconditional refresh rather than leaving it to valueChanged.
        """
        self.zoom_bar.setValue(zoom)
        self._show_scaled_size()

    def _probe_source(self) -> None:
        """Read the first video's resolution, so the diagram is about it.

        Anything unreadable -- an empty field, a path part-way through being
        typed, a folder with no videos in it -- falls back to the assumption
        rather than emptying the panel. The diagram is about the framing, and
        the framing is worth showing before there is a video to apply it to.
        """
        raw = self.source_edit.text().strip()
        if raw == self._probed:
            return
        self._probed = raw
        size, note = self.ASSUMED_SOURCE, "assumed — pick a folder to read the real size"
        if raw:
            try:
                first = pipeline.sources(Path(raw))[0]
                info = video_io.probe(first)
                if info.width > 0 and info.height > 0:
                    size, note = (info.width, info.height), f"from {first.name}"
            except Exception:  # noqa: BLE001 -- a half-typed path is not an error
                pass
        self._source_size, self._source_note = size, note
        # The width is the source's aspect applied to the height, so a new
        # source changes what the same zoom comes to.
        self.size_fields.set_source(size)
        self._show_scaled_size()
        self._refresh_preview()

    def _refresh_preview(self, *_ignored) -> None:
        width, height = self._source_size
        self.preview.show_framing(
            self._settings_now(), width, height, label=self._source_note
        )

    def _settings_now(self) -> shorts.Settings:
        return shorts.Settings.load(
            zoom=self.zoom_bar.value(),
            offset_y=self.offset_spin.value(),
            opacity=self.opacity_spin.value(),
            blur_sigma=self.blur_spin.value(),
        )

    def _defaults(self) -> shorts.Settings:
        """config/shorts.json, or something sane if it cannot be read."""
        try:
            return shorts.Settings.load()
        except Exception:  # noqa: BLE001 -- a missing config must not kill the tab
            return shorts.Settings(
                canvas_w=1080, canvas_h=1920, zoom=0.5, offset_y=0,
                opacity=0.75, blur_sigma=8.0, blur_downscale=8, crop=(0, 0, 0, 0),
            )

    def remembered(self) -> dict[str, object]:
        return {
            "zoom": self.zoom_bar.value(),
            "offset_y": self.offset_spin.value(),
            "opacity": self.opacity_spin.value(),
            "blur": self.blur_spin.value(),
        }

    def option_defaults(self) -> dict[str, object]:
        # The same config/shorts.json values the controls were built from.
        defaults = self._defaults()
        return {
            "zoom": defaults.zoom,
            "offset_y": defaults.offset_y,
            "opacity": defaults.opacity,
            "blur": defaults.blur_sigma,
        }

    def restore(self) -> None:
        super().restore()
        for name, widget, cast in (
            ("zoom", self.zoom_bar, float),
            ("offset_y", self.offset_spin, int),
            ("opacity", self.opacity_spin, float),
            ("blur", self.blur_spin, float),
        ):
            self.restore_option(name, widget, cast)
        self._refresh_resets()
        self._show_scaled_size()
        self._refresh_preview()

    def _current(self) -> tuple[shorts.Settings, float | None, float | None]:
        settings = self._settings_now()
        # Both spin boxes spell "unset" as their minimum, 0.
        start = self.start_spin.value() or None
        duration = self.duration_spin.value() or None
        return settings, start, duration

    def build_job(self, source: Path, dest: Path):
        settings, start, duration = self._current()
        cancel = self._cancel

        def job(log, progress) -> Outcome:
            result = pipeline.make_shorts_batch(
                source, dest, settings=settings, start=start,
                duration=duration, log=log, progress=progress, cancel=cancel,
            )
            return Outcome(f"{result.summary()} written to {dest}", folder=dest)

        return job

    def build_secondary_job(self, source: Path, dest: Path):
        settings, start, duration = self._current()
        cancel = self._cancel

        def job(log, progress) -> Outcome:
            # One frame, to check the framing before paying for the renders --
            # so it previews the first video rather than all of them.
            first = pipeline.sources(source)[0]
            still = pipeline.preview_short(
                first, dest, settings=settings, start=start,
                duration=duration, log=log, cancel=cancel,
            )
            return Outcome(
                f"preview written to {still}", folder=dest, preview=still
            )

        return job


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)

        self.short_tab = ShortTab()
        self.setCentralWidget(self.short_tab)

        self._build_status_bar()

        # Width is a starting point the user can change; the height is not,
        # it follows what the page needs (see showEvent).
        self._fitted = False
        # Wider than the controls alone need: the framing panel takes the
        # right-hand third, and the portrait diagram in it is only worth
        # drawing if it has the width to be read at a glance.
        self.resize(980, self.sizeHint().height())

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._fitted:
            # The one time the height is set outright: the window opens exactly
            # as tall as its content. It has to wait for real geometry -- before
            # the window is up the page has its default width, and wraps its
            # description to a different number of lines than it will once
            # shown. After this the height belongs to the user.
            self._fitted = True
            self.resize(self.width(), self._floor())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # What the page needs depends on its width. Without re-flooring here,
        # dragging the window narrower keeps a height that no longer fits and
        # the log gets squeezed off the bottom. Only the floor moves -- the
        # height is the user's once the window is up.
        if event.size().width() != event.oldSize().width():
            self._floor()

    def _floor(self) -> int:
        """Set the window's minimum height to what the page needs, and return it.

        The minimum has to be set explicitly, for the same reason the height
        does: the page's own hint does not account for the status bar and the
        window chrome around it.
        """
        page = self.short_tab
        needed = (
            page.heightForWidth(page.width())
            if page.hasHeightForWidth()
            else page.sizeHint().height()
        )
        height = needed + (self.height() - page.height())
        self.setMinimumHeight(height)
        return height

    def closeEvent(self, event) -> None:
        if self.short_tab.closing():
            self.short_tab.save()
            self._stop_monitoring()
            event.accept()
            return
        # Killing a running ffmpeg or a half-written mp4 on a stray click is
        # worth one confirmation.
        answer = QMessageBox.question(
            self,
            APP_NAME,
            "A job is still running. Stop it and quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        # Quitting has to take the work with it. ffmpeg is a separate process
        # and does not care that its parent has gone -- left alone it runs the
        # encode to the end, on the GPU, with no window to show for it.
        self.short_tab.stop_for_close()
        self.short_tab.save()
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
