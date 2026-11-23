import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from src.steps import RangeBar  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _click(bar, x, kind=QMouseEvent.Type.MouseButtonPress):
    event = QMouseEvent(kind, QPointF(x, 10), QPointF(x, 10), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    {QMouseEvent.Type.MouseButtonPress: bar.mousePressEvent, QMouseEvent.Type.MouseMove: bar.mouseMoveEvent,
     QMouseEvent.Type.MouseButtonRelease: bar.mouseReleaseEvent}[kind](event)


def test_clicking_the_bar_seeks_and_never_moves_the_bounds(app):
    bar = RangeBar()
    bar.resize(416, 34)                       # 400 px of track: 1 s per 4 px over 0..100 s
    bar.set_range(0, 100, 40, 60)
    seeks = []
    bar.seek.connect(seeks.append)
    _click(bar, 8 + 50 * 4)                   # 50 s: inside the clip
    _click(bar, 8 + 90 * 4, QMouseEvent.Type.MouseMove)   # dragged past the end: clamped
    _click(bar, 0, QMouseEvent.Type.MouseButtonRelease)
    _click(bar, 8 + 10 * 4, QMouseEvent.Type.MouseMove)   # not dragging any more: ignored
    assert seeks[0] == pytest.approx(50) and 59 <= seeks[1] <= 60 and len(seeks) == 2
    assert (bar.start, bar.end) == (40, 60)
