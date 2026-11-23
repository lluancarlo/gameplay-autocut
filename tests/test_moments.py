import numpy as np
import pytest

from src import pipeline, video_io
from src.moments import engine, signals
from src.moments.speech import Word

CLIP = {"min": 10, "ideal": 20, "max": 30, "setup": 5, "tail": 2, "extent": 0.5}


def test_normalize_ignores_case_accents_and_stretched_letters():
    assert signals.normalize("NÃÃÃO acredito!!") == signals.normalize("não acredito")
    assert signals.normalize("noooo") == "no"


def test_keywords_match_multi_word_phrases():
    words = [Word(5.0, 5.3, "Non"), Word(5.3, 5.5, "ci"), Word(5.5, 6.0, "credo!"), Word(9.0, 9.2, "ok")]
    curve, hits = signals.keywords(words, 12, {"non ci credo": 1.0, "dai": 0.5})
    assert hits == [(5.0, "non ci credo")]
    assert curve[5] == 1.0 and curve[9] == 0.0


def test_peaks_keep_their_distance():
    curve = np.zeros(100)
    curve[[10, 15, 60]] = [0.9, 0.5, 0.7]
    assert engine._peaks(curve, gap=20, floor=0.1) == [10, 60]


@pytest.mark.parametrize("peak", [3, 50, 98])
def test_window_respects_length_and_keeps_the_setup(peak):
    curve = np.zeros(100)
    curve[peak] = 1.0
    start, end = engine._window(curve, peak, CLIP)
    assert CLIP["min"] <= end - start <= CLIP["max"]
    assert start <= max(0, peak - CLIP["setup"]) and end > peak


def test_long_reaction_is_trimmed_to_the_maximum():
    curve = np.ones(200)
    start, end = engine._window(curve, 100, CLIP)
    assert end - start == pytest.approx(CLIP["max"])
    assert start <= 100 - CLIP["setup"]


def test_keyframe_before():
    keyframes = [0.0, 4.17, 8.33]
    assert pipeline.keyframe_before(keyframes, 6.0) == 4.17
    assert pipeline.keyframe_before(keyframes, 4.17) == 4.17
    assert pipeline.keyframe_before([], 3.0) == 0.0


def test_cut_copies_from_the_keyframe(tmp_path):
    source = tmp_path / "game.mkv"
    video_io._run([
        video_io.ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-c:a", "aac", "-shortest", str(source),
    ])
    keyframes = video_io.keyframes(source)
    assert keyframes == pytest.approx([0, 2, 4, 6, 8], abs=0.05)

    first, second = pipeline.cut_moments(source, [(0.5, 3.0), (3.0, 6.0)], keyframes, tmp_path / "out")
    assert second.name == "game_00m03s.mp4"
    assert video_io.probe(first).duration == pytest.approx(3.0, abs=0.15)
    assert video_io.probe(second).duration == pytest.approx(4.0, abs=0.15)
