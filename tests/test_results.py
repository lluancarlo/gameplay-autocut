"""Result improvements: captions, teaser/follow composition, voice markers, scene-cut snapping."""

from types import SimpleNamespace

import numpy as np

from src import captions, composer, shorts
from src.moments import engine, visual
from src.moments.speech import Word, _matches


def words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(start + i * step, start + i * step + step * 0.9, w, "pt") for i, w in enumerate(text.split())]


def test_marker_phrase_variants():
    assert _matches(words("bora clipar agora vai")) == [(0.5, 1.45)]
    assert _matches(words("Clipa, agora!"))
    assert _matches(words("clip a gora"))
    assert _matches(words("clicar agora"))
    assert not _matches(words("agora não"))
    assert not _matches(words("e para agora"))


def test_caption_chunks_break_on_pauses_and_size():
    ws = words("um dois tres quatro cinco") + [Word(10, 10.4, "seis")]
    groups = captions.chunks(ws, most=4)
    assert [len(g) for g in groups] == [4, 1, 1]


def test_voice_captions_highlight_each_word(tmp_path):
    ws = words("olha isso", start=101)
    path = captions.write(tmp_path / "a.ass", 100, 120, voice_words=ws, offset=1.5)
    text = path.read_text(encoding="utf-8")
    assert text.count("Style: Voice") == 1
    events = [l for l in text.splitlines() if l.startswith("Dialogue")]
    assert len(events) == 2
    assert "0:00:02.50" in events[0]                     # 101 - 100 + 1.5 teaser
    assert captions.YELLOW + "OLHA" in events[0] and captions.YELLOW + "ISSO" in events[1]


def test_game_subtitles_merge_and_clip():
    subs = [{"t": 10, "text": "Oi"}, {"t": 12, "text": "Oi"}, {"t": 30, "text": "Tchau"}]
    lines = captions.game_lines(subs)
    assert [l.text for l in lines] == ["Oi", "Tchau"]
    assert lines[0].end == 13
    assert len(captions.game_events(lines, [(25, 40)], 0)) == 1


def test_nothing_to_caption_writes_nothing(tmp_path):
    assert captions.write(tmp_path / "x.ass", 0, 10) is None


def test_filter_arg_escapes_windows_paths(tmp_path):
    arg = captions.filter_arg(tmp_path / "a.ass")
    assert "\\:" in arg and "\\\\" not in arg and arg.startswith("ass='")


def test_lerp_expression_is_piecewise():
    expr = composer._lerp_expr([(0, 0), (1, 10), (2, 10)])
    assert expr == "if(lt(t,1.00),0.0+(10.000)*(t-0.00),if(lt(t,2.00),10.0+(0.000)*(t-1.00),10.0))"
    assert composer._lerp_expr([(0, 5)]) == "5.0"


def test_follow_graph_is_full_height_9_16():
    info = SimpleNamespace(width=1920, height=1080)
    graph = composer.follow_graph("0:v", "v0", info, shorts.Settings.load(), [(0, 0.0), (1, 1.0)])
    assert "crop=w=608:h=1080" in graph and "scale=1080:1920" in graph and graph.endswith("[v0]")
    assert "1312.0" in graph            # clamped to the right edge (1920 - 608)


def test_teaser_adds_length():
    short = composer.Short(10, 30, teaser_at=20)
    assert short.length == 20 + composer.TEASER_S
    assert composer.Short(10, 30).length == 20


def test_filtergraph_labels_are_unique_per_input():
    from pathlib import Path

    from src import video_io
    info = video_io.VideoInfo(Path("x.mp4"), 1920, 1080, 30, 60)
    settings = shorts.Settings.load()
    a, _ = shorts.filtergraph(shorts.plan(info, settings), settings, "0:v", "v0", tag="0")
    b, _ = shorts.filtergraph(shorts.plan(info, settings), settings, "1:v", "v1", tag="1")
    assert "[bg0]" in a and "[bg1]" in b and "[bg0]" not in b


def test_scene_cuts_find_jumps():
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=8), rng.normal(size=8)
    vecs = np.array([first] * 20 + [second] * 20, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    emb = SimpleNamespace(frames=40, vectors=vecs, times=np.arange(40) / 10)
    assert visual.scene_cuts(emb, 0.8) == [2.0]


def test_snap_to_scene_cuts():
    clip = {"min": 10, "scene_cut": {"snap_s": 1.5}}
    assert engine._snap_cuts(20.0, 40.0, [19.0, 41.0], clip) == (19.05, 40.95)
    assert engine._snap_cuts(20.0, 40.0, [15.0, 45.0], clip) == (20.0, 40.0)
    # never snap the end so close that the clip gets shorter than min
    assert engine._snap_cuts(20.0, 31.0, [29.8], clip) == (20.0, 31.0)


TRIM = {"level": 0.25, "min_s": 3, "keep_s": 0.5, "edge_s": 2, "protect_s": 3, "min_length_s": 12}


def test_dead_seconds_are_cut_but_not_talk_peak_or_edges():
    curve = [1.0] * 5 + [0.05] * 8 + [1.0] * 10        # seconds 5-12 dead
    kept = composer.keep_segments(0, 23, curve, [], [20], TRIM)
    assert kept == [(0, 5.5), (12.5, 23)]
    # someone talking at 8 s: the run splits, only the part of 3 s or more is cut
    kept = composer.keep_segments(0, 23, curve, [(8.0, 8.5)], [20], TRIM)
    assert kept == [(0, 9.5), (12.5, 23)]                       # 5-6 s too short a run; 9-12 s cut
    assert composer.keep_segments(0, 23, curve, [(5, 13)], [20], TRIM) == [(0, 23)]
    assert composer.keep_segments(0, 23, curve, [], [9], TRIM) == [(0, 23)]


def test_trim_never_goes_below_min_length():
    curve = [1.0] * 3 + [0.0] * 10 + [1.0] * 3
    kept = composer.keep_segments(0, 16, curve, [], [], TRIM)
    assert sum(b - a for a, b in kept) >= 12


def test_captions_follow_the_cuts(tmp_path):
    ws = [Word(14.0, 14.4, "depois", "pt")]
    path = captions.write(tmp_path / "c.ass", 0, 20, voice_words=ws, pieces=[(0, 5), (12, 20)])
    line = next(l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue"))
    assert line.startswith("Dialogue: 0,0:00:07.00")            # 5 s kept, then 2 s into the second piece


def test_game_ducks_under_a_separate_voice_track():
    cfg = {"duck": True, "duck_threshold": 0.03, "duck_ratio": 6, "duck_release_ms": 400, "denoise_nf": -25}
    graph = composer._audio_graph(0, 2, 1, cfg)
    assert "sidechaincompress" in graph and "[0:a:1]afftdn" in graph and graph.endswith("[a0]")
    assert "sidechaincompress" not in composer._audio_graph(0, 2, None, cfg)
    assert composer._audio_graph(0, 1, None, cfg) == "[0:a:0]anull[a0]"
