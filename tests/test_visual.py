from types import SimpleNamespace

import numpy as np
import pytest

from src.moments import concepts, engine, motion, visual
from src.moments.vision import Embeddings

CFG = {"concepts": {"baseline_s": 20, "floor": 1.0, "span": 2.0, "present_floor": -5.0, "present_span": 2.0},
       "novelty": {"window_s": 5, "baseline_s": 20, "floor": 0.02, "span": 0.15},
       "menu": {"floor": 0.0, "span": 2.0}}


def test_relative_needs_both_a_rise_and_presence():
    z = np.full(60, -12.0)
    z[30:33] = -8.0           # a rise, but nowhere near present
    z[40:43] = -2.0           # a rise, and present
    rel = visual.relative(z, CFG["concepts"])
    assert rel[31] == 0.0 and rel[41] == pytest.approx(1.0)
    flat = visual.relative(np.full(60, -1.0), CFG["concepts"])
    assert flat.max() == 0.0  # always present is the usual level, not an event


def test_novelty_spikes_where_the_picture_changes():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    seconds = np.array([a] * 30 + [b] * 30)
    nov = visual.novelty(seconds, CFG["novelty"])
    assert int(nov.argmax()) == 30 and nov[30] == pytest.approx(1.0)
    assert nov[:29].max() == 0.0 and nov[40:].max() == 0.0


def _bank():
    names = ["menu", "ordinary", "water"]
    return concepts.Bank(names, {"menu": "suppress", "ordinary": "neutral", "water": "event"},
                         np.eye(3, dtype=np.float32), 10.0, -8.0, {})


def test_build_scores_menus_and_keeps_them_out_of_the_vision_curve():
    n = 60
    vectors = np.zeros((n, 3), dtype=np.float16)
    vectors[:, 1] = 1                            # ordinary gameplay
    vectors[20:25] = [1, 0, 0]                   # a menu
    vectors[40:45] = [0, 0, 1]                   # water
    emb = Embeddings(vectors, np.arange(n, dtype=np.float32), 1.0, "CPU", 0.0)
    vis = visual.build(emb, _bank(), n, CFG, {"water": "a person in water"})
    assert vis.menu[22] == pytest.approx(1.0) and vis.menu[5] == 0.0
    assert vis.relative["water"][42] == pytest.approx(1.0) and vis.vision[42] == pytest.approx(1.0)
    assert set(vis.relative) == {"water"}


def _texture(h, w, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (5, 5), 0)


def test_motion_separates_camera_pan_from_local_movement():
    big = _texture(180, 320)
    pan = np.stack([big[20:110, 20 + 3 * i:180 + 3 * i] for i in range(8)])   # the whole frame slides: camera
    m = motion.measure(pan, 10)
    assert np.median(m.camera) == pytest.approx(3.0, abs=0.8) and np.median(m.local) < 1.0
    patch = _texture(20, 20, seed=1)
    still = np.stack([big[20:110, 20:180].copy() for _ in range(8)])
    for i, frame in enumerate(still):                                         # one square moves: local
        frame[30:50, 20 + 8 * i:40 + 8 * i] = patch
    m2 = motion.measure(still, 10)
    assert np.median(m2.camera) < 1.0 and np.median(m2.local) > np.median(m.local)


def test_static_hud_pixels_are_masked():
    frames = np.random.default_rng(0).integers(0, 255, (5, 10, 10), dtype=np.uint8)
    frames[:, :2, :2] = 200
    mask = motion.static_mask(frames)
    assert not mask[:2, :2].any() and mask[5:, 5:].all()


def _moment(start, end, score, label):
    m = engine.Moment(start, end, (start + end) / 2, score, [])
    m.label = label
    return m


def test_dedupe_merges_touching_moments_and_drops_lookalikes():
    cfg = {"visual": {"merge_gap_s": 5, "dedupe_cos": 0.95}, "clip": {"max": 30}}
    per_second = np.zeros((200, 2))
    per_second[:, 0] = 1
    per_second[100:200] = [0, 1]
    moments = [_moment(10, 20, 50, "vehicle"), _moment(22, 30, 40, "vehicle"),   # touching: merged
               _moment(60, 70, 30, "vehicle"),                                   # looks like the first: dropped
               _moment(120, 130, 20, "vehicle"),                                 # looks different: kept
               _moment(40, 50, 10, "unlabelled")]
    out = engine._dedupe(moments, per_second, cfg)
    assert sorted((m.start, m.end) for m in out) == [(10, 30), (40, 50), (120, 130)]


def test_lead_in_starts_before_the_picture_shows_it():
    rel = np.zeros(100)
    rel[45:47] = 1.0
    rel[50:60] = 1.0           # a 3-second dropout at 47-49
    vis = SimpleNamespace(relative={"water": rel})
    moment = _moment(50, 70, 50, "unusual")
    moment.evidence = [engine.Evidence("vision", "see:water", 1.0, 52.0, "water", 10)]
    cfg = {"visual": {"lead": {"level": 0.25, "seconds": 8, "gap_s": 3}}, "clip": {"max": 30}}
    assert engine._lead_in(moment, vis, cfg, 100)
    assert moment.start == 37 and moment.end == 67  # 45 - 8, then capped at 30 s
    moment.label = "reaction"
    assert not engine._lead_in(moment, vis, cfg, 100)


CLIP = {"min": 10, "max": 45, "tail": 2,
        "onset": {"level": 0.3, "gap_s": 5, "lookback_s": 35, "pre_roll_s": 5}}


def test_onset_walks_back_to_where_the_action_starts():
    activity = np.zeros(200)
    activity[100:104] = 1.0      # first burst
    activity[108:120] = 1.0      # after a 4 s pause: same fight
    start, end = engine._onset(110.0, 125.0, 118, activity, CLIP)
    assert start == 95 and end == 125           # 100 - 5 s pre-roll
    activity[100:104] = 0.0
    activity[90:94] = 1.0        # 14 s before: a different event
    start, _ = engine._onset(110.0, 125.0, 118, activity, CLIP)
    assert start == 103                          # 108 - 5


def test_onset_never_cuts_the_peak():
    activity = np.ones(200)
    start, end = engine._onset(150.0, 170.0, 168, activity, CLIP)
    assert end >= 170 and end - start <= 45


def test_menu_runs_ignore_isolated_seconds():
    flags = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    assert engine._runs(flags, 3).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]


def test_clips_never_open_on_a_menu():
    menu = np.zeros(200)
    menu[80:95] = 1.0            # a map before the action
    menu[130:140] = 1.0          # and one after it
    start, end = engine._avoid_menus(70.0, 135.0, 110, menu, CLIP)
    assert (start, end) == (95.0, 130.0)
    start, end = engine._avoid_menus(90.0, 100.0, 96, menu, CLIP)
    assert start >= 95 and end - start >= 10     # grown back to the minimum on the gameplay side
