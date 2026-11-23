import json
from types import SimpleNamespace

import numpy as np

from src.moments import engine, judge, ocr, speech, voice
from src.moments.speech import Word

VCFG = {"game_language": "en", "track_min_speech": 0.05, "min_subtitle_s": 30, "subtitle_pad_s": 1,
        "commentary_ratio": 0.35, "commentary_min_s": 60, "no_voice_reaction": 0.5}


def test_voice_by_language_separates_player_from_game():
    words = [Word(1, 1.5, "hey", "en"), Word(10, 10.5, "caraca", "pt"), Word(11, 11.5, "olha", "pt")]
    heard = voice.attribute(words, np.zeros(100), VCFG)
    assert heard.mode == "mixed" and [w.text for w in heard.player_words] == ["caraca", "olha"]
    assert [w.text for w in heard.game_words] == ["hey"]
    only_game = voice.attribute([Word(1, 2, "hey", "en")], np.zeros(100), VCFG)
    assert only_game.mode == "none" and not only_game.player_words


def test_voice_falls_back_to_subtitles_without_language():
    subtitles = np.zeros(200)
    subtitles[0:100] = 1
    words = [Word(t, t + 0.5, "w") for t in range(0, 200, 2)]
    heard = voice.attribute(words, subtitles, dict(VCFG, commentary_min_s=10))
    assert heard.mode == "mixed" and all(w.start >= 100 for w in heard.player_words)
    assert voice.attribute(words, np.zeros(200), VCFG).mode == "unknown"
    assert voice.attribute([], subtitles, VCFG).mode == "none"


def test_ocr_zones_and_gta_events():
    assert ocr.zone([0.3, 0.85, 0.7, 0.9]) == "subtitle"
    assert ocr.zone([0.3, 0.4, 0.7, 0.5]) == "big"
    assert ocr.zone([0.02, 0.1, 0.2, 0.13]) == "hud"
    lexicon = {"wanted": ["a policia esta procurando voce"], "escaped": ["fugiram"], "failure": ["wasted"]}
    lines = [{"t": 10, "text": "A polícia está procurando você nessa", "zone": "hud"},
             {"t": 20, "text": "M-merda! Fugiram!", "zone": "hud"},
             {"t": 30, "text": "Ele fugiram de mim", "zone": "subtitle"},       # dialogue: never an event
             {"t": 40, "text": "WASTED", "zone": "big"},
             {"t": 50, "text": "wastedland", "zone": "big"}]                    # whole words only
    found = [(e["t"], e["kind"]) for e in ocr.match_events(lines, lexicon)]
    assert found == [(10, "wanted"), (20, "escaped"), (40, "failure")]


def test_subtitle_seconds_cover_the_sampling_gap():
    text = ocr.ScreenText(0.5, [{"t": 10, "text": "x", "zone": "subtitle"}, {"t": 50, "text": "y", "zone": "hud"}])
    covered = np.flatnonzero(text.subtitle_seconds(60)).tolist()
    assert 10 in covered and 11 in covered and 50 not in covered


def test_passages_join_close_segments():
    assert speech._passages([(0, 3), (4, 6), (20, 22), (22.5, 40), (41, 50)]) == [(0, 6), (20, 40), (41, 50)]


def test_chase_starts_at_the_wanted_notification():
    clip = {"max": 45, "tail": 2, "onset": {"pre_roll_s": 5}}
    assert engine._chase_start(100.0, 130.0, 125, [95.0], clip) == 90.0
    assert engine._chase_start(100.0, 130.0, 125, [40.0], clip) == 100.0   # too long ago to fit
    assert engine._chase_start(100.0, 130.0, 125, [], clip) == 100.0


def test_text_curve_marks_outcomes_only():
    curve = engine._text_curve([{"t": 10, "kind": "failure"}, {"t": 40, "kind": "wanted"}], 60,
                               ("achievement", "failure", "escaped"))
    assert curve[10] == 1 and curve[8] == 1 and curve[40] == 0


def test_judge_parse_is_forgiving():
    out = judge.parse('sure: {"score": 14, "event": "rocket", "title": " Fuga! ", "description": "d", '
                      '"action_starts_at": 4, "why": "w"} done')
    assert out["score"] == 10 and out["event"] == "other" and out["title"] == "Fuga!"


def test_judge_blends_scores_and_moves_the_start():
    cfg = {"weight": 0.6, "pre_roll_s": 3, "look_before_s": 10}
    moment = engine.Moment(100.0, 130.0, 120.0, 40, [])
    moment.detector_score = 40
    judge._apply(moment, {"score": 9, "event": "crash", "title": "t", "description": "d", "why": "w",
                          "action_starts_at": 5, "window_start": 90.0}, cfg)
    assert moment.score == round(0.6 * 90 + 0.4 * 40) and moment.ai["title"] == "t"
    assert moment.start == 92.0 and moment.ai["start_moved_from"] == 100.0
    late = engine.Moment(100.0, 130.0, 120.0, 40, [])
    late.detector_score = 40
    judge._apply(late, {"score": 2, "action_starts_at": 38, "window_start": 90.0, "event": "other",
                        "title": "", "description": "", "why": ""}, cfg)
    assert late.start == 100.0  # would leave less than 8 s: ignored


def test_judge_dossier_lists_what_is_known():
    heard = voice.Voice("mixed", player_words=[Word(101, 101.5, "caraca", "pt")],
                        game_words=[Word(105, 105.5, "freeze", "en")])
    screen = ocr.ScreenText(0.5, [{"t": 102, "text": "Parado aí!", "zone": "subtitle"},
                                  {"t": 103, "text": "A polícia está procurando você", "zone": "hud"}])
    moment = SimpleNamespace(explanation="Vehicle (police) — siren")
    text = judge.dossier(moment, 100, 130, heard, screen, "GTA 6")
    for piece in ("caraca", "freeze", "Parado aí!", "procurando você", "siren", "GTA 6"):
        assert piece in text


def test_judge_frames_split_a_jpeg_stream(tmp_path):
    from src import video_io

    clip = tmp_path / "c.mp4"
    import subprocess

    subprocess.run([video_io.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=s=320x180:r=30:d=4", "-pix_fmt", "yuv420p", str(clip)], check=True)
    pictures = judge.frames(clip, 0, 4, 4, 160, video_io.Canceller())
    assert len(pictures) == 4 and all(p[:2] == b"\xff\xd8" and p[-2:] == b"\xff\xd9" for p in pictures)


def test_schema_and_events_agree():
    assert set(judge.SCHEMA["properties"]["event"]["enum"]) == set(judge.EVENTS)
    json.dumps(judge.SCHEMA)


def test_every_weighted_curve_has_a_reason():
    from src import config

    cfg = config.load("moments")
    sound_like = {"reaction", "action", "vision", "vehicle", "siren", "crash", "keywords"}
    for name, weight in cfg["weights"].items():
        if weight > 0 and name not in sound_like:
            assert name in engine.REASONS, name


def test_old_data_folder_and_settings_are_adopted_after_the_rename(tmp_path, monkeypatch):
    from src import config, gui

    (tmp_path / "VVideo-data" / "models").mkdir(parents=True)
    (tmp_path / "VVideo.ini").write_text("[framing]\nzoom=0.4\n", encoding="utf-8")
    monkeypatch.setattr(config, "_home", lambda: tmp_path)
    assert config._data_root() == tmp_path / "GameplayAutoCut-data"
    assert (tmp_path / "GameplayAutoCut-data" / "models").is_dir() and not (tmp_path / "VVideo-data").exists()
    assert gui.settings_file() == tmp_path / "GameplayAutoCut.ini"
    assert "zoom=0.4" in (tmp_path / "GameplayAutoCut.ini").read_text(encoding="utf-8")
    assert config._data_root() == tmp_path / "GameplayAutoCut-data"   # second start: nothing to move
