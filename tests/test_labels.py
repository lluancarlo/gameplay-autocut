import json

import numpy as np
import pytest

from src import config
from src.moments import engine, labels, sounds
from src.moments.labels import UNLABELLED, Evidence

RULES = config.load("moments")["labels"]["rules"]


def tags(names, rows):
    return {"probs": np.array(rows, dtype=np.float32), "names": names}


def test_group_takes_the_best_member_and_names_it():
    t = tags(["Car", "Siren", "Laughter"], [[0.2, 0.7, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 0.0]])
    values, names = sounds.group(t, ["Car", "Siren"])
    assert values.tolist() == pytest.approx([0.7, 0.9, 0.0])
    assert names == ["Siren", "Car", ""]
    empty, blank = sounds.group(t, ["Not a class"])
    assert empty.tolist() == [0, 0, 0] and blank == ["", "", ""]


def test_every_configured_group_name_is_a_real_audioset_class():
    folder = config.DATA_DIR / "models" / "ced-mini"
    if not (folder / "class_labels_indices.csv").is_file():
        pytest.skip("sound model not downloaded")
    known = set(sounds.class_names(folder))
    for group, members in config.load("moments")["sound_groups"].items():
        assert set(members) <= known, (group, set(members) - known)


def ev(signal, value, seconds=1, detail=""):
    return Evidence("audio", signal, value, 0.0, detail or signal, seconds)


def test_classify_picks_the_label_whose_evidence_clears_its_bar_most():
    label, detail, strength, level = labels.classify(
        [ev("gunfire", 0.9, 5, "Machine gun"), ev("vehicle", 0.5, 6, "Car")], RULES)
    assert (label, detail, level) == ("combat", "machine gun", "high")
    assert strength == pytest.approx(0.9 / RULES["combat"]["signals"]["gunfire"][0], abs=1e-3)


def test_classify_needs_the_evidence_to_last():
    assert labels.classify([ev("vehicle", 0.9, 2, "Car")], RULES)[0] == UNLABELLED  # needs 4 s
    assert labels.classify([ev("vehicle", 0.9, 5, "Car")], RULES)[0] == "vehicle"


def test_siren_makes_a_vehicle_moment_a_siren_one():
    label, detail, *_ = labels.classify([ev("vehicle", 0.8, 6, "Car"), ev("siren", 0.5, 4, "Police car (siren)")], RULES)
    assert (label, detail) == ("vehicle", "siren")


def test_weak_evidence_stays_unlabelled():
    assert labels.classify([ev("crash", 0.1, 1)], RULES) == (UNLABELLED, "", 0.0, "")
    assert labels.classify([], RULES)[0] == UNLABELLED


def test_collect_measures_inside_the_moment_only():
    n = 30
    gun = np.zeros(n)
    gun[10:14] = 0.8
    gun[25] = 0.9  # outside the moment
    groups = {"gunfire": (gun, ["Machine gun"] * n)}
    curves = {"loudness": np.zeros(n), "speech": np.zeros(n)}
    found = labels.collect(5, 20, groups, curves, [(12.0, "no way"), (28.0, "lets go")], {"no way": 1.0}, 0.3)
    gunfire = next(e for e in found if e.signal == "gunfire")
    assert gunfire.seconds == 4 and gunfire.at == 10 and gunfire.value == pytest.approx(0.8)
    assert [e.detail for e in found if e.signal == "keyword"] == ["no way"]


def test_explanation_puts_supporting_evidence_first():
    evidence = [ev("speech", 1.0, 5), ev("gunfire", 0.6, 4, "Gunshot, gunfire")]
    text = labels.explain("combat", "gunshot", evidence, ("gunfire", "explosion"))
    assert text.startswith("Combat (gunshot) — gunshot for 4s")
    assert labels.explain(UNLABELLED, "", []) == "No clear event"


def test_zero_weight_curves_do_not_change_the_score():
    cfg = config.load("moments")
    rng = np.random.default_rng(1)
    base = {k: rng.random(120) for k in ("loudness", "reaction", "action", "speech", "keywords")}
    extra = dict(base, vehicle=rng.random(120), siren=rng.random(120), crash=rng.random(120))
    zero = dict(cfg, weights={**cfg["weights"], "vehicle": 0.0, "siren": 0.0, "crash": 0.0})
    assert np.array_equal(engine.fuse(base, zero)[0], engine.fuse(extra, zero)[0])
    some = dict(cfg, weights={**cfg["weights"], "siren": 0.3})
    assert not np.array_equal(engine.fuse(base, some)[0], engine.fuse(extra, some)[0])


def test_moment_json_carries_label_and_evidence():
    moment = engine.Moment(1.0, 11.0, 5.0, 60, ["loud"], label="combat", label2="gunshot", level="high",
                           label_score=2.1, evidence=[ev("gunfire", 0.7, 3)], explanation="Combat — gunshot")
    data = json.loads(json.dumps(engine.asdict(moment)))
    assert data["label"] == "combat" and data["evidence"][0]["signal"] == "gunfire"
    assert engine.Moment(0, 1, 0, 1, []).label == UNLABELLED  # older code paths keep working


def see(concept, value, seconds=3, detail="a picture"):
    return Evidence("vision", "see:" + concept, value, 0.0, detail, seconds)


def test_events_take_precedence_over_reaction():
    evidence = [ev("reaction", 0.95, 2, "Shout"), see("water", 0.8, 4, "a person jumping into water")]
    label, detail, _s, level = labels.classify(evidence, RULES)
    assert (label, detail) == ("unusual", "water")
    assert level != "low"  # the shout agrees with the picture
    assert labels.classify([ev("reaction", 0.95, 2, "Shout")], RULES)[0] == "reaction"


def test_picture_only_labels_are_capped_at_low():
    label, detail, strength, level = labels.classify([see("vehicle", 1.0, 5)], RULES)
    assert (label, detail, level) == ("vehicle", "", "low")  # and no "Vehicle (vehicle)"
    assert strength == pytest.approx(2.0)


def test_police_detail_and_motion_agreement():
    evidence = [see("vehicle", 1.0, 5), see("police", 0.6, 3), Evidence("motion", "motion", 0.7, 0, "6 px", 10)]
    label, detail, _s, level = labels.classify(evidence, RULES)
    assert (label, detail, level) == ("vehicle", "police", "high")


def test_collect_reads_vision_novelty_and_motion():
    from types import SimpleNamespace

    n = 20
    rel = np.zeros(n)
    rel[5:9] = 0.9
    novelty = np.zeros(n)
    novelty[6] = 0.8
    vis = SimpleNamespace(relative={"water": rel}, novelty=novelty, describe={"water": "a person jumping into water"})
    mcfg = {"floor": 2.0, "span": 6.0}
    found = labels.collect(0, 15, {}, {"loudness": np.zeros(n)}, [], {}, 0.3, vis,
                           {"camera": 1.0, "local": 6.0, "local_peak": 9.0}, mcfg)
    by = {e.signal: e for e in found}
    assert by["see:water"].seconds == 4 and by["see:water"].detail == "a person jumping into water"
    assert "novelty" in by and by["motion"].value == pytest.approx(4 / 6, abs=1e-3)
    calm = labels.collect(0, 15, {}, {"loudness": np.zeros(n)}, [], {}, 0.3, None,
                          {"camera": 1.0, "local": 3.0, "local_peak": 9.0}, mcfg)
    assert not any(e.signal == "motion" for e in calm)  # (3 - 2) / 6 is below 0.5
    assert labels.phrase(by["see:water"]) == "looks like a person jumping into water"


def test_escape_rule_needs_calm_after_and_is_off_by_default():
    n = 40
    curves = {"siren": np.r_[np.ones(20), np.zeros(20)], "action": np.zeros(n)}
    on = {"enabled": True, "calm_s": 8, "calm_below": 0.3}
    assert labels.escape(curves, 19.5, on)
    assert not labels.escape(curves, 10, on)  # the siren is still going after the moment
    assert not labels.escape(curves, 19.5, dict(on, enabled=False))
    assert not labels.escape(curves, 35, on)  # not enough recording left to tell
