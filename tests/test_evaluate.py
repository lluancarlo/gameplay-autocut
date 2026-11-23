import json
from pathlib import Path

import pytest

from src.evaluation import metrics as ek
from src.evaluation.dataset import EvalSet, merge_spans, subtract_span

ROOT = Path(__file__).resolve().parent.parent


def rec(truths, reviewed=((0, 600),), non_gameplay=()):
    return ek.Recording("r", "game", "tune", "x.mp4", "reviewed", 600.0,
                        list(reviewed), list(non_gameplay), truths)


def pred(start, end, score, peak=None, label=ek.UNLABELLED, confidence=None):
    return ek.Pred(start, end, (start + end) / 2 if peak is None else peak, score, label, confidence)


def test_iou_and_coverage():
    assert ek.iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert ek.iou((0, 10), (20, 30)) == 0.0
    assert ek.covered((0, 10), [(0, 5)]) == 0.5


def test_match_is_one_to_one_and_greedy_by_score():
    truths = [ek.Truth(100, 120, 3)]
    preds = [pred(95, 115, 50), pred(100, 120, 90)]
    assert ek.match(preds, truths) == {1: 0}  # higher score wins, the other stays unmatched


def test_peak_inside_truth_counts_as_match_even_with_low_iou():
    truths = [ek.Truth(100, 104, 2)]
    preds = [pred(90, 130, 10, peak=102)]
    assert ek.match(preds, truths) == {0: 0}
    assert ek.match(preds, truths, peak_counts=False) == {}


def test_precision_at_k_ignores_rating_one_matches():
    truths = [ek.Truth(10, 20, 3), ek.Truth(100, 110, 1), ek.Truth(200, 210, 2)]
    preds = [pred(10, 20, 90), pred(100, 110, 80), pred(300, 310, 70), pred(200, 210, 60)]
    out = ek.relevance(rec(truths), preds, ks=(2, 3))
    # rating-1 hit is dropped from the list: [hit, miss, hit]
    assert out["p@2"] == 0.5
    assert out["p@3"] == pytest.approx(2 / 3)
    assert out["recall"] == 1.0


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_otherwise():
    truths = [ek.Truth(10, 20, 3), ek.Truth(100, 110, 2)]
    perfect = ek.relevance(rec(truths), [pred(10, 20, 90), pred(100, 110, 80)], ks=(2,))
    swapped = ek.relevance(rec(truths), [pred(10, 20, 80), pred(100, 110, 90)], ks=(2,))
    assert perfect["ndcg@2"] == pytest.approx(1.0)
    assert swapped["ndcg@2"] < 1.0


def test_unreviewed_time_is_out_of_scope():
    truths = [ek.Truth(10, 20, 3), ek.Truth(500, 510, 3)]
    out = ek.relevance(rec(truths, reviewed=[(0, 300)]), [pred(10, 20, 90), pred(500, 510, 80)], ks=(1,))
    assert out["n_relevant"] == 1 and out["n_pred"] == 1 and out["p@1"] == 1.0


def test_no_relevant_truth_gives_none_not_zero():
    out = ek.relevance(rec([ek.Truth(10, 20, 1)]), [pred(10, 20, 90)], ks=(5,))
    assert out["p@5"] is None and out["recall"] is None


def test_non_gameplay_false_positives_per_hour():
    out = ek.relevance(rec([], reviewed=[(0, 3600)], non_gameplay=[(0, 60)]), [pred(10, 20, 50), pred(500, 510, 50)])
    assert out["non_gameplay_fp_per_h"] == 1.0


def test_classification_counts_only_matched_labelled_pairs():
    truths = [ek.Truth(10, 20, 3, "combat"), ek.Truth(100, 110, 2, "vehicle"), ek.Truth(200, 210, 2)]
    preds = [pred(10, 20, 90, label="combat"), pred(100, 110, 80, label="combat"),
             pred(200, 210, 70, label="vehicle"), pred(400, 410, 60, label="vehicle")]
    pairs = ek.classification(rec(truths), preds)["pairs"]
    assert sorted(pairs) == [("combat", "combat"), ("combat", "vehicle")]
    table = ek.label_table(pairs)
    assert table["combat"]["precision"] == 0.5 and table["combat"]["recall"] == 1.0
    assert table["vehicle"]["recall"] == 0.0
    assert ek.abstention([("unlabelled", "combat"), ("combat", "combat")]) == 0.5


def test_calibration_metrics():
    perfect = [(1.0, True), (0.0, False)]
    assert ek.ece(perfect) == 0.0 and ek.brier(perfect) == 0.0
    over = [(0.9, False)] * 10
    assert ek.ece(over) == pytest.approx(0.9) and ek.brier(over) == pytest.approx(0.81)
    assert ek.ece([]) is None


def test_bootstrap_needs_two_recordings_and_is_deterministic():
    assert ek.bootstrap_ci([0.5]) is None
    a = ek.bootstrap_ci([0.2, 0.4, 0.6, 0.8])
    assert a == ek.bootstrap_ci([0.2, 0.4, 0.6, 0.8]) and a[0] <= 0.5 <= a[1]


def test_committed_eval_files_are_valid():
    es = EvalSet(ROOT / "eval")
    for rec_id in es.entries():
        r = es.recording(rec_id)
        assert all(s < e for s, e in r.reviewed + r.non_gameplay)
        assert r.status in ("draft", "reviewed")
        assert Path(r.path).name  # resolves against the repo root


def test_invalid_rating_is_rejected(tmp_path):
    es = EvalSet(tmp_path)
    with pytest.raises(ValueError):
        es.write_labels("x", {"duration": 10, "reviewed_spans": [[0, 10]],
                              "moments": [{"start": 1, "end": 2, "rating": 5}]})


def test_add_recording_creates_entry_and_draft_labels(tmp_path):
    video = tmp_path / "clips" / "My Session.mp4"
    video.parent.mkdir()
    video.write_bytes(b"")
    es = EvalSet(tmp_path / "eval")
    rec_id = es.add_recording(video, "My Session", "The Finals", "tune", 600.0, voice_track=1)
    assert rec_id == "my-session"
    entry = es.entries()[rec_id]
    assert entry == {"path": "clips/My Session.mp4", "game": "the-finals", "split": "tune", "voice_track": 1}
    assert es.video(entry) == video
    assert es.read_labels(rec_id)["status"] == "draft"
    with pytest.raises(ValueError):
        es.add_recording(video, "my session", "x", "tune", 1.0)
    with pytest.raises(ValueError):
        es.add_recording(video, "other", "x", "validation", 1.0)


def test_write_labels_sorts_and_merges(tmp_path):
    es = EvalSet(tmp_path)
    data = {"duration": 100, "status": "draft", "reviewed_spans": [[50, 80], [0, 60]], "non_gameplay": [],
            "moments": [{"start": 30, "end": 40, "rating": 2}, {"start": 5, "end": 9, "rating": 1}]}
    es.write_labels("r", data)
    back = es.read_labels("r")
    assert back["reviewed_spans"] == [[0, 80]]
    assert [x["start"] for x in back["moments"]] == [5, 30]


def test_span_helpers():
    assert merge_spans([[5, 8], [0, 2], [1, 3]]) == [[0, 3], [5, 8]]
    assert subtract_span([[0, 10]], 3, 4) == [[0, 3], [4, 10]]
    assert subtract_span([[0, 10]], -1, 20) == []


def test_select_skips_drafts_unless_asked(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    es = EvalSet(tmp_path / "eval")
    es.add_recording(video, "a", "g", "holdout", 10.0)
    assert es.select("holdout", False) == ([], [es.recording("a")])
    assert [r.id for r in es.select("all", True)[0]] == ["a"]
    assert es.select("tune", True) == ([], [])


def test_report_from_saved_baseline_links_times(tmp_path):
    from src.evaluation import runner

    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    es = EvalSet(tmp_path / "eval")
    es.add_recording(video, "a", "g", "tune", 120.0)
    labels = es.read_labels("a")
    labels.update(reviewed_spans=[[0, 120]], moments=[{"start": 10, "end": 20, "rating": 3, "label": "combat"},
                                                     {"start": 60, "end": 70, "rating": 2}])
    es.write_labels("a", labels)
    es.baseline_dir.mkdir(parents=True)
    (es.baseline_dir / "a.json").write_text(json.dumps(
        {"moments": [{"start": 9, "end": 21, "peak": 15, "score": 80, "reasons": ["loud"]}]}))
    with pytest.raises(ValueError):
        runner.evaluate(es, "tune", include_drafts=False, run_new=False)
    text, source = runner.evaluate(es, "tune", include_drafts=True, run_new=False)
    assert source == es.baseline_dir
    assert "| p@5 | 20% |" in text  # 1 hit in the top 5
    assert "(vvideo://seek/a/9.00)" in text and "Missed highlights" in text
    assert (es.folder / "runs" / "latest-report.md").is_file()
