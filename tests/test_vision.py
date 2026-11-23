import json
import queue
import shutil
import subprocess

import numpy as np
import pytest

from src import config, video_io
from src.moments import concepts, vision


def test_to_pixels_matches_the_siglip_preprocessing():
    frames = np.array([[[[0, 128, 255]]]], dtype=np.uint8)  # 1 frame, 1x1 pixel
    out = vision.to_pixels(frames)
    assert out.shape == (1, 3, 1, 1) and out.dtype == np.float32
    assert out[0, :, 0, 0] == pytest.approx([-1.0, 128 / 127.5 - 1, 1.0], abs=1e-6)


def _video(tmp_path, seconds=2, colour="red"):
    path = tmp_path / "clip.mp4"
    subprocess.run([video_io.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c={colour}:s=320x180:r=30:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return path


def test_read_frames_samples_at_the_requested_rate_and_size(tmp_path):
    out: queue.Queue = queue.Queue()
    vision.read_frames(_video(tmp_path), 10, video_io.Canceller(), out, batch=8)
    batches = []
    while (item := out.get()) is not None:
        assert not isinstance(item, BaseException), item
        batches.append(item)
    frames = np.concatenate(batches)
    assert len(frames) == 20 and frames.shape[1:] == (224, 224, 3)
    assert [len(b) for b in batches] == [8, 8, 4]
    assert frames[:, 112, 112].mean(axis=0) == pytest.approx([254, 0, 0], abs=3)  # red


def test_read_frames_reports_errors(tmp_path):
    out: queue.Queue = queue.Queue()
    vision.read_frames(tmp_path / "missing.mp4", 10, video_io.Canceller(), out, batch=8)
    assert isinstance(out.get(), RuntimeError)


def test_per_second_pooling_averages_and_normalises():
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float16)
    emb = vision.Embeddings(vectors, np.array([0.0, 0.5, 1.2], dtype=np.float32), 2.0, "CPU", 0.0)
    pooled = emb.per_second(3)
    assert pooled[0] == pytest.approx([2 ** -0.5, 2 ** -0.5], abs=1e-3)
    assert pooled[1] == pytest.approx([1, 0]) and pooled[2].tolist() == [0, 0]


def test_scores_are_siglip_sigmoids():
    bank = concepts.Bank(["a", "b"], {"a": "event", "b": "neutral"}, np.eye(2, dtype=np.float32), 10.0, -5.0, {})
    out = concepts.scores(bank, np.array([[1.0, 0.0]]))
    assert out["a"][0] == pytest.approx(1 / (1 + np.exp(-5)))
    assert out["b"][0] == pytest.approx(1 / (1 + np.exp(5)))


def test_shipped_prompt_bank_matches_its_sources():
    bank = concepts.load()
    meta = bank.meta
    assert meta["image_encoder"]["sha256"] == vision.MODEL["sha256"]
    assert meta["lowercase"] is True and meta["text_len"] == 64
    events = json.loads((config.CONFIG_DIR / "events.json").read_text(encoding="utf-8"))
    assert bank.names == list(events["concepts"])
    assert np.allclose(np.linalg.norm(bank.vectors, axis=1), 1, atol=1e-5)


def test_bank_refuses_an_edited_events_json(tmp_path):
    shutil.copy(config.CONFIG_DIR / "events.npz", tmp_path)
    (tmp_path / "events.json").write_text("{}", encoding="utf-8")
    with pytest.raises(concepts.BankMismatch):
        concepts.load(tmp_path)


def test_embed_end_to_end_when_the_model_is_present(tmp_path):
    model = vision.models.models_dir() / vision.MODEL["name"] / "vision_model_fp16.onnx"
    if not model.is_file():
        pytest.skip("vision model not downloaded")
    cfg = {"fps": 5, "cpu_fps": 5, "batch": 4}
    result = vision.embed(_video(tmp_path, seconds=2), 2.0, cfg, log=lambda *_: None,
                          progress=lambda *_: None, cancel=video_io.Canceller())
    assert result.frames == 10 and result.vectors.shape == (10, 768)
    assert np.allclose(np.linalg.norm(result.vectors.astype(np.float32), axis=1), 1, atol=1e-2)
    vision.save(tmp_path, result)
    again = vision.load_cached(tmp_path, cfg)
    assert again is not None and again.frames == 10


def test_decoder_follows_the_measured_policy(monkeypatch, tmp_path):
    cfg = {"decode": "auto", "nvdec_codecs": ["hevc", "av1"]}
    monkeypatch.setattr(vision, "nvidia_present", lambda: True)
    for codec, expected in (("h264", "cpu"), ("hevc", "nvdec"), ("av1", "nvdec")):
        monkeypatch.setattr(vision, "video_codec", lambda _p, c=codec: c)
        assert vision.decoder(tmp_path, cfg) == expected
    monkeypatch.setattr(vision, "nvidia_present", lambda: False)
    assert vision.decoder(tmp_path, cfg) == "cpu"
    assert vision.decoder(tmp_path, dict(cfg, decode="nvdec")) == "nvdec"


def test_video_codec_reads_the_stream(tmp_path):
    assert vision.video_codec(_video(tmp_path)) == "h264"
