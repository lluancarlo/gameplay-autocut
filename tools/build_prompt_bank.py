"""Build config/events.npz: the text embeddings of every prompt in config/events.json.

Developer tool, never run by the app. The app ships only SigLIP 2's image
encoder; the text side runs here, once, and its output is versioned:

    python tools/build_prompt_bank.py

Downloads (once, into VVideo-data/dev-hf, not shipped): the ONNX text encoder
(1.1 GB) and the tokenizer, both pinned to the same revision as the image
encoder in src/moments/vision.py. `logit_scale` / `logit_bias` are SigLIP 2's
learned sigmoid parameters, read from google/siglip2-base-patch16-224 at the
pinned revision below (exp(4.7244531) and -16.7717247).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.moments import vision  # noqa: E402

GOOGLE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
LOGIT_SCALE_RAW = 4.724453449249268    # model.safetensors["logit_scale"] at GOOGLE_REVISION
LOGIT_BIAS = -16.771724700927734       # model.safetensors["logit_bias"] at GOOGLE_REVISION
TEXT_LEN = 64
TEXT_FILES = {
    "onnx/text_model.onnx": "baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293",
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
}
EVENTS = ROOT / "config" / "events.json"
BANK = ROOT / "config" / "events.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(vision.MODEL["repo"], name, revision=vision.MODEL["revision"],
                                cache_dir=str(ROOT / "VVideo-data" / "dev-hf")))
    if sha256(path) != TEXT_FILES[name]:
        raise SystemExit(f"{name}: SHA-256 does not match the pinned file")
    return path


def tokenize(tokenizer, texts: list[str]) -> np.ndarray:
    """Lowercased, EOS appended, padded to 64 -- how SigLIP 2's text tower was trained."""
    tokenizer.enable_padding(length=TEXT_LEN, pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(TEXT_LEN)
    ids = np.array([e.ids for e in tokenizer.encode_batch([t.lower() for t in texts])], dtype=np.int64)
    eos = tokenizer.token_to_id("<eos>")
    for row, text in zip(ids, texts):
        if eos not in row:
            raise SystemExit(f"Prompt too long for {TEXT_LEN} tokens: {text!r}")
    return ids


def main() -> int:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    events = json.loads(EVENTS.read_text(encoding="utf-8"))
    prompts, owners = [], []
    for concept, entry in events["concepts"].items():
        for prompt in entry["prompts"]:
            prompts.append(prompt)
            owners.append(concept)
    tokenizer = Tokenizer.from_file(str(fetch("tokenizer.json")))
    ids = tokenize(tokenizer, prompts)
    sess = ort.InferenceSession(str(fetch("onnx/text_model.onnx")), providers=["CPUExecutionProvider"])
    out = sess.run(["pooler_output"], {"input_ids": ids})[0]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    meta = {
        "image_encoder": vision.MODEL,
        "text_encoder": {"repo": vision.MODEL["repo"], "revision": vision.MODEL["revision"], "files": TEXT_FILES},
        "source_checkpoint": {"repo": "google/siglip2-base-patch16-224", "revision": GOOGLE_REVISION},
        "text_len": TEXT_LEN, "lowercase": True,
        "events_json_sha256": sha256(EVENTS),
        "onnxruntime": ort.__version__,
    }
    np.savez(BANK, embeddings=out.astype(np.float32), prompts=np.array(prompts), concepts=np.array(owners),
             kinds=np.array([events["concepts"][c]["kind"] for c in owners]),
             logit_scale=np.float32(math.exp(LOGIT_SCALE_RAW)), logit_bias=np.float32(LOGIT_BIAS),
             meta=json.dumps(meta))
    print(f"{len(prompts)} prompts, {len(events['concepts'])} concepts -> {BANK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
