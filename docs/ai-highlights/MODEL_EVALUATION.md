# Model evaluation

Tags: **[Verified]** primary source / this repo / measured · **[Secondary]** third-party source only ·
**[Unchecked]** from prior knowledge, not re-checked during this research · **[Assumption]** estimate.

Selection criteria, in order: licence compatible with an MIT desktop app (Apache-2.0 / MIT / BSD);
runs offline on Windows without PyTorch at runtime (ONNX Runtime, CTranslate2 or sherpa-onnx);
fits next to Whisper in 6–8 GB VRAM for ordinary users, comfortably in 16 GB; actual fit for
*gameplay* rather than general benchmarks.

---

## 1. Audio

| Model | Licence | Size | Runtime | Fit for VVideo | Verdict |
|---|---|---|---|---|---|
| **CED-mini** (in use) | Apache-2.0 on HF card [Verified via [HF card](https://www.huggingface.co/mispeech/ced-base/blob/main/README.md)] | 9.6 M params, 49.0 mAP AudioSet [Verified, [paper](https://arxiv.org/pdf/2308.11957)]; ~10 MB int8 file (repo) | sherpa-onnx, CPU (repo) | 527 AudioSet classes, incl. gunfire, explosion, siren, vehicle, skidding, crash, laughter, cheering — most classes needed already exist | **Keep.** Cache the full 527-d vector instead of two groups; add groups |
| CED-base | Apache-2.0 [Verified, same card] | 86 M params, 50.0 mAP [Verified] | ONNX | +1 mAP for ~9× compute | Not worth it |
| PANNs Cnn14 | Official repo licence not confirmed; a CoreML conversion states CC-BY-4.0 [Secondary] | 0.431 mAP [Verified, [paper](https://arxiv.org/pdf/1912.10211)] | PyTorch | Weaker than CED | Reject |
| BEATs | not checked | — | PyTorch | No advantage over CED for tagging | Reject |
| MS-CLAP (audio–text zero-shot) | Conflict: PyPI says MIT, HF card says MS-PL [Secondary, [HF](https://huggingface.co/microsoft/msclap/raw/main/README.md)] | — | PyTorch | Free-text audio queries ("victory jingle") — interesting but redundant with CED for known classes | Later, only if CED classes prove insufficient |
| LAION-CLAP | Not found in research | — | PyTorch | Same as above | Not now |
| **faster-whisper** (in use) | MIT [Unchecked] | large-v3-turbo 1.6 GB / small 0.5 GB (repo log strings) | CTranslate2 CUDA/CPU | Speech, keywords, language | **Keep.** Extend keyword lexicons; nothing else |

**Audio changes recommended (no new model):**
- new CED groups: `vehicle` (Car, Vehicle, Engine, Accelerating/revving, Skidding, Tire squeal),
  `siren` (Siren, Police car (siren), Emergency vehicle), `crash` (Crash, Smash/crash, Breaking, Glass, Shatter),
  split `explosion` from `gunfire`; `music_swell` / `fanfare` as an achievement hint **[Assumption: usefulness]**.
- store per-second 527-d probabilities (≈ 3.8 MB/hour in float16) → features for later classifiers.
- CED stays on CPU: measured audio extraction is 94× realtime and CED is small; moving it to GPU
  would add a sherpa-onnx CUDA build for little gain **[Assumption]**.

---

## 2. Visual — image embeddings / zero-shot

| Model | Licence | Size | Runtime | Notes | Verdict |
|---|---|---|---|---|---|
| **SigLIP 2 base/16 (224 or 256)** | Apache-2.0 [Secondary: Google JAX repos and timm ports tagged apache-2.0, [HF](https://huggingface.co/google/siglip2-large-patch16-512-jax/tree/main), [timm](https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli)] | ~375 M params total incl. text tower [Secondary]; image encoder 352 MB fp32 / 92.8 MB w8a16, text encoder 1.05 GB fp32 [Secondary, Qualcomm AI Hub listing via [HF](https://huggingface.co/qualcomm/SigLIP2)] | ONNX export via `optimum` or `torch.onnx` shown by community repos [Secondary] | Multilingual text tower; sigmoid loss gives independent per-prompt scores (good for multi-label). Strong zero-shot classification and retrieval per [paper](https://arxiv.org/abs/2502.14786v1) | **Prototype default.** Ship image tower only; precompute prompt embeddings |
| SigLIP 2 so400m/16 (256/384) | Apache-2.0 [Secondary] | ~400 M image tower [Unchecked] | ONNX | Better accuracy, ~4–5× compute of base [Assumption] | **Benchmark in M2** as "quality" option |
| SigLIP (v1) | Apache-2.0 [Unchecked] | ONNX exports exist ([deepghs](https://huggingface.co/deepghs/siglip_onnx)) [Secondary] | ONNX | Ready-made ONNX, superseded by v2 | Fallback if v2 export is troublesome |
| OpenAI CLIP ViT-B/16, L/14 | MIT [Unchecked] | — | ONNX | Softmax-trained → scores depend on the prompt set; weaker than SigLIP 2 | Reject |
| OpenCLIP (LAION) | Code MIT; weights per-model [Unchecked] | — | ONNX | Competitive, but licence must be checked per checkpoint | Not needed |
| NVIDIA TAO SigLIP 2 | NVIDIA Open Model License [Secondary, [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/siglip_v2)] | — | TensorRT | Different terms than Google's originals | Avoid; use Google's weights |

**Why SigLIP 2 and not a detector:** the categories we need are *scene-level*
("car chase", "explosion", "menu screen"), which is exactly what image–text embeddings answer. The
same embedding also feeds novelty, duplicate removal and the feedback kNN — one model, four uses.

Prompt strategy: per concept 4–8 prompt variants ("a video game screenshot of …", "gameplay footage
of …", first- and third-person phrasings), averaged embeddings; every concept paired with a neutral
counter-prompt set ("ordinary gameplay, walking") and scored relative to the recording baseline.

---

## 3. Visual — object detection

| Model | Licence | Verdict |
|---|---|---|
| Ultralytics YOLO (v8/11/…) | **AGPL-3.0** or paid Enterprise licence [Verified, [ultralytics.com/license](https://www.ultralytics.com/license)] | **Reject.** Using it would require releasing VVideo under AGPL or buying a licence |
| RF-DETR Nano/Small/Medium/Large | Apache-2.0; Atto/Femto/Pico/XL/2XL detection sizes under Roboflow Platform Model License [Verified, [Roboflow licence page](https://roboflow.com/model-licenses/rf-detr)] | Keep as option for M9 game profiles (e.g. trained kill-feed icon detector). Not needed for the core |
| COCO-class detection in general | — | COCO "car / person / truck" adds little over SigLIP scene scores for highlights, and fails on stylised games [Assumption] |

---

## 4. OCR / HUD text

| Option | Licence | Runtime | Verdict |
|---|---|---|---|
| **RapidOCR + PP-OCRv5 models** | Code Apache-2.0; models Apache-2.0 with Baidu copyright [Secondary, [mirror](https://gitee.com/byteuser/RapidOCR), [HF](https://huggingface.co/monkt/paddleocr-onnx)] | onnxruntime, Windows/Linux/macOS [Secondary] | **Recommended** (M6). Uses ONNX Runtime, no PaddlePaddle install |
| PaddleOCR (native) | Apache-2.0 [Unchecked] | PaddlePaddle framework | Reject for runtime: a second deep-learning framework in the bundle |
| Tesseract | Apache-2.0 [Unchecked] | native exe | Weak on stylised game fonts over busy backgrounds [Assumption] |
| VLM reading text (Qwen3-VL) | see §6 | — | Only inside the optional re-ranker |

Use: **only on candidate windows** (≈ 5 frames per candidate around the peak and the end), full
resolution centre region + top bar. Generic detector: text boxes whose height > 4 % of frame height
in the central 60 % → "big announcement text"; matched against a multilingual lexicon
(`victory, defeat, eliminated, you died, mission passed, wasted, vitória, derrota, eliminado …`).

---

## 5. Temporal / video models

| Model | Licence | Zero-shot? | Verdict |
|---|---|---|---|
| X-CLIP base/32 (8 or 16 frames), base/16 zero-shot | MIT [Verified, [HF card](https://huggingface.co/microsoft/xclip-base-patch32)] | Yes (video–text). base/32 trained on Kinetics-400 with 8 frames at 224×224 [Verified, card]; base/16 zero-shot: 44.6 % HMDB-51, 72.0 % UCF-101, 65.2 % Kinetics-600 top-1 [Secondary, [card](https://huggingface.co/microsoft/xclip-base-patch16-zero-shot/blob/main/README.md)] | **Comparative experiment M3b** (§10). Most relevant prior work: AMD fine-tuned X-CLIP for FPS event detection on 1-s clips ([arXiv 2505.07721](https://arxiv.org/abs/2505.07721)); accuracy figures not confirmed; dataset not public. Domain gap: trained on human-action video, not games |
| V-JEPA 2 | Code MIT, weights Apache-2.0 — both [Secondary only] ([index](https://awesome.ecosyste.ms/projects/github.com%2Ffacebookresearch%2Fvjepa2)) | No (needs trained probe) | Strong video features, large (ViT-L/g); no zero-shot → needs data. Later |
| VideoMAE / V-JEPA 1 | Non-commercial licences [Unchecked] | No | Reject on licence |
| Per-frame SigLIP + temporal pooling (ours) | — | Yes | **Recommended first** — cheapest, reuses embeddings, good enough for scene-level events |

---

## 6. Vision-language models (optional re-ranker / explainer)

| Model | Licence | Size | Verdict |
|---|---|---|---|
| Qwen3-VL 2B / 4B / 8B | Apache-2.0 [Secondary, [report summary](https://alphaxiv.org/overview/2511.21631v2), [Ollama](https://registry.ollama.ai/library/qwen3-vl:2b)] | 2–8 B params; dense; text timestamp tokens for video grounding [Secondary] | **Best candidate for M7**: label + one-line explanation for top 10–30 candidates. Needs a runtime (llama.cpp/GGUF or transformers+torch) — runtime choice and Windows support to be verified |
| Qwen2.5-VL 3B | Qwen research licence for 3B [Unchecked] | — | Avoid |
| Florence-2, SmolVLM2, moondream2 | MIT / Apache-2.0 [Unchecked] | 0.2–2 B | Smaller alternatives if Qwen3-VL is too heavy |
| VideoGameBunny | Licence of data not found [Secondary, [paper](https://arxiv.org/abs/2407.15295)] | LLaVA-style | Interesting (trained on 185k game images, 413 titles) but licence unknown → no |

A VLM is **not** in the first prototype: it is the heaviest dependency, slow per call, and its
value can only be judged once the cheap pipeline is measured.

---

## 7. Runtimes and packaging

### 7.1 What is installed today (investigated in `.venv`, 2026-11-18) [Verified]

| Package | Version | Pulled in by | Why | CUDA/DLL facts |
|---|---|---|---|---|
| `onnxruntime` (CPU) | 1.32.0 | **faster-whisper 1.2.1** (`onnxruntime<2,>=1.14` in its metadata) | Silero VAD: `faster_whisper/vad.py` imports `onnxruntime`; VVideo calls `transcribe(..., vad_filter=True)` and `detect_language(..., vad_filter=True)` | Providers available: `AzureExecutionProvider`, `CPUExecutionProvider` |
| `ctranslate2` | 4.9.0 | faster-whisper | Whisper | Ships `ctranslate2.dll`, **`cudnn64_9.dll`**, `libiomp5md.dll` in its own folder; sees 1 CUDA device |
| `sherpa-onnx` / `-core` | 2.0.1 | `requirements.txt` | CED-mini | Ships **its own `onnxruntime.dll`** plus `sherpa-onnx-c-api.dll` — a second ORT copy, separate from the Python `onnxruntime` package |
| `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-nvrtc-cu12` | 12.9 / 9.28 / 12.9 (per `models.py` URLs) | `requirements-cuda.txt` (source) or downloaded into `VVideo-data/cuda` (frozen) | Whisper on GPU | `cublas64_12`, `cublasLt64_12`, `cudnn*64_9`, `nvrtc64_120_0` — **no `cudart64_12.dll`, no cuFFT/cuRAND** |
| PyTorch | — | not installed | — | `VVideo.spec` explicitly excludes torch/transformers/nvidia |

`VVideo.spec` collects dynamic libs only for `ctranslate2` and `sherpa_onnx`; the Python `onnxruntime`
package is picked up through the import graph (faster-whisper → vad).

### 7.2 Compatibility facts

- CTranslate2 ≥ 4.5 uses cuDNN 9 and needs CUDA ≥ 12.3 [Secondary, [faster-whisper #1086](https://github.com/SYSTRAN/faster-whisper/issues/1086)]; the PyPI listing for 4.8.1 carries a CUDA 12.4 classifier [Secondary, [mirror](https://simple-repository.app.cern.ch/project/ctranslate2)]. 4.9 specifics not found → **verify in M2** (it runs on GPU on the target machine today [Verified: device count 1, project README]).
- ONNX Runtime GPU: 1.21.x–1.26.x PyPI builds use **CUDA 12.8 + cuDNN 9**; from 1.27 the default PyPI `onnxruntime-gpu` is **CUDA 13**; release notes for 1.26 say CUDA 12 publishing ends with 1.27, though the docs' CUDA-12 table lists 1.27–1.30 as well — **contradictory, verify against PyPI files** [Secondary, [ORT CUDA EP docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [install docs](https://onnxruntime.ai/docs/install/)].
- `pip install onnxruntime-gpu[cuda,cudnn]` pulls the CUDA/cuDNN runtime DLLs through pip [Secondary, install docs]. The ORT CUDA EP needs at least the CUDA runtime (`cudart64_12`) in addition to cuBLAS/cuDNN [Unchecked: exact DLL list — derive it from that extra in M2].
- `onnxruntime` and `onnxruntime-gpu` (and `onnxruntime-directml`) **install into the same `onnxruntime/` folder; the last one installed wins**; maintainers recommend installing only the GPU package, which also provides the CPU provider [Secondary, [ORT #10685](https://github.com/microsoft/onnxruntime/issues/10685), [fastembed #608](https://github.com/qdrant/fastembed/issues/608)].

### 7.3 Packaging options

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Replace CPU ORT with `onnxruntime-gpu` 1.26.x (CUDA 12.8)** | Constraints file / post-install step so `faster-whisper`'s `onnxruntime` requirement does not reinstall the CPU wheel; extend the CUDA download list with the ORT-required DLLs | Same CUDA 12 / cuDNN 9 family as CTranslate2 → one set of DLLs; fastest on NVIDIA | Pins ORT below current; pip resolver fights the dependency name; larger `.exe` for users without NVIDIA [Unchecked: wheel size] |
| **B. Replace CPU ORT with `onnxruntime-directml`** | Same replacement trick | Any DX12 GPU (NVIDIA/AMD/Intel), no CUDA DLLs to download | Slower than CUDA EP [Assumption]; DirectML is in maintenance mode upstream [Unchecked] |
| **C. Keep CPU ORT, run vision in a separate worker process** with its own ORT-GPU | Child process, own site-packages | No in-process conflict | Two Python runtimes in the bundle; IPC; much larger build |
| **D. CPU only for vision** | Keep today's packages | Zero packaging work | Viable only if M2 shows CPU SigLIP base is fast enough at 1–2 fps |

**Recommendation:** decide in **M2** by measurement, in this order: try **D** (cost of CPU inference
for the dense pass), then **A** for the NVIDIA path, with **B** as the fallback for non-NVIDIA GPUs.
Whichever is chosen must pass the **clean-environment validation** (§7.4). The second ORT copy inside
sherpa-onnx already coexists with the Python ORT today (both run in one process); whether it still does
with `onnxruntime-gpu` is **unverified** and part of the M2 spike.

### 7.4 Clean Windows environment validation (M2 and M8)

| Environment | What it proves |
|---|---|
| Windows Sandbox or fresh VM, **no NVIDIA GPU** | Frozen `.exe` starts, downloads only what it needs, CPU providers load, all models infer on fixtures |
| Fresh Windows user profile on a machine with an NVIDIA GPU, **driver only** (no CUDA toolkit, no Python) | CUDA DLLs downloaded into `VVideo-data/cuda` suffice for both CTranslate2 and ORT; provider list shows CUDA; parity of outputs vs. CPU |
| Target dev machine | Benchmarks |

Smoke test = a hidden `--self-test` flag (M8) that loads every model, runs one fixture through each,
prints providers, versions and timings, and exits non-zero on any failure.

### 7.5 Other runtimes

| Runtime | Licence | Verdict |
|---|---|---|
| TensorRT EP | — | Not worth it for ≤ 15k images per hour; per-GPU engine builds [Assumption] |
| CTranslate2 | MIT [Unchecked] | Keep (Whisper) |
| sherpa-onnx | Apache-2.0 [Unchecked] | Keep (CED) |
| OpenCV (already installed, `opencv-python` 5.0) | Apache-2.0 [Unchecked] | Use for DIS optical flow and affine fitting — no new dependency |
| PyTorch | BSD | Dev-only for ONNX export and the X-CLIP experiment; never shipped |

---

## 8. Preferred initial stack

| Role | Choice | Why |
|---|---|---|
| Sound events | CED-mini (existing), more groups, full vectors | Already works, CPU, covers gunfire/vehicle/siren/crash/laughter |
| Speech | faster-whisper (existing) | Already works on GPU |
| Image embeddings | **SigLIP 2 base/16-224 image tower (candidate), ONNX** — pinned revision, validated per §9; so400m benchmarked as option | Apache-2.0 [Secondary], zero-shot, multilingual prompts, one model serves concepts + novelty + dedupe + feedback |
| Prompt embeddings | Precomputed offline, shipped as `.npy` with the prompt bank | Avoids shipping the 1 GB text tower; custom prompts become an optional download |
| Semantic novelty | numpy on cached embeddings | No model |
| Motion | OpenCV DIS optical flow + RANSAC affine (camera vs local) on adaptive-pass frames | Already-installed OpenCV |
| Runtime | ONNX Runtime; CPU vs CUDA (1.26.x) vs DirectML decided by M2 measurement | §7.3 |
| Temporal model | None in the stack; X-CLIP evaluated in experiment M3b | §10 |
| OCR (M6) | RapidOCR + PP-OCRv5 on candidates | Apache-2.0, ONNX, no new framework |
| VLM (M7, optional) | Qwen3-VL 2B/4B | Only after measurement |
| Excluded | YOLO (AGPL), VideoMAE/V-JEPA 1 (NC), QVHighlights-trained weights (NC data) | Licence |

Every model entry in `models.py` gets: URL, SHA-256, size, licence id, licence URL. The UI's
first-run download notice lists them.

---

## 9. SigLIP 2 validation checklist and prompt-bank generation

SigLIP 2 stays the initial visual encoder **candidate**. Nothing below has been run yet; it is the
M2 checklist. Items marked [Verified] are documented facts, the rest must be confirmed.

### 9.1 Exact checkpoint

- Candidates: `google/siglip2-base-patch16-224` (default), `-base-patch16-256`, `so400m-patch16-256` (quality).
  Fixed-resolution variants only — the NaFlex variants take variable-size inputs and complicate ONNX export.
- Pin the Hugging Face **revision hash** of the checkpoint, not just the repo id.
- Confirm the licence on that exact repo page (current evidence is secondary, §2).

### 9.2 Tokenizer and text side

- Text length **64**, padding `max_length`, **lowercased** text — the Transformers docs say the model was trained this way and that a standalone Gemma tokenizer must be given `padding="max_length", max_length=64` [Verified, [Transformers SigLIP2 docs](https://huggingface.co/docs/transformers/v4.51.1/model_doc/siglip2)].
- Known pitfall: the checkpoints' `tokenizer_config.json` names `GemmaTokenizer`, which can bypass the
  lowercasing done by `Siglip2Tokenizer` → **lowercase prompts explicitly** in the export tool
  [Secondary, [HF discussion](https://huggingface.co/google/siglip2-base-patch16-224/discussions/22)].
- Probabilities need the model's learned `logit_scale` and `logit_bias` (sigmoid loss) — exported with the embeddings.

### 9.3 Image preprocessing

- Expected (per SigLIP family processors): resize to the model size without crop, rescale 1/255, normalise
  with mean = std = 0.5 [Unchecked — read `preprocessor_config.json` of the pinned revision].
- Reimplemented in numpy at runtime (no `transformers` in the bundle) and checked by a **parity test**:
  numpy preprocessing vs. the HF processor on 20 fixture frames → max abs diff < 1e-3.

### 9.4 ONNX export

- Dev-only `tools/export_siglip.py` (PyTorch + transformers, pinned in a dev requirements file):
  exports **only the vision tower** (pixel_values → pooled embedding), opset ≥ 17, dynamic batch, fp32 and fp16 files.
  Community repos show both `torch.onnx.export` and `optimum-cli` exports of SigLIP 2 [Secondary, [example](https://huggingface.co/AXERA-TECH/siglip2-base-patch16-224/blob/main/python/export_onnx.py)].
- Parity test: ONNX (CPU and GPU provider) vs. PyTorch embeddings, cosine ≥ 0.999 on fixtures; fp16 ONNX
  vs. fp32 cosine ≥ 0.995; zero-shot top-1 identical on fixtures.
- The exported file's SHA-256 becomes the encoder id used by the cache and the prompt bank.

### 9.5 Prompt-bank generation and versioning

```
config/events.json          (human-edited: labels, prompts, counter-prompts, thresholds)
        │  tools/build_prompt_bank.py   (dev-only, torch)
        ▼
config/events.npz           (shipped)
  embeddings[n_prompts, d]   float32, L2-normalised
  prompt_ids[n_prompts]      -> label / variant
  logit_scale, logit_bias
  meta: checkpoint repo + revision, image-encoder ONNX sha256, tokenizer id + file sha256,
        text_len=64, lowercase=true, events.json sha256, tool version, transformers/torch versions
```

- The bank is rebuilt whenever `events.json` or the encoder changes; CI test fails if the `events.json`
  hash in `events.npz` does not match the committed `events.json`.
- Runtime refuses a bank whose encoder SHA-256 differs from the installed encoder (ARCHITECTURE.md §9–10).
- User-defined prompts (later, optional) need the text tower at runtime — a separate optional download.

### 9.6 Not assumed

High image–text similarity is **not** treated as event recognition (ARCHITECTURE.md §6.0). M3 measures
per-label precision of concept-only labels; labels that fail stay as features only.

---

## 10. Experiment M3b — X-CLIP vs. pooled SigLIP frames

**Question:** does a video–text model that sees several frames at once recognise gameplay events
better than per-frame SigLIP scores pooled over time, enough to justify a second model?

| Arm | Input per candidate window | Model |
|---|---|---|
| A | dense-pass frames, mean/max pooled over the 5–10 s event window | SigLIP 2 base |
| B | 8 frames uniformly from the event window (adaptive pass), 224×224 | X-CLIP base/16 zero-shot (`microsoft/xclip-base-patch16-zero-shot`) |
| C | A + B score average | both |

- Same label set, same prompt wording (adapted to each tokenizer: X-CLIP uses the CLIP tokenizer, 77 tokens; processor wraps a VideoMAE image processor + CLIP tokenizer [Secondary, [Transformers X-CLIP docs](https://huggingface.co/docs/transformers/model_doc/xclip)]).
- Data: candidate windows of the **tuning** recordings only; final numbers on held-out.
- Metrics: per-label average precision and precision@fixed recall, per game; inference cost per window.
- Runs in a dev script with PyTorch — **no ONNX export unless arm B or C wins**. If it wins, ONNX export
  of X-CLIP (cross-frame attention, prompt generator) is a separate feasibility task [Unchecked].
- Decision rule: adopt X-CLIP only if C beats A by ≥ 0.05 AP on at least two labels on held-out data,
  without making any label worse by more than 0.02, at < 30 s added per 60-min recording.
