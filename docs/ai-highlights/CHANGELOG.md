# Planning changelog

Records architectural decisions and open risks for the AI Gaming Highlights plan. Planning only —
no application code, dependency or model has changed.

---

## Revision 7 — 2026-11-23 — M3 implemented (visual detection)

- `src/moments/visual.py`: per-second concept logits (max over the 10 frames of each second), **relative** curves
  (logit − 120 s rolling median, above `floor`) **gated by presence** (absolute logit ≥ −5 → 1 at −3). Without the
  gate "police cars chasing a car" fired on every GTA drive. Semantic novelty (1 − cos to the previous 10 s) is a
  separate curve. Menu/map/pause detector: (non_gameplay − ordinary) logit margin, AUC 0.914 on the GTA-6 draft
  menu spans (non_gameplay alone 0.840); at margin ≥ 1, precision 0.77 / recall 0.78.
- Fusion: new `vision` curve (strongest event concept, menus removed) at **weight 0.25**; `novelty` at **0** (it lowered
  nDCG@10 on the draft set); score multiplied by (1 − 0.9 × menu); peaks that are ≥ 0.75 sure to be a menu are dropped.
- `src/moments/motion.py`: DIS optical flow on the top 20 moments (decoded grey, 320 px, 10 fps), RANSAC similarity
  model = camera motion, residual = local motion, static-HUD mask. Calibration: the 90th-percentile residual was
  noise (16 px/frame on a parked-car scene); the **median** residual separates — meditation 0.2, driving 0.3, the
  bridge jump 5.6, a The Finals gunfight 7. Evidence only from (local − 2) / 6 ≥ 0.5; never labels on its own.
- Labels: rules are per signal `[min_value, min_seconds]`; new label `unusual` (see:water); **event labels take
  precedence over `reaction`**; **picture-only labels are capped at "low"** unless sound, speech or motion agree;
  `escape` rule implemented, **off** (no reviewed chase data to prove it).
- Bounds: picture-led moments start `lead` (8 s) before the concept's run begins, stepping over dropouts up to
  3 s (the murky underwater frame read as a "menu" for 3 s). Same-label moments that nearly touch are merged;
  same-label look-alikes (cos > 0.95) dropped.
- UI: step 3 curve greys menus/maps/pause; summary says how much was left out; new type 🌀 unusual; explanations
  include "looks like …", "sudden change on screen", "fast movement in the scene".
- Measured on GTA-6: the bridge jump — missed by the audio detector — is now **#1**, "Unusual (water) — looks like a
  person jumping into water, much louder than usual, sudden change on screen", high confidence, 03:57–04:27 (the
  fall is at ~04:00). 1:55 of menus and maps detected; 3 menu peaks dropped. Re-analysis of a cached recording: ~8 s.
- **Gate: not passable yet.** On the single draft-labelled recording, relevance did not move (p@5 40 %, p@10 30 %,
  recall 60 % at every vision weight tried; nDCG@10 0.45 → 0.42; strict recall 0.40 → 0.20 — one moment out of five).
  The draft label for the jump (03:40–04:10, drawn from 5 s contact sheets, including the dare beforehand) is
  matched to the neighbouring "prank" truth instead, which is a label-quality problem as much as a detector one.
  **Decision D35:** keep vision on by default (qualitatively clearly better, no labelled evidence against it) and
  re-run the gate as soon as reviewed labels and a holdout recording exist. Reverting is one config line
  (`weights.vision: 0`).
- M3b (X-CLIP comparison) not run: it is a separate, dev-only experiment.

## Revision 6 — 2026-11-23 — M2 implemented (vision foundation)

Decisions (all measured on the target machine unless marked):

- **D27 — sampling at 10 frames/s** for the whole recording (user decision; plan said 1–2 fps dense + 4–8 fps
  adaptive). Cost measured below; `cpu_fps` = 2 on machines without a usable GPU. Configurable in
  `config/moments.json` → `vision`.
- **D28 — model:** `google/siglip2-base-patch16-224` (Apache-2.0, confirmed from the repo's own metadata), image tower
  only, as the `onnx-community` ONNX export pinned at revision `ba1f3b0`, fp16 file (186 MB, SHA-256 checked).
  **No PyTorch anywhere** — the text tower runs once in the dev tool `tools/build_prompt_bank.py`.
- **D29 — runtime: ONNX Runtime DirectML 1.24.4** replaces the CPU `onnxruntime` wheel (option B of
  MODEL_EVALUATION §7.3). Throughput, batch 64: CPU 22 img/s (fp32) / 18 (fp16); **DirectML 312 (fp32) / 726 (fp16)**;
  CUDA EP (`onnxruntime-gpu` 1.26 with pip CUDA/cuDNN) **failed** to initialise cuDNN and fell back to CPU. DirectML
  needs no CUDA DLLs and works on any DX12 GPU. CED (sherpa-onnx's own ORT), Whisper's VAD (now on the DirectML wheel's
  CPU provider) and the vision model were verified together in one process (risk R3 closed on this machine).
- **D30 — preprocessing:** ffmpeg `format=rgb24,scale=224:224:flags=bilinear` (RGB conversion **before** resizing).
  Against the reference PIL bilinear: pixel MAE 0.07–0.10/255, embedding cos ≥ 0.9993 on 4 GTA-6 frames
  (resizing in YUV first gave 0.976–0.997). DirectML fp16 vs CPU fp32 embeddings: cos 1.0000.
- **D31 — tokenizer:** Gemma `tokenizer.json` pads to 64 and appends EOS, but does **not** lowercase ("A Photo" and
  "a photo" differ) — the bank builder lowercases explicitly. `logit_scale` = exp(4.72445) and `logit_bias` = −16.77172
  read from the Google checkpoint's safetensors header at revision `75de2d5`.
- **D32 — decode policy (M2b):** 10 fps pipeline, 60 s samples, CPU vs NVDEC: H.264 3440×1440p60 12× vs 6× realtime,
  HEVC 5× vs 12×, **AV1 0.8× vs 5.6×**; 1080p: H.264 72× vs 31×, HEVC 37× vs 40×, AV1 4.7× vs 25×. → `decode: auto`
  uses NVDEC for HEVC/AV1 when an NVIDIA GPU is present, the CPU otherwise.
- **D33 — prompt bank** `config/events.json` → `config/events.npz` (24 prompts, 8 concepts), carrying the image encoder's
  SHA-256 and the events.json hash; the app refuses a mismatched bank.
- **D34 — `VVideo.exe --self-test`** writes `VVideo-data/self-test.json` (ffmpeg, ORT providers, a real vision
  inference, prompt bank, sound model, Whisper runtime). Passed on the frozen build on the target machine.

Measured: GTA-6 (15:32, 1080p30 H.264) visual pass 9,318 frames in 19 s (481 frames/s, decode-bound); 60 s of
3440×1440p60 AV1 in 10.5 s, H.264 in 4.8 s. Cache: ~14 MB per 15 min at 10 fps (float16).

Sanity check against the draft GTA-6 labels (per-second concept scores, no tuning, indicative only): bridge-jump
`water` AUC 0.97 — the 5 highest seconds of the whole video are 04:05–04:20, the moment the audio detector missed;
`vehicle` vs driving stretches AUC 0.92; `non_gameplay` vs menu spans AUC 0.83.

Not affecting the moments yet: the embeddings are cached and reported (step 2 log, step 3 summary); M3 fuses them.

Still open: clean-environment validation (no-GPU VM, NVIDIA driver-only profile) — run `VVideo.exe --self-test` there.
New risk **R15**: onnxruntime-directml is pinned at 1.24.4 (DirectML is in maintenance upstream); CUDA EP remains a
fallback to revisit if DirectML stalls.

## Revision 5 — 2026-11-23 — M1 implemented (audio labels)

- CED now keeps all 527 class probabilities per second (`sounds.npz`, float16, ~4 MB/hour); sound groups
  (`reaction`, `gunfire`, `explosion`, `vehicle`, `siren`, `crash`) are defined by AudioSet class name in
  `config/moments.json` and can change without re-listening. A test checks every name exists in the model.
- New relative curves `vehicle`, `siren`, `crash`, at **weight 0** (`engine.fuse` ignores zero-weight curves, so
  relevance is identical to the baseline — verified on the evaluation set and by a unit test). On the draft GTA-6
  labels, weights 0.2–0.3 lowered p@10 (30 → 20 %) and recall (60 → 40 %), so they stay at 0 until reviewed data
  says otherwise.
- Every moment gets `label` (combat / vehicle / destruction / reaction / unlabelled), `label2`, `level`
  (low / medium / high — not a calibrated probability), `label_score`, `evidence` and a one-line `explanation`
  (`src/moments/labels.py`, rules in `config/moments.json`).
- UI: step 3 cards show the type, a **Type** filter with counts, and the explanation with its level under the
  player; step 4 lists each clip's type. `find-moments.py` prints label and explanation.
- Gate (draft labels, 1 recording — indicative only): relevance unchanged; classification on matched moments went
  from 0 % labelled to: reaction P 80 % / R 67 % (n = 6), vehicle 1/1; 25 % left unlabelled.
- Old analysis caches (`sounds.json`) are not reused: the first analysis of a recording after this update re-runs
  the sound model (seconds), Whisper's cache is kept.

## Revision 4 — 2026-11-23 — app restructured as a four-step auto-clipper

- Decision **D26**: the app is one guided flow — 1 Video → 2 Find clips → 3 Pick clips → 4 Make shorts — in
  `src/gui.py` (window, non-clickable step header) and `src/steps.py` (the steps). Steps advance themselves;
  Back / New video / Cancel are the only ways around.
- Step 4 cuts and reframes in one encode (`pipeline.clip_shorts`), so cuts are exact (no keyframe snapping);
  `pipeline.preview_clip` renders the vertical preview; framing is adjustable in a dialog with live preview.
- The "Make shorts" folder converter and the "Moments" tab are gone from the window (`make-short.py` keeps the
  converter on the command line).
- D25 refined: **user-facing** features ship with UI in the app; evaluation is **developer tooling** and stays out
  of the app and the `.exe` (`python tools/evaluate.py --gui` opens its window, now `src/evaluation/window.py`).

## Revision 3 — 2026-11-23 — M0 implemented

- Decision **D25**: every feature ships with UI in the app; the user tests only through the window.
- Evaluation code lives in the app package `src/evaluation/` (`dataset.py`, `metrics.py`, `runner.py`) and is shared by
  the new **Evaluation tab** (`src/evaluation_tab.py`) and `tools/evaluate.py`. Also `tests/test_evaluate.py`,
  `eval/splits.json`, `eval/labels/`, `docs/ai-highlights/labelling-guide.md`.
- Evaluation folder: repo `eval/` from source, `VVideo-data\eval` in the `.exe`, selectable in the tab; relative
  recording paths resolve against the folder's parent.
- Baseline frozen in `eval/baseline/` from the current audio-only detector on `GTA-6.mp4` (example recording,
  split `tune`, **draft** labels written from 5-s contact sheets + transcript, not watched with sound).
- Reproducibility checked: re-scoring saved predictions and a full re-analysis without cache give identical metrics.
- Finding: on this recording the "speech" and "keywords" signals fire on **game dialogue** (single mixed audio track,
  no player microphone), so the detector ranks NPC arguments first and misses the best visual moment (bridge jump).
  New risk **R14**: recordings without a separate mic track turn the reaction/keyword signals into dialogue detectors.

## Revision 2 — 2026-11-18 — architecture review corrections

### Decisions

| # | Decision | Rationale | Where |
|---|---|---|---|
| D1 | SigLIP 2 remains the visual encoder **candidate**, not a settled choice; adoption requires the M2c validation checklist (pinned revision, tokenizer: Gemma, 64 tokens, `max_length` padding, explicit lowercasing; preprocessing parity; ONNX parity) | Tokenizer config of the checkpoints can bypass lowercasing; preprocessing/export errors produce plausible but wrong scores | MODEL_EVALUATION §9 |
| D2 | Precomputed text embeddings are a **versioned artefact** (`config/events.npz`) carrying checkpoint revision, encoder SHA-256, tokenizer hash, `logit_scale`/`logit_bias`, prompt-file hash; runtime refuses mismatches | Silent drift between image encoder and text embeddings is undetectable from the output | MODEL_EVALUATION §9.5, ARCHITECTURE §9 |
| D3 | Image–text similarity is a **feature, not event recognition**; labels need temporal persistence plus a second modality or OCR, and abstain otherwise | A still frame cannot show what happened | ARCHITECTURE §6.0 |
| D4 | New experiment **M3b: X-CLIP vs pooled SigLIP**, dev-only, with an explicit adoption rule | Test temporal video–text modelling instead of assuming either way | MODEL_EVALUATION §10, IMPLEMENTATION M3b |
| D5 | Sampling = dense 1–2 fps over the whole file + **adaptive 4–8 fps** inside top-N candidate windows | Fine temporal detail where it matters at a cost independent of recording length | ARCHITECTURE §6.1 |
| D6 | **Semantic novelty** (embedding change) and **motion** (optical flow) are separate modules and curves | They answer different questions and fail differently | ARCHITECTURE §6.2 |
| D7 | Motion split into **camera vs local** (RANSAC affine on DIS flow) with a static-HUD mask; motion never labels alone | Player camera movement dominates raw flow in FPS/TPS games | ARCHITECTURE §6.3 |
| D8 | Temporal context only through **named rules with visible inputs**, auto-disabled below precision 0.6 / 10 occurrences on `tune` | Avoid narrative assumptions such as "calm after a chase = escape" without evidence | ARCHITECTURE §6.4 |
| D9 | Separate metric sets for **relevance** (precision@k, nDCG@k, temporal recall) and **classification** (per-label/per-game P/R/F1 on matched moments, abstention), plus **calibration** (ECE, Brier, reliability) | Ranking quality and label correctness are different failures | DATASET_RESEARCH §5 |
| D10 | Splits by **whole recording**: train / tune / holdout, fixed in `eval/splits.json`; holdout only at gates | Prevent temporal leakage and tuning on test data | DATASET_RESEARCH §4.2 |
| D11 | Relevance ratings 0–3; graded via nDCG, binary at ≥ 2; second annotator on ≥ 1 h per game with weighted κ | Relevance is subjective and must be measured as such | DATASET_RESEARCH §4.3–4.5, §5.1 |
| D12 | **No per-label calibration below 30/30 examples**; shared calibrator up to 100/100; otherwise low/medium/high levels | Small-sample calibrators overfit | ARCHITECTURE §7 |
| D13 | Every detection change must beat the **frozen audio-only baseline on holdout** (bootstrap CI over recordings, or consistent across all holdout games while holdout is small) | Require measurable improvement, not anecdotes | DATASET_RESEARCH §5.4, IMPLEMENTATION gates |
| D14 | Dataset collection is **incremental (P1–P4)**: The Finals first (4 recordings, ~3–4 h), then WarDogs, a driving stand-in, then GTA VI when footage exists — reaching 10–15 h by P3 | Reconciles "3–5 recordings" with "10–15 h"; keeps all targets in scope | DATASET_RESEARCH §4.4 |
| D15 | Three label kinds — **evaluation**, **training**, **weak** — never mixed; weak labels never used as negatives | Prevent contamination of metrics | DATASET_RESEARCH §4.1 |
| D16 | **Unreviewed or un-cut moments are never negatives**; only explicit 👎 / "wrong label" (the latter only for that label) | Absence of action is not a judgement | DATASET_RESEARCH §4.1, IMPLEMENTATION M4 |
| D17 | kNN activates at ≥ 5 positives + 5 explicit negatives per label (≥ 2 recordings); logistic at ≥ 30 + 30 (≥ 3 recordings); bounded adjustment | Defined minimums; small evidence cannot override strong signals | DATASET_RESEARCH §6.2 |
| D18 | Personalisation evaluated by leave-one-recording-out and held-out learning curves | Prove gains on recordings whose feedback was not used | DATASET_RESEARCH §6.3 |
| D19 | Feedback and feature vectors stay **local**; the "opt-in sharing" idea from revision 1 is removed | Privacy; no upload path | ARCHITECTURE §9, DATASET_RESEARCH §6.1 |
| D20 | ONNX Runtime source identified: CPU `onnxruntime` 1.32 is required by **faster-whisper** (Silero VAD, used via `vad_filter=True`); sherpa-onnx carries its own `onnxruntime.dll`; CTranslate2 4.9 carries `cudnn64_9.dll` | Needed before any packaging decision | MODEL_EVALUATION §7.1 |
| D21 | Packaging decided by measurement in **M2a** in the order D (CPU only) → A (`onnxruntime-gpu` 1.26.x, CUDA 12.8, replacing the CPU wheel) → B (DirectML) | CPU/GPU ORT wheels overwrite each other; ORT ≥ 1.27 targets CUDA 13 while CTranslate2 is on CUDA 12 | MODEL_EVALUATION §7.2–7.3 |
| D22 | **Clean Windows environment validation** (no-GPU VM; driver-only NVIDIA profile) is an acceptance criterion in M2a and M8; `--self-test` flag planned | The dev machine hides missing-DLL problems | MODEL_EVALUATION §7.4 |
| D23 | Decode default is **CPU unless measured otherwise**; M2b benchmarks CPU vs NVDEC across H.264/HEVC/AV1 and resolutions | Measured: CPU 16–86× realtime vs NVDEC 5.6–30× on H.264 here | PERFORMANCE §1, IMPLEMENTATION M2b |
| D24 | Estimates revised: prototype ~6–7 weeks (was 5–6), MVP ~14–18 weeks (was 13–17) | Runtime spike, validation, adaptive pass and evaluation rigour added | IMPLEMENTATION Totals |

### Unresolved risks

| Risk | Impact | Mitigation / when resolved |
|---|---|---|
| R1 ORT CUDA-12 availability: docs contradict each other on CUDA-12 builds after 1.26 | Could force CUDA 13 next to CTranslate2's CUDA 12, or pin an ageing ORT | M2a: check PyPI files; fall back to D or B |
| R2 `onnxruntime-gpu` replacing the CPU wheel while faster-whisper declares `onnxruntime` | pip may reinstall the CPU wheel and silently remove CUDA EP | Constraints/post-install step; startup check that CUDA EP is listed when expected |
| R3 Two ORT copies in one process (sherpa-onnx's DLL + Python ORT-GPU) | DLL clash or crash on load | M2a test; fallback B or C |
| R4 SigLIP 2 licence confirmed only via secondary sources | Shipping risk | Check exact pinned repo before M2c completes |
| R5 Zero-shot concept scores may not transfer across art styles | Labels stay at low precision | Relative scoring, abstention, auto-demotion of failing labels |
| R6 Rule-defined events (kill/death/victory) may remain unreliable without profiles | Reduced feature parity with Eklipse | OCR (M6), optional profiles (M9) |
| R7 Camera-motion separation fails in parallax-heavy scenes | Noisy motion curve | Motion never labels alone; measured per label in M3 |
| R8 Small holdout (P1: 1 recording) gives wide CIs | Gates may be inconclusive early | Consistency-across-games rule; grow holdout in P2–P3 |
| R9 Annotation effort (20–30 person-hours) and rater subjectivity | Delays; noisy ground truth | Phased collection; second annotator; κ reported |
| R10 No GTA VI footage yet; WarDogs footage must be supplied | Target-game claims untested | P2/P4 phases; holdout-first for unseen games |
| R11 X-CLIP ONNX export feasibility unknown | Even a winning experiment may be costly to ship | Only attempted if M3b decision rule passes |
| R12 HEVC/AV1 recordings not yet benchmarked | Decode policy may differ by codec | M2b matrix |
| R13 Download size grows beyond today's ~3 GB first run | User friction | Base encoder default; optional models; sizes shown before download |

---

## Revision 1 — 2026-11-18 — initial plan

Initial five documents: architecture, model evaluation, dataset research, implementation plan,
performance plan; measured decode throughput on the target machine.
