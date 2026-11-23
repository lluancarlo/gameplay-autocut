# Planning changelog

Records architectural decisions and open risks for the AI Gaming Highlights plan. Planning only —
no application code, dependency or model has changed.

---

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
