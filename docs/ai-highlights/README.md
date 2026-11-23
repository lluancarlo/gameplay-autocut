# AI Gaming Highlights — planning documents

Status: **proposal only.** Nothing here has been implemented, no dependency has been
installed and no model has been downloaded. Written 2026-11-18 against the working
tree of `master` (`fc9a896` plus the uncommitted Moments feature).

| Document | What it answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, data flow, module boundaries, data structures, fusion, temporal analysis, cache, failure handling |
| [MODEL_EVALUATION.md](MODEL_EVALUATION.md) | Audio, vision, OCR, video and VLM candidates: licence, size, hardware, Windows fit, recommendation |
| [DATASET_RESEARCH.md](DATASET_RESEARCH.md) | Public datasets and open-source projects, what is actually usable, and the smallest dataset we must build ourselves |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Milestones with files, dependencies, acceptance criteria, tests and estimates |
| [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) | Measured decode numbers on the target machine, labelled estimates for the rest, CPU/GPU/VRAM strategy |
| [CHANGELOG.md](CHANGELOG.md) | Architectural decisions per revision and unresolved risks |

Conventions used throughout:

- **[Verified]** — checked in this repository, measured on the target machine, or stated by the
  model card / official repository / paper (source linked).
- **[Secondary]** — stated by a third-party source only (mirror, aggregator, blog). Must be re-checked
  against the primary source before shipping.
- **[Unchecked]** — from prior knowledge; not re-checked during this research.
- **[Assumption]** — engineering estimate. No number marked this way has been measured.

---

## Final technical assessment

### 1. Is a genuinely game-agnostic detector feasible with current pretrained models?

**Partly.** It is feasible for *interestingness* and for *coarse, visually or acoustically
distinctive* situations: gunfire, explosions, vehicles at speed, sirens, fire/destruction,
menus/loading screens (to exclude), sudden scene changes, loud reactions. Image–text models
(SigLIP 2) and AudioSet taggers (CED, already in VVideo) know these concepts without knowing the game.

It is **not** feasible zero-shot for events defined by game rules: *who* got the kill, a
*death* vs. a knock, *victory* vs. *defeat*, a *clutch*, an *escape* that succeeded. Those
live in HUD text, kill feeds and round state, and they differ per game. The realistic
generic route for them is OCR of large on-screen text plus a multilingual lexicon; anything
beyond that needs either user feedback (learned per user) or optional game profiles.

### 2. Accuracy to expect without custom training

No measurement exists yet. Ranges below are **[Assumption]** and the first milestone
(M0) exists to replace them with numbers:

| Task | Expected without training |
|---|---|
| Rank "worth watching" moments (precision of top 10) | Moderate. Today's audio detector already does this; vision should mainly remove false positives (menus, loud talking over nothing). |
| Gameplay vs. non-gameplay (menu, lobby, loading, desktop) | Good — a very distinctive visual concept. |
| Coarse categories (combat, vehicle/chase, explosion/destruction) | Fair to good when the category is visible in most frames of the window. |
| Rule-defined events (kill, death, victory/defeat) | Poor without OCR; fair with OCR when the game shows large text; unreliable otherwise. |
| Funny / unexpected / clutch | Poor visually; relies on audio reactions and speech, as today. |

### 3. Models for the first prototype

- **Keep:** CED-mini (sound events) and faster-whisper (speech) — already integrated.
- **Add (candidate, pending M2c validation):** **SigLIP 2 image encoder** (Apache-2.0 per secondary
  sources), run through ONNX Runtime, at 1–2 frames/s plus 4–8 frames/s around candidates, with
  *precomputed, versioned* text-prompt embeddings so the text tower is not shipped.
- **Compare, don't assume:** X-CLIP (MIT) against pooled SigLIP frames in experiment M3b.
- **Add later, on candidates only:** **RapidOCR** with PP-OCRv5 models (Apache-2.0) for large on-screen text.
- **Optional, gated:** a small VLM (Qwen3-VL 2B/4B, Apache-2.0 per secondary sources) as a re-ranker
  and explainer of the top candidates — not in the first prototype.
- **Excluded:** Ultralytics YOLO (AGPL-3.0 — incompatible with an MIT desktop app unless an
  enterprise licence is bought, and COCO classes add little over SigLIP here).

### 4. Biggest technical bottleneck

**Evaluation data, not compute.** There is no public, licence-clean dataset of labelled
highlights for these games (see DATASET_RESEARCH.md). Without a small labelled evaluation set
from real recordings, every tuning decision is guesswork. Second bottleneck: rule-defined
events (kill/death/win/loss) are not visible to general-purpose models.

### 5. Time to a convincing prototype

**~6–7 weeks** for one developer (M0–M3): evaluation harness and first labels, audio quick wins,
runtime/packaging spike, SigLIP validation, dense + adaptive sampling, zero-shot curves and motion
fused into the existing engine. Plus 0.5–1 week for the X-CLIP experiment (M3b).

### 6. Time to a production-quality MVP

**~14–18 weeks** (M0–M8, without the optional VLM): plus review UI for labels and evidence,
learning from user corrections, generic OCR, packaging, VRAM management, CPU-only fallback and tests.
Add ~2 weeks if the VLM re-ranker is approved.

### 7. What can be reused

Nearly the whole Moments engine: audio loading, CED and Whisper stages, the per-second
`curves` dictionary and its weighted fusion, peak picking, windowing, word snapping, overlap
removal, the per-file cache, cancellation, `Span` progress, model download with SHA-256, CUDA DLL
loading (cuBLAS/cuDNN 9 are already downloaded for Whisper and are what ONNX Runtime's CUDA
provider also needs), the review UI (curve, cards, looping player, range bar) and the cut/Shorts
pipeline. The new work plugs in as **additional curves and additional fields on `Moment`**.

### 8. What to implement first

**M0: an evaluation harness and 3–5 hand-labelled recordings** (including The Finals, which
is already on the machine). Then the audio quick wins (M1), because they are cheap and immediately
measurable, and only then vision (M2–M3).

### 9. Most significant technical risks

1. Zero-shot scores are poorly calibrated across games (art style, HUD, colour grading) —
   mitigated by per-recording baselines, as the current `action` signal already does.
2. Rule-defined events (kill, death, victory/defeat) may stay unreliable without game profiles.
3. Packaging: `onnxruntime` (CPU, required by faster-whisper for its VAD) and `onnxruntime-gpu`
   overwrite each other; ORT ≥ 1.27 targets CUDA 13 by default while CTranslate2 uses CUDA 12;
   sherpa-onnx carries a second ORT DLL. Settled by measurement in M2a (CHANGELOG R1–R3).
4. Download size for users: more models on top of the current ~3 GB first-run download.
5. Licensing drift: several licences below are only confirmed by secondary sources.
6. No GTA VI footage is available to test with yet, and WarDogs footage must be supplied.

### 10. What it would take to approach Eklipse.gg

Eklipse says it uses kill-feed activity, audio spikes, chat velocity and engagement cues, with
per-game support ([eklipse.gg/ai-highlights](https://eklipse.gg/ai-highlights)) **[Secondary]**. Matching it means:
per-game kill-feed/HUD detectors for the most popular titles, a large labelled corpus of clips
(they have years of user clips; we have none), and ideally live-stream signals (chat) that a local
recorder does not have. VVideo can compete on *offline, private, free, any game*, and on
learning from the individual user, not on per-game precision across hundreds of titles.

---

## Recommended implementation sequence

1. Commit the current Moments work (it is uncommitted today).
2. M0 — evaluation harness + P1 data (4 The Finals recordings, train/tune/holdout). Freeze the audio-only baseline.
3. M1 — audio quick wins (full CED vectors, vehicle/siren/crash groups, `Moment` gets label/confidence/evidence).
4. M2 — runtime/packaging spike with clean-environment validation, decode benchmark matrix, SigLIP 2 validation, frames + cache.
5. M3 — zero-shot concept curves, non-gameplay filter, semantic novelty, adaptive 4–8 fps pass with camera/local motion,
   temporal rules → **prototype demo** (gate vs baseline). M3b — X-CLIP experiment.
6. M4 — review UI shows labels and evidence, captures *explicit* accept/reject/relabel.
7. M5 — learn from corrections (kNN → logistic, with activation thresholds).
8. M6 — generic OCR on candidates.
9. M8 — productization. M7 (VLM) and M9 (game profiles) only if the numbers justify them.

## Decisions that require human approval

1. **P1 footage:** which 4 The Finals recordings (~3–4 h) form P1 and their split; who supplies WarDogs footage for P2.
2. **New runtime dependency:** permission to run the M2a spike in a throw-away venv, and afterwards the chosen option (CPU-only / `onnxruntime-gpu` 1.26.x / DirectML). Also **dev-only downloads:** SigLIP 2 and X-CLIP checkpoints plus PyTorch in a separate dev environment, for export and the M3b experiment.
3. **Model downloads and sizes** users will see on first run (SigLIP 2 variant choice: base vs. so400m).
4. **Licensing posture:** accept Apache-2.0/MIT only (recommended), exclude AGPL (YOLO) and non-commercial data/weights (QVHighlights, V-JEPA 1, VideoMAE).
5. **PyTorch:** allowed only as a *development* tool for one-off ONNX export, never shipped (recommended), or not at all.
6. **VLM re-ranker (M7):** whether a 2–8 GB optional model is acceptable in a desktop tool.
7. **Labelling effort:** who labels the ~10–15 h of evaluation footage, and whether user corrections may be stored locally for learning.
8. **Taxonomy:** the final list of event labels shown to users (proposal in ARCHITECTURE.md §5).
9. Whether to commit the current uncommitted Moments work before starting.
