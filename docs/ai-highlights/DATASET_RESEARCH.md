# Dataset research

Tags as in [README.md](README.md). Key question for each item: can it help VVideo — an MIT-licensed,
offline desktop app — **train, tune or evaluate** detection for games like WarDogs, The Finals and GTA VI?

## Conclusion first

**No public dataset fits.** The relevant ones are either non-commercial, of unclear licence,
built from other people's Reddit/YouTube/Twitch videos, about other games (LoL, Dota 2, CS:GO), or
not released. The practical path: **pretrained zero-shot models + a small, incrementally grown,
self-labelled evaluation set from our own recordings + learning from explicit user corrections.**
Public datasets serve as *inspiration for taxonomies and prompts* and, where licence permits, for
internal sanity checks, never as training data for shipped weights.

---

## 1. Highlight / moment datasets

| Dataset | Content | Licence / availability | Relevance | Use |
|---|---|---|---|---|
| QVHighlights (Moment-DETR) | 10k+ YouTube vlog/news videos, query-based moments and saliency | Annotations **CC BY-NC-SA 4.0**, "research purposes only"; code MIT [Verified, [repo](https://github.com/jayleicn/moment_detr), [paper](https://arxiv.org/pdf/2107.09609)] | Not gaming; query-driven | **No** (non-commercial; weights trained on it inherit the problem) |
| Fu et al. 2017, LoL highlights from chat | LoL broadcasts (NALCS 218, LMS 103 videos), frames labelled highlight/not via community highlights | Release stated in abstract, not confirmed [Secondary, [paper](https://aclanthology.org/D17-1102/)] | Esports broadcast, chat-driven | No (other game, chat signal we don't have) |
| LIGHTOR | Twitch chat + play data for Dota 2 and LoL, crowdsourced labels | Data said to be in their GitHub repo [Secondary, [arXiv 1910.12201](https://arxiv.org/pdf/1910.12201)]; licence not checked | Chat-based | No |
| AMD "Gameplay Highlights Generation" | 5 FPS games (CS:GO, Valorant, PUBG, Fortnite, OW2 per summaries), 7 event types, 1-s clips, ~110 GB | **In-house, not public** [Secondary, [arXiv 2505.07721](https://arxiv.org/abs/2505.07721)] | **Highest** — same problem, fine-tuned X-CLIP, tested on an unseen game (Apex) | Method reference only: event taxonomy, 1-s clip framing, cross-game training result |
| Yahoo esports highlights | Visual-effect-based highlight detection on 3 titles | [Secondary, [arXiv 1611.08780](https://arxiv.org/pdf/1611.08780v1.pdf)] | Broadcast overlays | No |
| TVSum / SumMe / YouTube Highlights | Generic video summarisation | Various, mostly research use [Unchecked] | Not gaming | No |

## 2. Gameplay vision datasets

| Dataset | Content | Licence / availability | Relevance | Use |
|---|---|---|---|---|
| GamePhysics (CLIP meets GamePhysics) | 26,954 Reddit r/GamePhysics videos, 1,873 games | **Conflicting:** HF card says CreativeML OpenRAIL-M, loader script says MIT; underlying videos belong to Reddit posters [Secondary, [HF](https://huggingface.co/datasets/asgaardlab/GamePhysics/raw/main/README.md), [paper](https://arxiv.org/abs/2203.11096v2)] | **Methodologically very relevant:** CLIP zero-shot retrieval of physics events across 1,873 games | Method + prompt ideas; internal evaluation of `unusual` only after licence check |
| GlitchBench | ~593 glitch frames + glitch-free frames, 205 games | **MIT** on HF [Secondary, [HF](https://huggingface.co/datasets/glitchbench/GlitchBench/blob/main/README.md)]; frames are from commercial games | Glitch detection; best LMM (GPT-4V) got 43.4 % [Secondary, [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Taesiri_GlitchBench_Can_Large_Multimodal_Models_Detect_Video_Game_Glitches_CVPR_2024_paper.html)] | Sanity test for `unusual`; also a warning that large models are weak at "something is off" |
| VideoGameBunny data | 185,259 images, 413 titles, 389k instruction pairs | Licence **not found** [Secondary, [arXiv 2407.15295](https://arxiv.org/abs/2407.15295)] | Game-image understanding | No until licence is clear |
| Kinetics-700 | Human-action YouTube clips, 700 classes | Annotations CC BY 4.0, videos remain uploaders' [Secondary, [arXiv 1907.06987](https://arxiv.org/abs/1907.06987v2)] | Real-world actions; X-CLIP was trained on Kinetics-400 | Only indirectly (pretrained models) |

## 3. Open-source projects

| Project | Approach | Licence | Takeaway |
|---|---|---|---|
| EZClips ([github](https://github.com/corp14x3/EZClips)) | Kill-feed template matching + edge detection, ROI settings, merges consecutive kills | Not checked | Kill detection in practice = per-game HUD templates → optional profiles |
| WatchDog ([site](https://thrtn70.github.io/WatchDog/)) | Windows tray app; game audio; native detection for CS2/Valorant/OW2/Siege + audio for others | GPL-2.0 [Secondary] | Same audio-first idea as VVideo today; GPL → do not copy code |
| montage_maker ([github](https://github.com/rlmsinclair/montage_maker)) | Kill detection with game + in-game name | Not checked | Per-game |
| CLIPxGamePhysics ([github](https://www.github.com/asgaardlab/CLIPxGamePhysics)) | CLIP zero-shot event retrieval in gameplay video | Not checked | Closest to our zero-shot approach |
| Commercial: Eklipse, FragCut | Eklipse: kill feed, audio spikes, chat velocity, engagement [Secondary, [eklipse.gg](https://eklipse.gg/ai-highlights)]; FragCut: CLIP visual + Gemini audio [Secondary] | Closed | Leaders combine per-game HUD detection with audio |

---

## 4. Our own data

### 4.1 Three kinds of labels — never mixed

| Kind | Source | Used for | Never used for |
|---|---|---|---|
| **Evaluation labels** | Annotators following the protocol (§4.3), on recordings assigned to the *tuning* or *held-out* split | Metrics, thresholds, calibration (tuning split only), go/no-go decisions | Training feedback heads |
| **Training examples** | **Explicit** user decisions in the app: 👍 accept, 👎 reject, label changed, bounds edited; plus annotator labels on *training* recordings | kNN prototypes, logistic heads (M5) | Reporting accuracy |
| **Weak labels** | Automatic: high-confidence CED events, keywords, OCR matches | Prompt selection, sanity checks, candidate pre-fill for annotators | Evaluation; **never as negatives** |

**Unreviewed ≠ negative.** A moment the user never looked at, or looked at and did not cut, is
*unknown*. Only an explicit 👎 / "not a highlight" / "wrong label" counts as a negative, and a
"wrong label" is a negative for that label only, not for relevance. A cut moment is a positive for
relevance; its label is unconfirmed unless the user saw and kept it.

### 4.2 Splits — by recording, fixed in advance

Splits are by **whole recording** (never seconds of one recording in two splits — neighbouring seconds
are near-duplicates and would leak). Each recording gets a split when it is added, recorded in
`eval/splits.json`, and never moves:

| Split | Purpose | Rule |
|---|---|---|
| `train` | Source of training examples for M5 experiments | May be looked at freely |
| `tune` | Thresholds, weights, prompt choices, rule enable/disable, calibration fitting | Looked at during development |
| `holdout` | Final numbers only | Run **only** at milestone gates; results logged with the commit hash; no tuning after looking |

### 4.3 Annotation protocol

- **Exhaustive within a reviewed span.** Evaluation recordings are watched end to end (at up to 2×) so
  that recall can be computed. Each file stores `reviewed_spans`; anything outside is excluded from metrics.
- Pre-fill is allowed (the current detector's moments shown as suggestions), but the annotator must also
  add missed moments — pre-fill alone would bias recall.
- Per moment: `start`, `end`, `label` (taxonomy level 1, optional level 2), `relevance` rating
  **0 = not a highlight, 1 = mildly interesting, 2 = good, 3 = must-clip**, optional free-text note.
- Guidelines file with 2–3 example clips per rating level per game, written before labelling starts.

### 4.4 Incremental collection plan

The earlier proposal (3–5 recordings for M0 vs. 10–15 h overall) is reconciled as **phases**; each
phase is gated on the previous one being complete and used.

| Phase | When | Footage | Split assignment | Unlocks |
|---|---|---|---|---|
| **P1** | M0 | **The Finals**, 4 recordings, ~3–4 h total | 1 train · 2 tune · 1 holdout | Baseline of the current audio detector; M1–M3 development |
| **P2** | during M3 | + **WarDogs** 2–3 recordings (~2–3 h) + 1 more The Finals | WarDogs: 1 tune · 1–2 holdout | First cross-game numbers |
| **P3** | M4–M5 | + 1 open-world driving game as GTA-style stand-in (~2 h) + more of both games to reach **10–15 h** | ≥ 30 % of hours in holdout, every game present in holdout | Per-game/per-category numbers with usable counts; M5 personalisation tests |
| **P4** | when available | **GTA VI** footage | holdout first (tests generalisation to an unseen game), then tune | Confirms game-agnostic claim on a target game |

The 71-s `The Finals.mp4` on the machine is too short for evaluation; it is a fixture for tests.
Effort **[Assumption]**: ~1.5–2× footage length for exhaustive review → P1 ≈ 6–8 person-hours, all phases ≈ 20–30.

### 4.5 Second annotator

Relevance is subjective. On ≥ 1 hour per game, a second person labels independently; agreement is
reported (§5.1). If agreement is low, metrics use only moments both rate ≥ 2 as "highlights", and the
spread is reported rather than hidden.

---

## 5. Evaluation methodology

Two separate questions, two separate metric sets.

### 5.1 Highlight relevance — "did we find the moments worth clipping?"

| Metric | Definition |
|---|---|
| Precision@k (k = 5, 10, 20) | Share of the top-k predicted moments matching a ground-truth moment with rating ≥ 2 |
| nDCG@k | Uses the 0–3 ratings as graded gain — rewards ranking must-clips above mildly-interesting |
| Temporal recall | Share of ground-truth moments (rating ≥ 2) matched by any prediction |
| Matching | A prediction matches if temporal IoU ≥ 0.3 **or** its peak lies inside the GT interval; one-to-one, greedy by score. IoU ≥ 0.5 reported as a stricter secondary number |
| Non-gameplay false positives | Predictions inside annotated non-gameplay spans, per hour |
| Rater agreement | Weighted Cohen's κ on ratings of jointly reviewed spans (when §4.5 applies) |

Ratings are handled as **graded relevance** (nDCG) and as a **binary threshold at ≥ 2** (precision/recall);
rating-1 moments are neither rewarded nor penalised in the binary metrics.

### 5.2 Event classification — "is the label right?"

Evaluated **only on predictions that matched a ground-truth moment** (so localisation errors do not count twice).

| Metric | Definition |
|---|---|
| Per-label precision / recall / F1 | Level-1 labels; level-2 separately where ≥ 10 GT examples exist |
| Per-game breakdown | Same table per game; never only pooled |
| Confusion matrix | Level-1 labels + `unlabelled` |
| Abstention | Share of matched moments left `unlabelled` — a label system that abstains is preferred to one that guesses |
| Counts | Every cell reports *n*; cells with n < 10 are shown greyed and excluded from decisions |

### 5.3 Confidence calibration

- **Expected calibration error** (10 bins), **Brier score**, and a reliability diagram per level-1 label
  with ≥ 30 positives, on held-out data.
- Calibrators are fitted **only on `tune`** and only when the data thresholds in ARCHITECTURE.md §7 are met;
  below them, the system shows low/medium/high levels and reports ranking metrics (AP) only.

### 5.4 Uncertainty and the improvement requirement

- 95 % confidence intervals by **bootstrap over recordings** (not over seconds).
- **Every detection change must beat the existing audio-only baseline** (frozen at M0, re-run at each gate
  with the same code tag) on `holdout`: precision@10 and nDCG@10 improve, and the bootstrap CI of the
  difference excludes zero, **or**, while holdout has < 6 recordings and CIs are necessarily wide, the
  improvement holds on every game present in holdout. Changes that only help on `tune` do not ship.
- Per-label additions must not reduce relevance metrics (labels are additive; they must not cost recall).

---

## 6. Learning from corrections

### 6.1 What is stored (locally only)

Every explicit decision stores the moment's pooled feature vector (SigLIP mean embedding + CED vector +
signal summary, ~1.3k floats), the decision, the label, the recording key and the app/model versions,
in `VVideo-data/feedback/`. **Nothing is uploaded; there is no sharing feature in the plan.**

### 6.2 Minimum data before each method activates

| Method | Activates for a label when | Behaviour below the threshold |
|---|---|---|
| kNN / prototype adjustment | ≥ **5 explicit positives and ≥ 5 explicit negatives** for that label, from ≥ 2 recordings | No adjustment; zero-shot score only |
| Per-user logistic head | ≥ **30 explicit positives and ≥ 30 explicit negatives**, from ≥ 3 recordings | kNN only |
| Relevance head (label-independent) | ≥ 50 explicit accepts and ≥ 50 explicit rejects | Fused relevance as today |

k = min(5, n_smallest_class); distance = cosine on L2-normalised vectors; adjustment bounded (e.g. ±0.2 on the
score) so a handful of examples cannot override strong zero-shot evidence **[Assumption: bound, tuned in M5]**.

### 6.3 Evaluating personalisation

- **Leave-one-recording-out** over `train`+`tune` to choose settings; final number on `holdout` recordings
  whose feedback was **never** used.
- Simulated feedback: replay annotator labels from training recordings as if they were user clicks, in time order,
  and plot held-out metrics vs. number of feedback items (learning curve).
- Ship only if the learning curve shows improvement over zero-shot on held-out data without degrading
  any game by more than the CI width.

### 6.4 Licensing of our own data

Recordings are the user's; gameplay footage contains publisher-copyrighted content. Keep footage local,
do not redistribute it, commit only label JSON, split files and derived metrics.
