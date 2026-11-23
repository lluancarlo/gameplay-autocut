# Labelling guide (M0)

How evaluation labels are written. The rules come from
[DATASET_RESEARCH.md](DATASET_RESEARCH.md) §4–5; this page is the practical version.

## The evaluation window (developer tool)

Not part of the app or the `.exe`. Open it from the repo with `python tools/evaluate.py --gui`.

1. **Evaluation folder** — defaults to the repo's `eval/`.
2. **Add recording...** — video, name, game, split (cannot change later), voice track if the mic is separate.
3. **Label** page — play the video (Space, J/L = ∓5 s, speed up to 2×). Press **I** and **O** at the playhead
   to set Start and End, choose rating and label, **Add moment** (Ctrl+Enter). Select a row to edit or delete it.
   *Start–End is menu / non-gameplay* marks menus, maps and pause screens. Mark what you watched with
   *Start–End reviewed* or *Whole video reviewed*, and set **Labels status** to *reviewed* when done.
   Everything is saved to the label file immediately. *Show detector suggestions* is off by default on purpose.
4. **Results** page — choose a split, **Run evaluation** (runs the detector, compares with the baseline),
   **Score saved baseline** (no models), **Freeze last run as baseline...**. Times in the report jump to the Label page.

The same can be done from a terminal with `python tools/evaluate.py --help`.

## Files

| File | What | Committed |
|---|---|---|
| `eval/splits.json` | Every recording: path, game, split (`train` / `tune` / `holdout`), tracks | yes |
| `eval/labels/<id>.json` | Labels for one recording | yes |
| `eval/baseline/` | Frozen predictions + report of the audio-only detector | yes |
| `eval/runs/` | Ad-hoc predictions | no (ignored) |
| the videos | wherever they are; `path` points at them | **never** |

A recording's split is chosen when it is added and **never changes**. Holdout recordings are only
scored at milestone gates.

## Label file

```json
{
  "version": 1,
  "recording": "the-finals-01",
  "status": "draft",               // "reviewed" only after a human watched it end to end
  "annotator": "luan",
  "duration": 1834.2,
  "reviewed_spans": [[0, 1834.2]],  // what was actually watched; nothing outside counts
  "non_gameplay": [[0, 42], [610, 655]],   // menus, map, loading, lobby, pause, desktop
  "moments": [
    {"start": 312, "end": 331, "rating": 3, "label": "combat", "label2": "multi-elimination", "note": "3 kills in a row"}
  ]
}
```

## Procedure

1. Watch the whole reviewed span (up to 2× speed is fine) **with sound**. Detector suggestions may
   be used as a starting point, but you must also add moments it missed.
2. For every moment worth mentioning write `start`, `end` (seconds, ± 1 s is fine), `rating`, `label`.
3. Mark every non-gameplay stretch longer than ~2 s.
4. Set `status` to `"reviewed"` only when the whole `reviewed_spans` was watched.

## Rating (relevance)

| Rating | Meaning | Ask yourself |
|---|---|---|
| 0 | Not a highlight | Only write a 0 when you want to record that something *looks* like a highlight but isn't (e.g. loud cutscene) |
| 1 | Mildly interesting | Would keep it only in a long compilation |
| 2 | Good | Would post it as a short |
| 3 | Must clip | The best moments of the session |

Unlabelled time inside `reviewed_spans` means "nothing worth clipping". Rating 1 is neither
rewarded nor penalised in precision/recall; it only counts in nDCG.

## Labels (level 1)

`combat` · `vehicle` · `destruction` · `achievement` · `failure` · `reaction` · `unusual`.
Leave `label` out if none fits. `label2` is free text for now (e.g. `chase`, `escape`, `victory`,
`stunt`, `funny`); it becomes a fixed list once there are enough examples.

## Rules

- **Never label by what the detector says.** Label what happened.
- One moment = one event. Two events 10 s apart are two moments.
- Game dialogue and cutscenes are rated like anything else — by whether you'd clip them.
- A second annotator, when available, labels the same recording **without** seeing the first file.
