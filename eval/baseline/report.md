# Evaluation — baseline (audio-only, pre-M1)

- Date: 2026-11-23  
- Code: `fc9a896+dirty`  
- Split: `all` (drafts included — not valid for decisions)  
- Predictions: `eval\baseline`

## Recordings

| Recording | Game | Split | Labels | Reviewed | Highlights (rating ≥ 2) |
|---|---|---|---|---|---|
| gta6-sample-01 | gta6 | tune | draft | 15.5 min | 5 |

## Highlight relevance

| Metric | All (mean of recordings) | 95 % CI | gta6 |
|---|---|---|---|
| p@5 | 40% | n/a (< 2 recordings) | 40% |
| p@10 | 30% | n/a (< 2 recordings) | 30% |
| p@20 | 15% | n/a (< 2 recordings) | 15% |
| ndcg@5 | 45% | n/a (< 2 recordings) | 45% |
| ndcg@10 | 45% | n/a (< 2 recordings) | 45% |
| ndcg@20 | 51% | n/a (< 2 recordings) | 51% |
| recall | 60% | n/a (< 2 recordings) | 60% |
| recall_strict | 40% | n/a (< 2 recordings) | 40% |
| non_gameplay_fp_per_h | 3.86 | n/a (< 2 recordings) | 3.86 |

## Event classification

Matched moments with a ground-truth label: **8**. Abstention (`unlabelled`): **100%**.

| Label | Precision | Recall | F1 | n predicted | n true |
|---|---|---|---|---|---|
| vehicle (n < 10) | – | 0% | – | 0 | 1 |
| reaction (n < 10) | – | 0% | – | 0 | 6 |
| unusual (n < 10) | – | 0% | – | 0 | 1 |

Calibration: ECE –, Brier – — no confidences emitted.

## Per recording

#### gta6-sample-01 (gta6, tune, labels: draft)

| # | Score | Start–End | Matches | Non-gameplay | Reasons |
|---|---|---|---|---|---|
| 1 | 49 | 07:26–07:46 | rating 2, reaction (confrontation with bloodied naked NPCs ('slap contest')) |  | "what the", "fuck", fast speech |
| 2 | 41 | 04:19–04:39 | rating 2, reaction ('It's just a prank, bro!' after the jump) |  | "bro", "lets go", fast speech |
| 3 | 40 | 08:06–08:26 | rating 1, reaction (electrolytes joke) |  | loud, fast speech |
| 4 | 37 | 05:38–05:58 | rating 1, reaction (argument at the camp entrance) |  | "what the", "shit", fast speech |
| 5 | 36 | 09:09–09:30 | — |  | fast speech, loud |
| 6 | 34 | 14:56–15:16 | — | yes | loud |
| 7 | 31 | 00:00–00:10 | — |  | loud |
| 8 | 26 | 02:46–03:08 | rating 1, vehicle (car sliding sideways with dust, ~02:50 -- verify) |  | fast speech |
| 9 | 26 | 07:06–07:26 | rating 1, reaction (comic dialogue with the naturist NPCs) |  | loud, fast speech |
| 10 | 26 | 12:36–12:56 | — |  | fast speech, loud |
| 11 | 25 | 01:39–01:59 | — |  | fast speech, "fucking" |
| 12 | 25 | 06:08–06:28 | rating 1, reaction ('Shut your ass up. You a fraud.') |  | fast speech |
| 13 | 22 | 10:23–10:43 | rating 2, unusual (meditation minigame with psychedelic colour effect) |  | "shit", "fucking", fast speech |
| 14 | 20 | 03:22–03:42 | — |  | "come on" |
| 15 | 20 | 04:56–05:18 | — |  | fast speech, "fuck" |
| 16 | 18 | 09:32–09:52 | — |  | fast speech |
| 17 | 14 | 14:34–14:54 | — |  | fast speech |

Missed highlights (rating ≥ 2, no prediction matched):

- 03:40–04:10 rating 3, unusual: NPCs dare them; jump off the bridge into the river (~03:55-04:05)
- 11:28–11:48 rating 2, reaction: 'organ trafficker' joke during meditation

