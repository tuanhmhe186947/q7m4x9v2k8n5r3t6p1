# Paired realtime Pareto evidence — 2026-07-19

This table pairs the same 13 videos, detector, GT, evaluator and
`include_hidden=true` contract. `Quality` is retained as post-video evidence;
its delay `-1` result is not a realtime claim. Values are from the primary
authority CSVs; the independent repeat is used for the runtime and semantic
repeatability gates.

Contract: `iou0_area0_condarea0_merge0`, gap tolerance `15`, zero generated
MP4/preview/overlay/event clip. `IDSW` is remapped identity-switch count.

## Aggregate and operational contract

| mode | timing / delay | IDSW | HOTA | IDF1 |
|---|---|---:|---:|---:|
| `bytetrack_raw` | causal / 0 | 145 | 88.91% | 88.47% |
| `realtime_fast` | causal / 0 | 69 | 94.35% | 93.91% |
| `realtime_balanced` | causal / 0 | 121 | 95.68% | 95.76% |
| `realtime_quality_delayed` | post-video / -1 | 166 | 97.66% | 97.58% |
| `hybrid_bytetrack_best` | offline / n.a. | 0 | 98.35% | 99.15% |

Detection/continuity totals are raw `FP/FN` and strict `fragments`:

- Raw: `1640/1640`, `496`; Fast: `506/630`, `110`.
- Balanced: `448/586`, `127`; Quality: `449/587`, `130`.
- Hybrid: `1593/1593`, `426`.

Runtime evidence (not all values authorize a speed claim):

- Raw: loop-FPS `22.65/27.03`, p95 `108.04/54.31 ms`.
- Fast: effective FPS `27.40/28.16`, p95 `62.49/49.29 ms`.
- Balanced: loop-FPS `26.75` primary, common-harness repeat pending.
- Quality: loop-FPS `12.09`, p95 `139.81 ms`, post-video upper bound only.
- Hybrid: offline; realtime speed claim is not applicable.

No row is selected from accuracy alone. Fast is the current causal reference
because it has the lowest causal IDSW and passes causal, integrity,
repeatability and no-MP4 checks. Its profile-specific `000302` guard is still
open (`6` observed versus a ceiling of `2`), so it is not a final winner.
Balanced remains non-dominated on HOTA/IDF1 and FP/FN, while Quality is not
realtime-valid under its current timing contract. A final speed claim requires
one common runtime harness.

## Paired per-video IDSW

| video | raw | fast | balanced | quality* | hybrid |
|---|---:|---:|---:|---:|---:|
| `000085` | 8 | 2 | 2 | 4 | 0 |
| `000114` | 13 | 0 | 6 | 18 | 0 |
| `000216` | 8 | 4 | 4 | 6 | 0 |
| `000225` | 0 | 2 | 0 | 0 | 0 |
| `000226` | 0 | 4 | 4 | 4 | 0 |
| `000231` | 26 | 12 | 26 | 37 | 0 |
| `000233` | 20 | 17 | 27 | 23 | 0 |
| `000263` | 38 | 18 | 38 | 42 | 0 |
| `000302` | 0 | 6 | 0 | 0 | 0 |
| `000327` | 16 | 4 | 6 | 24 | 0 |
| `000328` | 6 | 0 | 4 | 4 | 0 |
| `000329` | 6 | 0 | 2 | 0 | 0 |
| `000330` | 4 | 0 | 2 | 4 | 0 |

`*Quality` is the simple-gain post-video authority and is shown for scientific
comparison only. Full HOTA, IDF1, FP/FN and continuity fields remain in each
linked `tracking_metrics.csv`; this compact table must not replace paired
per-video audit when a new candidate is promoted.

## Authority sources

- Raw authority run: `20260719_7e38234_raw_repeatability_v1`.
- Fast authority run: `20260718_7f36b57_r1_visible_prefer_fast_full13_primary_v1`.
- Balanced authority run: `20260718_80e4600_gain017_alt025_full13`.
- Quality authority run: `20260717_9a2979d_..._dccad96_gain003_v1`.
- Hybrid authority run: `20260719_h5b_h4_full13_combined_v2`.
- Selection authority:
  `docs/TRACKING_REALTIME_PARETO_SELECTION_DECISION_20260719.json`.
