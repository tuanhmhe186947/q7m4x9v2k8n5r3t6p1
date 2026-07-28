# B0/B1 detector replay blocker

Date: 2026-07-28

## Decision

The authorized B0/B1 regeneration stopped before tracker execution with
`FAIL_COMMON_DETECTOR_REPLAY_CONTRACT`.

The surviving R0 detector cache is valid for R0 reuse, but it is not complete
detector evidence for either current B0 or current B1.

## Mathematical execution mismatch

- The locked range is frame 0 through frame 1799 for each of 13 videos.
- R0 uses `detect_every_n_frames=2`; its cache contains exactly the 900 even
  indices `0, 2, ..., 1798` per video.
- `runner.py` makes every frame a detector frame when the active mode is
  `bytetrack_raw` or `hybrid_bytetrack`.
- `ReplayDetector.set_frame_context()` fails closed when the requested frame
  is absent from the cache.
- Therefore exact B0/B1 execution would request 1,800 detector records per
  video, but the R0 cache supplies only 900.

Treating an uncached odd frame as an empty detection result, changing the
profile cadence, or carrying forward an even-frame result would alter the
detector population or tracker semantics. None is authorized.

## Consequence

No B0/B1 tracker was run, no detector inference was invoked, no prediction
artifact was created, and no metric evaluator was invoked. A future task must
authorize full-frame detector-evidence generation for the same locked
development population before exact B0/B1 regeneration can proceed.

The execution topology remains `SEPARATE_EXACT_PROFILE_EXECUTIONS`: promoted
B1 has a hybrid-specific pre-repair association path and is not equivalent to
applying only the frozen repair stack to B0 output.
