# Legacy Diff Notes

## Key difference

Legacy 21/06 and current `hybrid_bytetrack` are not the same behavior.

## Legacy

File:

```text
src/pig_behavior/data_preparation/tracking_engine.py
```

Characteristics:

- One tracking flow.
- No `cfg.mode`.
- No `hybrid_bytetrack`.
- No `bytetrack_raw`.

## Current

Files:

```text
src/pig_behavior/tracking/config.py
src/pig_behavior/tracking/runner.py
src/pig_behavior/tracking/detections.py
src/pig_behavior/tracking/association.py
src/pig_behavior/tracking/refinement.py
```

Important differences:

1. Mode-based behavior.
2. `bytetrack` and `gt_export` map to `hybrid_bytetrack`.
3. `hybrid_bytetrack` may use raw ByteTrack ID logic.
4. `hybrid_bytetrack` may match all detections too early.
5. Post-processing may be forced for `hybrid_bytetrack`.
6. Detection filtering differs by mode.
7. Output paths may include mode subdirectories.

## Highest-priority suspect for `000263`

`association.py`:

- raw_id owner/penalty/bypass for `hybrid_bytetrack`
- `all_detection_indices` matching for `hybrid_bytetrack`
