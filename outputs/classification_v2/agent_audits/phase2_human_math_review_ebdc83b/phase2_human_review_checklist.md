# Phase 2 human sign-off checklist

Reviewer: ____________________  
Review date: ____________________  
Decision: `ACCEPT_PHASE_2` / `REJECT_PHASE_2`

- [ ] Confirm the exact 12-feature order in `phase2_motion_schema_table.csv`.
- [ ] Recompute the schema hash from `phase2_motion_schema.json`.
- [ ] Verify midpoint sample times and the `1.5 s` irregular-time example.
- [ ] Verify tangential acceleration is not called vector magnitude.
- [ ] Verify `(1,0)` to `(0,1)` has zero tangential but positive vector acceleration.
- [ ] Verify exact zero speed makes direction unavailable without an epsilon.
- [ ] Verify angle change uses the signed shortest wrapped difference.
- [ ] Verify missing derivatives use false masks and measured zero uses true masks.
- [ ] Inspect all 22 golden numerical scenarios and unit denominators.
- [ ] Run `phase2_independent_reference_verifier.py` and require exit code zero.
- [ ] Confirm all six negative schema families fail closed.
- [ ] Confirm producer/exporter preflight is 12D with empty error lists.
- [ ] Inspect 10 CVAT and 10 legacy production-unit summaries.
- [ ] Confirm at least one valid stationary pair and valid nonzero motion exist.
- [ ] Confirm Phase 1 reset, identity, and invalid-pair invariants remain intact.
- [ ] Confirm Phase 3/4 gaps remain open and no official artifacts were promoted.

Phase 2 is not accepted merely because automated tests pass. `READY_FOR_PHASE3`
must remain `NO` until the reviewer records `ACCEPT_PHASE_2`.
