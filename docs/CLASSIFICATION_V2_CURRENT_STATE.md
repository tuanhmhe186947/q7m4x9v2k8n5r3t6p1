# Classification V2 Current State

## Authority

This file is the authoritative status snapshot for the active
`classification_v2` lineage as of 2026-07-13. It records current gates, not
historical intent. When an older memory, plan, report, or runbook conflicts with
this file, use this file together with `02_CURRENT_DECISION.md` and the data
rebuild runbook.

The active objective is a leakage-safe, reproducible 10-class multimodal
spatio-temporal classifier. The accepted claim boundary is internal
recording-date/video-safe validation, not external-farm generalization.

## Active Data Lineage

The current rebuild starts from 245,664 enhanced frame/object rows:

- 172,800 `cvat_tracking_xml` rows;
- 72,864 `legacy_recovered` rows;
- 4,181 rows currently marked `Hidden=Yes`;
- 241,483 rows currently marked `Hidden=No`.

The versioned Hidden template is:

```text
outputs/classification_v2/rebuilds/hidden_review_v5_full_20260713
```

It contains 5,171 unique review items:

- 4,121 Hidden Yes confirmations;
- 211 high-risk Hidden No items;
- 647 stratified-random Hidden No items;
- 192 clean negative controls;
- 4,649 CVAT items and 522 legacy items.

Independent coverage reports zero missing untrusted Hidden Yes items, zero
trusted-Yes quota mismatches, and zero high-risk cap violations. Human Hidden
decisions are not complete, so no v5 hidden-reviewed frame artifact is final.

The existing behavior manifest contains 4,670 mandatory review units. The
current decision files contain three rows: one accept, one exclude, and one
pending. There are 4,667 missing decisions. Behavior apply is fail-closed and
must not emit a reviewed dataset from this incomplete payload.

## Active Gate Status

| Gate | Status | Evidence or blocker |
|---|---|---|
| Raw `data/` immutable | PASS | No rebuild writes under `data/` |
| Enhanced frame features | PASS | 245,664 rows audited |
| Hidden v5 template | PASS | 5,171 unique items, independent audit clean |
| Hidden human decisions | FAIL | Full decision coverage is incomplete |
| Hidden decision apply | BLOCKED | Requires resolved coverage |
| Temporal rebuild from Hidden-reviewed data | BLOCKED | Upstream apply missing |
| Behavior human decisions | FAIL | 4,667 missing, one pending |
| Behavior decision apply | BLOCKED | Complete gate fails |
| Reviewed train-ready snapshot | FAIL | No complete reviewed lineage exists |
| Model smoke on active snapshot | NOT RUN | Snapshot hashes are not frozen |
| Full OOF on active snapshot | NOT AUTHORIZED | Smoke and launch gates not reached |
| Q2 result claim | NOT ALLOWED | Active reviewed evaluation is absent |

## Historical Model Evidence

A full engineering OOF run exists under:

```text
outputs/classification_v2/model_full/full_multimodal_oof
```

It was produced at commit `18d6692` from the previous artifact lineage. It has
73,668 window predictions and 32,727 native temporal predictions, with accuracy
`0.5216793473` and supported macro-F1 `0.4156053847`.

This run remains useful for debugging compute, cache, fold, and model wiring. It
is not the final thesis result because its input lineage predates the current
two-sided Hidden review and complete behavior-review gates. Old reports that
label it paper-facing or `PASS_PARTIAL_ROADMAP` are lineage-local historical
evidence and do not authorize a current Q2 claim.

## Required Execution Order

1. Validate v5 Hidden media on a bounded sample, then on the full template.
2. Complete and audit all v5 Hidden decisions.
3. Apply Hidden decisions with row/schema preservation checks.
4. Rebuild temporal harmonization and windows from the Hidden-reviewed artifact.
5. Rebuild behavior review units for the same versioned lineage.
6. Complete all behavior decisions and pass the fail-closed coverage gate.
7. Apply behavior decisions without dropping rows or overwriting enhanced data.
8. Rebuild reviewed windows, native units, X/y/masks/weights, and image indexes.
9. Freeze data, cache, feature-whitelist, and fold hashes.
10. Run one-batch, tiny-overfit, resume, runtime, and one-fold smoke gates.
11. Obtain a new full-run authorization bound to the frozen hashes and code SHA.
12. Run finalists only, then grouped native-unit evaluation and completion gates.

Detailed commands are in
`CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md`. Script ownership
and stage order are documented in `scripts/classification_v2/README.md` and
`classification_v2_q2_multimodal_workflow.md`.

## Non-Negotiable Contracts

- Legacy native/review units are 16-frame bursts.
- CVAT anchors represent six-frame intervals `k..k+5`.
- `pig_id` is annotation-local, not cross-video biological identity.
- Training windows are created only after temporal harmonization.
- No label, review, path, stable ID, fold ID, or target-derived field enters X.
- No random frame/window split and no overlapping windows across split roles.
- No silent row drop, silent relabel, silent trust promotion, or raw-data edit.
- Every changed semantic lineage must pass a short representative run before
  broader processing or training.
