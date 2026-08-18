# FULL-T6/46D remote training-input status — 2026-08-18

## Current status

`REMOTE_FULL_T6_46D_BYTE_PARITY = PASS`
`REMOTE_R128_RGB_BYTE_PARITY = PASS`
`TRAINING_DATA_INPUT_GATE = PASS`
`TRAINING_RUN = NOT_STARTED`

The remote files were read directly, byte-by-byte, on
`training-pig-project-L4` in Teamspace `pig-project`. The old
`HASH_PENDING_REMOTE_READ` field in
`remote_package_verification_20260817.json` is historical metadata and is
superseded by this live verification.

## Authority and paths

- Local FULL-T6/46D authority: `docs/classification_v2/full_t6_46d_final_authority_20260817.json`
- Local input map:
  `docs/classification_v2/full_t6_training_authority_20260817/`
  `full_t6_training_input_map_20260817.json`
- Local upload manifest:
  `outputs/classification_v2/full_t6_training_authority_20260817/`
  `upload_manifest_20260817.json`
- Local cross-binding audit:
  `outputs/classification_v2/full_t6_training_authority_20260817/`
  `local_cross_binding_audit.json`
- Remote FULL-T6 package:
  `lit://ironheart211224/pig-project/uploads/`
  `classification_v2/full_t6_training_authority_20260817`
- Remote R128 RGB cache:
  `lit://ironheart211224/pig-project/uploads/`
  `classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache`

## Verified contract

- View: T6; targets: `33,287`; train/validation: `27,834/5,453`.
- Feature width: `46`; active schema SHA:
  `18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4`.
- Target/frame/mask/split/order parity: `PASS`; duplicate/missing target: `0/0`.
- RGB coverage: `33,287/33,287`; required contexts: `199,722`; missing/duplicate contexts: `0/0`.

## Seed and namespace guard

- Current S1 temporal-screening seed evidence is the `s1_view_seed_metrics.csv`
  under `.codex_worktrees/classification_v2_s1_temporal_screening_20260814/`,
  `outputs/classification_v2/s1_temporal_screening_20260814/`,
  `s1_final_authority_20260816/`. Its rows are the current S1 seeds
  `20260814`, `20260815`, and `20260816` across T6/T8/T12/T16.
- `docs/classification_v2/s1_04_canonical_r128_3seed_summary_20260817.json`
  binds seeds `20260804`, `20260805`, and `20260806` to the historical
  namespace `s1_post_temporal_closure_20260809` and validation `6154/6154`.
  Mark that result `NOT_COMPATIBLE_FOR_CURRENT_FULL_T6_46D`; do not use it as
  current evidence or compare it with a fresh R54/R128 run.
- A seed number alone is not data provenance: a fresh run using
  `20260804/05/06` is a new result only when its manifest binds the current
  FULL-T6/46D authority, schema SHA, and remote input hashes above.
- Keep this FULL-T6 full-pool authority separate from S1 matched-seed
  authority; never infer one from the other.

## Remote SHA results

The following remote hashes exactly match the local expected hashes:

- `full_t6_canonical_46d.npz`: 16,822,477 bytes
  - SHA-256: `fa4a9f26135271717115355b0ba2a71058b506d05e7cb70b560dca299f14b7d7`
- `full_t6_row_manifest.csv`: 19,218,966 bytes
  - SHA-256: `6737b4437074a1d4021d3749c980797c9dbf145691778d6a6b1075fcfacee6e0`
- `full_temporal_window_manifest_release.csv`: 67,504,559 bytes
  - SHA-256: `c992568cb4e6fe5fe2486072bffd614d1419806b3430961af35d212a2e1c246a`
- `target_split_roles_release.csv`: 38,948,758 bytes
  - SHA-256: `eb4a41753658c52910ff42de98b65a9aff542b89ae817761c07af78273820e80`
- `packed_image_cache_index.csv`: 47,781,243 bytes
  - SHA-256: `9ccef8607973cfb8c8377474665af5d62874b5beea39ad716872b187f8d29d68`
- `packed_rgb_128_letterbox.npy`: 12,075,663,488 bytes
  - SHA-256: `c352a74cade4587e9dcbb8c3eead0c095c992306549b53da6d8b2a361691f5ee`

The complete direct read covered 13/13 FULL-T6/46D package artifacts plus
2/2 R128 RGB artifacts, with no mismatch. Do not re-upload or re-hash these
unchanged paths. Revalidate only if the path, bytes, expected hash, authority
schema, or remote prefix changes.

This status verifies the data-input package only. It does not authorize or claim a training run.
