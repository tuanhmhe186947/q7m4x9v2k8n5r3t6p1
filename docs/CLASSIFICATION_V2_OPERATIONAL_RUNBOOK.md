# Classification V2 operational runbook

Scientific authority is `a35e0b9aae8b55167b4562cfc7e26a45e2b4e312`.
Operational-final execution is configuration-driven and fail-closed. The
default config keeps every authorization flag false.

## Preflight

From the repository root:

```bat
set PYTHONPATH=%CD%\src
set PYTHONDONTWRITEBYTECODE=1
set PYTHONHASHSEED=0
python scripts/classification_v2/lineage_preflight.py --config configs/classification_v2/lineage_rebuild_v1.yaml
```

Expected output is `status=PASS`, 72,880 legacy rows, 72,880 crop files, 12
behavior XMLs, 172,800 CVAT boxes, projected 245,680 rows, and
`release_authority_all_false=true`. Any hash, path, schema or authority error
stops the run.

## Operational-final validation profile

Run the complete relevant profile from a clean checkout with:

```bat
python -m pytest -p no:cacheprovider tests -k classification_v2
```

The command must exit zero. Tests that bind optional, non-versioned integration
artifacts may skip only with one of these stable reason-code prefixes:

```text
OPTIONAL_EXTERNAL_HIDDEN_V6_FIXTURE_UNAVAILABLE
OPTIONAL_EXTERNAL_LEGACY_GOAL_BUNDLE_UNAVAILABLE
```

The Hidden tests run normally when the versioned local human-review bundle is
present and readable. The legacy goal-completion integration test runs normally
when every hash-bound L0-L8 development artifact is supplied. Their synthetic,
configuration, transaction, and fail-closed tests always run. Missing local
cache outputs are not a skip condition; bounded synthetic cache manifests test
repeat-comparison logic in a clean checkout.

## Stage-by-stage operation

Each stage is invoked separately:

```bat
python scripts/classification_v2/run_lineage_stage.py --config configs/classification_v2/lineage_rebuild_v1.yaml --stage source_merge
python scripts/classification_v2/validate_lineage_stage.py --config configs/classification_v2/lineage_rebuild_v1.yaml --stage source_merge
```

After source acceptance, a human changes only the corresponding authorization
flag from `false` to `true`, reviews the diff, and then runs:

```bat
python scripts/classification_v2/run_lineage_stage.py --config configs/classification_v2/lineage_rebuild_v1.yaml --stage frame_local
python scripts/classification_v2/validate_lineage_stage.py --config configs/classification_v2/lineage_rebuild_v1.yaml --stage frame_local
```

Repeat the same pair for:

```text
hidden_design
hidden_decision_migration
hidden_coverage_gate
hidden_apply
temporal_harmonization
native_evidence
pig_strenet_evidence
behavior_review_units
behavior_decision_apply
train_ready
tensor_export
model_input
```

For every name above, use these exact templates:

```bat
python scripts/classification_v2/run_lineage_stage.py ^
  --config configs/classification_v2/lineage_rebuild_v1.yaml ^
  --stage STAGE_NAME
python scripts/classification_v2/validate_lineage_stage.py ^
  --config configs/classification_v2/lineage_rebuild_v1.yaml ^
  --stage STAGE_NAME
```

Replace `STAGE_NAME` with exactly one listed stage ID. The runner validates the
complete source bundle and each upstream manifest as current-authoritative,
rejects unknown or unauthorized stages, refuses existing candidate paths,
invokes only the selected stage, writes one production candidate manifest, and
stops. The `train_ready` stage has two bounded operations within that stage:
reviewed-window construction and explicit-whitelist tabular export. It never
runs the next stage and never promotes a candidate to official.

Human-provided inputs remain separate from generated candidates:

```text
decision_inputs/hidden/previous_hidden_review_unit_manifest.csv
decision_inputs/hidden/previous_hidden_review_decisions.csv
decision_inputs/behavior/behavior_unit_review_decisions.csv
decision_inputs/final/data_contract.json
```

Those paths are below the external run root, not the repository. Copying,
migrating, or accepting decisions is never automatic. Place each file only
after its human gate and validate it before enabling the corresponding stage.
A stage authorization does not change release or training authorization.

## Stop boundaries

Stop on any source fingerprint mismatch, stale path, missing upstream manifest,
candidate collision, failed stage-local validator, incomplete human review,
missing release authorization, or semantic/schema mismatch. Human decisions
are never copied or applied automatically. Temporal, native, Pig-STRENet,
Behavior, export, tensor and model-input stages remain unauthorized by default.
