# Classification V2 S9 Native Temporal Metrics Gate

Muc tieu S9 la khoa chat duong tu engineering smoke sang paper-facing
experiment. Dataset co the duoc huan luyen bang overlapping sequence windows,
nhung metric chinh cho paper Q2 phai duoc tong hop ve native temporal unit /
review unit de tranh pseudoreplication.

## Claim boundary

- Claim cho phep: Q2 strong, improved pig behavior recognition under
  session/video-safe validation.
- Claim khong cho phep: Q1 generalization, cross-farm, cross-camera,
  cross-cohort neu chua co external validation.
- `pig_id` chi la annotation-local identity, khong duoc coi la cung ca the
  sinh hoc xuyen video.

## Required evaluation contract

Moi paper-facing experiment record phai co `evaluation_contract`:

- `paper_claim_level = Q2_strong`
- `primary_metric_unit = native_temporal_unit`
- `split_policy` thuoc nhom video/session-safe, mac dinh
  `recording_group_oof`
- `source_domain_control_required = true`
- `native_temporal_metrics_required = true`
- `window_metrics_are_secondary = true`
- `feature_leakage_guard_required = true`
- `review_unit_decisions_required = true`
- `pig_identity_scope = annotation_local_not_cross_video`
- `interaction_context = full_frame_partner_context`
- `external_generalization_claim = false`

## Result kinds

- `protocol_gate`, `data_gate`, `review_gate`, `engineering_smoke`: duoc dung
  de ghi evidence ve pipeline/checker, khong dai dien cho ket qua model chinh.
- `model_evaluation`, `baseline_evaluation`, `ablation_evaluation`: bat buoc co
  native temporal prediction metrics/audit trong metrics payload.

## PASS criteria

Mot model record paper-facing PASS khi:

1. Provenance tro toi frozen dataset snapshot, paper protocol, paper audit,
   source-domain control, native OOF folds va trainer contract.
2. Evaluation contract khai bao native temporal unit la primary metric unit.
3. Metrics payload co `native_temporal_metrics`,
   `native_temporal_prediction_audit`, `review_unit_metrics` hoac
   `temporal_unit_metrics`.
4. Neu co window metrics, chung chi la secondary evidence.
5. Split policy la video/session-safe va source-domain control bat buoc.
6. Khong claim external generalization.

## FAIL criteria

Experiment bi chan neu:

- Thieu `evaluation_contract`.
- Paper-facing model result chi co `window_metrics`.
- `primary_metric_unit` khac `native_temporal_unit`.
- Split policy random/window-only.
- Tat source-domain control hoac feature leakage guard.
- Gan `pig_id` la biological identity xuyen video.
- Bat `external_generalization_claim` khi chua co external cohort/farm/camera.

## Module and script design

- `src/pig_behavior/classification_v2/evaluation/native_temporal_metrics_gate.py`
  chua logic gate doc lap, co the dung lai cho registry va checker khac.
- `src/pig_behavior/classification_v2/experiments/registry.py` ghi
  `evaluation_contract` vao moi record moi.
- `src/pig_behavior/classification_v2/experiments/record_contract.py` doc record,
  validate provenance va goi native temporal gate.
- `scripts/behavior_review_tools/classification_v2_register_experiment.py`
  expose `--result-kind`, `--primary-metric-unit`, `--split-policy`,
  `--external-generalization-claim`.
- `scripts/dev_tools/check_classification_v2_experiment_registry.py` la gate
  CLI chinh cho tung record JSON.

## Next implementation steps after S9

1. Tao native temporal prediction aggregation script tu window predictions ve
   review/native temporal unit.
2. Dinh nghia metrics schema chung cho baseline/model/ablation:
   per-class F1, balanced accuracy, macro-F1, confusion pairs, source/session
   stratified metrics, confidence intervals.
3. Tao checker fail neu metrics payload khong co CI/SESOI cho paper-facing
   model result.
4. Them loader/sampler smoke audit de xac nhan source-domain mask duoc dung
   trong training ma khong dua source/path/id columns vao X.
5. Sau do moi chay training smoke E0/E1/E3, chua train full neu gate tren chua
   PASS.
