# Classification V2 Data Rebuild And Human Review Runbook

## Current authority and stop state (2026-07-22)

This section is the current operational authority. If an older section below
conflicts with it, this section wins. Code authority
`198767700833b8e00f79a5c69382e186bc06d799` is stopped because the production
Hidden carry-forward dry-run treated 13 expected old-only decisions as fatal.
Never copy a stale SHA from this prose; v6 must bind the new clean patched HEAD.

Lineage `c2v2_human_review_20260721_reviewer01_v3` is frozen with all of these
statuses:

- `STOPPED_AFTER_TEMPORAL_HARMONIZATION`
- `SPATIOTEMPORAL_SCIENTIFIC_AUDIT_FAILED`
- `NOT_RESUMABLE_AFTER_SEMANTIC_CHANGE`
- `NOT_BEHAVIOR_REVIEW_AUTHORITY`
- `NOT_TRAIN_READY`

Do not continue Pig-STRENet, native behavior-unit creation, behavior GUI,
behavior decisions, final sequence windows, train-ready export, or training
from v3. Do not overwrite its workspace. Its `harmonized_frames.csv` and
enhanced spatiotemporal frame features may be read only for before/after audit;
they are not authority after the semantic patch.

Any content under `data/04a_pig_strenet_review_evidence` from the stopped run is
classified as `FAILED_DIAGNOSTIC_PRE_MOTION_FIX`, `NOT_REUSABLE`, and
`NOT_REVIEW_EVIDENCE`. Inventory it by path, size, modification time, and hash,
but never resume or promote it.

Lineage `c2v2_human_review_20260722_reviewer01_v5` is stopped at the Hidden
carry-forward dry-run. It is not resumable after this carry-contract patch.
Do not overwrite, repair, resume, or promote it. The next candidate lineage is
`c2v2_human_review_20260722_reviewer01_v6`, created only after the new clean
code SHA passes the production 5,240-to-5,233 carry-forward integration gate.

Current decision flags are:

- `READY_TO_CREATE_V6_LINEAGE=YES`
- `READY_FOR_HIDDEN_V6=YES`
- `READY_FOR_BEHAVIOR_GUI=NO`
- `READY_FOR_FULL_T6_T8_T12_T16_BUILD=NO`
- `READY_FOR_TRAINING=NO`
- `PROVISIONAL_PRIMARY_VIEW=T6_CONTIGUOUS`
- `FINAL_PRIMARY_VIEW_LOCKED=NO`

The bounded gate is implemented by
`classification_v2_run_pre_behavior_review_smoke.py`. The old smoke hash did
not exercise the corrected production carry-forward boundary and is not v6
authority.
The patched representative smoke must remove any inherited key before the
builder, derive it again, pass `hidden_structure_audit.json`, and repeat
deterministically. Its scope remains `representative_smoke_only` and never
authorizes behavior GUI.

Every subsection explicitly marked `Historical` or `cấm chạy cho v6` is
non-executable evidence even when it still contains a fenced command block.

## Required v6 execution order

After a new code authority exists, execute v6 in this exact order:

1. Rebuild source/frame-local geometry, ROI, social/partner, pen primitives,
   and structural `temporal_unit_key` from immutable source authority. The key
   identifies the exact legacy 16f burst or CVAT 6f interval; it is not motion,
   a pair-derived value, or a temporal aggregate.
2. Rebuild and audit the Hidden sampling manifest with fixed code.
3. Carry forward Hidden decisions only by exact stable review key with identical
   frame/object identity, span, and visual-media authority.
4. Human-review every new-only Hidden item, then dry-run, apply, and independently
   check the complete fixed manifest.
5. Run temporal harmonization from fixed, Hidden-reviewed frame-local rows.
6. Recompute `NATIVE_UNIT_REVIEW_EVIDENCE` inside each exact legacy 16-frame
   burst or CVAT 6-frame interval. Reset every pair at `temporal_unit_key`.
7. Build Pig-STRENet evidence from the same-lineage harmonized/native evidence;
   do not reuse any v3 diagnostic output.
8. Build native-only behavior-review units directly from temporal intervals.
   Do not build or pass a pre-review sequence-window manifest.
9. Validate media and GUI contracts, then create the official clean-code
   review-authority manifest. Open behavior GUI only when it reports
   `authorizes_behavior_gui=true`.
10. Complete behavior review, audit decisions, apply them, and independently
    check reviewed-frame authority.
11. Recompute final T6/T8/T12/T16 and declared ablation views from reviewed
    frame-local primitives.
12. Run the independent sequence checker, leakage gates, determinism checks,
    snapshot/hash gates, and only then create train-ready exports.

No positional Hidden matching is permitted. Old-only decisions remain audit
evidence and never migrate to another unit. New-only items are never accepted
automatically. Behavior review cannot begin until Hidden is complete on the
fixed manifest, and training cannot begin until behavior review and final-view
recompute are complete.

The 2026-07-22 fixed-code read-only comparison records 5,240 old v1 items and
5,233 rebuilt items. There are 5,227 exact-common review keys, 13 old-only
keys, and 6 new-only keys. All 5,227 common keys have identical identity, span,
and visual-media authority, so exactly 5,227 decisions are carried. The 13
old-only decisions are nonfatal audit evidence and must never be written to the
current decision CSV. The 6 new-only items remain undecided for human Hidden
review in v6. Risk/stratum/priority rationale changes remain audit evidence and
scientific support must be recomputed on the new manifest.

Every dry-run and apply audit must contain these explicit fields:
`previous_manifest_items`, `current_manifest_items`, `exact_common_items`,
`carried_decision_items`, `old_only_items`, `old_only_decision_items`,
`new_only_items`, `unknown_decision_items`, `identity_mismatches`,
`span_mismatches`, `media_mismatches`, `positional_matches`, and `errors`.
Dry-run exits zero only when the fixed partition is 5,227/13/6 with zero
unknown decisions, mismatches, positional matches, and errors. Apply publishes
the 5,227-row current decision CSV last; any failure must leave it unpromoted.

Hidden selection no longer consumes external pair/motion columns. It derives
adjacent visibility geometry directly from frame-local bbox primitives and
resets that evidence at `temporal_unit_key`. Full-data perturbation of every
available motion-derived input leaves all 5,233 rebuilt keys, strata,
priorities, scores, and reasons unchanged. The old 5,240-item manifest is still
not interchangeable with the fixed manifest because removing historical
pair contamination changes the selected key set.

## Scientific motion, timing, and mask contract

Frame/native-unit pair evidence resets at source, dataset, video, actor, track,
and `temporal_unit_key`. The first row of every native unit has no inherited
displacement, speed, acceleration, ROI transition, partner-distance delta, or
pen-motion delta. Final windows recompute pair features from the exact selected
rows and never consume a pair whose first endpoint lies before the window.

Two pair classes are distinct:

- `adjacent_motion_pair_valid`: `motion_delta_frames == 1` and positive finite
  elapsed time;
- `sparse_velocity_pair_valid`: `motion_delta_frames > 1` and positive finite
  elapsed time.

Sparse pairs may estimate velocity with real `motion_delta_seconds`, but they
are not contiguous path pairs. Non-positive frame/time deltas and boundary
crossings invalidate the pair. Primary thresholds use
`speed_n_per_second`, `acceleration_n_per_second2`, and corresponding physical
social/pen rates. Historical per-frame columns are audit/ablation fields only.

Pen velocity, inward normal, tangent, and signed-distance delta must share one
image-diagonal metric coordinate system. Compute normal and parallel motion by
dot product; do not combine independently normalized scalars with Pythagoras.

`observed_mask`, `spatial_quality_mask`, ROI validity, social validity, pen
validity, Hidden/review eligibility, and motion-pair masks are separate. A slot
with `spatial_quality_mask=0` has zeroed spatial values and contributes nothing
to the encoder. Replacing the quality mask with the observed mask is forbidden.

## Canonical three-grain computation order

The pipeline has three separate computation grains:

1. `FRAME_LOCAL_PRIMITIVES` contains only one-frame geometry, ROI/pen/partner
   distances, image/posture evidence, Hidden state, labels, and timestamps. It
   has no diff, shift, rolling, pair-derived value, or temporal aggregate.
2. `NATIVE_UNIT_REVIEW_EVIDENCE` recomputes pairs inside one exact legacy
   16-frame burst or one exact CVAT 6-frame interval. Its pair scope is the
   `temporal_unit_key`; it is review evidence, not a final-window aggregate.
3. `FINAL_VIEW_FEATURES` is created only after behavior apply. It selects exact
   frames for one declared view, recomputes all pairs inside that view, then
   aggregates and computes masks/eligibility for that view alone.

The canonical flow is:

```text
raw/source data -> frame-local primitives -> Hidden apply
-> temporal harmonization -> native-unit review evidence
-> behavior review/apply -> reviewed frame-local data
-> exact-view selection -> exact-view pair recomputation
-> exact-view aggregation -> masks/eligibility -> train-ready export
```

A final view may reuse only frame-local primitives. It must not sum native-unit
aggregates, import native-unit pair columns, or reuse another view's aggregate.
Every derived artifact carries `feature_computation_grain`, `pair_scope_key`,
`view_type`, `sampling_pattern`, `selected_frame_indices`,
`pair_delta_frames`, `pair_delta_seconds`, `constituent_native_unit_keys`,
`pair_recomputed_for_view`, and `aggregate_recomputed_for_view` as applicable.

## Cross-length and temporal-sampling views

`T6_contiguous`, `T8_contiguous`, `T12_contiguous`, and `T16_contiguous` are
distinct exact-span views. `S6@16` is a distinct sparse legacy-only ablation
with offsets `[0,3,6,9,12,15]`; it is never named or treated as contiguous T6.
Its five pairs have frame deltas `[3,3,3,3,3]`, use real elapsed seconds, and
carry a sparse-pair mask. Sparse path length is separate from full T16 path.

For a CVAT T8/T12/T16 view, enumerate every native interval intersecting the
exact span. All constituent intervals must be human-reviewed, label-resolved,
train-eligible, and share one final label. A pair across two intervals is valid
only when the selected frames are consecutive and elapsed time is positive.

Each view records `selected_frame_offsets`, `selected_frame_indices`,
`selected_timestamps_seconds`, `physical_span_seconds`, `expected_slot_count`,
`observed_slot_count`, and `constituent_native_unit_keys`. Declared duration,
observed timestamp span, adjacent observed duration, effective observation
rate, and adjacent-pair coverage remain separate timing fields.

Hard-fail a build or checker when any of these occurs:

- `pair_scope_key` differs from the current native unit or final `window_id`;
- a final window consumes a pair whose first endpoint is outside the view;
- a final view imports pair/aggregate values from another grain or view;
- `S6@16` is labeled as contiguous T6;
- a CVAT multi-unit final window includes an unreviewed, unresolved,
  ineligible, or differently labeled native unit;
- primary training silently mixes `view_type` or `sampling_pattern`.

Do not promote `S6@16` as a cross-source primary view when only legacy supplies
it. Treat it as `legacy_only_ablation`. Select a primary cross-source view only
after reporting physical-span distributions and source predictability from
view metadata; a frame-count match alone is not a physical-time match.

The 2026-07-22 pre-review structural audit is explicitly not review or training
authority. At 30 FPS it finds identical cross-source physical spans for each
contiguous view: T6 `0.166667 s`, T8 `0.233333 s`, T12 `0.366667 s`, and T16
`0.500000 s`. All four are structurally available in both sources without a
view-metadata timing shortcut. It recommends T6 as the primary representation
to validate after behavior apply; T8/T12/T16 remain cross-length ablations.
Only legacy supplies 4,555 S6@16 candidates, so S6@16 remains a legacy-only
ablation. Final counts and eligibility must be recomputed after human behavior
review; the structural audit consumes no behavior label.

The earlier apparent source separation by physical span was a timestamp-authority
bug: legacy used `times.txt`, while CVAT used `frame_index / 30`. The user has
confirmed that the source videos are 30 FPS, 1,800-frame, 60-second recordings.
The full read-only audit opened all 678 active videos: every container reports
30 FPS, 1,800 frames, and 60 seconds. Both source types have source-frame stride
one. The old median deltas were `0.162406 s` for legacy and `0.033333 s` for
CVAT; both become `0.033333 s` under the decoded-frame clock.

The canonical motion formula is therefore
`timestamp_sec = source_frame_index / source_fps`, with `source_fps = 30` for
this active authority. Preserve the original `times.txt` value as
`acquisition_timestamp_sec` and its source as audit-only provenance; never use
it as the default motion clock or let source type select the clock. This makes
cross-source T6 physical meaning equivalent and requires no source-driven
resampling. It does not make `S6@16` contiguous or cross-source.

The reproducible evidence is in `timestamp_provenance_audit_v2.json` and
`active_data_before_after_v3.json` under
`outputs/classification_v2/audits/spatiotemporal_semantic_patch_20260722`.

## Historical Classification V2 blocker patch (2026-07-21)

Operator lineage `c2v2_human_review_20260721_reviewer01_v2` is frozen as
`STOPPED_AT_HIDDEN_COMPLETE_UNIT_SMOKE`. Its source/frame rebuild passed, but
the Hidden builder incorrectly inferred full scientific support validation
from the absence of row caps. The two CSV files left in
`data\00_smoke\hidden_review` are failed-build evidence only: they are not a
review manifest authority and must not be renamed, overwritten, or resumed.
No GUI, Hidden carry-forward, behavior decision, temporal full build, sequence
build, or training step is authorized on v2. A semantic fix requires a new
code SHA and the proposed versioned v3 lineage.

The active order is now: Hidden-reviewed frames and temporal harmonization,
native review units (legacy complete 16f bursts and CVAT complete 6f
intervals), native-unit evidence, full behavior review/apply, then a
full-recompute of 6/8/12/16-frame windows. Before behavior review, only
synthetic or complete-unit smoke is permitted; do not build the full
unreviewed window population in a new lineage.

The v1 `data\\04_sequence_unreviewed` root (`%SEQ0%`) is retained as
`PROVISIONAL_UNREVIEWED`. It is diagnostic evidence only, must not be copied
or promoted to training, and must not be overwritten. Final windows belong in
the new `%SEQ1%` root and must be rebuilt from `reviewed_frame_features.csv`.
The main policy is `FULL_NATIVE_UNIT_BEHAVIOR_REVIEW_REQUIRED`: every retained
legacy 16f burst and CVAT 6f interval needs a decision; excluded/uncertain or
unreviewed rows remain in audits with `include=false` and zero weight.

For every apply/check command, pass the same-lineage manifest explicitly:
`--review-unit-manifest-csv %REV%\\full_review_unit_manifest.csv`.

The standalone temporal producer uses the exact CLI flags
`--input-csv`, `--output-csv`, `--intervals-csv`, and `--audit-json`. Its
canonical native-unit output for the new pre-review lineage is
`%SEQ0%\\temporal_intervals_standalone.csv`; native-only review consumes that
file directly and does not invoke the sequence-window builder:

```bat
%PY% %S0%\classification_v2_build_temporal_harmonization.py ^
  --input-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-csv %SEQ0%\harmonized_frames.csv ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --audit-json %SEQ0%\temporal_harmonization_audit.json ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16
%PY% %S1%\classification_v2_build_review_units.py ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --output-dir %REV% ^
  --native-only ^
  --full-native-unit-behavior-review
%PY% %S1%\check_review_unit_gui_contract.py ^
  --review-units-csv %REV%\full_review_unit_manifest.csv ^
  --frame-features-csv %SEQ0%\harmonized_frames.csv ^
  --audit-json %REV%\gui_contract_audit.json
```

## Model-search authority after reviewed-data handoff

Human review and source freezing remain prerequisites. Once the selected main
lineage is review-complete, use this scientific order:

1. establish a strong, stable base with only enough tuning to act as a reliable
   measurement instrument;
2. freeze it and screen every modality/fusion using parameter-matched-zero,
   availability-only, and real controls on identical grouped folds;
3. select fusion candidates from all ten behavior rows, paired cluster
   uncertainty, support, calibration, availability, and harm bounds;
4. jointly tune backbone, temporal model, and selected fusion on rented GPUs;
5. repeat matched confirmatory modality ablations on the tuned finalist.

The screening pass narrows hypotheses but does not lock the final model. Deep
tuning begins only after useful inputs and fusion candidates have evidence.
Local RTX 3050 capacity is a correctness constraint, not an architecture limit.
Preserve valid legacy/main runs and reusable caches, predictions, checkpoints,
and diagnostics. Rerun only when semantics change or artifact integrity fails.
Current legacy A128/all-seven evidence does not prove that steps 4 or 5 ran.

### Active mixed reviewed lineage decision ngày 2026-07-20

Mục tiêu active của runbook này là tạo một mixed lineage gồm legacy 16f P0-P10
đã PASS và đúng 12 behavior XML mới trong
`data\annotations\classification`. Video tương ứng được resolve từ
`data\videos`; source provenance của legacy và XML phải được giữ riêng trong
merged manifest.

Không dùng XML trong `data\annotations\tracking` làm behavior authority. Đây
là artifact tracking cũ/forensic và có thể khác behavior/Hidden của XML mới.
Không gọi mixed export là reviewed hoặc train-ready trước khi Hidden review,
behavior review, snapshot và downstream leakage gates đều PASS.

Tài liệu này là active mixed integration path từ
`07_export\legacy_frame_object_annotations.csv` và 12 XML classification đến
reviewed data, leakage-safe model inputs và model gates.

Luồng tạo lại legacy 16f từ raw/provenance/CVAT đến frame-object export nằm ở
`docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`. Không dùng các lệnh trong
tài liệu này để đọc CSV legacy ở project root đã được dọn; chỉ consume một
export versioned sau khi runbook 16f đạt short và full data gates.

Trạng thái PASS/FAIL hiện hành nằm trong
`docs/CLASSIFICATION_V2_CURRENT_STATE.md`. Audit runbook ngày 2026-07-16 khóa
trạng thái vận hành như sau: người dùng chưa bắt đầu human review, vì vậy số
decision được người dùng xác nhận cho lineage mới là **0**. Các CSV decision cũ
chỉ là artifact pilot/legacy chưa xác minh và không phải review authority.

Bounded legacy+CVAT chain đã PASS kỹ thuật cho identifier-v2, alignment và
feature contract. Legacy L0-L8 cũng đã hoàn tất riêng cho profile
`legacy-only-unreviewed-development`. Hai bằng chứng này không thay thế human
review của legacy hoặc nhánh chính, không cấp quyền gọi data là reviewed và
không tự cấp quyền chạy full OOF.

### Historical boundary của legacy-only lane ngày 2026-07-19

Các ghi chú legacy-only bên dưới là historical development evidence. Chúng
không chuyển metric hoặc quyền promotion sang mixed reviewed lineage mới.
Mixed lineage phải tạo source manifest, hash, folds và snapshot riêng.

Lane `legacy-only-unreviewed-development` trước đây được tạo để thử prompt/goal
orchestration và sàng lọc cấu hình hoặc giả thuyết cho nhánh chính. Kết quả của
lane này chỉ là bounded development evidence. Goal hoặc handback của legacy
không tự kích hoạt, resume, PASS hay cấp quyền chạy goal P0-P8 của nhánh chính;
mọi cấu hình muốn chuyển sang nhánh chính phải được test lại trên snapshot,
folds, review gates và short gate của chính nhánh đó.

Legacy 16f vẫn cần review Hidden và behavior nếu muốn gọi chính lineage này là
reviewed hoặc dùng nó cho train-ready evidence. Review đó độc lập với review của
nhánh chính và không được tính thay cho coverage của nhánh chính.

### Trạng thái canonical của legacy 16f ngày 2026-07-19

Run canonical là
`outputs/legacy_16f_rebuild/legacy_16f_rebuild_20260718_v2`. P0-P10 đã PASS:
27.665 raw CVAT rows, loại đúng 5 source rows thuộc 3 actor lỗi, giữ 27.660
source rows, loại 330 rows theo reviewed-video policy, còn 27.330 anchors,
4.555 actors, 666 groups và 72.880 export rows. Mỗi actor được giữ có đủ 16
frames và sáu anchors `0,3,6,9,12,15`.

`P0-P10 PASS` hoặc `technically clean` chỉ xác nhận cấu trúc, nguồn, policy,
lineage và export nhất quán. Nó không xác nhận nhãn behavior đúng về sinh học
và không biến Hidden thành human-trusted. Ba actor bị loại là source-quality
policy, không phải human-review decision:

- `burst_color_11c02639_300 / ID_3`
- `burst_color_5532ba8c_200 / ID_5`
- `burst_color_77fe4f70_33 / ID_1`

Legacy 16f vẫn phải đi qua cả hai lớp review trước khi train-ready:

1. review Hidden hai chiều ở grain frame/object sau enhanced frame features;
2. review behavior ở complete native unit 16-frame.

Coverage được xác minh của lineage mới hiện vẫn là `Hidden = 0 decisions` và
`behavior = 0 decisions`. Không được dùng trạng thái sạch kỹ thuật để bypass
hai gate này, tạo final reviewed snapshot, chạy training trên active data hoặc
chạy full OOF.

## C6 screening record ngày 2026-07-19

Sau handoff kỹ thuật của rebuild mới, thứ tự chạy đúng là: C6 temporal
controls (18/18 fresh repeats), freeze A128, dựng modality inputs/cache, rồi
C6 modality matrix (22 mode x 2 fresh processes) và paired evaluation. Decision
đạt `PASS` với 44 packet hợp lệ, 14 paired comparisons, 2.000 bootstrap draws
cho mỗi comparison, 241 validation native units, 32 video clusters và không có
lỗi packet.

Config là
`configs/classification_v2/legacy_development_c6_modality_matrix_rebuild_20260719_v2.json`;
output là
`outputs/classification_v2/legacy_only_unreviewed_development/c6mm_20260719_v1`.
Đây vẫn là `legacy-only-unreviewed-development` và quality status là
`TECHNICALLY_CLEAN_UNREVIEWED_DOUBLE_CHECK_PENDING`. Hidden/behavior review
chỉ là double-check; không có full development, full OOF hoặc Q2 claim nào
được mở. Input model dùng các feature cache `.npy` hash-bound và training đọc
zero source media.

### C6 behavior-conditional interpretation and main-lineage retest

The C6 screen contains all seven optional branches and all ten behavior
classes. A branch that fails the global legacy promotion gate is `deferred`,
not removed and not proven useless for every class. Preserve its per-class
precision, recall, F1, confusion, support, and paired predictions.

After the exact main lineage reaches `REVIEW_STAGE=behavior_complete`, freeze
its source manifest, snapshot, native units, folds, actor base, temporal view,
feature whitelist, seeds, and metric contract. Then:

1. retest geometry, motion, ROI, numeric social, pen, union, and full-frame
   context with parameter-matched-zero, availability-only, and real controls;
2. report all ten behaviors, behavior groups, sources, availability strata,
   NLL/calibration, and paired video-cluster intervals per class;
3. predeclare target behaviors and non-target harm bounds for each modality;
4. test behavior-conditional or residual fusion only on development folds,
   without selecting weights from outer-fold predictions;
5. repeat correctness and short gates after every semantic change, then lock
   finalists before requesting full OOF authorization.

No legacy-only point estimate transfers to the mixed reviewed lineage. No
mixed data/model run starts while Hidden or behavior review is incomplete.

## 1. Trạng thái và quyền chạy full

Người dùng đã cho phép chạy full. Quyền này là **có điều kiện** và không bỏ qua
gate. Với mỗi lineage hoặc cấu hình có thay đổi ý nghĩa dữ liệu, thứ tự bắt buộc
là:

1. `py_compile`, unit test và audit chỉ đọc.
2. Chạy synthetic hoặc tiny smoke.
3. Chạy short representative chain bằng complete temporal units của cả hai
   source; không dùng leading-row truncation cho temporal/review validation.
4. Kiểm schema, row count, key uniqueness, hash, output và runtime.
5. Chỉ chạy full khi tất cả kiểm tra trên PASS.

Nếu short run FAIL thì dừng, sửa module chính, tạo output smoke mới và chạy lại.
Không dùng full run để dò lỗi. Thay đổi threshold, temporal contract, source
allowlist, resize policy hoặc review policy đều tạo một cấu hình mới và phải qua
short run mới.

Trạng thái dữ liệu hiện có chưa đủ để gọi là human-reviewed final. Các file cũ
có 30 Hidden row và 3 behavior row, nhưng người dùng xác nhận chưa thực hiện
review; vì vậy chúng bị loại khỏi authority mới. Không migrate, carry hoặc copy
chúng vào decision root sạch. Mọi `reviewed_frame_features.csv` cũ chỉ là
artifact kỹ thuật. CVAT No vẫn không được coi là visible trusted chỉ vì
tracking đã xuất thuộc tính đó.

## 2. Bất biến khoa học

- Không sửa, xóa, đổi tên hoặc overwrite bất kỳ file nào dưới `data\`.
- Không drop row để làm số đẹp. Source-defect policy được phép loại row khỏi
  retained universe, nhưng phải giữ đầy đủ policy/audit accounting. Exclusion
  do human review phải giữ row và ghi mask/weight/action.
- Không đổi label ngoài GUI decision và apply audit.
- Legacy dùng native/review unit 16 frame; decision áp cho cả burst.
- CVAT dùng anchor `k` cho interval `k..k+5`; decision áp cho cả 6 frame.
- Non-anchor CVAT frame kế thừa target anchor, không phải frame không nhãn.
- Training window chỉ được sinh sau temporal harmonization.
- `pig_id` chỉ là ID trong annotation/video, không phải biological identity
  xuyên video hoặc session.
- Không dùng `manual_*`, `review_*`, label, ID, path, policy text hoặc split
  field trong model X.
- Không chọn mọi numeric column. Chỉ dùng whitelist có audit.
- Normalization, prior, class weight, threshold và calibration chỉ fit từ
  training partition của từng fold.
- Không drop mixed/transition window. Giữ row, status và main-train mask.
- Không dùng global class weights trong bước tạo data.
- Hidden là visibility attribute cấp frame/object, không phải behavior target.
- CVAT Hidden là tracking-derived và untrusted cho tới khi human review.
- Audit phải kiểm cả `Yes -> No` và false negative `No -> Yes`.
- Không lan một hidden decision sang cả interval 6/16 frame nếu reviewer không
  khai báo rõ span; mặc định decision chỉ áp đúng frame/object item.
- Identifier-v2 dùng `scene_frame_uid` cho scene/frame và object-level
  `frame_uid` cho đúng một frame/object row. Không dùng hai key thay thế nhau.
- Join/apply ở object grain phải chứng minh one-to-one bằng object-level
  `frame_uid` hoặc composite provenance tương đương; final lineage phải chạy
  lại identifier audit dù bounded current-code audit đã PASS.

## 3. Sơ đồ dữ liệu

```text
CVAT task_0..task_2 XML + task_3 JSON + manifest.jsonl
  -> audit first-task-frame behavior authority và sáu bbox anchor k0..k5
  -> legacy center/scaffold CSV + six-anchor bbox CSV
  -> dense legacy recovery: 16 frame, mười frame xen giữa được khôi phục
  -> legacy_dense_tracklet_map.csv mới trong versioned root
  -> legacy_frame_object_annotations.csv
  -> merge với 12 CVAT behavior XML
  -> context policy -> geometry -> ROI -> motion/social/posture
  -> complete-unit scientific smoke scope (legacy 16f, CVAT scene block 6f)
  -> two-sided Hidden review: Yes + risk/random/control No
  -> hidden_reviewed_frame_features.csv
  -> temporal harmonization: legacy 16f, CVAT 6f
  -> native-unit review evidence, reset at temporal_unit_key
  -> causal Pig-STRENet review evidence with validity masks
  -> native-only review_unit_manifest + 4 policy templates
  -> official review-authority hash
  -> GUI decision CSVs
  -> complete-decision audit
  -> apply decisions, không drop frame
  -> reviewed_frame_features.csv
  -> exact reviewed T6/T8/T12/T16 + legacy-only S6@16
  -> native temporal units
  -> Q2 outer/inner folds + optional native leave-one-group sensitivity
  -> whitelisted X/y/masks/event weights/spatial sequences
  -> provisional T6 primary + T8/T12/T16 and S6@16 ablations
  -> grouped source/length/missingness shortcut controls
  -> image context index
  -> reusable actor and interaction letterbox caches
  -> versioned data contract + immutable hash snapshot
  -> one-batch -> tiny-overfit -> resume -> runtime -> one-fold
  -> finalist lock -> preflight -> authorization -> full OOF
  -> native-unit evaluation -> calibration -> confusion -> registry
```

### 3.1. Thứ tự short trước full

Mỗi stage có gate riêng; PASS ở stage trước không truyền quyền sang stage sau:

```text
legacy CVAT input audit -> one-burst recovery smoke -> full legacy recovery
-> legacy export smoke -> legacy full export
merge/parser smoke -> full merge
feature parser smoke -> full frame features -> complete-unit science smoke
Hidden short/media/apply -> full Hidden review/apply
temporal/window short -> full temporal/window build
review-unit/GUI pilot -> full behavior review/apply
cache preview/subset -> full cache/integrity/pack
model one-batch/tiny/resume/one-fold -> authorized full OOF
```

`full` ở giữa sơ đồ chỉ nghĩa là xử lý toàn bộ artifact của stage đó, không phải
quyền chạy training dài. Complete-unit smoke dùng để xác nhận contract liên
stage; leading-row smoke chỉ kiểm parser/schema. Không bỏ một gate vì stage
trước đã exit 0.

## 4. Chuẩn bị môi trường và lineage

### 4.1. Ranh giới output khi agent đang chuẩn bị

Trong giai đoạn hiện tại, agent chỉ được ghi audit và artifact phát sinh vào
`outputs/classification_v2/agent_audits/<AUDIT_RUN_ID>`. Agent không mở GUI,
không chạy apply/rebuild vào root review và không ghi vào các thư mục output
canonical đang có dữ liệu. Root `human_review_workspace` chỉ thuộc operator;
agent chỉ được đọc sau khi người dùng handoff đúng `RUN_ID` và `REVIEW_STAGE`.

Mọi command writer phải nhận path rõ ràng. Không bỏ qua biến root để script tự
suy ra đường dẫn canonical. Trước handoff, chỉ chạy static/synthetic hoặc
`--dry-run`; sau handoff vẫn phải giữ cùng `AUDIT_RUN_ID` cho toàn bộ chain.

Mở **CMD**, không chạy các lệnh pipeline trong PowerShell. Đổi `RUN_ID` cho mỗi
lineage mới; không tái dùng thư mục của một cấu hình khác.

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
set RUN_ID=c2v2_human_review_20260722_reviewer01_v6
set REVIEWER_NAME=reviewer01
set CODE_AUTHORITY_SHA=replace_with_clean_git_head
set UROOT=human_review_workspace\classification_v2\%RUN_ID%
set R=%UROOT%\data
set SM=%R%\00_smoke
set SRC=%R%\01_source_full
set FEAT=%R%\02_frame_features
set FRAMELOCAL=%FEAT%\frame_local_primitives.csv
set HREV=%R%\03_hidden_review
set SEQ0=%R%\04_sequence_unreviewed
set PIGREV=%R%\04a_pig_strenet_review_evidence
set REV=%R%\05_review_units
set HSMDEC=%UROOT%\human_decisions\hidden_smoke
set HDEC=%UROOT%\human_decisions\hidden
set DEC=%UROOT%\human_decisions\behavior
set RFRAME=%R%\07_reviewed_frames
set L16RUN=outputs\legacy_16f_rebuild\legacy_16f_rebuild_20260718_v2
set L16EXPORT=%L16RUN%\07_export
set L16CROPS=%L16RUN%\06_full_recovery\crops
set L16POLICY=%L16RUN%\02_video_policy
set L16SCAFFOLD=%L16POLICY%\nodup\old_burst_center_keyframes_nodup_videos.csv
set L16ACTORPOLICY=%L16POLICY%\excluded_actor_keys.csv
set L16P10=%L16RUN%\08_audits\legacy_16f_rebuild_completion_audit.json
```

`%UROOT%` intentionally stops at `%RFRAME%`. Do not declare reviewed sequence,
fold, train-ready, cache, snapshot, or model output below the human root.

Không bắt đầu rebuild chỉ vì đã có tài liệu này. Chỉ tạo v6 và chạy Hidden v6
sau khi patch có SHA mới và integration gate PASS; chưa cho phép behavior GUI.
Trước lệnh đầu tiên, operator
chạy `git rev-parse HEAD`, điền kết quả vào `CODE_AUTHORITY_SHA` và xác nhận
worktree sạch. Nếu code classification đang có thay đổi chưa commit hoặc SHA
khác handoff, dừng để tránh đổi code giữa lineage.

Không chạy nguyên văn khi `RUN_ID` còn chứa placeholder `YYYYMMDD` hoặc
`reviewer`, và không dùng lại
một `%UROOT%` đã có artifact. Root này nằm ngoài `outputs\`, thuộc quyền vận
hành của người review; agent chỉ được đọc trong lúc review và phải ghi audit
riêng dưới một `outputs\classification_v2\agent_audits\<AUDIT_RUN_ID>` độc lập.
Việc tách vật lý này ngăn lệnh operator xung đột với output agent. Không phải
mọi builder đều chặn output tồn tại. Khi khởi tạo lineage mới, chạy guard sau
đúng một lần trước khi tạo artifact:

```bat
if exist "%UROOT%" (echo ERROR: RUN_ID already exists: %UROOT% & exit /b 2)
```

Agent không dùng `%UROOT%` làm working hoặc audit root. Chỉ sau khi người dùng
handoff một stage, agent tạo namespace riêng bằng lệnh sau; `AUDIT_RUN_ID` mới
được dùng cho mỗi audit có semantic config khác:

```bat
set AUDIT_RUN_ID=c2v2_agent_audit_YYYYMMDD_vN
set AROOT=outputs\classification_v2\agent_audits\%AUDIT_RUN_ID%
set HANDOFF=%AROOT%\review_handoff
set DROOT=%AROOT%\data
set SEQ1=%DROOT%\08_sequence_reviewed
set NATIVE=%DROOT%\09_native_units
set SPLIT=%DROOT%\10_grouped_splits
set TRAIN=%DROOT%\11_train_ready
set CACHE=%DROOT%\12_actor_cache_224_letterbox
set VCACHE=%DROOT%\13_interaction_cache_224_letterbox
set SNAP=%DROOT%\14_training_snapshot
set MODEL=%DROOT%\15_model_development
if exist "%AROOT%" ^
  (echo ERROR: AUDIT_RUN_ID already exists: %AROOT% & exit /b 2)
```

Chỉ thêm `--overwrite` khi lặp lại đúng semantic config sau short PASS. Đổi
config phải dùng `RUN_ID` mới để không trộn lineage. Resume GUI/cache là ngoại
lệ có chủ ý và phải dùng đúng manifest/hash của cùng lineage.

Khai báo script root ngắn để lệnh dễ đọc và tránh lỗi dòng dài:

```bat
set S0=scripts\classification_v2\00_source_feature_temporal
set S1=scripts\classification_v2\01_review_units_gui
set S2=scripts\classification_v2\02_train_ready_exports
set S3=scripts\classification_v2\03_image_cache_context
set S4=scripts\classification_v2\04_baselines_smokes
set S5=scripts\classification_v2\05_preflight_authorization
set S6=scripts\classification_v2\06_full_oof_training
set S7=scripts\classification_v2\07_postrun_evaluation
set S8=scripts\classification_v2\08_publication_reporting
set S9=scripts\classification_v2\09_final_release_audit
```

Không tạo nhiều folder tên `smoke`, `resume_smoke`, `letterbox_smoke` ở cấp
`outputs\classification_v2` hoặc `human_review_workspace\classification_v2`.
Human-review artifact kết thúc dưới `%RFRAME%`; decision chỉ nằm dưới
`%HSMDEC%`, `%HDEC%` hoặc `%DEC%`. Agent-derived artifact từ reviewed
harmonization trở đi chỉ nằm dưới `%DROOT%`. Smoke và full có tên theo vai trò,
không theo lỗi thử nghiệm.

`%UROOT%` là operator-owned: agent không chạy GUI, apply, rebuild hoặc checker
ghi vào root này. Sau khi người dùng xác nhận một stage và gửi đúng `RUN_ID`,
agent chỉ đọc decision/artifact, rồi ghi hash và checker mirror vào `%HANDOFF%`
thuộc `%AROOT%`. Root review của người dùng không nhận output do agent tạo.

Handoff không cần copy file. Người dùng chỉ gửi bốn giá trị:

```text
RUN_ID=<exact folder name>
REVIEW_STAGE=hidden_complete | behavior_complete
REVIEWER_NAME=<reviewer id used for this root>
REVIEW_CODE_SHA=<exact Git SHA supplied at launch>
```

Agent phải resolve đúng `%UROOT%`, đọc coverage/hash và đặt mọi audit output ở
`%AROOT%`. Agent không tự apply hoặc sửa `%UROOT%`; các lệnh apply/rebuild trong
tài liệu là lệnh operator chạy. Không tự dò và chọn CSV ở folder khác có cùng
tên. Một downstream agent run chỉ được consume artifact người dùng handoff và
phải ghi sang root versioned riêng.

Ownership cố định:

| Root | Quyền ghi | Vai trò |
|---|---|---|
| `data/` | Không ai trong workflow này | Raw input bất biến |
| `%UROOT%` ngoài `outputs/` | Operator | Rebuild và review lineage |
| `%AROOT%` | Agent | Handoff audit và mọi downstream artifact sau review |
| canonical/rebuild cũ | Không ghi | Technical/forensic reference chỉ đọc |

Không dùng canonical folder hiện có làm authority cho rebuild mới. Audit đã thấy
`frame_features\geometry_audit.json` chỉ có 173.664 row, gồm 100.800 CVAT row và
còn chứa dataset `Tracking_annotation_Pigs291119_000263_30fps`. Trong khi đó,
enhanced audit cùng canonical tree có 245.664 row, gồm 172.800 CVAT row từ
allowlist 12 XML. Đây là historical mixed artifact, không phải active input.
Active mixed output dự kiến có 72.880 legacy export rows cộng 172.800 XML rows
trước các downstream review masks. Mọi stage phải đọc artifact được map chính
xác từ cùng cặp `RUN_ID`/`AUDIT_RUN_ID`; không được ghép một canonical
intermediate vào lineage mới.

## 5. Kiểm kê nguồn bất biến

Kiểm tra file tồn tại trước khi chạy. Lệnh chỉ đọc:

```bat
dir /b "%L16EXPORT%"
dir /b "%L16CROPS%"
dir /b "%L16POLICY%"
dir /b data\annotations\classification
dir /b data\videos
certutil -hashfile ^
"%L16EXPORT%\legacy_frame_object_annotations.csv" SHA256
certutil -hashfile "%L16P10%" SHA256
certutil -hashfile "%L16SCAFFOLD%" SHA256
certutil -hashfile "%L16ACTORPOLICY%" SHA256
certutil -hashfile data\data\task_0\annotations.xml SHA256
certutil -hashfile data\data\task_0\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_1\annotations.xml SHA256
certutil -hashfile data\data\task_1\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_2\annotations.xml SHA256
certutil -hashfile data\data\task_2\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_3\annotations.json SHA256
certutil -hashfile data\data\task_3\data\manifest.jsonl SHA256
certutil -hashfile data\annotations\roi\ROI_annotations.coco.json SHA256
certutil -hashfile data\annotations\classification\Pigs281119_000085.xml SHA256
certutil -hashfile data\annotations\classification\Pigs281119_000114.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000216.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000225.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000226.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000231.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000233.xml SHA256
certutil -hashfile data\annotations\classification\Pigs291119_000302.xml SHA256
certutil -hashfile data\annotations\classification\Pigs301119_000327.xml SHA256
certutil -hashfile data\annotations\classification\Pigs301119_000328.xml SHA256
certutil -hashfile data\annotations\classification\Pigs301119_000329.xml SHA256
certutil -hashfile data\annotations\classification\Pigs301119_000330.xml SHA256
```

Dense map cũ chỉ là reference so sánh; không còn là bbox/behavior input của
legacy rebuild khi CVAT đã được sửa. CSV center cũ chỉ cung cấp group/video/path
metadata và valid-group universe; actor behavior và sáu bbox lấy lại từ CVAT.

Thư mục tracking cũ có 13 XML và không được dùng làm behavior authority. XML
behavior authority hiện tại là allowlist 12 file trong
`data\annotations\classification`; **không** truyền cả directory bằng
`--cvat-tracking-dir`:

```text
data\annotations\tracking\Tracking_annotation_Pigs291119_000263_30fps.xml
```

Không thêm `000263` vào mixed lineage. Task này thuộc nhánh tracking cũ và bị
loại khỏi behavior source hiện tại.

Behavior XML allowlist hiện tại dưới `data\annotations\classification`:

```text
Pigs281119_000085.xml
Pigs281119_000114.xml
Pigs291119_000216.xml
Pigs291119_000225.xml
Pigs291119_000226.xml
Pigs291119_000231.xml
Pigs291119_000233.xml
Pigs291119_000302.xml
Pigs301119_000327.xml
Pigs301119_000328.xml
Pigs301119_000329.xml
Pigs301119_000330.xml
```

Ghi lại SHA256 của input trong audit notebook hoặc run manifest. Hash khác ở
lần chạy sau nghĩa là lineage mới và phải chạy short chain lại.

## 6. Rebuild legacy 16f và xuất frame-object annotations

### 6.0. Reference hiện có và lineage mới

Canonical handoff hiện hành là output P10 đã khóa của run 2026-07-18:

```text
%L16EXPORT%\legacy_frame_object_annotations.csv
SHA256=fbd6300fca8fdab0b2c644626397ec6c6aa79f80b48a383f54e745cbcbcbcad3
```

Artifact này có 72.880 rows, 4.555 actors và 666 groups. Completion audit là
`%L16P10%`; export discrepancy rows bằng 0. Downstream human-review rebuild
phải consume export này, không tự chạy lại full recovery. Chỉ rebuild P0-P10
khi source, code hoặc semantic configuration thay đổi.

Reference 72.864 rows/4.554 actors và hash
`adbdb572b976e9f63cff5f9b29ced649f37fa80dd382336b3678f71ac50ff636` chỉ là
archive lịch sử trước rebuild; không dùng làm expected count hoặc authority.
Xác nhận canonical handoff bằng lệnh chỉ đọc:

```bat
certutil -hashfile "%L16EXPORT%\legacy_frame_object_annotations.csv" SHA256
type "%L16P10%"
```

Không overwrite canonical run. Một rebuild mới phải dùng root versioned mới và
được so với P10 bằng input hash, code SHA, config và policy diff.

Nhãn behavior authority của legacy được nạp từ toàn bộ CVAT native export
`data\data\task_0..task_3`. Task 0-2 dùng `annotations.xml`; task 3 dùng
`annotations.json`. Frame ID phải map qua `data\manifest.jsonl` của đúng task.
Authority là CVAT task frame nhỏ nhất trong từng group, không nhất thiết là slot
`k0`. Trong canonical export, 147/666 groups có authority khác k0 và 1.069/4.555
actors thuộc các group này; phân bố slot là k0=519, k1=34, k2=45, k3=68.

Chính sách legacy 16f:

- behavior ở first task frame của group là authority duy nhất và được giữ
  nguyên trên đủ 16 dense frames;
- behavior ở các task frame còn lại chỉ dùng làm disagreement audit;
- bbox ở từng `k0..k5` là sáu GT độc lập từ CVAT, không copy bbox `k0`;
- mười frame không phải anchor được detector/tracker khôi phục có ràng buộc
  giữa các GT lân cận; tại anchor, bbox CVAT luôn thắng;
- Hidden không được suy ra từ behavior. Tại `k0..k5`, dense row phải giữ đúng
  Hidden của chính bbox CVAT đó. Với hai frame xen giữa: nếu hai anchor biên
  đồng ý thì dùng giá trị chung; nếu khác nhau thì seed bảo thủ là `Yes`;
- mọi Hidden sinh từ CVAT phải giữ `hidden_is_trusted=False`, status
  `seed_unreviewed`/`untrusted_cvat_seed`; downstream frame-level Hidden review
  mới có quyền chuyển nó thành trusted;
- actor không xuất hiện ở behavior-authority frame hoặc thiếu một trong sáu
  anchors thì bị loại khỏi recovery input theo policy khai báo;
- duplicate `(group_id, slot, pig_id)`, behavior authority không hợp lệ hoặc
  bbox không hợp lệ làm audit dừng trước khi sinh recovery CSV;
- không fallback về behavior hoặc bbox dense cũ khi CVAT authority bị thiếu.

### 6.1. Audit và dựng recovery input từ CVAT

Chạy audit-only trước. Lệnh này đọc đủ `task_0..task_3`, ghi audit vào root
versioned, nhưng không sinh center/anchor CSV khi còn lỗi.

```bat
set LCVAT_AUDIT=%SM%\legacy_cvat_input_audit
if exist "%LCVAT_AUDIT%" ^
  (echo ERROR: legacy CVAT audit root exists & exit /b 2)
%PY% %S0%\classification_v2_rebuild_legacy_cvat_recovery_inputs.py ^
  --cvat-export-root data\data ^
  --metadata-scaffold-csv "%L16SCAFFOLD%" ^
  --exclude-actor-key-csv "%L16ACTORPOLICY%" ^
  --output-dir "%LCVAT_AUDIT%" ^
  --behavior-authority-policy first_task_frame_per_group ^
  --min-anchor-count 6 ^
  --audit-only
```

PASS yêu cầu `errors=[]`, không duplicate anchor identity và không invalid
first-frame behavior/bbox/Hidden. Chỉ actor có đúng sáu slot `k0..k5` và xuất
hiện ở first task frame mới được đưa vào recovery input. Actor thiếu anchor,
vắng ở authority frame, sai slot hoặc sai frame map phải bị loại có khai báo.

Sau khi audit PASS, dùng root versioned mới và bỏ `--audit-only`:

```bat
set LCVAT=%SRC%\legacy_cvat_rebuild
if exist "%LCVAT%" (echo ERROR: legacy CVAT rebuild root exists & exit /b 2)
%PY% %S0%\classification_v2_rebuild_legacy_cvat_recovery_inputs.py ^
  --cvat-export-root data\data ^
  --metadata-scaffold-csv "%L16SCAFFOLD%" ^
  --exclude-actor-key-csv "%L16ACTORPOLICY%" ^
  --output-dir "%LCVAT%" ^
  --behavior-authority-policy first_task_frame_per_group ^
  --min-anchor-count 6
```

Output:

```text
legacy_center_keyframes_from_cvat.csv
legacy_six_anchor_bboxes_from_cvat.csv
legacy_recovery_input_manifest.json
```

### 6.1.1. Scaffold authority and derived spatial columns

The metadata scaffold is used only to resolve the legacy group and source:

```text
group_id
day_final
video_final
frames
source_video_key
```

The old actor-level values in the scaffold are not annotation authority. In
particular, do not copy or validate old `trigger_type`, `roi_name`,
`near_roi`, ROI distances, center-derived coordinates, or other stale spatial
features. CVAT is authoritative for each actor's six native boxes,
first-task-frame behavior and anchor Hidden value. `classification_v2` recomputes ROI,
geometry, motion, social, and pen features from the rebuilt frame/object rows.

If `source_video_key` is blank, the rebuild may derive it deterministically
from `video_final` (including the `pigs...a` and `pigs...b` session suffix).
The derived key is recorded in the audit. A supplied key that disagrees with
the parseable video path or `day_final` is an error; no default key is used.

The center row is always the exact CVAT k0 row. Its bbox, Hidden, image name,
`center_frame_from_img`, and `center_frame_final` must come from k0, with
`bbox_anchor_slot=0` and `frame_mismatch=False`. The old scaffold center frame
must never select a different anchor. Đây chỉ là center-bbox authority; nó không
ghi đè behavior authority nếu first task frame của group là k1, k2 hoặc k3.

### 6.2. Một-burst dense recovery smoke

Smoke dùng một group đủ tám actor và full 16 frame. Đổi `LEGACY_DRIVE_ROOT`
nếu Drive được mount ở nơi khác.

```bat
set LEGACY_DRIVE_ROOT=G:\My Drive
set LEGACY_SMOKE_GROUP=burst_color_000dacf1_400
set LSMREC=%SM%\legacy_recovery
if exist "%LSMREC%" (echo ERROR: recovery smoke root exists & exit /b 2)
%PY% -m legacy_burst_recovery.main ^
  --input-csv "%LCVAT%\legacy_center_keyframes_from_cvat.csv" ^
  --legacy-burst-bbox-csv ^
  "%LCVAT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --drive-root "%LEGACY_DRIVE_ROOT%" ^
  --output-root "%LSMREC%" ^
  --detector-weights models\detector\pig_detector_yolov8.pt ^
  --scene-mask data\annotations\scene\mask.png ^
  --mask-filter-detections --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside --track-end-mode full_legacy_burst ^
  --sequence-views legacy_old_pattern_6 ^
  --extract-crops --filter-group-id "%LEGACY_SMOKE_GROUP%" ^
  --progress --log-file "%LSMREC%.log"
%PY% %S0%\check_classification_v2_legacy_cvat_recovery_output.py ^
  --center-csv "%LCVAT%\legacy_center_keyframes_from_cvat.csv" ^
  --anchor-csv "%LCVAT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --dense-csv "%LSMREC%\legacy_dense_tracklet_map.csv" ^
  --audit-json "%LSMREC%\cvat_recovery_output_audit.json" ^
  --filter-group-id "%LEGACY_SMOKE_GROUP%"
```

Checker lọc center/anchor theo chính group của smoke; không dùng `head` hoặc
leading rows vì có thể cắt dở actor/native unit.

### 6.3. Full dense recovery sau smoke PASS

Chỉ chạy exact config này sau khi smoke và checker đều PASS. Không thêm
`--resume` cho lần chạy đầu vào root mới.

```bat
set LFREC=%SRC%\legacy_recovery
if exist "%LFREC%" (echo ERROR: full recovery root exists & exit /b 2)
%PY% -m legacy_burst_recovery.main ^
  --input-csv "%LCVAT%\legacy_center_keyframes_from_cvat.csv" ^
  --legacy-burst-bbox-csv ^
  "%LCVAT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --drive-root "%LEGACY_DRIVE_ROOT%" ^
  --output-root "%LFREC%" ^
  --detector-weights models\detector\pig_detector_yolov8.pt ^
  --scene-mask data\annotations\scene\mask.png ^
  --mask-filter-detections --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside --track-end-mode full_legacy_burst ^
  --sequence-views legacy_old_pattern_6 ^
  --extract-crops --progress --flush-every 500 ^
  --log-file "%LFREC%.log"
%PY% %S0%\check_classification_v2_legacy_cvat_recovery_output.py ^
  --center-csv "%LCVAT%\legacy_center_keyframes_from_cvat.csv" ^
  --anchor-csv "%LCVAT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --dense-csv "%LFREC%\legacy_dense_tracklet_map.csv" ^
  --audit-json "%LFREC%\cvat_recovery_output_audit.json"
```

Checker phải PASS với mọi dense behavior bằng first-task-frame authority, mọi
bbox anchor khớp CVAT trong tolerance, đủ frame liên tục, không duplicate và
không mất actor. Nếu fail, không export và không dùng `--resume` để che lỗi.

### 6.4. Frame-object export smoke

```bat
if exist "%SM%\legacy_export\legacy_frame_object_annotations.csv" ^
  (echo ERROR: smoke legacy export exists & exit /b 2)
%PY% src\legacy_burst_recovery\export_legacy_annotations.py ^
  --dense-csv "%LSMREC%\legacy_dense_tracklet_map.csv" ^
  --cvat-behavior-authority-root data\data ^
  --behavior-authority-policy first_task_frame_per_group ^
  --output-dir %SM%\legacy_export ^
  --expected-sequence-length 16 ^
  --anchor-relative-frames 0,3,6,9,12,15 ^
  --expected-pig-count 8
```

Smoke một group đủ tám actor phải tạo 128 object row và tám tracklet, mỗi
tracklet đủ 16 frame. Nếu số khác, đọc summary và xác định do input thay đổi
hay do regression. Không thêm
`--training-only`: context row phải còn để tạo social feature. Không thêm
`--require-full-8-for-eval`: thiếu full-pen context không được làm mất sample.
Overlay audit phải không đổi behavior vì dense recovery đã dùng cùng
first-task-frame authority; thay đổi là dấu hiệu lineage/config không đồng bộ.

### 6.5. Full export sau khi recovery và export smoke PASS

```bat
if exist "%SRC%\legacy_export\legacy_frame_object_annotations.csv" ^
  (echo ERROR: full legacy export exists & exit /b 2)
%PY% src\legacy_burst_recovery\export_legacy_annotations.py ^
  --dense-csv "%LFREC%\legacy_dense_tracklet_map.csv" ^
  --cvat-behavior-authority-root data\data ^
  --behavior-authority-policy first_task_frame_per_group ^
  --output-dir %SRC%\legacy_export ^
  --expected-sequence-length 16 ^
  --anchor-relative-frames 0,3,6,9,12,15 ^
  --expected-pig-count 8
```

Output chính:

```text
%SRC%\legacy_export\legacy_frame_object_annotations.csv
%SRC%\legacy_export\legacy_frame_object_export_audit.json
%SRC%\legacy_export\legacy_cvat_behavior_authority_audit.json
%SRC%\legacy_export\legacy_cvat_behavior_discrepancies.csv
```

Exporter fail nếu artifact đích đã tồn tại; chỉ dùng `--overwrite` khi cố ý chạy
lại đúng lineage. Gate: CSV không rỗng; row count bằng `%LFREC%` dense input;
không duplicate `group_id,pig_id,frame_index`; recovery checker, export audit và
overlay audit đều có `errors=[]`.
`legacy_behavior_changed_by_cvat_authority` phải bằng 0 ở mọi row và
Hidden/provenance phải giữ nguyên qua export. Bất kỳ mismatch nào nghĩa là
center/anchor/dense/CVAT hash không cùng lineage. Ghi hash của CVAT inputs,
recovery input manifest, dense output, recovery audit, export audit và
frame-object output ở mục 17.

Canonical P5 đã PASS với `errors=[]`, `warnings=[]` và retained incomplete actor
count bằng 0. Explicit actor policy loại đúng 5 source rows thuộc ba actor lỗi;
reviewed-video policy loại 330 rows, còn 27.330 anchors. Ba actor đó phải vắng
trong mọi downstream artifact nhưng vẫn hiện diện trong policy/audit accounting.
Đây là source-quality exclusion, không tạo Hidden hoặc behavior review coverage.

Các cột legacy như `behavior_coarse` và `use_for_*_training` là metadata
target-derived để tương thích/audit, không phải feature. Context policy hiện
hành phải recompute eligibility theo `schema.py`; review policy mới là authority
cho interaction/ROI/motion/posture. Cấm đưa các cột legacy này vào X hoặc dùng
chúng để route modality, partner hay image context.

## 7. Merge mixed legacy 16f và behavior XML

Đây là bước tạo source manifest active cho mixed reviewed lineage. Legacy input
chỉ được lấy từ `%L16EXPORT%` của run P0-P10 đã khóa; XML phải lấy đúng 12 biến
`X01..X12` từ `data\annotations\classification`. Không truyền directory và
không dùng XML tracking cũ. Mọi input path, SHA256, row count và source_type
phải được ghi trong merge audit trước khi chạy feature chain.

Khai báo allowlist một lần trong cùng cửa sổ CMD:

```bat
set X01=data\annotations\classification\Pigs281119_000085.xml
set X02=data\annotations\classification\Pigs281119_000114.xml
set X03=data\annotations\classification\Pigs291119_000216.xml
set X04=data\annotations\classification\Pigs291119_000225.xml
set X05=data\annotations\classification\Pigs291119_000226.xml
set X06=data\annotations\classification\Pigs291119_000231.xml
set X07=data\annotations\classification\Pigs291119_000233.xml
set X08=data\annotations\classification\Pigs291119_000302.xml
set X09=data\annotations\classification\Pigs301119_000327.xml
set X10=data\annotations\classification\Pigs301119_000328.xml
set X11=data\annotations\classification\Pigs301119_000329.xml
set X12=data\annotations\classification\Pigs301119_000330.xml
```

Không dùng `--trust-hidden` mặc định. Hidden vẫn được bảo tồn, nhưng không tự
động reject/downweight. Chỉ bật flag đó khi provenance chứng minh Hidden đã qua
review đáng tin và policy mới có audit riêng.

### 7.1. Short merge

```bat
%PY% %S0%\classification_v2_merge_sources.py ^
  --legacy-csv "%L16EXPORT%\legacy_frame_object_annotations.csv" ^
  --cvat-tracking-xml %X01% --cvat-tracking-xml %X02% ^
  --cvat-tracking-xml %X03% --cvat-tracking-xml %X04% ^
  --cvat-tracking-xml %X05% --cvat-tracking-xml %X06% ^
  --cvat-tracking-xml %X07% --cvat-tracking-xml %X08% ^
  --cvat-tracking-xml %X09% --cvat-tracking-xml %X10% ^
  --cvat-tracking-xml %X11% --cvat-tracking-xml %X12% ^
  --fps 30 ^
  --max-rows-per-source 96 ^
  --output-csv %SM%\merged_frame_objects.csv ^
  --audit-json %SM%\merged_frame_objects_audit.json ^
  --lineage-json %SM%\merged_frame_objects_lineage.json
%PY% %S0%\check_classification_v2_mixed_source_lineage.py ^
  --lineage-json %SM%\merged_frame_objects_lineage.json ^
  --legacy-export "%L16EXPORT%\legacy_frame_object_annotations.csv" ^
  --classification-dir data\annotations\classification ^
  --expected-xml-count 12 ^
  --output-json %SM%\mixed_source_lineage_gate.json
```

Smoke PASS khi audit có `errors=[]`, hai source đều xuất hiện, behavior nằm
trong 10 lớp hợp lệ, key được tạo và không source nào mất toàn bộ row. Limit là
theo từng source, chỉ dùng để kiểm parser/schema, không dùng để ước lượng phân
bố lớp.

### 7.2. Full candidate merge

```bat
%PY% %S0%\classification_v2_merge_sources.py ^
  --legacy-csv "%L16EXPORT%\legacy_frame_object_annotations.csv" ^
  --cvat-tracking-xml %X01% --cvat-tracking-xml %X02% ^
  --cvat-tracking-xml %X03% --cvat-tracking-xml %X04% ^
  --cvat-tracking-xml %X05% --cvat-tracking-xml %X06% ^
  --cvat-tracking-xml %X07% --cvat-tracking-xml %X08% ^
  --cvat-tracking-xml %X09% --cvat-tracking-xml %X10% ^
  --cvat-tracking-xml %X11% --cvat-tracking-xml %X12% ^
  --fps 30 ^
  --output-csv %SRC%\merged_frame_objects.csv ^
  --audit-json %SRC%\merged_frame_objects_audit.json ^
  --lineage-json %SRC%\merged_frame_objects_lineage.json
%PY% %S0%\check_classification_v2_mixed_source_lineage.py ^
  --lineage-json %SRC%\merged_frame_objects_lineage.json ^
  --legacy-export "%L16EXPORT%\legacy_frame_object_annotations.csv" ^
  --classification-dir data\annotations\classification ^
  --expected-xml-count 12 ^
  --output-json %SRC%\mixed_source_lineage_gate.json
```

Gate full mixed merge:

- audit `errors=[]`;
- source distribution có `legacy_recovered` và `cvat_tracking_xml`;
- đúng 12 CVAT behavior dataset dự kiến;
- không có `Tracking_annotation_Pigs291119_000263_30fps` trong lineage;
- lineage ghi `canonical_source_fps=30`, formula
  `source_frame_index/source_fps`, và `times.txt` là acquisition-audit only;
- invalid bbox, unknown label và row count đều được ghi, không bị xóa;
- không dùng `--require-full-8-for-eval`.

## 8. Tạo `FRAME_LOCAL_PRIMITIVES` cho v6

V6 phải tách frame-local khỏi native pair evidence. Frame-local chứa context,
geometry, all-ROI, same-frame partner/social geometry, pen distance, media,
Hidden provenance, source-frame index, canonical timestamp và structural
`temporal_unit_key`. Key này phải được suy deterministic từ source/video/actor,
CVAT 6f anchor hoặc legacy 16f burst ngay trước Hidden. Nó không phải motion và
không cho phép diff, shift, rolling, speed, acceleration, transition hoặc
aggregate xuất hiện trong frame-local.

`%FRAMELOCAL%` được khai báo ở section 4.1 và là đường dẫn authority duy nhất.

Production builder và independent checker là hai lệnh sau. Builder đọc trực
tiếp merged source, khóa clock decoded-frame 30 FPS, giữ `times.txt` ở trường
acquisition audit-only, tính geometry/ROI/same-frame social/static pen và ghi
atomically. Checker đọc lại source độc lập, kiểm row order/key, timestamp, range,
schema registry và yêu cầu `errors=[]`.

Builder và checker phải hard-fail nếu key trống, sai công thức, đổi actor/video,
hoặc không tạo đúng unit CVAT 6f và legacy 16f. Production Hidden smoke phải đọc
trực tiếp output này và có `structural_audit.errors=[]`; không được chạy temporal
harmonization trước Hidden để bổ sung key.

```bat
%PY% %S0%\classification_v2_build_frame_local_primitives.py ^
  --input-csv %SRC%\merged_frame_objects.csv ^
  --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --pen-mask data\annotations\scene\mask.png ^
  --output-csv %FRAMELOCAL% ^
  --schema-json %FEAT%\frame_local_schema.json ^
  --audit-json %FEAT%\frame_local_audit.json ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA%
%PY% %S0%\check_classification_v2_frame_local_primitives.py ^
  --source-csv %SRC%\merged_frame_objects.csv ^
  --frame-local-csv %FRAMELOCAL% ^
  --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --pen-mask data\annotations\scene\mask.png ^
  --schema-json %FEAT%\frame_local_schema.json ^
  --builder-audit-json %FEAT%\frame_local_audit.json ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --output-json %FEAT%\frame_local_checker.json
```

Không dùng `--overwrite` trong lần build authority đầu tiên. Nếu output đã tồn
tại, dừng và audit lineage thay vì resume âm thầm. Enhanced artifact cũ chứa
pair-derived columns, không được đổi tên thành frame-local hoặc đưa vào Hidden.

### 8.1. Historical combined feature smoke — không chạy cho v6

```bat
set FSM=%SM%\frame_features
%PY% %S0%\classification_v2_apply_context_policy.py ^
  --input-csv %SM%\merged_frame_objects.csv ^
  --output-csv %FSM%\frame_context.csv ^
  --audit-json %FSM%\frame_context_audit.json
%PY% %S0%\classification_v2_build_geometry_features.py ^
  --input-csv %FSM%\frame_context.csv ^
  --output-csv %FSM%\frame_geometry.csv ^
  --audit-json %FSM%\frame_geometry_audit.json
%PY% %S0%\classification_v2_build_roi_features.py ^
  --input-csv %FSM%\frame_geometry.csv ^
  --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --output-csv %FSM%\frame_roi.csv ^
  --audit-json %FSM%\frame_roi_audit.json
%PY% %S0%\classification_v2_build_enhanced_spatiotemporal_features.py ^
  --input-csv %FSM%\frame_roi.csv ^
  --pen-mask data\annotations\scene\mask.png ^
  --expected-pen-mask-sha256 ^
  b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658 ^
  --output-csv %FSM%\frame_enhanced.csv ^
  --audit-json %FSM%\frame_enhanced_audit.json
```

Chain này dùng merge bị giới hạn theo leading rows, nên chỉ kiểm parser, schema
và row preservation. Nó không chứng minh temporal/review correctness vì có thể
cắt giữa CVAT interval hoặc legacy burst. PASS khi row count giữ nguyên qua bốn
bước, audit không có error, bbox/all-ROI columns tồn tại và source vẫn đủ.
Warning về thiếu context phải được đếm, không được thay bằng drop.

### 8.2. Historical combined full chain — không chạy cho v6

```bat
%PY% %S0%\classification_v2_apply_context_policy.py ^
  --input-csv %SRC%\merged_frame_objects.csv ^
  --output-csv %FEAT%\frame_context.csv ^
  --audit-json %FEAT%\frame_context_audit.json
%PY% %S0%\classification_v2_build_geometry_features.py ^
  --input-csv %FEAT%\frame_context.csv ^
  --output-csv %FEAT%\frame_geometry.csv ^
  --audit-json %FEAT%\frame_geometry_audit.json
%PY% %S0%\classification_v2_build_roi_features.py ^
  --input-csv %FEAT%\frame_geometry.csv ^
  --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --output-csv %FEAT%\frame_roi.csv ^
  --audit-json %FEAT%\frame_roi_audit.json
%PY% %S0%\classification_v2_build_enhanced_spatiotemporal_features.py ^
  --input-csv %FEAT%\frame_roi.csv ^
  --pen-mask data\annotations\scene\mask.png ^
  --expected-pen-mask-sha256 ^
  b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658 ^
  --output-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --audit-json %FEAT%\spatiotemporal_frame_features_enhanced_audit.json
```

Khối lệnh này chỉ giữ làm historical diagnostic. Enhanced output chứa cả pair
features, không phải frame-local authority và không được đưa vào Hidden v6.

Enhanced mặc định thêm `pen_context` từ mask calibration cố định. Audit phải
khớp SHA-256 và kích thước frame; mask được threshold ở 127 rồi chỉ resize bằng
nearest-neighbor. Output giữ nguyên row/key và thêm khoảng cách có dấu tới biên,
tỷ lệ bbox nằm trong chuồng, near-boundary, vận tốc tiến ra xa biên và vận tốc
song song biên. Path/hash, availability, quality, inward-normal,
`pen_center_inside` và binary `pen_near_boundary` chỉ là audit/derivation,
không thuộc model-X. Binary near-boundary chỉ dùng để đo tỷ lệ, episode và
longest run ở mức window. `--no-pen-context` chỉ dùng cho ablation khai báo
trước.

Spatial export có thể tạo group `pen_boundary_context`, nhưng trainer contract
chuẩn chưa bật group này. So sánh promotion đầu tiên bắt buộc là paired
`actor_geometry_motion -> actor_geometry_motion_pen`; cả hai cùng nhận
`motion_delta`, nhờ vậy chỉ thay đổi pen context. Không đổi đồng thời backbone,
resolution, loss, sampler hay temporal encoder. Cặp
`actor_geometry -> actor_geometry_pen` chỉ là diagnostic, không đủ làm bằng
chứng promotion vì candidate có thêm tín hiệu chuyển động theo biên.

### 8.3. Historical complete-unit smoke — không chạy cho v6

Khối này phụ thuộc combined enhanced artifact cũ và không phải v6 authority:

```bat
set SSCOPE=%SM%\complete_unit_scope
if exist "%SSCOPE%\frame_features_complete_units.csv" ^
  (echo ERROR: complete-unit smoke exists & exit /b 2)
%PY% %S0%\classification_v2_build_temporal_smoke_scope.py ^
  --input-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --output-csv %SSCOPE%\frame_features_complete_units.csv ^
  --audit-json %SSCOPE%\temporal_smoke_scope_audit.json ^
  --blocks-per-source 4 --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16
```

Gate: `errors=[]`, hai source, complete CVAT 6-frame actor units, complete legacy
16-frame tracklets và không selected incomplete block. Scope 688 rows/63 native
units là reference lịch sử; canonical lineage phải lấy expected count từ audit
mới và không cắt row để ép về số cũ.

Reference scope 688-row không chứa case anchor 1020. Vì vậy technical smoke này
không được dùng làm bằng chứng cho case bắt buộc; checker ở mục 10.2 phải chạy
trên chính full versioned enhanced/interval/review-unit artifacts trước khi mở
full human behavior review hoặc model smoke.

## 8A. Hidden review hai chiều trước temporal harmonization

Hidden review độc lập với behavior review. Mục tiêu không chỉ xác nhận các row
đã có `Hidden=Yes`, mà còn phát hiện false negative trong `Hidden=No`. CVAT là
nguồn yếu nhất vì Hidden chủ yếu đến từ tracking; row CVAT chưa review phải giữ
`hidden_trust_status=untrusted_tracking_derived`.

Canonical legacy 16f sạch kỹ thuật vẫn nằm trong population review này. Builder
và coverage audit phải chứng minh có legacy frame/object items ở cả hai chiều;
P10 PASS hoặc đủ 72.880 crop files không được tính là Hidden decision.

Bốn cohort không được trộn ý nghĩa thống kê:

- `hidden_yes_confirmation`: census `Hidden=Yes` chưa tin cậy, đồng thời lấy
  mẫu phân tầng từ `Hidden=Yes` trusted để kiểm tra lại prior review;
- `hidden_no_high_risk`: targeted enrichment theo overlap, proximity,
  bbox/shape change, pair geometry và temporal visibility evidence; tuyệt đối
  không dùng behavior hoặc interaction label;
- `hidden_no_random_audit`: random phân tầng để ước lượng false-negative rate;
- `hidden_no_clean_control`: kiểm specificity ở nhóm risk thấp.

Random audit lưu population, inclusion probability và inverse sampling weight.
Chỉ post-stratified random estimate được diễn giải như prevalence. Correction
yield của high-risk cohort không phải prevalence.

Từ commit `2c0cf21`, `_hidden_false_negative_risk()` và sampling strata không
dùng `behavior`, `fight`, `social-nose` hoặc target-derived field. Behavior chỉ
là descriptive metadata. Test hoán vị toàn bộ behavior phải giữ nguyên item,
cohort, risk, stratum, probability và priority; template audit phải báo
`target_derived_fields=[]`. High-risk cohort vẫn chỉ là enrichment yield, không
phải population prevalence.

Temporal Hidden risk chỉ được truyền giữa frame thật sự kề nhau:
absolute frame-index delta phải bằng 1. Sorted sparse CVAT rows không tự động
được coi là adjacent. Persistent pair contact/overlap và bbox instability cũng
chỉ được cộng khi cặp frame này hợp lệ; toàn bộ logic vẫn độc lập behavior.

### 8A.1. Historical combined-input smoke — không chạy cho v6

```bat
set HSM=%SM%\hidden_review
%PY% %S1%\classification_v2_build_hidden_review_units.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --output-dir %HSM% ^
  --design-scope smoke
```

```bat
%PY% %S1%\check_hidden_review_template_coverage.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --audit-json %HSM%\hidden_review_coverage_audit.json
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HSM%\hidden_review_frame_context.csv ^
  --output-dir %HSMDEC% --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root "%L16CROPS%" ^
  --validation-audit-json %HSM%\hidden_media_validation_audit.json ^
  --validate-only
```

Không thêm `--max-rows-per-source` ở đây vì nó có thể cắt complete-unit scope
lần thứ hai. Short PASS khi hai source đều có mặt, input có cả Yes/No, không thiếu
untrusted Yes, trusted Yes đạt quota phân tầng, negative cohorts tồn tại, key
unique và media missing bằng 0. Builder xuất frame-context subset để GUI không
đọc lại full frame-local CSV.

`--design-scope` là semantic contract bắt buộc, độc lập với input bounding.
`smoke` vẫn chạy tất cả structural checks nhưng không yêu cầu full-support
quota. `--max-rows` và `--max-rows-per-source` chỉ giới hạn input debug; chúng
không đổi design scope hay scientific threshold. Canonical complete-unit smoke
không dùng row cap. `--design-scope full` kết hợp với bất kỳ row cap nào phải
hard-fail.

Builder chỉ publish canonical manifest, frame context, templates, template
audit và scientific design sau khi toàn bộ validation PASS. Failure phải
rollback toàn bộ canonical output set và chỉ có thể ghi
`hidden_review_build_failure.json` với `no_outputs_published=true`. Không reuse
output directory của một failed build làm authority.

Mở GUI pilot sau media gate:

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HSM%\hidden_review_frame_context.csv ^
  --output-dir %HSMDEC% --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root "%L16CROPS%" ^
--max-items 5
```

Sau pilot 5 item, chạy lại đúng manifest smoke và đúng `%HSMDEC%`, nhưng bỏ
`--max-items`, để hoàn tất toàn bộ decision của short scope. Đây vẫn chỉ là
Hidden smoke; tuyệt đối chưa ghi vào `%HDEC%`. `%HDEC%` chỉ bắt đầu ở mục
8A.2a với full manifest `%HREV%`:

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HSM%\hidden_review_frame_context.csv ^
  --output-dir %HSMDEC% --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root "%L16CROPS%"
```

Chỉ sau khi short-scope GUI hoàn tất mới chạy coverage checker và apply smoke
ngay bên dưới. Không dùng artifact này làm full review authority và không tạo
fake decision để ép smoke hoặc full review PASS.

```bat
%PY% %S1%\check_hidden_review_decision_coverage.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HSMDEC%\hidden_review_decisions.csv ^
  --audit-json %HSM%\hidden_review_decision_coverage_audit.json
%PY% %S1%\classification_v2_apply_hidden_review_decisions.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HSMDEC%\hidden_review_decisions.csv ^
  --output-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --audit-json %HSM%\apply_hidden_review_audit.json ^
  --confusion-audit-json %HSM%\hidden_confusion_audit.json
%PY% %S1%\check_apply_hidden_review_decisions_output.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --output-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --audit-json %HSM%\check_apply_hidden_review_output.json
```

### 8A.2. V6 full manifest và human review

Chỉ chạy khi `%FRAMELOCAL%` đã qua frame-local schema gate. Cap high-risk kiểm
soát workload nhưng audit vẫn ghi toàn bộ population và số chưa được chọn.

```bat
%PY% %S1%\classification_v2_build_hidden_review_units.py ^
  --input-csv %FRAMELOCAL% ^
  --output-dir %HREV% ^
  --design-scope full ^
  --trusted-yes-per-stratum 1 ^
  --random-no-per-stratum 10 ^
  --clean-control-per-stratum 1 ^
  --max-high-risk-per-stratum 16
%PY% %S1%\check_hidden_review_template_coverage.py ^
  --input-csv %FRAMELOCAL% ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --audit-json %HREV%\hidden_review_coverage_audit.json
```

V6 phải rebuild manifest bằng code authority mới và lấy support từ audit.
Fixed comparison hiện có: old `5,240`, rebuilt `5,233`, exact intersection
`5,227`, old-only `13`, new-only `6`. Chỉ 5.227 exact identities có cùng span
và media authority được carry-forward; 13 old-only giữ audit-only, còn 6
new-only phải được human review. Không điều chỉnh quota sau khi xem outcome mà
vẫn gọi đó là cùng predeclared design.

Trước khi xem kết quả wave đầu, ghi vào review-design manifest: ngưỡng chấp
nhận false-negative, phương pháp confidence interval và strata sẽ báo cáo.
Không khóa Hidden review chỉ vì point estimate thấp. Upper confidence bound của
random weighted estimate và correction yield ở wave high-risk cuối đều phải
đạt ngưỡng đã khai báo; nếu không thì mở rộng wave hoặc census. Không suy ngược
prevalence từ high-risk cohort và không đổi ngưỡng sau khi xem kết quả.

Commits `6949ad0` và `e9a585d` đã cài gate machine-readable. Random cohort dùng
Hájek inverse-probability weighting. Uncertainty dùng source-stratified
recording-cluster bootstrap, bao bởi native-unit Kish-effective Wilson interval.
High-risk yield được báo cáo riêng và không mang nghĩa prevalence. Policy v1
khóa upper bound random `0.05`, high-risk `0.10`, cùng minimum item/native/
recording support. Đây không còn là implementation blocker; human evidence còn
thiếu mới là blocker. Gate chỉ PASS khi coverage, support và threshold cùng đạt.

#### 8A.2a. Carry-forward exact identity và review 6 new-only

Không positional matching và không carry từ pilot/unverified CSV. Bind rõ old
v1 manifest cùng verified human decisions, giữ nguyên old-only evidence và tạo
decision authority mới tại `%HDEC%`.

Trước lần mở GUI đầu tiên, bind hai authority cũ rồi dry-run và apply. Output
root mới phải chưa có decision CSV.

```bat
if not defined OLD_HIDDEN_MANIFEST exit /b 2
if not defined OLD_HIDDEN_DECISIONS exit /b 2
if exist "%HDEC%\hidden_review_decisions.csv" ^
  (echo ERROR: v6 Hidden decision file already exists & exit /b 2)
%PY% %S1%\classification_v2_carry_forward_hidden_review_decisions.py ^
  --previous-manifest-csv "%OLD_HIDDEN_MANIFEST%" ^
  --current-manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv "%OLD_HIDDEN_DECISIONS%" ^
  --output-decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --audit-json %HREV%\hidden_carry_forward_dry_run_audit.json ^
  --dry-run
%PY% %S1%\classification_v2_carry_forward_hidden_review_decisions.py ^
  --previous-manifest-csv "%OLD_HIDDEN_MANIFEST%" ^
  --current-manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv "%OLD_HIDDEN_DECISIONS%" ^
  --output-decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --audit-json %HREV%\hidden_carry_forward_apply_audit.json ^
  --apply
```

Hai audit phải xác nhận đúng 5.227 carried rows, không payload drift, không
unknown key và không output positional. Sau đó GUI chỉ còn 6 new-only items
chưa có decision; không tự accept các item này.

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HREV%\hidden_review_frame_context.csv ^
  --output-dir %HDEC% --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root "%L16CROPS%" ^
  --validation-audit-json %HREV%\hidden_media_validation_audit.json ^
  --validate-only
```

Chỉ mở full GUI khi media audit schema v2 báo `media_missing=0` và SHA256 của
manifest/frame-context còn khớp bytes hiện tại:

Đây là **Hidden handoff point**. Agent dừng sau validate-only và báo `RUN_ID`,
manifest hash cùng media-audit path. Chỉ người dùng mở GUI trong `%HDEC%`.

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HREV%\hidden_review_frame_context.csv ^
  --output-dir %HDEC% --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root "%L16CROPS%"
```

GUI hiển thị full frame với actor và bbox context, kèm actor crop letterbox.
Reviewer chỉ cần chọn `Hidden=Yes`, `Hidden=No (visible)` hoặc `Unclear`; GUI
tự ghi confidence và reason mặc định để không tạo thêm thao tác. GUI chỉ ghi
decision CSV, không sửa XML/CSV nguồn.

#### 8A.2b. Final Hidden coverage metadata-drift policy

Coverage và scientific gate dùng policy
`hidden_review_metadata_drift_v1`. Chỉ hai GUI-copied sampling fields
`hidden_false_negative_risk_score` và
`hidden_false_negative_risk_reasons` là mutable audit metadata. Mismatch của
chúng phải được giữ trong `decision_metadata_drift_counts`, affected unique
items, warnings và policy fields, nhưng không làm gate fail và không được dùng
để rewrite decision CSV.

Mọi shared field khác vẫn fail-closed, gồm canonical key, source/video/media,
actor/track/pig, frame/span/native-unit, `hidden_before_review`, cohort và mọi
unapproved metadata. Missing, unknown, duplicate/conflicting, blank,
unsupported, malformed, pending hoặc unclear decisions vẫn chặn gate. Audit
checker phải ghi checker HEAD riêng với immutable input hashes và
`data_lineage_authority_preserved=true`; checker patch không được tuyên bố đã
regenerate manifest hoặc decisions.

```bat
%PY% %S1%\check_hidden_review_decision_coverage.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --audit-json %HREV%\hidden_review_decision_coverage_audit.json
%PY% %S1%\check_hidden_review_scientific_gate.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --design-json %HREV%\hidden_review_scientific_design.json ^
  --audit-json %HREV%\hidden_scientific_gate.json
```

#### Hidden review tối thiểu và nhanh

`hidden_review_confidence` là cột tương thích provenance, không phải một nhãn
reviewer phải chọn. GUI tự ghi `high` cho Yes/No và `low` cho Unclear. Cột này
không phải chất lượng ảnh, confidence của behavior, detector/tracker confidence
hay confidence interval thống kê. Nó không đổi mask, sample weight, quyền đưa
row vào training và bị cấm khỏi model-X.

Nút Yes tự ghi reason chung `occluded_or_not_visible`; reviewer chỉ chọn reason
chi tiết khi thật sự cần audit nguyên nhân. Nút No tự ghi `clearly_visible`, còn
Unclear tự ghi `ambiguous`. Như vậy quyết định thông thường chỉ cần một phím:
`H`, `V` hoặc `U`.

Coverage vẫn fail-closed với confidence trống/sai hoặc payload tự sửa thành
`reviewed + Yes/No + low`. Đây là kiểm tra nhất quán CSV, không phải yêu cầu
reviewer thao tác thêm. `Unclear` vẫn chặn final gate; dùng nó thay vì đoán khi
không đủ bằng chứng rồi quay lại item đó sau.

Blur, actor nhỏ/xa và ánh sáng yếu không được thu thập trong Hidden GUI vì các
flag đó không đi vào training và làm chậm review. Nếu chúng khiến không thể xác
định occlusion, chọn Unclear. Nếu vẫn nhìn đủ để quyết định, chỉ chọn Yes/No.
Occlusion chính là đối tượng của Hidden; có thể chọn reason chi tiết
`occluded_by_pig` hoặc `occluded_by_scene` khi thông tin đó có giá trị audit.

### 8A.3. Complete gate và apply

```bat
%PY% %S1%\check_hidden_review_decision_coverage.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --audit-json %HREV%\hidden_review_decision_coverage_audit.json
```

Default là fail-closed: missing, duplicate, pending và `Unclear` đều làm gate
FAIL. `--allow-unresolved` chỉ bỏ yêu cầu resolve `pending/Unclear`; nó vẫn FAIL
khi thiếu decision row, duplicate hoặc unknown item. Flag này không tạo được
fake coverage và không được dùng cho training snapshot. Non-selected CVAT No
vẫn là untrusted, không âm thầm thành trusted No.

Sau complete coverage, chạy gate khoa học không có `--report-only`:

```bat
%PY% %S1%\check_hidden_review_scientific_gate.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --design-json %HREV%\hidden_review_scientific_design.json ^
  --audit-json %HREV%\hidden_review_scientific_gate_audit.json
```

Trong lúc review dở, có thể thêm `--report-only` để ghi blocker. Output đó
không cấp quyền apply/snapshot. Final apply yêu cầu coverage checker và
scientific gate đều PASS.

```bat
%PY% %S1%\classification_v2_apply_hidden_review_decisions.py ^
  --input-csv %FRAMELOCAL% ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HDEC%\hidden_review_decisions.csv ^
  --output-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --audit-json %HREV%\apply_hidden_review_audit.json ^
  --confusion-audit-json %HREV%\hidden_confusion_audit.json
%PY% %S1%\check_apply_hidden_review_decisions_output.py ^
  --input-csv %FRAMELOCAL% ^
  --output-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --audit-json %HREV%\check_apply_hidden_review_output.json
```

Apply PASS khi output rows bằng frame-local rows, non-Hidden source columns không
đổi, decision match đúng frame/object key và audit ghi `Yes->No`, `No->Yes`,
trust status, random false-negative estimate cùng high-risk correction yield.

Coverage, scientific gate, apply validation và confusion audit dùng chung
metadata policy `hidden_review_metadata_drift_v1`. Chỉ
`hidden_false_negative_risk_reasons` và
`hidden_false_negative_risk_score` là sampling/audit metadata có thể drift mà
không làm apply FAIL. Apply vẫn phải so sánh, ghi exact drift counts, unique
affected items và warning; tuyệt đối không rewrite manifest hoặc decision CSV.
Mọi mismatch khác vẫn fail-closed. Ba output apply được stage và validate cùng
một transaction; failure không được publish output CSV hay hai audit JSON.

## 9. Temporal harmonization và native evidence trước behavior review

Temporal harmonization chỉ bắt đầu từ Hidden-reviewed frame-local authority.
Trước behavior review, tuyệt đối không build full T6/T8/T12/T16/S6@16 corpus.
Sau harmonization, recompute pair evidence trong từng `temporal_unit_key`.

### 9.0. Active v6 native-evidence chain

```bat
%PY% %S0%\classification_v2_build_temporal_harmonization.py ^
  --input-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-csv %SEQ0%\harmonized_frames.csv ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --audit-json %SEQ0%\temporal_harmonization_audit.json ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16
%PY% %S0%\classification_v2_build_enhanced_spatiotemporal_features.py ^
  --input-csv %SEQ0%\harmonized_frames.csv ^
  --output-csv %SEQ0%\native_review_evidence.csv ^
  --audit-json %SEQ0%\native_review_evidence_audit.json
```

Frame đầu mỗi native unit phải có pair validity false và zero inherited
motion/ROI/social/pen transition. Grain phải là
`NATIVE_UNIT_REVIEW_EVIDENCE`; pair scope phải là `temporal_unit_key`.

### 9.1. Historical pre-review window smoke — cấm chạy cho v6

```bat
set TSM=%SM%\sequence_unreviewed
%PY% %S0%\classification_v2_build_temporal_harmonization.py ^
  --input-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --output-csv %TSM%\harmonized_frames.csv ^
  --intervals-csv %TSM%\temporal_intervals_standalone.csv ^
  --audit-json %TSM%\temporal_harmonization_audit.json ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16
%PY% %S0%\classification_v2_build_sequence_windows.py ^
  --input-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --output-dir %TSM% ^
  --window-lengths 6,8,12,16 ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16 ^
  --max-windows-per-track 2 ^
  --disable-fast-reuse
```

`--disable-fast-reuse` là bắt buộc với lineage versioned; nếu bỏ, script có thể
tái dùng window canonical cũ. Không dùng `--exclude-mixed-windows`. Mixed và
transition phải còn trong manifest nhưng `window_valid_for_main_train` phản ánh
eligibility.

### 9.2. Historical pre-review full windows — cấm chạy cho v6

```bat
%PY% %S0%\classification_v2_build_temporal_harmonization.py ^
  --input-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-csv %SEQ0%\harmonized_frames.csv ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --audit-json %SEQ0%\temporal_harmonization_audit.json ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16
```

```bat
%PY% %S0%\classification_v2_build_sequence_windows.py ^
  --input-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %SEQ0% ^
  --window-lengths 6,8,12,16 ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16 ^
  --disable-fast-reuse
```

Hai file interval được tạo độc lập để audit determinism:

```text
%SEQ0%\temporal_intervals_standalone.csv
%SEQ0%\temporal_label_intervals.csv
```

Chúng phải có cùng native-unit key, label và interval boundary. Gate bắt buộc:

- duplicate `temporal_unit_key = 0`;
- CVAT interval dài 6, legacy interval dài 16;
- anchor `Pigs281119_000085_30fps / ID_4 / 1020` là `social-nose`;
- `window_id` unique, không có `window_uid`;
- window status stable/mixed/transition được đếm;
- source và label distribution được ghi;
- standalone interval và window-builder interval khớp key/label/boundary;
- không row bị mất mà không có audit reason.

Case anchor bắt buộc có một nuance quan trọng. Raw rows `1020..1025` của
`Pigs281119_000085_30fps / ID_4` chứa cả `social-nose` và `stand`; anchor 1020
là `social-nose`. Sau harmonization, target của đủ sáu frame là `social-nose`
và review group là `interaction`. Đây là propagation đúng contract, không phải
đổi âm thầm raw non-anchor annotation.

Window audit phải tách `hidden_ratio_raw`, `hidden_ratio_trusted`, review
coverage và longest consecutive Hidden run. Canonical policy mặc định bật và
dùng Hidden ở mức frame/object sau apply, trước khi quyết định training tier:

| View | Main: tổng Hidden / longest run | Robust tối đa: tổng / run |
| --- | --- | --- |
| T6 | `1 / 1` | `3 / 2` |
| T8 | `2 / 1` | `4 / 3` |
| T12 | `3 / 2` | `6 / 4` |
| T16 | `4 / 3` | `8 / 6` |

Tương đương, `main_train` yêu cầu Hidden ratio `<=0.25` và longest-run ratio
`<=0.20`. Vượt một ngưỡng main nhưng vẫn `<=0.50` tổng và `<=0.40` run chỉ được
`robust_train_only`. Vượt một ngưỡng robust phải `exclude`, không được dùng cho
main hay robustness training và `window_sample_weight` bị ép về `0.0`. Phép so
sánh là `>` nên giá trị đúng biên vẫn ở tầng thấp hơn.

Policy bảo thủ dùng `hidden_ratio_raw_window` của cột `hidden` hiện hành sau
apply; untrusted Hidden=Yes vẫn được tính để không vô tình nhận ảnh thiếu bằng
chứng. `hidden_ratio_trusted_window` và review coverage chỉ dùng audit. Toàn bộ
Hidden/review field bị cấm khỏi model-X.

Khi policy bật, window builder bắt buộc rebuild từ frame rows và không được
fast-reuse window cũ. `--no-exclude-high-hidden-from-main` chỉ dành cho ablation
được khai báo rõ; không dùng cho canonical reviewed lineage.

## 10. Tạo review unit và template

Review unit là đơn vị human decision, không phải training window. Mỗi unit chỉ
thuộc một template chính: interaction, ROI, motion hoặc posture. `playwithtoy`
luôn nằm trong ROI review. `stand` thuộc motion/context; `fight` thuộc
interaction; posture chỉ có `lying` và `sitting`.

Builder gắn review-only conflict/insufficiency/priority evidence từ motion,
all-ROI, posture proxy và social persistence. Các score này chỉ sắp xếp/cung
cấp ngữ cảnh cho reviewer; chúng không tự đổi label, weight hoặc model target và
mọi `review_*` field bị cấm khỏi X. ROI bbox contact chỉ là proxy không gian,
không được diễn giải như bằng chứng chắc chắn về ăn/uống nếu thiếu head/snout.

Behavior queue có năm cohort rời nhau: mandatory census, high-risk,
stratified random residual audit, clean control và not-selected. Mandatory gồm
interaction/rare/temporal-unstable theo policy và toàn bộ retained legacy khi
bật complete-legacy. Random lưu population, probability và inverse weight.
Behavior not-selected và clean control không phải human-verified clean.

### 10.0. Pig-STRENet review evidence

Build causal evidence after native-unit recomputation and before review-unit
selection. V6 input is `%SEQ0%\native_review_evidence.csv`. Every temporal unit
must match exactly; missing or extra evidence keys fail closed. Transition
evidence is zero unless history and target are both complete. The `%PIGSM%`
smoke commands immediately below are historical and must not be run for v6;
the active v6 command starts at `%PIGREV%`.

~~~bat
set PIGSM=%TSM%\pig_strenet_review_evidence
%PY% %S3%\classification_v2_build_pig_strenet_artifacts.py ^
  --input-csv %TSM%\harmonized_frames.csv --output-dir %PIGSM% ^
  --history-length 6 --target-length 6 --legacy-target-starts 6 ^
  --top-k-neighbors 3 --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --video-root data\videos --legacy-crop-root "%L16CROPS%" ^
  --run-scope smoke
%PY% %S3%\check_classification_v2_pig_strenet_artifacts.py ^
  --artifact-dir %PIGSM% --input-csv %TSM%\harmonized_frames.csv ^
  --expected-run-scope smoke ^
  --output-json %PIGSM%\pig_strenet_artifact_gate.json
%PY% %S3%\classification_v2_build_pig_strenet_artifacts.py ^
  --input-csv %SEQ0%\native_review_evidence.csv --output-dir %PIGREV% ^
  --history-length 6 --target-length 6 --legacy-target-starts 6 ^
  --top-k-neighbors 3 --roi-coco data\annotations\roi\ROI_annotations.coco.json ^
  --video-root data\videos --legacy-crop-root "%L16CROPS%" ^
  --run-scope full
%PY% %S3%\check_classification_v2_pig_strenet_artifacts.py ^
  --artifact-dir %PIGREV% --input-csv %SEQ0%\native_review_evidence.csv ^
  --expected-run-scope full ^
  --output-json %PIGREV%\pig_strenet_artifact_gate.json
~~~

The builder and checker are both mandatory; a builder exit 0 is not a media
gate. Difference maps use actor pixels from crop or source video. ROI visual
evidence uses source-video scene frames. Static background.png or Image #1 is
never accepted as temporal scene evidence. Pair labels are ignored by the
review bridge, and all emitted review_pig fields remain forbidden from model-X.

### 10.1. Historical window-based builder smoke — không chạy cho v6

```bat
%PY% %S1%\classification_v2_build_review_units.py ^
  --intervals-csv %TSM%\temporal_label_intervals.csv ^
  --sequence-window-manifest-csv %TSM%\sequence_window_manifest.csv ^
  --output-dir %SM%\review_units ^
  --max-units-per-template 100000 ^
  --pig-strenet-artifact-dir %PIGSM% ^
  --disable-window-review-overlay
%PY% %S1%\check_review_unit_template_coverage.py ^
  --review-unit-dir %SM%\review_units ^
  --allow-incomplete-label-coverage
%PY% %S1%\classification_v2_write_behavior_scientific_design.py ^
  --manifest-csv %SM%\review_units\full_review_unit_manifest.csv ^
  --review-audit-json %SM%\review_units\review_unit_audit.json ^
  --output-json %SM%\review_units\behavior_scientific_design.json ^
  --smoke
```

### 10.2. Full review-unit build

```bat
%PY% %S1%\classification_v2_build_review_units.py ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --native-only ^
  --full-native-unit-behavior-review ^
  --output-dir %REV% ^
  --max-units-per-template 100000 ^
  --pig-strenet-artifact-dir %PIGREV% ^
  --include-all-retained-legacy-units
%PY% %S1%\check_review_unit_template_coverage.py ^
  --review-unit-dir %REV% ^
  --require-complete-legacy ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --audit-json %REV%\review_unit_coverage_gate.json
%PY% %S1%\classification_v2_write_behavior_scientific_design.py ^
  --manifest-csv %REV%\full_review_unit_manifest.csv ^
  --review-audit-json %REV%\review_unit_audit.json ^
  --output-json %REV%\behavior_review_scientific_design.json
%PY% %S0%\check_classification_v2_cvat_anchor_case.py ^
  --enhanced-csv %SEQ0%\native_review_evidence.csv ^
  --intervals-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --review-units-csv %REV%\review_unit_manifest.csv ^
  --output-json %REV%\cvat_anchor_1020_audit.json
%PY% %S1%\check_review_unit_gui_contract.py ^
  --review-units-csv %REV%\full_review_unit_manifest.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --audit-json %REV%\gui_contract_audit.json
```

Flag `--include-all-retained-legacy-units` đưa mọi native unit legacy 16-frame
được giữ vào behavior-review manifest, kể cả unit ổn định không có evidence
conflict. Flag `--require-complete-legacy` đối chiếu
`review_unit_manifest.csv` với `full_review_unit_manifest.csv` và fail nếu thiếu
bất kỳ legacy unit nào. Đây là cặp gate bắt buộc khi gọi legacy 16f là complete
behavior-reviewed.

Hai flag này chỉ kiểm tra các `legacy_recovered` unit đã hiện diện trong
`%SEQ0%\temporal_intervals_standalone.csv`; chúng không tự nhập legacy ngoài source
manifest. Với active mixed lineage, sự hiện diện của legacy là bắt buộc và
`--require-complete-legacy` phải PASS. Nếu sau này chạy CVAT-only sensitivity,
phải bỏ cả hai flag và không gọi kết quả đó là complete legacy review.

Native-only mode không đọc window-review overlay và không cần pre-review window
manifest. Nếu code yêu cầu một window artifact ở đây, dừng vì contract đã drift.

Gate: duplicate `review_unit_id=0`, không có `window_uid`, template labels đúng
policy, mọi retained legacy unit đều có trong full review, và
`full_review_unit_manifest.csv` bằng union của các template. Builder vẫn tạo
`temporal_consistency_review_unit_template.csv`; với 10 behavior canonical file
này phải rỗng. Nếu nó có row thì dừng thay vì bỏ qua. Temporal evidence audit
không được phát hiện label/review leakage vào trainer whitelist.

Behavior scientific design phải được ghi trước decision đầu tiên. Full design
bind exact manifest hash, policy hash, cohort support và residual estimand.
Smoke design chỉ kiểm tra contract; nó luôn non-authorizing.

### 10.3. Official review-authority gate trước behavior GUI

Ba file `%SEQ0%\timestamp_fps_contract.json`, `%REV%\evidence_semantics.json`
và `%REV%\behavior_review_media_authority.json` phải được build và independently
check trong chính v6. Không copy artifact từ representative smoke.

```bat
%PY% %S0%\classification_v2_write_timestamp_fps_contract.py ^
  --frame-local-csv %FRAMELOCAL% --video-root data\videos ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --source-lineage-artifact merged=%SRC%\merged_frame_objects_lineage.json ^
  --output-json %SEQ0%\timestamp_fps_contract.json
%PY% %S0%\check_classification_v2_timestamp_fps_contract.py ^
  --frame-local-csv %FRAMELOCAL% --video-root data\videos ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --source-lineage-artifact merged=%SRC%\merged_frame_objects_lineage.json ^
  --contract-json %SEQ0%\timestamp_fps_contract.json ^
  --output-json %SEQ0%\timestamp_fps_contract_checker.json
%PY% %S1%\classification_v2_write_evidence_semantics.py ^
  --frame-local-csv %FRAMELOCAL% ^
  --native-evidence-csv %SEQ0%\native_review_evidence.csv ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --output-json %REV%\evidence_semantics.json
%PY% %S1%\check_classification_v2_evidence_semantics.py ^
  --frame-local-csv %FRAMELOCAL% ^
  --native-evidence-csv %SEQ0%\native_review_evidence.csv ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --semantics-json %REV%\evidence_semantics.json ^
  --output-json %REV%\evidence_semantics_checker.json
```

```bat
%PY% %S1%\classification_v2_build_behavior_review_media_authority.py ^
  --review-units-csv %REV%\full_review_unit_manifest.csv ^
  --native-evidence-csv %SEQ0%\native_review_evidence.csv ^
  --video-root data\videos --legacy-crop-root "%L16CROPS%" ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --output-index-csv %REV%\behavior_review_media_authority_index.csv ^
  --output-json %REV%\behavior_review_media_authority.json
%PY% %S1%\check_classification_v2_behavior_review_media_authority.py ^
  --review-units-csv %REV%\full_review_unit_manifest.csv ^
  --native-evidence-csv %SEQ0%\native_review_evidence.csv ^
  --video-root data\videos --legacy-crop-root "%L16CROPS%" ^
  --lineage-id %RUN_ID% --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --index-csv %REV%\behavior_review_media_authority_index.csv ^
  --authority-json %REV%\behavior_review_media_authority.json ^
  --output-json %REV%\behavior_review_media_authority_checker.json

if not exist "%SEQ0%\timestamp_fps_contract.json" exit /b 2
if not exist "%REV%\evidence_semantics.json" exit /b 2
if not exist "%REV%\behavior_review_media_authority.json" exit /b 2
if not defined CODE_AUTHORITY_SHA exit /b 2
%PY% %S1%\classification_v2_build_review_authority_manifest.py ^
  --code-authority-sha %CODE_AUTHORITY_SHA% ^
  --lineage-id %RUN_ID% ^
  --authority-scope official_v6_pre_behavior_review ^
  --source-artifact merged_source=%SRC%\merged_frame_objects_lineage.json ^
  --frame-local-csv %FRAMELOCAL% ^
  --hidden-reviewed-frame-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --harmonized-frame-csv %SEQ0%\harmonized_frames.csv ^
  --temporal-native-unit-manifest-csv ^
  %SEQ0%\temporal_intervals_standalone.csv ^
  --pig-strenet-evidence-manifest %PIGREV%\pig_strenet_artifact_gate.json ^
  --behavior-review-unit-manifest-csv %REV%\full_review_unit_manifest.csv ^
  --media-authority-manifest %REV%\behavior_review_media_authority.json ^
  --timestamp-fps-contract-json %SEQ0%\timestamp_fps_contract.json ^
  --evidence-semantics-json %REV%\evidence_semantics.json ^
  --component-gate frame_local=%FEAT%\frame_local_checker.json ^
  --component-gate hidden_coverage=%HREV%\hidden_review_coverage_audit.json ^
  --component-gate hidden_scientific=%HREV%\hidden_scientific_gate.json ^
  --component-gate hidden_apply=%HREV%\check_apply_hidden_review_output.json ^
  --component-gate temporal_harmonization=%SEQ0%\temporal_harmonization_audit.json ^
  --component-gate native_evidence=%SEQ0%\native_review_evidence_audit.json ^
  --component-gate pig_strenet=%PIGREV%\pig_strenet_artifact_gate.json ^
  --component-gate native_review_unit_coverage=%REV%\review_unit_coverage_gate.json ^
  --component-gate timestamp_fps=%SEQ0%\timestamp_fps_contract_checker.json ^
  --component-gate evidence_semantics=%REV%\evidence_semantics_checker.json ^
  --component-gate media_authority=%REV%\behavior_review_media_authority_checker.json ^
  --output-json %REV%\behavior_review_authority.json
%PY% %S1%\check_classification_v2_review_authority_manifest.py ^
  --manifest-json %REV%\behavior_review_authority.json ^
  --code-authority-sha %CODE_AUTHORITY_SHA% --lineage-id %RUN_ID% ^
  --source-artifact merged_source=%SRC%\merged_frame_objects_lineage.json ^
  --frame-local-csv %FRAMELOCAL% ^
  --hidden-reviewed-frame-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --harmonized-frame-csv %SEQ0%\harmonized_frames.csv ^
  --temporal-native-unit-manifest-csv %SEQ0%\temporal_intervals_standalone.csv ^
  --pig-strenet-evidence-manifest %PIGREV%\pig_strenet_artifact_gate.json ^
  --behavior-review-unit-manifest-csv %REV%\full_review_unit_manifest.csv ^
  --media-authority-manifest %REV%\behavior_review_media_authority.json ^
  --timestamp-fps-contract-json %SEQ0%\timestamp_fps_contract.json ^
  --evidence-semantics-json %REV%\evidence_semantics.json ^
  --component-gate frame_local=%FEAT%\frame_local_checker.json ^
  --component-gate hidden_coverage=%HREV%\hidden_review_coverage_audit.json ^
  --component-gate hidden_scientific=%HREV%\hidden_scientific_gate.json ^
  --component-gate hidden_apply=%HREV%\check_apply_hidden_review_output.json ^
  --component-gate temporal_harmonization=%SEQ0%\temporal_harmonization_audit.json ^
  --component-gate native_evidence=%SEQ0%\native_review_evidence_audit.json ^
  --component-gate pig_strenet=%PIGREV%\pig_strenet_artifact_gate.json ^
  --component-gate native_review_unit_coverage=%REV%\review_unit_coverage_gate.json ^
  --component-gate timestamp_fps=%SEQ0%\timestamp_fps_contract_checker.json ^
  --component-gate evidence_semantics=%REV%\evidence_semantics_checker.json ^
  --component-gate media_authority=%REV%\behavior_review_media_authority_checker.json ^
  --output-json %REV%\behavior_review_authority_checker.json
```

Không truyền `--code-dirty`. Dừng nếu manifest invalid, tham chiếu v3, chứa
pair columns trong frame-local schema hoặc không báo
`authorizes_behavior_gui=true`. Đây là gate cuối trước section 11.

## 11. GUI smoke và human review đầy đủ

GUI smoke là kiểm tra bắt buộc trước khi review hàng nghìn unit. Dùng đúng
output directory sẽ dùng cho full review; lần chạy sau tự resume, không ghi đè
decision cũ. Không dùng `--fresh` và không xóa CSV giữa các session.

Khi causal history complete, GUI hiển thị H frames trước T target frames và
ghi rõ frame range. History thiếu hoặc có gap không được bịa ảnh hay dùng
transition score; GUI chỉ hiển thị target và validity metadata. Review-Pig
evidence là ngữ cảnh để con người kiểm tra, không phải auto-label authority.

Behavior review phải bao gồm mọi retained legacy actor ở complete native unit
16-frame, ngoài các CVAT unit theo contract. Không review từng frame rời và
không coi ba source-policy exclusions là behavior decisions. Gate phải kiểm
coverage theo source và native-unit grain trước khi apply.

Behavior decision schema hiện có 24 cột nhưng chưa nhúng reviewer/timestamp.
Vì vậy `RUN_ID` phải chứa reviewer ID, `%UROOT%` chỉ thuộc một reviewer và
handoff phải khóa path/hash. Double review phải dùng một `RUN_ID` khác. Trước
lần mở GUI đầu tiên, xác nhận bốn decision root đều sạch:

```bat
if exist "%DEC%\roi\behavior_unit_review_decisions.csv" exit /b 2
if exist "%DEC%\motion\behavior_unit_review_decisions.csv" exit /b 2
if exist "%DEC%\posture\behavior_unit_review_decisions.csv" exit /b 2
if exist "%DEC%\interaction\behavior_unit_review_decisions.csv" exit /b 2
```

### 11.1. Năm unit mỗi nhóm

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\roi_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\roi --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --roi-coco-json data\annotations\roi\ROI_annotations.coco.json ^
  --max-items 5 --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\motion_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\motion --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --max-items 5 --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\posture_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\posture --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --max-items 5 --copy-contact-sheets
```

Interaction cần scene/partner context rộng. GUI interaction hiện hiển thị
full-frame CVAT context trực tiếp; `--padding` không mở rộng interaction vì
nhánh này không dùng actor crop padding. Legacy chỉ có scene/partner context
nếu resolver tìm được video/frame-specific scene; actor crop đơn lẻ không được
coi là đủ bằng chứng. Nếu actor/partner/role vẫn không đủ rõ, chọn
`review_later`; không đoán label.

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\interaction_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\interaction --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --max-items 5 --copy-contact-sheets
```

Sau smoke, chạy coverage ở chế độ chưa bắt complete. Missing unit là warning;
schema, duplicate, invalid action hoặc corrected-label lỗi vẫn là FAIL.

```bat
%PY% %S1%\check_review_unit_decision_coverage.py ^
  --review-manifest-csv %REV%\full_review_unit_manifest.csv ^
  --decisions-csv ^
  %DEC%\roi\behavior_unit_review_decisions.csv ^
  %DEC%\motion\behavior_unit_review_decisions.csv ^
  %DEC%\posture\behavior_unit_review_decisions.csv ^
  %DEC%\interaction\behavior_unit_review_decisions.csv ^
  --audit-json %DEC%\decision_coverage_smoke_audit.json
```

Smoke PASS khi cả legacy crop và CVAT video+bbox hiển thị đúng, ROI overlay đọc
được, quyết định được lưu và lần mở lại hiển thị đúng decision cũ. Kiểm riêng
video `Pigs291119_000231_30fps.mp4`; resolver phải mở được key không có suffix.

### 11.2. Full human review có resume

Chạy lại cùng output directory và bỏ `--max-items`. Có thể đóng/mở nhiều lần;
GUI nạp CSV cũ, chặn blank/duplicate ID và ghi deterministic order.
Đây là **behavior handoff point**; agent không mở GUI hoặc ghi `%DEC%`.

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\roi_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\roi --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --roi-coco-json data\annotations\roi\ROI_annotations.coco.json ^
  --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\motion_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\motion --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\posture_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\posture --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --copy-contact-sheets
```

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\interaction_review_unit_template.csv ^
  --frame-features-csv %SEQ0%\native_review_evidence.csv ^
  --output-dir %DEC%\interaction --video-root data\videos ^
  --raw-root "%L16CROPS%" ^
  --copy-contact-sheets
```

Quy tắc quyết định:

- `accept`: giữ nhãn gốc; thường `main_train`, weight 1.
- `corrected`: chọn đúng một trong 10 behavior và ghi note ngắn.
- `exclude`: không xóa row; apply đặt include false và weight 0.
- `low_weight_train`: giữ row với weight giảm, phải có lý do quality rõ.
- `review_later`: fail-closed, không vào training và làm complete gate FAIL.
- `fight`: chỉ actor trực tiếp tham gia, không bystander.
- `social-nose`: actor-only mặc định, không hard-propagate sang receiver.

Số review unit phải lấy từ manifest mới sau khi merge canonical 72.880-row
legacy export; không dùng estimate 4.670 lịch sử. Nên double-review 10-20% nhóm
hiếm/confusion và báo agreement riêng; GUI decision không được dùng làm model
feature.

Với kết quả paper-facing, subset double-review phải được chọn theo strata trước
khi xem disagreement, review lần hai ở chế độ blind và lưu reviewer/run riêng.
Báo action agreement, corrected-label agreement và per-confusion disagreement;
adjudication tạo artifact mới, không sửa âm thầm decision gốc. Nếu chỉ có một
reviewer, phải ghi đây là limitation thay vì claim annotation reliability cao.

## 12. Audit decision và apply review

### 12.1. Complete gate bắt buộc

```bat
%PY% %S1%\check_review_unit_decision_coverage.py ^
  --review-manifest-csv %REV%\full_review_unit_manifest.csv ^
  --decisions-csv ^
  %DEC%\roi\behavior_unit_review_decisions.csv ^
  %DEC%\motion\behavior_unit_review_decisions.csv ^
  %DEC%\posture\behavior_unit_review_decisions.csv ^
  %DEC%\interaction\behavior_unit_review_decisions.csv ^
  --audit-json %DEC%\decision_coverage_final_audit.json ^
  --require-complete
%PY% %S1%\check_behavior_review_scientific_gate.py ^
  --manifest-csv %REV%\full_review_unit_manifest.csv ^
  --decisions-csv ^
  %DEC%\roi\behavior_unit_review_decisions.csv ^
  %DEC%\motion\behavior_unit_review_decisions.csv ^
  %DEC%\posture\behavior_unit_review_decisions.csv ^
  %DEC%\interaction\behavior_unit_review_decisions.csv ^
  --design-json %REV%\behavior_review_scientific_design.json ^
  --audit-json %DEC%\behavior_scientific_gate.json
```

Scientific gate dùng inverse-probability weighting cho random residual cohort,
cluster uncertainty theo source/video/native unit và báo high-risk yield riêng.
Nếu gate không PASS, phần not-selected chưa đủ bằng chứng để authorize training
snapshot; không được đổi tên low-risk thành clean.

Không chạy apply nếu lệnh này FAIL. Điều kiện PASS gồm đủ 24 column, không
duplicate/missing/unexpected ID, không pending, không `review_later`, corrected
behavior hợp lệ và không `window_uid`.

### 12.2. Apply decisions

```bat
%PY% %S1%\classification_v2_apply_review_unit_decisions.py ^
  --frame-features-csv ^
  %HREV%\hidden_reviewed_frame_features.csv ^
  --review-unit-manifest-csv %REV%\full_review_unit_manifest.csv ^
  --decisions-csv ^
  %DEC%\roi\behavior_unit_review_decisions.csv ^
  %DEC%\motion\behavior_unit_review_decisions.csv ^
  %DEC%\posture\behavior_unit_review_decisions.csv ^
  %DEC%\interaction\behavior_unit_review_decisions.csv ^
  --output-csv %RFRAME%\reviewed_frame_features.csv ^
  --combined-decisions-csv %RFRAME%\review_unit_decisions_combined.csv ^
  --audit-json %RFRAME%\apply_review_unit_decisions_audit.json
```

```bat
%PY% %S1%\check_apply_review_unit_decisions_output.py ^
  --reviewed-csv %RFRAME%\reviewed_frame_features.csv ^
  --audit-json %RFRAME%\apply_review_unit_decisions_audit.json ^
  --combined-csv %RFRAME%\review_unit_decisions_combined.csv ^
  --source-frame-features-csv ^
  %HREV%\hidden_reviewed_frame_features.csv
```

Apply không overwrite frame-local hoặc native-evidence CSV. Nó giữ nguyên số
frame row, lưu
`behavior_before_review` và `behavior_after_review`, đồng thời thêm action,
include flag và weight. Corrected decision áp toàn bộ 16 frame legacy hoặc 6
frame CVAT qua `temporal_unit_key`. Exclude đặt mask/weight, không xóa row.

Gate:

- reviewed rows bằng Hidden-reviewed frame-local rows;
- audit `errors=[]` và unmatched decision bằng 0;
- duplicate `review_unit_id=0` trong combined decisions;
- applied/accepted/corrected/excluded counts được ghi;
- excluded/corrected frame counts được ghi;
- label distribution trước/sau được kiểm và mọi delta truy về decision.

### 12.3. Handoff bắt buộc trước downstream rebuild

Sau khi section 12.1 và 12.2 PASS, operator dừng và gửi handoff
`behavior_complete`. Không tự chạy section 13 trở đi trong `%UROOT%`.

Agent chỉ đọc `%RFRAME%\reviewed_frame_features.csv`, decisions và audit đã
handoff. Agent tạo một `AUDIT_RUN_ID` mới, sinh artifact map hash-bound, rồi ghi
mọi output section 13-17 dưới `%AROOT%`. Không copy hay sửa file trong
`%UROOT%`; input human được tham chiếu bằng exact path và SHA256.

Hard stop:

```bat
if not defined AROOT ^
  (echo ERROR: wait for behavior_complete handoff and agent root & exit /b 2)
if /i "%AROOT%"=="%UROOT%" ^
  (echo ERROR: human and agent roots must differ & exit /b 2)
```

## 13. Rebuild reviewed windows và native units

Rebuild reviewed sequence phải full-recompute vì corrected label có thể đổi
target và temporal status. Từ section này, input `%RFRAME%` là read-only và mọi
output dùng `%DROOT%`; không dùng fast overlay canonical.

### 13.1. Reviewed sequence windows

```bat
%PY% %S0%\classification_v2_build_sequence_windows.py ^
  --input-csv %RFRAME%\reviewed_frame_features.csv ^
  --output-dir %SEQ1% ^
  --harmonized-frame-csv %SEQ1%\harmonized_frames.csv ^
  --temporal-intervals-csv %SEQ1%\temporal_label_intervals.csv ^
  --window-lengths 6,8,12,16 ^
  --include-legacy-sparse-s6-at16 ^
  --behavior-review-requirement full_native_unit_review_required ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16 ^
  --disable-fast-reuse
```

Không dùng `--exclude-mixed-windows`. `window_sample_weight=0` và
`window_valid_for_main_train=false` phải phản ánh review-excluded frame mà
không làm mất window row.

Builder phải recompute exact pairs và aggregates riêng cho từng view. Gate yêu
cầu `pair_recomputed_for_view=true`, `aggregate_recomputed_for_view=true`,
`pair_scope_key=window_id`; S6@16 phải là legacy-only sparse ablation và không
được trộn vào T6 primary corpus.

### 13.2. Native temporal units

```bat
%PY% %S0%\classification_v2_build_native_temporal_units.py ^
  --intervals-csv %SEQ1%\temporal_label_intervals.csv ^
  --reviewed-frame-csv %RFRAME%\reviewed_frame_features.csv ^
  --output-dir %NATIVE%
%PY% %S2%\check_classification_v2_native_temporal_units.py ^
  --manifest-csv %NATIVE%\native_temporal_unit_manifest.csv ^
  --output-json %NATIVE%\check_native_temporal_units_audit.json
```

Native unit là statistical prediction unit chính. Overlapping window chỉ là
training augmentation; không được coi mỗi window là quan sát độc lập khi báo
metric paper-facing.

Gate: duplicate `temporal_unit_key=0`, CVAT non-6f=0, legacy non-16f=0,
negative weight=0, excluded/corrected unit được đếm và label đã review khớp
frame-level apply.

## 14. Recording groups và leakage-safe folds

Primary protocol dùng `recording_date` vì metadata session/farm/camera chưa được
chuẩn hóa đầy đủ. Nếu có `recording_metadata.csv` được human-validated, có thể
chuyển sang `recording_session`; thay đổi đó tạo lineage mới và phải smoke lại.

### 14.1. Recording-group manifest

```bat
%PY% %S2%\classification_v2_build_recording_groups.py ^
  --input-csv %NATIVE%\native_temporal_unit_manifest.csv ^
  --output-dir %SPLIT%\recording_groups ^
  --group-level recording_date
```

Không dùng `pig_id` để group xuyên video. Unknown farm/pen/camera/cohort phải
ghi `unknown`, không suy đoán.

### 14.2. Native-unit publication split audit

```bat
%PY% %S2%\classification_v2_build_publication_folds.py ^
  --manifest-csv %NATIVE%\native_temporal_unit_manifest.csv ^
  --recording-group-manifest-csv ^
  %SPLIT%\recording_groups\recording_group_manifest.csv ^
  --output-dir %SPLIT%\publication_native ^
  --group-level recording_date ^
  --label-col behavior_label ^
  --valid-col native_unit_valid_for_main_eval ^
  --id-col temporal_unit_key
```

```bat
%PY% %S2%\check_classification_v2_publication_folds.py ^
  --split-manifest-csv ^
  %SPLIT%\publication_native\publication_split_manifest.csv ^
  --recording-group-manifest-csv ^
  %SPLIT%\publication_native\recording_group_manifest.csv ^
  --output-json %SPLIT%\publication_native\split_check_audit.json ^
  --id-col temporal_unit_key
```

### 14.3. Q2 outer/inner folds

Năm grouped folds này là protocol khoa học chính. Với từng outer fold, `test`
phải bất khả kiến; candidate selection, early stopping, normalization, class
weight và calibration fit chỉ được dùng `train`/`validation` của chính fold đó.
Outer test chỉ đánh giá selection rule đã khai báo trước. Mọi model paired
comparison phải dùng cùng assignment/hash và cùng native units.

```bat
%PY% %S2%\classification_v2_build_q2_folds.py ^
  --native-unit-csv ^
  %SPLIT%\publication_native\publication_split_manifest.csv ^
  --output-dir %SPLIT%\q2_grouped_folds ^
  --folds 5 --seed 20260710
%PY% %S2%\check_classification_v2_q2_folds.py ^
  --fold-dir %SPLIT%\q2_grouped_folds
```

Không chọn global finalist bằng pooled outer-test metrics rồi báo lại chính các
metric đó như unbiased evidence. Hoặc khóa finalist trên development groups
tách biệt, hoặc predeclare candidate-selection rule chạy hoàn toàn trong inner
roles của từng outer fold. Dataset hiện chưa có untouched external session.

### 14.4. Native leave-one-group-out engineering manifest

Current OOF runner đọc manifest leave-one-recording-group-out riêng và không
dùng inner validation roles. Build/audit manifest này để tái lập runner hiện
tại, nhưng không gọi nó là confirmatory protocol sau khi cùng recording groups
đã được dùng để chọn architecture:

```bat
%PY% %S2%\classification_v2_build_native_oof_folds.py ^
  --native-split-manifest ^
  %SPLIT%\publication_native\publication_split_manifest.csv ^
  --output-dir %SPLIT%\native_oof_folds
%PY% %S2%\check_classification_v2_native_oof_folds.py ^
  --manifest %SPLIT%\native_oof_folds\native_oof_fold_manifest.csv ^
  --audit-json %SPLIT%\native_oof_folds\native_oof_fold_check.json
```

Q2 outer/inner roles là authority khoa học. Native leave-one-group-out chỉ là
engineering baseline hoặc sensitivity analysis được khai báo trước. Final Q2
runner phải nhận Q2 roles trực tiếp, fit mọi transform trong outer-train và dùng
inner validation; không được chạy cả hai protocol rồi chọn metric tốt hơn. Muốn
có confirmatory test độc lập thật sự phải khóa recording/session chưa từng tham
gia review design, feature/model selection hoặc threshold tuning; hiện chưa có
manifest như vậy.

Gate: một `recording_group_id` và recording date không xuất hiện ở nhiều split
trong cùng comparison; cùng `temporal_unit_key` không nằm ở nhiều outer test
fold; class-by-fold và source-by-fold support được báo cáo. Fold thiếu lớp không
được che giấu.

Recording date và source hiện có tương quan mạnh; một số fold có thể chỉ chứa
legacy hoặc CVAT. Grouped split ngăn leakage nhưng không tự loại domain
confounding. Không phá recording group để cân source. Phải báo cáo pooled,
per-source và source-by-fold metrics, đồng thời giữ claim ở internal known-domain
validation, không gọi là source/farm/camera generalization.

Không dùng split random theo frame, row hoặc overlapping window. Không dùng
outer-fold prediction để chọn architecture, threshold hoặc hyperparameter.

## 15. Train-ready tabular, spatial và weights

### 15.1. Window metadata split theo recording date

Tạo `split_manifest.csv` để loader có row/key metadata. Đây là split theo ngày,
không phải random window. Q2 outer/inner roles ở mục 14 vẫn là authority cho
OOF và model selection.

```bat
%PY% %S2%\classification_v2_build_publication_folds.py ^
  --manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --output-dir %TRAIN%\window_split_protocol ^
  --split-output-csv %TRAIN%\split_manifest.csv ^
  --group-level recording_date ^
  --label-col behavior_window_label ^
  --valid-col window_valid_for_main_train ^
  --id-col window_id
%PY% %S2%\check_classification_v2_split_group_leakage.py ^
  --split-manifest-csv %TRAIN%\split_manifest.csv ^
  --split-audit-json ^
  %TRAIN%\window_split_protocol\publication_split_audit.json ^
  --output-json %TRAIN%\window_split_group_leakage_audit.json
```

Gate phải chứng minh mọi window của cùng recording/video/native unit chỉ có một
split role. `window_id` là row key, không phải grouping key khoa học.

### 15.2. Whitelisted tabular X/y/mask/weight

```bat
%PY% %S2%\classification_v2_export_train_ready_windows.py ^
  --input-csv %SEQ1%\sequence_window_features.csv ^
  --output-dir %TRAIN% ^
  --trainer-contract-json configs\classification_v2\trainer_contract_v1.json
%PY% %S2%\check_classification_v2_train_ready_windows.py ^
  --audit-json %TRAIN%\train_ready_audit.json
%PY% %S2%\check_classification_v2_q2_feature_whitelist.py ^
  --output-json %TRAIN%\q2_feature_whitelist_audit.json
```

Exporter fail nếu thiếu, thừa hoặc sai thứ tự whitelist; nó không còn suy luận X
từ mọi numeric column hoặc prefix. `X_window_features.csv` chỉ chứa feature
whitelist. `y_behavior.csv`,
`train_mask.csv` và `sample_weight.csv` là artifact riêng, không join ngược vào
X. Audit phải có `forbidden_selected=[]` và row count X/y/mask/weight bằng nhau.
`check_classification_v2_q2_feature_whitelist.py` chỉ đối chiếu các contract
JSON; nó không tự kiểm `%TRAIN%\X_window_features.csv`. Candidate-specific check
được chạy sau spatial export ở mục 15.4.

### 15.3. Event-balanced weights

```bat
%PY% %S2%\classification_v2_build_event_weights.py ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --output-dir %TRAIN%
%PY% %S2%\check_classification_v2_event_weights.py ^
  --event-weight-csv %TRAIN%\event_weight_manifest.csv ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --output-json %TRAIN%\check_event_weight_audit.json
```

Event weight chia mass của một native event cho các overlapping windows. Nó
không phải class weight. Không chạy global class-weight builder ở giai đoạn
này; class prior/weight phải tính riêng từ train role của từng outer fold.

### 15.4. Spatial sequence tensors

```bat
%PY% %S2%\classification_v2_export_spatial_sequences.py ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --frame-features-csv %RFRAME%\reviewed_frame_features.csv ^
  --output-dir %TRAIN% --compress
%PY% %S2%\check_classification_v2_spatial_sequences.py ^
  --npz %TRAIN%\X_spatial_sequences.npz ^
  --audit-json %TRAIN%\spatial_sequence_audit.json ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --train-mask-csv %TRAIN%\train_mask.csv ^
  --output-json %TRAIN%\spatial_sequence_validation.json
%PY% %S2%\check_classification_v2_feature_semantics.py ^
  --contract-json configs\classification_v2\feature_semantics_v2.json ^
  --tabular-x-csv %TRAIN%\X_window_features.csv ^
  --spatial-npz %TRAIN%\X_spatial_sequences.npz ^
  --output-json %TRAIN%\feature_semantics_audit.json
```

Không truyền `trainer_contract_v2.json` vào
`check_classification_v2_trainer_contract.py`: checker hiện chỉ hiểu schema v1
và đọc canonical paths. Candidate `%TRAIN%` đã được kiểm trực tiếp bằng
feature-semantics command trên; trainer-contract checker phải được nâng cấp
nhận versioned path trước snapshot.

NPZ chứa numeric arrays và masks, không phải ảnh xem trực tiếp. Phải có
`length_mask`, `observed_mask`, `quality/missing` semantics và row order khớp
`window_id`. Padding không được coi là frame thật. Mọi window mask-true phải có
`trainable_rows_with_missing_slots=0`; missing slot ở mask-false vẫn được giữ
và báo cáo, không xóa row để làm audit đẹp.

### 15.5. Auxiliary y cho hierarchy

```bat
%PY% %S2%\classification_v2_build_auxiliary_targets.py --root %TRAIN% ^
  --behavior-label-authority FROZEN_HUMAN_REVIEWED
%PY% %S2%\check_classification_v2_auxiliary_targets.py ^
  --csv %TRAIN%\y_auxiliary_targets.csv ^
  --audit-json %TRAIN%\auxiliary_targets_audit.json
```

Đây là deterministic decomposition của behavior y, không phải annotation độc
lập. Chỉ dùng làm auxiliary target/mask; không đưa vào X và không dùng hard
argmax cascade vào final 10-class head. Attribute reviewed độc lập, nếu bổ sung
sau này, phải có confidence/mask và một ablation riêng.

> Superseding posture note: posture is now an independent masked burst target.
> Only `lying`, `sitting`, `stand`, and fixed-feeder `eat` have bounded safe
> derivations after Behavior authority is frozen. Other behaviors remain
> unresolved until independent posture authority is available.

### 15.6. Provisional primary và source-shortcut controls — post-review only

Section 13 là authority tạo exact-view features. Legacy packet builder dưới đây
chỉ giữ làm Group-B compatibility reference; không chạy cho v6 cho tới khi nó
đọc per-view outputs mà không resample, reuse aggregate hoặc đổi view identity:

```bat
set TVIEW=%TRAIN%\temporal_views
%PY% %S2%\classification_v2_build_temporal_views.py ^
  --window-manifest %SEQ1%\sequence_window_manifest.csv ^
  --harmonized-frame-csv %SEQ1%\harmonized_frames.csv ^
  --temporal-interval-csv %SEQ1%\temporal_label_intervals.csv ^
  --output-dir %TVIEW%
%PY% %S2%\check_classification_v2_temporal_view_shortcuts.py ^
  --temporal-view-dir %TVIEW% ^
  --output-json %TVIEW%\temporal_shortcut_audit.json
```

Khi được nâng cấp, packaging phải giữ nguyên `view_type`, sampling pattern,
selected frames/timestamps, pair deltas, masks và ordered-key hash từ section
13. Không được biến S6@16 thành fixed-six/T6 hoặc tạo aggregate mới từ một view
khác.

Sau đó tạo source-matched masks và grouped spatial probe bổ sung:

```bat
%PY% %S2%\classification_v2_build_source_matched_views.py ^
  --window-manifest %TRAIN%\split_manifest.csv ^
  --output-dir %TRAIN%\source_matched_views
%PY% %S2%\check_classification_v2_source_matched_views.py ^
  --input-csv %TRAIN%\split_manifest.csv ^
  --view-csv %TRAIN%\source_matched_views\source_matched_view_manifest.csv
%PY% %S4%\check_classification_v2_grouped_spatial_source_probe.py ^
  --root %TRAIN% ^
  --grouped-roles %SPLIT%\q2_grouped_folds\q2_outer_inner_roles.csv ^
  --output-json %TRAIN%\domain_controls\spatial_source_probe_audit.json
```

Tabular source probe và availability-only behavior probe phải đợi image và
interaction manifests ở mục 16.5. Không dùng artifact cũ 39 feature hoặc số row
hard-code làm authority cho lineage mới.

Thiết kế trước review là `PROVISIONAL_PRIMARY_VIEW=T6_CONTIGUOUS`; final primary
chưa khóa. T6 của cả hai source phải chọn sáu source frames liên tiếp. Ở 30 FPS,
mọi pair có delta frame 1, delta time `1/30 s` và physical span `5/30 s`.
Không lấy sáu quantile rải trên legacy burst rồi gọi là T6. T8, T12 và T16 là
cross-length ablations. `S6@16` dùng offsets `[0,3,6,9,12,15]`, là legacy-only
diagnostic và không được trộn vào primary corpus.

Sau behavior review, chỉ khóa primary nếu source × behavior coverage, class
balance, session-safe folds, Hidden/review eligibility và source-shortcut probes
đều PASS. Không chọn primary bằng view có test accuracy cao nhất.

Trong primary X, `window_length_frames` phải là hằng số. Chỉ giữ duration,
effective FPS hoặc frame-delta khi provenance thời gian của hai source đã được
calibrate và cùng ý nghĩa; nếu không chúng là source proxy và phải bỏ khỏi
primary view. Chạy riêng `availability-only` probe cho ROI/social/context masks
trước khi diễn giải gain từ real context.

Checker đo exact structural signatures cho source/length/padding/observed,
quality/timing/availability và association audit-only tới behavior. Near-direct
shortcut chưa có mitigation evidence hợp lệ là hard stop. Không tự thêm
`--mitigation-evidence-json`; file đó phải là output versioned của control riêng.
Context gain còn phải có actor-only, availability-only, real-context,
matched-context subset và modality-dropout controls.

## 16. Image context và cache tái sử dụng

### 16.0. Disk và cache-lineage preflight

Ước lượng 37 GB actor, 26 GB interaction và khoảng 126 GB khi giữ cả individual
và packed NPY dựa trên lineage lịch sử 245.664 rows. Canonical rebuild phải lấy
row count từ reviewed manifest mới rồi tính lại dung lượng; không dùng estimate
lịch sử làm gate cứng. Trước khi có estimate mới, 160 GB trống chỉ là mức
preflight bảo thủ:

```bat
%PY% -c "import shutil; print(shutil.disk_usage('C:/').free // 2**30)"
```

Nếu không đủ, dừng trước cache full và đổi storage root trong versioned contract.
Không xóa cache cũ hoặc raw data để giải phóng chỗ mà chưa kiểm lineage/hash.

### 16.1. Resolver/index smoke trước cache

```bat
%PY% %S3%\check_classification_v2_image_loader.py ^
  --input-csv %SEQ1%\sequence_window_manifest.csv ^
  --video-root data\videos ^
  --legacy-crop-root "%L16CROPS%" ^
  --output-audit %TRAIN%\source_image_loader_smoke_audit.json ^
  --sample-per-source 24
%PY% %S3%\classification_v2_build_image_context_index.py ^
  --frame-features-csv %RFRAME%\reviewed_frame_features.csv ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --output-dir %TRAIN% --video-root data\videos ^
  --legacy-crop-root "%L16CROPS%"
%PY% %S3%\check_classification_v2_image_context_index.py ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --window-context-csv %TRAIN%\image_window_context_manifest.csv ^
  --audit-json %TRAIN%\image_context_index_audit.json
```

Checker cuối kiểm trực tiếp sáu row của case
`Pigs291119_000231 / ID_4 / 678..683` và hiện fail nếu chúng không loadable.
Manifest đã xác nhận path thực tế là
`data\videos\Pigs291119_000231_30fps.mp4`. Tuy nhiên checker hiện chưa assert
exact basename; phải bổ sung assertion này trước snapshot để một file sai nhưng
vẫn loadable không thể PASS. Missing media/bbox được đếm, không thay bằng ảnh
zero âm thầm.

### 16.2. Actor cache short run trong chính cache root

```bat
%PY% %S3%\classification_v2_build_image_cache.py ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --window-context-csv %TRAIN%\image_window_context_manifest.csv ^
  --output-dir %CACHE% --image-size 224 ^
  --max-contexts 256 --preview-jpg --preview-limit 64 ^
  --checkpoint-every 64
```

Mở các JPEG dưới `%CACHE%\preview_jpg_224_letterbox`. Pig phải giữ tỷ lệ bbox,
không bị ép kéo thành vuông. Canvas 224x224 chỉ là vùng padded đen. Audit phải
ghi policy:

```text
letterbox_preserve_aspect_rgb_pad_black_v1
```

File `.npy` là mảng `uint8 RGB HWC` để load nhanh và không mất dữ liệu vì JPEG.
Metadata không nằm bên trong từng NPY; nó nằm trong `manifest.csv`: context ID,
source/video/frame/bbox, aspect ratio, scale và padding. Folder con hash là key
deterministic chống trùng tên; preview JPEG có tên người đọc được.

### 16.3. Full actor cache sau preview PASS

```bat
%PY% %S3%\classification_v2_build_image_cache.py ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --window-context-csv %TRAIN%\image_window_context_manifest.csv ^
  --output-dir %CACHE% --image-size 224 ^
  --preview-jpg --preview-limit 500 --checkpoint-every 1000 ^
  --resume-from-partial
```

Lệnh dùng cùng `%CACHE%`, không tạo `smoke/resume_smoke` folder mới. Nếu partial
không tồn tại, bỏ `--resume-from-partial`; các NPY đã có vẫn được skip theo key.
Không dùng `--overwrite` trừ khi input hash hoặc resize policy đổi và đã tạo
`RUN_ID` mới.

```bat
%PY% %S3%\check_classification_v2_image_cache.py ^
  --cache-manifest %CACHE%\manifest.csv ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --window-context-csv %TRAIN%\image_window_context_manifest.csv ^
  --image-size 224 --sample-windows 24 ^
  --source-equivalence-contexts 24 ^
  --output-json %CACHE%\check_image_cache_audit.json
```

### 16.4. Hash và packed cache

```bat
%PY% %S3%\classification_v2_build_image_cache_integrity.py ^
  --cache-manifest %CACHE%\manifest.csv ^
  --workers 4 --checkpoint-every 5000 --resume
%PY% %S3%\check_classification_v2_image_cache_integrity.py ^
  --cache-manifest %CACHE%\manifest.csv ^
  --integrity-manifest %CACHE%\image_cache_integrity_manifest.csv ^
  --full --output-audit %CACHE%\integrity_release_check.json
%PY% %S3%\classification_v2_build_packed_image_cache.py ^
  --cache-manifest %CACHE%\manifest.csv ^
  --image-size 224 --output-dir %CACHE% ^
  --workers 4 --checkpoint-every 5000
%PY% %S3%\check_classification_v2_packed_image_cache.py ^
  --root %CACHE% --sample-size 64
```

Packed `.npy` là tensor memory-mapped duy nhất để training không mở hàng trăm
nghìn file nhỏ. `packed_image_cache_index.csv` ánh xạ `image_context_id` sang
row, nên tensor không phải file rỗng hoặc thiếu metadata.

Lệnh fresh không có `--resume`: packed builder yêu cầu tensor và partial audit
đã tồn tại khi flag này được bật. Chỉ thêm `--resume` sau một run bị ngắt và khi
source-manifest hash, shape, image size cùng partial audit đều khớp. Không dùng
`--overwrite` để che lineage mismatch.

### 16.5. Interaction context index

```bat
%PY% %S3%\classification_v2_build_interaction_context_index.py ^
  --root %TRAIN% --output-dir %TRAIN%
```

Sau khi hai window-context manifest tồn tại, chạy canonical source và
availability probes ở native-unit grain:

```bat
%PY% %S7%\classification_v2_evaluate_domain_controls.py ^
  --root %TRAIN% ^
  --native-mapping %SEQ1%\sequence_window_manifest.csv ^
  --grouped-roles %SPLIT%\q2_grouped_folds\q2_outer_inner_roles.csv ^
  --trainer-contract-json configs\classification_v2\trainer_contract_v1.json ^
  --train-ready-audit-json %TRAIN%\train_ready_audit.json ^
  --image-window-manifest %TRAIN%\image_window_context_manifest.csv ^
  --interaction-window-manifest ^
  %TRAIN%\interaction_window_context_manifest.csv ^
  --output-dir %TRAIN%\domain_controls
%PY% %S2%\check_classification_v2_domain_controls.py ^
  --source-probe-audit ^
  %TRAIN%\domain_controls\grouped_source_probe_audit.json ^
  --availability-probe-audit ^
  %TRAIN%\domain_controls\grouped_availability_behavior_probe_audit.json ^
  --feature-shift-audit ^
  %TRAIN%\domain_controls\domain_feature_shift_audit.json ^
  --spatial-source-probe-audit ^
  %TRAIN%\domain_controls\spatial_source_probe_audit.json ^
  --output-json %TRAIN%\domain_controls\check_domain_controls.json
```

Context readiness/missingness là audit/mask, không phải evidence label. Partner
selection phải dựa trên geometry và cùng frame/video, không dựa vào target
`fight` hoặc `social-nose`.

Availability probe chỉ nhận `window_image_context_complete`,
`scene_context_ready` và `scene_partner_context_ready`. Không thêm
`interaction_context_ready`: implementation hiện tại gate field này bằng
interaction label, nên dùng nó sẽ tạo target-derived shortcut.

### 16.6. Interaction visual cache: short rồi full

```bat
%PY% %S3%\classification_v2_build_visual_interaction_cache.py ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --output-dir %VCACHE% --image-size 224 ^
  --padding-ratio 0.15 --max-contexts 128 ^
  --preview-limit 64 --checkpoint-every 32
```

Kiểm preview có actor và partner/context đúng, không truyền label sang
bystander. Sau PASS, tiếp tục cùng folder:

```bat
%PY% %S3%\classification_v2_build_visual_interaction_cache.py ^
  --frame-context-csv %TRAIN%\image_frame_context_manifest.csv ^
  --output-dir %VCACHE% --image-size 224 ^
  --padding-ratio 0.15 --preview-limit 500 ^
  --checkpoint-every 1000
%PY% %S3%\check_classification_v2_visual_interaction_cache.py ^
  --cache-dir %VCACHE% --sample-tensors 128 ^
  --output-json %VCACHE%\check_visual_interaction_cache.json
%PY% %S3%\classification_v2_build_packed_image_cache.py ^
  --cache-manifest %VCACHE%\visual_context_manifest.csv ^
  --available-column visual_context_available ^
  --image-size 224 --output-dir %VCACHE% ^
  --workers 4 --checkpoint-every 5000
```

Full visual-cache run cố ý không resume từ subset 128. Builder hiện áp
`max_contexts` trước khi dựng same-frame partner lookup; actor ở biên subset có
thể bị đánh `missing_nearest_partner_bbox` dù partner nằm ngoài subset. Full run
không `--resume` sẽ dựng lookup từ toàn manifest và tính lại các row đó. Resume
chỉ hợp lệ sau khi chính full-selection run bị ngắt. Packed interaction cache
cũng chỉ dùng `--resume` khi packed tensor và partial audit đã tồn tại, khớp hash.

Nếu context thiếu, giữ row và availability mask. Không được dùng
`context_available=true` như một proxy trực tiếp cho interaction label; model
phải có modality dropout và missingness ablation ở giai đoạn training.

## 17. Final data gate và snapshot

### 17.0. Technical reference gate

Current bounded code/data-generation evidence is checked independently from
human coverage. Hai reference root và `%UROOT%` chỉ được đọc; audit mới được
ghi dưới `%HANDOFF%` thuộc `%AROOT%`, không overwrite reference hoặc human
output:

```bat
set S9=scripts\classification_v2\09_final_release_audit
set BASE=outputs\classification_v2\rebuilds
set REF_ROOT=%BASE%\scientific_smoke_identifier_v2_20260713
set REF_REPEAT=%BASE%\scientific_smoke_identifier_v2_repeat_20260713
if not exist "%HANDOFF%" mkdir "%HANDOFF%"
%PY% %S9%\check_classification_v2_identifier_v2_lineage.py ^
  --root %REF_ROOT% ^
  --repeat-root %REF_REPEAT% ^
  --output-json %HANDOFF%\technical_reference_identifier_audit.json
%PY% %S9%\check_classification_v2_technical_smoke_gate.py ^
  --root %REF_ROOT% ^
  --repeat-root %REF_REPEAT% ^
  --output-json %HANDOFF%\technical_reference_smoke_gate.json
```

Expected statuses are `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED` and
`PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED`, with 688 frame rows, 63 native
units, 438 ordered windows, exact 110-feature X, zero trainable spatial gaps,
and 8/8 deterministic stage pairs. These gates must never bypass sections 8A,
11, or 12. Final reviewed data still requires complete human decisions.

### 17.1. Hash artifact chính

```bat
certutil -hashfile %SRC%\merged_frame_objects.csv SHA256
certutil -hashfile %SRC%\merged_frame_objects_audit.json SHA256
certutil -hashfile %SRC%\merged_frame_objects_lineage.json SHA256
certutil -hashfile %SRC%\mixed_source_lineage_gate.json SHA256
certutil -hashfile %FRAMELOCAL% SHA256
certutil -hashfile %HREV%\hidden_review_unit_manifest.csv SHA256
certutil -hashfile %HDEC%\hidden_review_decisions.csv SHA256
certutil -hashfile %HREV%\hidden_reviewed_frame_features.csv SHA256
certutil -hashfile %SEQ0%\native_review_evidence.csv SHA256
certutil -hashfile %REV%\behavior_review_authority.json SHA256
certutil -hashfile %RFRAME%\reviewed_frame_features.csv SHA256
certutil -hashfile %SEQ1%\sequence_window_manifest.csv SHA256
certutil -hashfile %NATIVE%\native_temporal_unit_manifest.csv SHA256
certutil -hashfile %SPLIT%\q2_grouped_folds\q2_outer_inner_roles.csv SHA256
if exist "%SPLIT%\native_oof_folds\native_oof_fold_manifest.csv" ^
  certutil -hashfile ^
    %SPLIT%\native_oof_folds\native_oof_fold_manifest.csv SHA256
certutil -hashfile %TRAIN%\X_window_features.csv SHA256
certutil -hashfile %TRAIN%\X_spatial_sequences.npz SHA256
certutil -hashfile %TRAIN%\feature_semantics_audit.json SHA256
certutil -hashfile %PIGREV%\pig_strenet_artifact_audit.json SHA256
certutil -hashfile %PIGREV%\pig_strenet_artifact_gate.json SHA256
certutil -hashfile %PIGREV%\run_manifest.json SHA256
certutil -hashfile %CACHE%\manifest.csv SHA256
certutil -hashfile %CACHE%\packed_rgb_224_letterbox.npy SHA256
certutil -hashfile %VCACHE%\visual_context_manifest.csv SHA256
certutil -hashfile %VCACHE%\packed_rgb_224_letterbox.npy SHA256
```

Ghi hash, row count, schema version, `RUN_ID`, Git SHA, dirty-worktree status,
Python/PyTorch/OpenCV version và input hash vào run manifest trước training.
Không gọi một folder là final nếu hash/audit chưa khóa.

### 17.2. Versioned snapshot hard stop

Artifact dưới `%AROOT%` là candidate versioned của agent. `%R%` chỉ là data
root thuộc human-review lineage. Không copy đè canonical để né path contract.
Phải phân biệt component PASS với integration PASS và luôn giữ hash lineage:

1. Identifier-v2, target-independent Hidden design, exact video resolver,
   per-view loader, fold-local preprocessing, native-event weighting và inner
   native-unit selection đã PASS ở code/fixture hoặc bounded audit.
2. Generated reviewed-Q2 contract v2 được build từ artifact map explicit và
   không dùng `configs/classification_v2/data_contract_v2.json` làm fallback.
3. `classification_v2_write_model_input_manifest.py` nhận generated contract và
   ghi manifest vào agent root, không tự suy child folder canonical.
4. Snapshot writer và P0 checker bind path, order, hash, lineage và project root
   explicit; integration này đã PASS bằng fixture tests.
5. Historical runner/OOF vẫn là gate riêng. Code contract PASS không tự cấp
   quyền chạy model smoke, full training hoặc full OOF.
6. Canonical/historical outputs có mixed lineage hoặc known positional mismatch
   nên không được dùng để lấp các path còn thiếu.
7. Final reviewed lineage vẫn phải chạy lại identifier, exact basename,
   source/missingness và ordered-interaction gates trên chính bytes của nó.

Vì vậy **không có lệnh snapshot/full nào được phép ghi trực tiếp vào `%R%`**.
Sau khi người dùng handoff đủ Hidden và behavior review, agent dùng sequence
explicit dưới đây; mọi output phát sinh đều nằm dưới `%AROOT%`.

Không chạy snapshot riêng lẻ trước contract/manifest. Thứ tự executable duy
nhất cho phase này nằm ở mục 17.2.1.

Contract phải tham chiếu duy nhất artifact của cùng `RUN_ID`, fixed temporal
view, 224 caches và Q2 primary roles; native OOF manifest chỉ bắt buộc nếu chạy
engineering sensitivity đã khai báo. Checkpoint phải ghi snapshot, config,
feature-whitelist, cache và fold hashes.

Trước freeze final, chạy identifier audit với
`--require-interaction-lineage`. Preflight block `05` bắt buộc nhận
`--lineage-audit-json`; audit phải bind đúng bytes của X/y/mask/weights,
spatial, image và interaction trong snapshot. Authorization v2 tiếp tục bind
snapshot ID, snapshot SHA, lineage SHA, ordered-window SHA, config và Git SHA.
Audit bounded hiện tại có human authorization false nên P0 phải FAIL.

### 17.2.1. Command sequence sau handoff, chỉ ghi dưới agent root

Chỉ chạy block này sau khi người dùng gửi `RUN_ID`,
`REVIEW_STAGE=behavior_complete`,
reviewer và Git SHA. Không tự dò một root khác có cùng tên. Các writer dưới đây
đều phải dùng cùng `AUDIT_RUN_ID`; không đổi root giữa các bước:

```bat
set S2=scripts\classification_v2\02_train_ready_exports
set S5=scripts\classification_v2\05_preflight_authorization
set TEMPLATE=configs\classification_v2\reviewed_q2_data_contract_template_v1.json
set LAYOUT=configs\classification_v2\reviewed_q2_artifact_layout_v1.json
set CONTRACTS=%AROOT%\contracts
set SNAP=%AROOT%\data\14_training_snapshot
set PREFLIGHT=%AROOT%\preflight

%PY% %S2%\classification_v2_write_reviewed_q2_artifact_map.py ^
  --human-review-run-id %RUN_ID% ^
  --agent-audit-run-id %AUDIT_RUN_ID% ^
  --template-json %TEMPLATE% ^
  --layout-json %LAYOUT% ^
  --output-json %CONTRACTS%\reviewed_q2_artifact_map.json ^
  --project-root %CD%
%PY% %S2%\classification_v2_build_versioned_data_contract.py ^
  --template-json %TEMPLATE% ^
  --artifact-map-json %CONTRACTS%\reviewed_q2_artifact_map.json ^
  --output-json %CONTRACTS%\data_contract.json ^
  --project-root %CD%
%PY% %S2%\classification_v2_write_model_input_manifest.py ^
  --data-contract-json %CONTRACTS%\data_contract.json ^
  --output-json %CONTRACTS%\model_input_contract.json ^
  --project-root %CD%
%PY% %S2%\classification_v2_freeze_training_snapshot.py ^
  --contract-json %CONTRACTS%\data_contract.json ^
  --output-json %SNAP%\snapshot.json
%PY% %S2%\check_classification_v2_training_snapshot.py ^
  --snapshot-json %SNAP%\snapshot.json ^
  --contract-json %CONTRACTS%\data_contract.json ^
  --output-json %SNAP%\training_snapshot_check.json
%PY% %S5%\check_classification_v2_reviewed_q2_p0_preflight.py ^
  --data-contract-json %CONTRACTS%\data_contract.json ^
  --snapshot-json %SNAP%\snapshot.json ^
  --output-json %PREFLIGHT%\reviewed_q2_p0_preflight.json ^
  --project-root %CD%
```

Block trên chỉ tạo contract, manifest, snapshot và audit; nó không chạy GUI,
không tự apply decision và không authorize full OOF. `model_smoke_authorized`
phải được kiểm riêng trước model smoke; `full_oof_authorized` vẫn false cho
đến khi các gate tiếp theo được hoàn tất.

### 17.3. Model finalist gate

Factory từ commit `07ed768` hỗ trợ `smoke_cnn`, ResNet18 và ResNet34 bằng một
contract chung. Exact pretrained enum và ImageNet RGB normalization được khóa;
unit test và structural audit chỉ dùng random-init nên không tải weight. Full
runner lịch sử vẫn là engineering runner 64 px và không tự trở thành finalist.
Không gọi interface PASS hoặc một random-init forward là final Q2 classifier.

Trên cùng frozen snapshot/folds, development phải tách từng biến:

```text
resolution: ResNet18-160 -> ResNet18-224
backbone:   ResNet18-224 -> ResNet34-224
temporal:   masked pooling -> TCN -> small Transformer nếu TCN chưa đủ
modality:   actor -> geometry/motion -> all-ROI -> social -> union context
imbalance:  event CE | effective-number CE | Balanced Softmax, chọn một
```

Kiểm interface, shape và capacity trước khi có snapshot bằng dry-run không
optimizer, không data I/O và không pretrained download:

```bat
%PY% %S4%\check_classification_v2_visual_backbones.py --dry-run
```

Gate synthetic tiếp theo không đọc project data, dùng 20 event cân bằng và
chạy hai lần để kiểm deterministic semantic signature:

```bat
%PY% %S4%\check_classification_v2_visual_tiny_overfit.py ^
  --backbone-name resnet18 --image-size 160 --steps 30 ^
  --device cuda --repeatability-runs 2 --overwrite
```

Gate này yêu cầu gradient hữu hạn/nonzero ở backbone và final head, accuracy
tiny-overfit tối thiểu 0,95, loss ratio tối đa 0,25 và resume logit delta bằng
0. BatchNorm được recalibrate sau khi optimizer dừng để metric chấm ở eval
mode; đây chỉ là correctness policy của tiny fixture, không khóa freeze/BN
policy cho active-data pilot. Audit luôn có `synthetic_only=true`,
`training_snapshot_allowed=false` và `full_oof_allowed=false`.

Mỗi candidate phải qua one-batch forward/backward, tiny overfit 16-64 native
event, checkpoint resume, AMP/runtime/VRAM, cache-only I/O và representative
one-fold development run. Context candidate phải qua missingness controls ở mục
15.6. Khóa tối đa F0 actor-temporal, F1 actor+geometry/motion/ROI, F2 final
multimodal và một F2-no-hierarchy ablation nếu cần.

### 17.4. Full OOF và postrun gate

Quyền full đã được cấp có điều kiện; semantic/config/hash đổi thì short gates
phải chạy lại. Chỉ sau finalist lock và clean snapshot mới chạy block `05`
preflight, authorization file, launch-packet writer và execution gate. Không tự
gõ một lệnh `--full` thủ công; dùng exact command/hash do launch packet sinh ra.

Sau full, bắt buộc kiểm prediction count/schema, collapse về native unit,
pooled 10-class macro-F1, rare/interaction/ROI/posture/locomotion metrics,
per-source/video/recording support, paired uncertainty, cross-fit calibration,
confusion-focus, experiment registry và block `09` completion gate. Outer OOF
prediction không được quay lại chọn architecture, threshold hoặc loss.

## 18. Lỗi thường gặp

**Có `reviewed_frame_features.csv` nhưng review coverage FAIL**

File chỉ chứng minh apply script chạy, không chứng minh con người đã review đủ.
Tiếp tục GUI cho tới khi `--require-complete` PASS; không hạ gate.

**Merge có 13 CVAT XML**

Đã dùng cả directory. Chạy lại bằng allowlist 12 file; audit riêng `000263`
trước khi thêm. Không xóa XML nguồn.

**Reviewed rebuild chạy rất nhanh và dùng output cũ**

Thiếu `--disable-fast-reuse`. Xóa không phải giải pháp. Tạo output versioned
mới và rebuild đúng input/hash.

**Preflight/full đọc canonical thay vì `%R%`**

Đây là blocker code, không phải lý do copy artifact sang canonical. Dừng ở mục
17.2, bổ sung contract/path overrides và tests, rồi freeze lại snapshot. Không
chạy full với cache 224 nhưng X/folds/event weights từ lineage 64/canonical cũ.

**Ảnh 224x224 trông như vuông**

Kiểm `resize_policy` và preview. Letterbox giữ nguyên tỷ lệ pig rồi padding;
square canvas không đồng nghĩa square-stretch. Nếu pig bị bóp méo, gate FAIL.

**NPY không có tên video/label khi mở**

NPY chỉ chứa pixel/tensor. Dùng `manifest.csv` hoặc packed index để tra metadata;
label cố ý không nằm trong image tensor để tránh leakage.

**Tên folder cache là hash**

Đó là deterministic storage key. Dùng `preview_jpg_*` để review bằng tên
source/video/pig/frame. Không rename file cache thủ công vì sẽ phá manifest.

**Hidden=No của CVAT được xem là visible trusted**

Đây là lỗi contract. Trước human decision, CVAT phải là
`untrusted_tracking_derived`. Không bật `--trust-hidden`, không tự chuyển toàn bộ
No thành trusted và không dùng availability/trust field như behavior feature.

**Hidden random audit có correction rate khác high-risk**

Đây là kết quả dự kiến. Chỉ random cohort với sampling weight dùng để ước lượng
false-negative prevalence. High-risk cohort dùng đo correction yield và tìm lỗi,
không được báo như prevalence của toàn dataset.

**GUI mất decision sau khi mở lại**

Phải dùng cùng output directory. Bản GUI hiện tại resume CSV và fail trên
duplicate ID. Không chạy script cũ/wrapper hoặc dùng `--fresh`.

**`review_later` còn trong final decisions**

Complete gate phải FAIL. Unit đó không được vào training cho tới khi có quyết
định cuối. Không đổi `review_later` thành accept hàng loạt.

**Mixed/transition làm giảm row hợp lệ**

Đúng policy: giữ row nhưng main-train mask có thể false. Không dùng
`--exclude-mixed-windows` và không xóa row để cân bằng số liệu.

**Class imbalance nặng**

Không oversample theo raw window count. Dùng event weight; class weight/loss chỉ
fit từ training fold. So sánh CE, effective-number CE và Balanced Softmax từng
thí nghiệm, không cộng nhiều cơ chế cùng lúc.

## 19. Tiêu chí PASS cuối

Dataset chỉ được gọi là `reviewed train-ready candidate` khi tất cả mục sau
PASS:

- [ ] Input hashes và allowlist 12 behavior XML đã khóa.
- [ ] Mọi artifact thuộc cùng `RUN_ID`; không canonical intermediate lẫn lineage.
- [ ] Raw `data\` không thay đổi.
- [ ] P10 bind đúng canonical export hash `fbd6300...cad3` và 72.880 rows.
- [ ] Ba actor source-policy lỗi vắng downstream nhưng còn đủ audit accounting.
- [ ] Legacy export giữ 4.555 native actor bursts, mỗi burst đủ 16 frames.
- [ ] CVAT anchor interval đúng 6 frame và non-anchor kế thừa target.
- [ ] Scientific smoke dùng complete units, không leading-row temporal scope.
- [ ] Case `000085 / ID_4 / anchor 1020 = social-nose + interaction` PASS.
- [ ] Mọi untrusted `Hidden=Yes` có item; trusted Yes đạt quota phân tầng.
- [ ] `Hidden=No` có high-risk, stratified-random và clean-control audit.
- [ ] CVAT chưa review giữ `untrusted_tracking_derived`, không silent trust.
- [ ] Legacy Hidden frame/object coverage đầy đủ; P10 không thay thế decision.
- [ ] Hidden decisions unique, resolved và áp đúng frame/object key.
- [ ] Hidden apply giữ nguyên row count và non-Hidden source columns.
- [ ] `Yes->No`, `No->Yes`, weighted false-negative rate có audit.
- [ ] Hidden weighted/high-risk CI và predeclared threshold gate PASS.
- [ ] Final Hidden manifest xác nhận target-independent và target-derived audit rỗng.
- [ ] Frame-local, Hidden-reviewed và behavior-reviewed rows bằng nhau.
- [ ] Duplicate `temporal_unit_key=0` và duplicate `review_unit_id=0`.
- [ ] Không output mới dùng `window_uid`.
- [ ] ROI/motion/posture/interaction templates đúng policy.
- [ ] `playwithtoy` review coverage đầy đủ.
- [ ] Bốn decision CSV đủ schema, không pending/missing/review_later.
- [ ] Legacy behavior coverage đầy đủ ở complete native unit 16-frame.
- [ ] Applied, excluded và corrected frame/unit counts có audit.
- [ ] Label distribution trước/sau truy được về decision.
- [ ] Reviewed windows rebuild bằng `--disable-fast-reuse`.
- [ ] Stable/mixed/transition và main-train-valid counts được báo cáo.
- [ ] Native CVAT 6f, legacy 16f, không duplicate key.
- [ ] Temporal evidence/determinism audit PASS, review scores không vào X.
- [ ] Recording-date/session leakage bằng 0 trong split/folds.
- [ ] Class-by-fold và source-by-fold support được báo cáo.
- [ ] Q2 outer/inner roles được khóa/hash; native manifest chỉ khi sensitivity
      run đã predeclare.
- [ ] X whitelist không có label/review/manual/ID/path/policy/split field.
- [ ] X/y/mask/weights/spatial/image/interaction row/order/key hash khớp.
- [ ] Identifier-v2 có `scene_frame_uid`/object `frame_uid` và lineage audit PASS.
- [ ] Không có global normalization/class weight fit trước fold.
- [ ] Legacy crop và CVAT video+bbox loader smoke PASS.
- [ ] Case `000231` exact basename `_30fps.mp4` được checker assert và PASS.
- [ ] Actor cache dùng letterbox, preview không méo, checksum PASS.
- [ ] Packed cache index/tensor equivalence PASS.
- [ ] Interaction context giữ missing mask và không label-select partner.
- [ ] T6 provisional primary, T8/T12/T16 ablations và legacy-only S6@16 có
      identity, pair scope và aggregate riêng.
- [ ] Grouped source/length/padding/missingness controls đã được audit.
- [ ] Final artifact hashes, config, code SHA và environment đã ghi.

Nếu bất kỳ mục nào FAIL, kết luận là `NOT TRAIN-READY`. Không dùng số row lớn,
training accuracy hoặc việc script exit 0 để thay cho gate thiếu. Sau PASS, bước
kế tiếp chỉ là model/local smoke theo roadmap, chưa phải full OOF.

Đặc biệt, `technically clean`, P0-P10 PASS hoặc export hash đúng không thể làm
hai checklist Hidden/behavior tự PASS.

Full OOF chỉ được phép khi thêm các mục sau PASS:

- [ ] Reviewer-agreement audit đã có, hoặc single-review limitation được khóa.
- [ ] Versioned data contract trỏ hoàn toàn tới một `RUN_ID`, không fallback.
- [ ] Snapshot/checkpoint bind dataset, cache, fold, whitelist và config hashes.
- [ ] ResNet resolution/backbone controls và temporal baseline đã chạy đúng cặp.
- [ ] F0/F1/F2 và inner selection rule được khóa trước outer predictions.
- [ ] One-batch, tiny-overfit, resume, AMP/runtime/VRAM và one-fold PASS.
- [ ] Preflight/authorization/execution cùng snapshot, lineage, ordered-window,
      config và Git SHA.
- [ ] Full predictions đủ count và collapse về native unit không mất unit.
- [ ] Calibration, confusion, grouped/source metrics, registry và completion PASS.

Theo authority hiện tại, synthetic và representative exact-view contract đã
PASS, nhưng full views chưa được phép build trước behavior apply. Active
reviewed packet, final primary lock và model-loader integration chưa tồn tại.
Vì vậy full final-view build, training và scientific full OOF vẫn
`FAIL/BLOCKED` cho tới khi Hidden và behavior review hoàn tất.
