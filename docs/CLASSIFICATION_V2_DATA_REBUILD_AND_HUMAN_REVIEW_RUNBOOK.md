# Classification V2 Data Rebuild And Human Review Runbook

Tài liệu này hướng dẫn tái tạo dữ liệu `classification_v2` từ việc xuất
`legacy_frame_object_annotations.csv` đến reviewed data, leakage-safe model
inputs, model smoke và ranh giới cho phép chạy full OOF. Tài liệu không tự khởi
chạy full training hoặc full OOF.

Trạng thái PASS/FAIL hiện hành nằm trong
`docs/CLASSIFICATION_V2_CURRENT_STATE.md`. Hiện tại luồng đang dừng ở human
Hidden review; không nhảy tới temporal rebuild hoặc model training.

Audit ngày 2026-07-13 kết luận: data-generation short chain đang PASS kỹ thuật,
nhưng end-to-end reviewed lineage tới full OOF chưa PASS. Human review chưa đủ;
versioned snapshot và current full runner chưa có cùng path/resolution/model
contract. Các hard stop tương ứng được ghi ngay tại từng stage bên dưới.

Audit lần hai cùng ngày xác nhận luồng **chưa hoàn toàn scientific-release
ready**. Các lỗi còn lại không làm sai reference legacy, nhưng đang chặn snapshot:
canonical outputs lẫn lineage, identifier `frame_uid` lệch contract object-row,
Hidden stopping rule chưa được code hóa và cache/full runner chưa dùng cùng một
versioned contract.

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

Trạng thái dữ liệu hiện có chưa đủ để gọi là human-reviewed final. Audit gần
nhất có 4.670 review unit bắt buộc nhưng chỉ 3 decision row, trong đó 2 active,
1 pending và còn thiếu 4.667 unit. File `reviewed_frame_features.csv` hiện tại
chỉ là artifact kỹ thuật đã đi qua apply logic, chưa phải ground truth sạch.
Hidden lineage cũng chưa hoàn tất human decision; đặc biệt CVAT No hiện không
được coi là visible trusted chỉ vì tracking đã xuất thuộc tính đó.

## 2. Bất biến khoa học

- Không sửa, xóa, đổi tên hoặc overwrite bất kỳ file nào dưới `data\`.
- Không drop row để làm số đẹp. Exclude phải giữ row và ghi mask/weight/action.
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
- Identifier contract mong muốn là `frame_uid` cho frame/object row. Code hiện
  dùng `frame_uid` như scene-frame key dùng chung giữa nhiều pig; vì vậy không
  được deduplicate hoặc join object row chỉ bằng cột này trước khi migration.
- Trong trạng thái chuyển tiếp, object row phải dùng composite provenance key
  hoặc `image_context_id`; mọi join/apply vẫn phải chứng minh one-to-one.

## 3. Sơ đồ dữ liệu

```text
legacy_dense_tracklet_map.csv
  -> legacy_frame_object_annotations.csv
  -> merge với 12 CVAT behavior XML
  -> context policy -> geometry -> ROI -> motion/social/posture
  -> complete-unit scientific smoke scope (legacy 16f, CVAT scene block 6f)
  -> two-sided Hidden review: Yes + risk/random/control No
  -> hidden_reviewed_frame_features.csv
  -> temporal harmonization: legacy 16f, CVAT 6f
  -> unreviewed sequence windows
  -> review_unit_manifest + 4 policy templates
  -> GUI decision CSVs
  -> complete-decision audit
  -> apply decisions, không drop frame
  -> reviewed_frame_features.csv
  -> reviewed sequence windows
  -> native temporal units
  -> Q2 outer/inner folds + optional native leave-one-group sensitivity
  -> whitelisted X/y/masks/event weights/spatial sequences
  -> fixed-6 primary temporal view + native-length ablation
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
legacy export smoke -> legacy full export
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

Mở **CMD**, không chạy các lệnh pipeline trong PowerShell. Đổi `RUN_ID` cho mỗi
lineage mới; không tái dùng thư mục của một cấu hình khác.

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PYTHONPATH=%CD%\src
set PY=C:\Users\ironh\anaconda3\envs\pig_project\python.exe
REM Replace the placeholders and use a new directory for every semantic rebuild.
set RUN_ID=c2v2_rebuild_YYYYMMDD_vN
set REVIEWER_NAME=replace_with_reviewer_id
set R=outputs\classification_v2\rebuilds\%RUN_ID%
set SM=%R%\00_smoke
set SRC=%R%\01_source_full
set FEAT=%R%\02_frame_features
set HREV=%R%\03_hidden_review
set SEQ0=%R%\04_sequence_unreviewed
set REV=%R%\05_review_units
set DEC=%R%\06_review_decisions
set RFRAME=%R%\07_reviewed_frames
set SEQ1=%R%\08_sequence_reviewed
set NATIVE=%R%\09_native_units
set SPLIT=%R%\10_grouped_splits
set TRAIN=%R%\11_train_ready
set CACHE=%R%\12_actor_cache_224_letterbox
set VCACHE=%R%\13_interaction_cache_224_letterbox
set SNAP=%R%\14_training_snapshot
set MODEL=%R%\15_model_development
```

Không chạy nguyên văn khi `RUN_ID` còn chứa `YYYYMMDD_vN`, và không dùng lại
một `%R%` đã có artifact. Active Hidden v5 là lineage review hiện hành riêng;
trạng thái và đường dẫn của nó nằm trong `CLASSIFICATION_V2_CURRENT_STATE.md`.
Không phải mọi builder đều chặn output tồn tại. Exporter legacy, temporal smoke
selector và một số builder CSV có thể ghi đè. Khi khởi tạo lineage mới, chạy
guard sau đúng một lần trước khi tạo artifact:

```bat
if exist "%R%" (echo ERROR: RUN_ID already exists: %R% & exit /b 2)
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
`outputs\classification_v2`. Mọi artifact của lần rebuild này nằm dưới `%R%`.
Smoke và full có tên theo vai trò, không theo lỗi thử nghiệm.

Không dùng canonical folder hiện có làm authority cho rebuild mới. Audit đã thấy
`frame_features\geometry_audit.json` chỉ có 173.664 row, gồm 100.800 CVAT row và
còn chứa dataset `Tracking_annotation_Pigs291119_000263_30fps`. Trong khi đó,
enhanced audit cùng canonical tree có 245.664 row, gồm 172.800 CVAT row từ
allowlist 12 XML. Đây là mixed lineage. Mọi stage phải đọc artifact ngay trước
nó dưới cùng `%R%`; không được ghép một canonical intermediate vào lineage mới.

## 5. Kiểm kê nguồn bất biến

Kiểm tra file tồn tại trước khi chạy. Lệnh chỉ đọc:

```bat
dir /b data\raw\legacy_full_multigt_masked_nodup_16f
dir /b data\annotations\tracking
dir /b data\videos
certutil -hashfile ^
  data\raw\legacy_full_multigt_masked_nodup_16f\legacy_dense_tracklet_map.csv ^
  SHA256
certutil -hashfile data\annotations\roi\ROI_annotations.coco.json SHA256
```

Thư mục tracking hiện có 13 XML nhưng lineage hành vi đã xác nhận chỉ có 12
video. **Không** truyền cả directory bằng `--cvat-tracking-dir`, vì file sau có
thể là tracking-only hoặc stale cho behavior:

```text
Tracking_annotation_Pigs291119_000263_30fps.xml
```

Chỉ thêm `000263` sau một audit riêng chứng minh XML có Behavior labels đúng,
temporal contract đúng và không làm thay đổi source lineage ngoài dự kiến.

Behavior XML allowlist hiện tại:

```text
Pigs281119_000085_30fps.xml
Pigs281119_000114_30fps.xml
Pigs291119_000216_30fps.xml
Pigs291119_000225_30fps.xml
Pigs291119_000226_30fps.xml
Pigs291119_000231_30fps.xml
Pigs291119_000233_30fps.xml
Pigs291119_000302_30fps.xml
Pigs301119_000327_30fps.xml
Pigs301119_000328_30fps.xml
Pigs301119_000329_30fps.xml
Pigs301119_000330_30fps.xml
```

Ghi lại SHA256 của input trong audit notebook hoặc run manifest. Hash khác ở
lần chạy sau nghĩa là lineage mới và phải chạy short chain lại.

## 6. Xuất legacy frame-object annotations

### 6.0. Reference hiện có và lineage mới

File người dùng nêu là derived reference, không phải raw input bất biến:

```text
outputs\legacy_full_multigt_masked_nodup_16f\exports_v2\
legacy_frame_object_annotations.csv
```

Audit hiện tại của reference: 72.864 row, 4.554 tracklet, mỗi tracklet đúng 16
frame, 0 duplicate tracklet/frame, 0 behavior thay đổi trong burst, 0 bbox
invalid và đủ 10 behavior. SHA256 hiện tại là:

```text
adbdb572b976e9f63cff5f9b29ced649f37fa80dd382336b3678f71ac50ff636
```

Xác nhận lại bằng lệnh chỉ đọc:

```bat
certutil -hashfile ^
  outputs\legacy_full_multigt_masked_nodup_16f\exports_v2\^
legacy_frame_object_annotations.csv SHA256
```

Không overwrite reference này khi rebuild. Mỗi run phải xuất bản mới dưới
`%SRC%\legacy_export`, rồi so row/schema/hash với reference. Hash khác không tự
động là lỗi, nhưng phải giải thích bằng code SHA, input hash hoặc config diff.

### 6.1. Short smoke bắt buộc

```bat
if exist "%SM%\legacy_export\legacy_frame_object_annotations.csv" ^
  (echo ERROR: smoke legacy export exists & exit /b 2)
%PY% src\legacy_burst_recovery\export_legacy_annotations.py ^
  --dense-csv ^
  data\raw\legacy_full_multigt_masked_nodup_16f\legacy_dense_tracklet_map.csv ^
  --output-dir %SM%\legacy_export ^
  --expected-sequence-length 16 ^
  --anchor-relative-frames 0,3,6,9,12,15 ^
  --expected-pig-count 8 ^
  --max-rows 32
```

Với input hiện tại, smoke tham chiếu tạo 32 object row, 16 frame, 2 tracklet,
không có bbox invalid và các sequence đủ 16 frame. Nếu số khác, phải đọc summary
và xác định do input hash thay đổi hay do regression. Không thêm
`--training-only`: context row phải còn để tạo social feature. Không thêm
`--require-full-8-for-eval`: thiếu full-pen context không được làm mất sample.

### 6.2. Full export sau khi smoke PASS

```bat
if exist "%SRC%\legacy_export\legacy_frame_object_annotations.csv" ^
  (echo ERROR: full legacy export exists & exit /b 2)
%PY% src\legacy_burst_recovery\export_legacy_annotations.py ^
  --dense-csv ^
  data\raw\legacy_full_multigt_masked_nodup_16f\legacy_dense_tracklet_map.csv ^
  --output-dir %SRC%\legacy_export ^
  --expected-sequence-length 16 ^
  --anchor-relative-frames 0,3,6,9,12,15 ^
  --expected-pig-count 8
```

Output chính:

```text
%SRC%\legacy_export\legacy_frame_object_annotations.csv
```

Exporter hiện không có `--overwrite` guard hoặc audit JSON; hai `if exist` trên
là bắt buộc. Gate: file không rỗng, bbox invalid được báo cáo, sequence
thiếu/ngoài range được ghi rõ và không row nào bị lọc âm thầm. Hash output phải
được ghi ở mục 17; downstream merge audit phải xác nhận row/schema/source.

Các cột legacy như `behavior_coarse` và `use_for_*_training` là metadata
target-derived để tương thích/audit, không phải feature. Context policy hiện
hành phải recompute eligibility theo `schema.py`; review policy mới là authority
cho interaction/ROI/motion/posture. Cấm đưa các cột legacy này vào X hoặc dùng
chúng để route modality, partner hay image context.

## 7. Merge legacy và CVAT

Khai báo allowlist một lần trong cùng cửa sổ CMD:

```bat
set X01=data\annotations\tracking\Pigs281119_000085_30fps.xml
set X02=data\annotations\tracking\Pigs281119_000114_30fps.xml
set X03=data\annotations\tracking\Pigs291119_000216_30fps.xml
set X04=data\annotations\tracking\Pigs291119_000225_30fps.xml
set X05=data\annotations\tracking\Pigs291119_000226_30fps.xml
set X06=data\annotations\tracking\Pigs291119_000231_30fps.xml
set X07=data\annotations\tracking\Pigs291119_000233_30fps.xml
set X08=data\annotations\tracking\Pigs291119_000302_30fps.xml
set X09=data\annotations\tracking\Pigs301119_000327_30fps.xml
set X10=data\annotations\tracking\Pigs301119_000328_30fps.xml
set X11=data\annotations\tracking\Pigs301119_000329_30fps.xml
set X12=data\annotations\tracking\Pigs301119_000330_30fps.xml
```

Không dùng `--trust-hidden` mặc định. Hidden vẫn được bảo tồn, nhưng không tự
động reject/downweight. Chỉ bật flag đó khi provenance chứng minh Hidden đã qua
review đáng tin và policy mới có audit riêng.

### 7.1. Short merge

```bat
%PY% %S0%\classification_v2_merge_sources.py ^
  --legacy-csv %SM%\legacy_export\legacy_frame_object_annotations.csv ^
  --cvat-tracking-xml %X01% --cvat-tracking-xml %X02% ^
  --cvat-tracking-xml %X03% --cvat-tracking-xml %X04% ^
  --cvat-tracking-xml %X05% --cvat-tracking-xml %X06% ^
  --cvat-tracking-xml %X07% --cvat-tracking-xml %X08% ^
  --cvat-tracking-xml %X09% --cvat-tracking-xml %X10% ^
  --cvat-tracking-xml %X11% --cvat-tracking-xml %X12% ^
  --max-rows-per-source 96 ^
  --output-csv %SM%\merged_frame_objects.csv ^
  --audit-json %SM%\merged_frame_objects_audit.json
```

Smoke PASS khi audit có `errors=[]`, hai source đều xuất hiện, behavior nằm
trong 10 lớp hợp lệ, key được tạo và không source nào mất toàn bộ row. Limit là
theo từng source, chỉ dùng để kiểm parser/schema, không dùng để ước lượng phân
bố lớp.

### 7.2. Full merge

```bat
%PY% %S0%\classification_v2_merge_sources.py ^
  --legacy-csv %SRC%\legacy_export\legacy_frame_object_annotations.csv ^
  --cvat-tracking-xml %X01% --cvat-tracking-xml %X02% ^
  --cvat-tracking-xml %X03% --cvat-tracking-xml %X04% ^
  --cvat-tracking-xml %X05% --cvat-tracking-xml %X06% ^
  --cvat-tracking-xml %X07% --cvat-tracking-xml %X08% ^
  --cvat-tracking-xml %X09% --cvat-tracking-xml %X10% ^
  --cvat-tracking-xml %X11% --cvat-tracking-xml %X12% ^
  --output-csv %SRC%\merged_frame_objects.csv ^
  --audit-json %SRC%\merged_frame_objects_audit.json
```

Gate full merge:

- audit `errors=[]`;
- source distribution có `legacy_recovered` và `cvat_tracking_xml`;
- đúng 12 CVAT behavior dataset dự kiến;
- không có `Tracking_annotation_Pigs291119_000263_30fps` trong lineage;
- invalid bbox, unknown label và row count đều được ghi, không bị xóa;
- không dùng `--require-full-8-for-eval`.

## 8. Tạo feature frame-level

Thứ tự là contract, không đổi: context policy -> geometry -> all-ROI -> enhanced
motion/social/posture. Context policy chuẩn hóa eligibility nhưng không xóa row.
Geometry tạo bbox/location chuẩn hóa. ROI tạo quan hệ tới feeder, drinker và
toy cho mọi row, không chọn ROI theo label. Enhanced tạo motion, temporal,
partner/social và posture proxy.

### 8.1. Parser/schema feature smoke

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
  --output-csv %FSM%\frame_enhanced.csv ^
  --audit-json %FSM%\frame_enhanced_audit.json
```

Chain này dùng merge bị giới hạn theo leading rows, nên chỉ kiểm parser, schema
và row preservation. Nó không chứng minh temporal/review correctness vì có thể
cắt giữa CVAT interval hoặc legacy burst. PASS khi row count giữ nguyên qua bốn
bước, audit không có error, bbox/all-ROI columns tồn tại và source vẫn đủ.
Warning về thiếu context phải được đếm, không được thay bằng drop.

### 8.2. Full feature chain

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
  --output-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --audit-json %FEAT%\spatiotemporal_frame_features_enhanced_audit.json
```

Không truyền `--max-rows` ở full. So sánh số row của enhanced với merge; mọi
chênh lệch phải có error hoặc audit reason cụ thể. Enhanced là input immutable
của nhánh review trong lineage này; apply review phải ghi file khác.

### 8.3. Complete-unit scientific smoke scope

Sau full feature extraction, tạo short scope từ complete scene/native blocks.
Đây mới là input cho Hidden, temporal và review smoke:

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
16-frame tracklets và không selected incomplete block. Reference hiện tại có
688 row, 63 native unit và đủ 10 behavior; số khác phải được giải thích, không
ép về 688 bằng cách cắt row.

Reference scope 688-row không chứa case anchor 1020. Vì vậy technical smoke này
không được dùng làm bằng chứng cho case bắt buộc; checker ở mục 10.2 phải chạy
trên chính full versioned enhanced/interval/review-unit artifacts trước khi mở
full human behavior review hoặc model smoke.

## 8A. Hidden review hai chiều trước temporal harmonization

Hidden review độc lập với behavior review. Mục tiêu không chỉ xác nhận các row
đã có `Hidden=Yes`, mà còn phát hiện false negative trong `Hidden=No`. CVAT là
nguồn yếu nhất vì Hidden chủ yếu đến từ tracking; row CVAT chưa review phải giữ
`hidden_trust_status=untrusted_tracking_derived`.

Bốn cohort không được trộn ý nghĩa thống kê:

- `hidden_yes_confirmation`: census `Hidden=Yes` chưa tin cậy, đồng thời lấy
  mẫu phân tầng từ `Hidden=Yes` trusted để kiểm tra lại prior review;
- `hidden_no_high_risk`: targeted enrichment theo overlap, proximity,
  bbox/shape change và, trong implementation hiện tại, original interaction
  label;
- `hidden_no_random_audit`: random phân tầng để ước lượng false-negative rate;
- `hidden_no_clean_control`: kiểm specificity ở nhóm risk thấp.

Random audit lưu population, inclusion probability và inverse sampling weight.
Chỉ post-stratified random estimate được diễn giải như prevalence. Correction
yield của high-risk cohort không phải prevalence.

Hàm `_hidden_false_negative_risk()` hiện cộng score khi `behavior` là `fight`
hoặc `social-nose`, dù docstring gọi risk là label-independent. Manifest v5 có
333 row mang reason `interaction_scene`. Trước final Hidden lock phải chọn một
trong hai contract: bỏ target term khỏi visibility risk, hoặc đổi tên cohort
thành target-informed enrichment và báo cáo tách biệt. Tuyệt đối không dùng
cohort này để ước lượng population false-negative prevalence.

### 8A.1. Short builder và media gate

```bat
set HSM=%SM%\hidden_review
%PY% %S1%\classification_v2_build_hidden_review_units.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --output-dir %HSM%
```

```bat
%PY% %S1%\check_hidden_review_template_coverage.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --audit-json %HSM%\hidden_review_coverage_audit.json
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HSM%\hidden_review_frame_context.csv ^
  --output-dir %HSM%\gui --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --validation-audit-json %HSM%\hidden_media_validation_audit.json ^
  --validate-only
```

Không thêm `--max-rows-per-source` ở đây vì nó có thể cắt complete-unit scope
lần thứ hai. Short PASS khi hai source đều có mặt, input có cả Yes/No, không thiếu
untrusted Yes, trusted Yes đạt quota phân tầng, negative cohorts tồn tại, key
unique và media missing bằng 0. Builder xuất frame-context subset để GUI không
đọc lại full enhanced CSV.

Mở GUI pilot sau media gate:

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HSM%\hidden_review_frame_context.csv ^
  --output-dir %HSM%\gui --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --max-items 5
```

Sau pilot, chạy lại không `--max-items` để hoàn thành short manifest, rồi kiểm
và apply. Không tạo fake decision để ép smoke PASS.

```bat
%PY% %S1%\check_hidden_review_decision_coverage.py ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HSM%\gui\hidden_review_decisions.csv ^
  --audit-json %HSM%\hidden_review_decision_coverage_audit.json
%PY% %S1%\classification_v2_apply_hidden_review_decisions.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --manifest-csv %HSM%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HSM%\gui\hidden_review_decisions.csv ^
  --output-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --audit-json %HSM%\apply_hidden_review_audit.json ^
  --confusion-audit-json %HSM%\hidden_confusion_audit.json
%PY% %S1%\check_apply_hidden_review_decisions_output.py ^
  --input-csv %SSCOPE%\frame_features_complete_units.csv ^
  --output-csv %HSM%\hidden_reviewed_frame_features.csv ^
  --audit-json %HSM%\check_apply_hidden_review_output.json
```

### 8A.2. Full manifest và human review

Chỉ chạy sau short PASS. Cap high-risk kiểm soát workload nhưng audit vẫn ghi
toàn bộ high-risk population và số chưa được chọn. Thay cap tạo review design
mới và phải lưu trong lineage.

```bat
%PY% %S1%\classification_v2_build_hidden_review_units.py ^
  --input-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --output-dir %HREV% ^
  --trusted-yes-per-stratum 1 ^
  --random-no-per-stratum 3 ^
  --clean-control-per-stratum 1 ^
  --max-high-risk-per-stratum 1
%PY% %S1%\check_hidden_review_template_coverage.py ^
  --input-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --audit-json %HREV%\hidden_review_coverage_audit.json
```

Cap 1 là wave đầu, không phải ngưỡng khoa học cố định. Sau mỗi wave, kiểm
high-risk correction yield. Nếu yield còn cao, tăng cap theo chuỗi 1, 2, 4, 8
hoặc bỏ cap; giữ cùng seed để selection lồng nhau và resume decision cũ. Chỉ
khóa final cap khi correction yield đã ổn định thấp và random weighted estimate
có uncertainty được báo cáo. Mọi lần mở rộng phải rebuild coverage audit.

Trước khi xem kết quả wave đầu, ghi vào review-design manifest: ngưỡng chấp
nhận false-negative, phương pháp confidence interval và strata sẽ báo cáo.
Không khóa Hidden review chỉ vì point estimate thấp. Upper confidence bound của
random weighted estimate và correction yield ở wave high-risk cuối đều phải
đạt ngưỡng đã khai báo; nếu không thì mở rộng wave hoặc census. Không suy ngược
prevalence từ high-risk cohort và không đổi ngưỡng sau khi xem kết quả.

Đây hiện là **implementation blocker**, không phải gate đã có sẵn. Apply audit
mới chỉ xuất Wilson 95% CI cho tỷ lệ random không trọng số, một weighted point
estimate và một high-risk point estimate. Nó chưa có design-based/clustered CI
cho weighted estimate, CI cho high-risk yield, hoặc machine-readable threshold
decision. Correlation của các frame cùng native unit/video phải được phản ánh
trong uncertainty. Không được gọi Hidden scientifically locked cho tới khi các
output và fail-closed checker này được cài đặt, test và bind vào snapshot.

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HREV%\hidden_review_frame_context.csv ^
  --output-dir %HREV%\gui --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --validation-audit-json %HREV%\hidden_media_validation_audit.json ^
  --validate-only
```

Chỉ mở full GUI sau khi media audit trên báo `media_missing=0`:

```bat
%PY% %S1%\review_hidden_quality_gui.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --frame-features-csv %HREV%\hidden_review_frame_context.csv ^
  --output-dir %HREV%\gui --reviewer %REVIEWER_NAME% ^
  --video-root data\videos ^
  --crop-root data\raw\legacy_full_multigt_masked_nodup_16f\crops
```

GUI hiển thị full frame với actor và bbox context, kèm actor crop letterbox.
Chọn `Hidden=Yes`, `Hidden=No (visible)` hoặc `Unclear`; confidence và reason
là bắt buộc về mặt quy trình. GUI chỉ ghi decision CSV, không sửa XML/CSV
nguồn.

### 8A.3. Complete gate và apply

```bat
%PY% %S1%\check_hidden_review_decision_coverage.py ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HREV%\gui\hidden_review_decisions.csv ^
  --audit-json %HREV%\hidden_review_decision_coverage_audit.json
```

Default là fail-closed: missing, duplicate, pending và `Unclear` đều làm gate
FAIL. `--allow-unresolved` chỉ bỏ yêu cầu resolve `pending/Unclear`; nó vẫn FAIL
khi thiếu decision row, duplicate hoặc unknown item. Flag này không tạo được
fake coverage và không được dùng cho training snapshot. Non-selected CVAT No
vẫn là untrusted, không âm thầm thành trusted No.

```bat
%PY% %S1%\classification_v2_apply_hidden_review_decisions.py ^
  --input-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --manifest-csv %HREV%\hidden_review_unit_manifest.csv ^
  --decisions-csv %HREV%\gui\hidden_review_decisions.csv ^
  --output-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --audit-json %HREV%\apply_hidden_review_audit.json ^
  --confusion-audit-json %HREV%\hidden_confusion_audit.json
%PY% %S1%\check_apply_hidden_review_decisions_output.py ^
  --input-csv %FEAT%\spatiotemporal_frame_features_enhanced.csv ^
  --output-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --audit-json %HREV%\check_apply_hidden_review_output.json
```

Apply PASS khi output rows bằng enhanced rows, non-Hidden source columns không
đổi, decision match đúng frame/object key và audit ghi `Yes->No`, `No->Yes`,
trust status, random false-negative estimate cùng high-risk correction yield.

## 9. Temporal harmonization và window chưa review

Temporal harmonization chỉ bắt đầu từ hidden-reviewed artifact. CVAT anchor `k`
đại diện `k..k+5`; legacy burst có 16 frame. Hidden vẫn là frame/object quality,
không được broadcast theo behavior interval. Sequence window được tạo sau bước
này. Window 6, 8, 12 và 16 frame là candidate pool để audit/ablation, không phải
cam kết rằng final model được dùng đồng thời mọi length. Primary model view phải
được khóa riêng ở mục 15.6 để sequence length/padding không tiết lộ source.

### 9.1. Short temporal/window chain

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

### 9.2. Full temporal/window chain

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

Window audit phải tách `hidden_ratio_raw`, `hidden_ratio_trusted` và review
coverage. Default không loại/downweight window chỉ vì hidden ratio cao. Flag
`--exclude-high-hidden-from-main` là opt-in policy experiment, không dùng trong
lineage chuẩn nếu chưa có ablation và phê duyệt riêng.

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

### 10.1. Builder smoke

```bat
%PY% %S1%\classification_v2_build_review_units.py ^
  --intervals-csv %TSM%\temporal_label_intervals.csv ^
  --sequence-window-manifest-csv %TSM%\sequence_window_manifest.csv ^
  --output-dir %SM%\review_units ^
  --max-units-per-template 100000 ^
  --disable-window-review-overlay
%PY% %S1%\check_review_unit_template_coverage.py ^
  --review-unit-dir %SM%\review_units ^
  --allow-incomplete-label-coverage
```

### 10.2. Full review-unit build

```bat
%PY% %S1%\classification_v2_build_review_units.py ^
  --intervals-csv %SEQ0%\temporal_label_intervals.csv ^
  --sequence-window-manifest-csv %SEQ0%\sequence_window_manifest.csv ^
  --output-dir %REV% ^
  --max-units-per-template 100000 ^
  --disable-window-review-overlay
%PY% %S1%\check_review_unit_template_coverage.py ^
  --review-unit-dir %REV%
%PY% %S0%\check_classification_v2_cvat_anchor_case.py ^
  --enhanced-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --intervals-csv %SEQ0%\temporal_label_intervals.csv ^
  --review-units-csv %REV%\review_unit_manifest.csv ^
  --output-json %REV%\cvat_anchor_1020_audit.json
%PY% %S0%\check_classification_v2_temporal_evidence.py ^
  --enhanced-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --intervals-csv %SEQ0%\temporal_label_intervals.csv ^
  --windows-csv %SEQ0%\sequence_window_manifest.csv ^
  --review-units-csv %REV%\review_unit_manifest.csv ^
  --trainer-contract-json configs\classification_v2\trainer_contract_v1.json ^
  --output-json %REV%\temporal_evidence_audit.json
```

Flag `--disable-window-review-overlay` ngăn builder đọc window-review artifact
canonical cũ. Chỉ bật overlay khi có file review window thuộc đúng `RUN_ID` và
hash đã khóa.

Gate: duplicate `review_unit_id=0`, không có `window_uid`, template labels đúng
policy, `full_review_unit_manifest.csv` bằng union của các template và temporal
evidence audit không phát hiện label/review leakage vào trainer whitelist.

## 11. GUI smoke và human review đầy đủ

GUI smoke là kiểm tra bắt buộc trước khi review hàng nghìn unit. Dùng đúng
output directory sẽ dùng cho full review; lần chạy sau tự resume, không ghi đè
decision cũ. Không dùng `--fresh` và không xóa CSV giữa các session.

### 11.1. Năm unit mỗi nhóm

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\roi_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\roi --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --roi-coco-json data\annotations\roi\ROI_annotations.coco.json ^
  --max-items 5 --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\motion_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\motion --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --max-items 5 --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\posture_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\posture --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --max-items 5 --copy-contact-sheets
```

Interaction cần scene/partner context rộng. `--padding 10` clamp crop về gần
full frame cho CVAT. Nếu actor/partner/role vẫn không đủ rõ, chọn
`review_later`; không đoán label.

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\interaction_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\interaction --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --padding 10 --max-items 5 --copy-contact-sheets
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

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\roi_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\roi --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --roi-coco-json data\annotations\roi\ROI_annotations.coco.json ^
  --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\motion_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\motion --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --copy-contact-sheets
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\posture_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\posture --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --copy-contact-sheets
```

```bat
%PY% %S1%\review_temporal_unit_gui.py ^
  --review-units-csv %REV%\interaction_review_unit_template.csv ^
  --frame-features-csv %HREV%\hidden_reviewed_frame_features.csv ^
  --output-dir %DEC%\interaction --video-root data\videos ^
  --raw-root data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --padding 10 --copy-contact-sheets
```

Quy tắc quyết định:

- `accept`: giữ nhãn gốc; thường `main_train`, weight 1.
- `corrected`: chọn đúng một trong 10 behavior và ghi note ngắn.
- `exclude`: không xóa row; apply đặt include false và weight 0.
- `low_weight_train`: giữ row với weight giảm, phải có lý do quality rõ.
- `review_later`: fail-closed, không vào training và làm complete gate FAIL.
- `fight`: chỉ actor trực tiếp tham gia, không bystander.
- `social-nose`: actor-only mặc định, không hard-propagate sang receiver.

Full review hiện tại tương đương khoảng 4.670 unit. Con số phải lấy từ manifest
mới, không hard-code. Nên double-review 10-20% nhóm hiếm/confusion và báo
agreement riêng; GUI decision không được dùng làm model feature.

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
```

Không chạy apply nếu lệnh này FAIL. Điều kiện PASS gồm đủ 24 column, không
duplicate/missing/unexpected ID, không pending, không `review_later`, corrected
behavior hợp lệ và không `window_uid`.

### 12.2. Apply decisions

```bat
%PY% %S1%\classification_v2_apply_review_unit_decisions.py ^
  --frame-features-csv ^
  %HREV%\hidden_reviewed_frame_features.csv ^
  --review-unit-manifest-csv %REV%\review_unit_manifest.csv ^
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

Apply không overwrite enhanced CSV. Nó giữ nguyên số frame row, lưu
`behavior_before_review` và `behavior_after_review`, đồng thời thêm action,
include flag và weight. Corrected decision áp toàn bộ 16 frame legacy hoặc 6
frame CVAT qua `temporal_unit_key`. Exclude đặt mask/weight, không xóa row.

Gate:

- reviewed rows bằng enhanced rows;
- audit `errors=[]` và unmatched decision bằng 0;
- duplicate `review_unit_id=0` trong combined decisions;
- applied/accepted/corrected/excluded counts được ghi;
- excluded/corrected frame counts được ghi;
- label distribution trước/sau được kiểm và mọi delta truy về decision.

## 13. Rebuild reviewed windows và native units

Rebuild reviewed sequence phải full-recompute vì corrected label có thể đổi
target và temporal status. Không dùng fast overlay canonical.

### 13.1. Reviewed sequence windows

```bat
%PY% %S0%\classification_v2_build_sequence_windows.py ^
  --input-csv %RFRAME%\reviewed_frame_features.csv ^
  --output-dir %SEQ1% ^
  --window-lengths 6,8,12,16 ^
  --cvat-label-stride 6 ^
  --legacy-expected-sequence-length 16 ^
  --disable-fast-reuse
```

Không dùng `--exclude-mixed-windows`. `window_sample_weight=0` và
`window_valid_for_main_train=false` phải phản ánh review-excluded frame mà
không làm mất window row.

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
  --contract-json configs\classification_v2\feature_semantics_v1.json ^
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
%PY% %S2%\classification_v2_build_auxiliary_targets.py --root %TRAIN%
%PY% %S2%\check_classification_v2_auxiliary_targets.py ^
  --csv %TRAIN%\y_auxiliary_targets.csv ^
  --audit-json %TRAIN%\auxiliary_targets_audit.json
```

Đây là deterministic decomposition của behavior y, không phải annotation độc
lập. Chỉ dùng làm auxiliary target/mask; không đưa vào X và không dùng hard
argmax cascade vào final 10-class head. Attribute reviewed độc lập, nếu bổ sung
sau này, phải có confidence/mask và một ablation riêng.

### 15.6. Primary temporal view và source-shortcut controls

Tạo non-destructive 6-frame/source-matched masks và grouped source probes:

```bat
%PY% %S2%\classification_v2_build_source_matched_views.py ^
  --window-manifest %TRAIN%\split_manifest.csv ^
  --output-dir %TRAIN%\source_matched_views
%PY% %S2%\check_classification_v2_source_matched_views.py ^
  --input-csv %TRAIN%\split_manifest.csv ^
  --view-csv %TRAIN%\source_matched_views\source_matched_view_manifest.csv
%PY% %S7%\classification_v2_evaluate_domain_controls.py ^
  --root %TRAIN% ^
  --grouped-roles %SPLIT%\q2_grouped_folds\q2_outer_inner_roles.csv ^
  --output-dir %TRAIN%\domain_controls
%PY% %S4%\check_classification_v2_grouped_spatial_source_probe.py ^
  --root %TRAIN% ^
  --grouped-roles %SPLIT%\q2_grouped_folds\q2_outer_inner_roles.csv ^
  --output-json %TRAIN%\domain_controls\spatial_source_probe_audit.json
```

Primary protocol là một fixed 6-slot view với cùng sampling rule và observed
time semantics cho hai source. `view_matched_6frame` hiện chỉ kiểm số slot; nó
chưa tự chứng minh cadence/duration tương đương. Native 6/16 và mọi-length
training chỉ là ablation cho tới khi source, sequence-length, padding và
effective-time probes được audit.

Trong primary X, `window_length_frames` phải là hằng số. Chỉ giữ duration,
effective FPS hoặc frame-delta khi provenance thời gian của hai source đã được
calibrate và cùng ý nghĩa; nếu không chúng là source proxy và phải bỏ khỏi
primary view. Chạy riêng `availability-only` probe cho ROI/social/context masks
trước khi diễn giải gain từ real context.

Không chạy full nếu model loader chưa nhận explicit temporal-view mask hoặc nếu
source/length gần như suy ra label mà chưa có matched analysis. Context gain còn
phải có actor-only, availability-only, real-context, matched-context subset và
modality-dropout controls.

## 16. Image context và cache tái sử dụng

### 16.0. Disk và cache-lineage preflight

Với 245.664 actor row và 172.800 CVAT visual-context row ở 224 px, raw `uint8`
pixel chiếm xấp xỉ 37 GB cho actor và 26 GB cho interaction. Giữ cả individual
NPY và packed NPY nhân đôi thành khoảng 126 GB, chưa tính preview, manifest,
integrity, partial checkpoint và filesystem overhead. Khuyến nghị có ít nhất
160 GB trống trước khi build cả hai cache:

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
  --legacy-crop-root ^
  data\raw\legacy_full_multigt_masked_nodup_16f\crops ^
  --output-audit %TRAIN%\source_image_loader_smoke_audit.json ^
  --sample-per-source 24
%PY% %S3%\classification_v2_build_image_context_index.py ^
  --frame-features-csv %RFRAME%\reviewed_frame_features.csv ^
  --window-manifest-csv %SEQ1%\sequence_window_manifest.csv ^
  --output-dir %TRAIN% --video-root data\videos ^
  --legacy-crop-root ^
  data\raw\legacy_full_multigt_masked_nodup_16f\crops
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

Context readiness/missingness là audit/mask, không phải evidence label. Partner
selection phải dựa trên geometry và cùng frame/video, không dựa vào target
`fight` hoặc `social-nose`.

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
human coverage:

```bat
set S9=scripts\classification_v2\09_final_release_audit
set BASE=outputs\classification_v2\rebuilds
set ROOT=%BASE%\scientific_smoke_identifier_v2_20260713
set REPEAT=%BASE%\scientific_smoke_identifier_v2_repeat_20260713
%PY% %S9%\check_classification_v2_identifier_v2_lineage.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
%PY% %S9%\check_classification_v2_technical_smoke_gate.py ^
  --root %ROOT% ^
  --repeat-root %REPEAT% ^
  --overwrite
```

Expected statuses are `PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED` and
`PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED`, with 688 frame rows, 63 native
units, 438 ordered windows, exact 110-feature X, zero trainable spatial gaps,
and 8/8 deterministic stage pairs. These gates must never bypass sections 8A,
11, or 12. Final reviewed data still requires complete human decisions.

### 17.1. Hash artifact chính

```bat
certutil -hashfile %SRC%\merged_frame_objects.csv SHA256
certutil -hashfile ^
  %FEAT%\spatiotemporal_frame_features_enhanced.csv SHA256
certutil -hashfile %HREV%\hidden_review_unit_manifest.csv SHA256
certutil -hashfile %HREV%\gui\hidden_review_decisions.csv SHA256
certutil -hashfile %HREV%\hidden_reviewed_frame_features.csv SHA256
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
certutil -hashfile %CACHE%\manifest.csv SHA256
certutil -hashfile %CACHE%\packed_rgb_224_letterbox.npy SHA256
certutil -hashfile %VCACHE%\visual_context_manifest.csv SHA256
certutil -hashfile %VCACHE%\packed_rgb_224_letterbox.npy SHA256
```

Ghi hash, row count, schema version, `RUN_ID`, Git SHA, dirty-worktree status,
Python/PyTorch/OpenCV version và input hash vào run manifest trước training.
Không gọi một folder là final nếu hash/audit chưa khóa.

### 17.2. Versioned snapshot hard stop

Artifact dưới `%R%` là candidate versioned. Không copy đè canonical để né path
contract. Audit hiện tại phát hiện các blocker sau trước snapshot/full:

1. `data_contract_v2.json` đang trỏ canonical train-ready/native/fold paths và
   packed cache 64 px, trong khi runbook tạo lineage `%R%` và cache 224 px.
2. `classification_v2_write_model_input_manifest.py` giả định canonical child
   folder names, không khớp layout `09_native_units/10_grouped_splits` ở đây.
3. Trainer-contract checker chỉ hiểu v1/canonical, chưa kiểm được candidate v2
   bằng explicit `%TRAIN%` paths.
4. Preflight/full runner chỉ override cache paths; CLI chưa nhận train-ready
   root, reviewed sequence manifest, native OOF manifest và event weights.
5. Full runner hiện dùng leave-one-group-out, không consume Q2 outer/inner roles
   và không có fold-local inner model selection/early stopping.
6. `view_matched_6frame` mới là selection mask; trainer chưa consume một fixed-6
   primary view đã loại/calibrate length, cadence và availability shortcuts.
7. Canonical feature artifacts đang lẫn ít nhất hai source allowlist; không có
   một canonical root hiện tại đủ điều kiện làm snapshot authority.
8. `frame_uid` đang là scene-frame key. Image-context audit có 213.376 duplicate
   theo cột này vì nhiều pig cùng frame, trái object-row identifier contract.
   Phải migrate sang `scene_frame_uid` cộng object-level `frame_uid`, hoặc sửa
   chính thức schema/checkers với compatibility và composite row key.
9. Hidden high-risk score còn target-informed; weighted/high-risk uncertainty
   và predeclared threshold gate chưa được code hóa fail-closed.
10. Image-context checker chỉ assert case `000231` loadable, chưa assert exact
    resolved basename `Pigs291119_000231_30fps.mp4`.

Vì vậy **chưa có lệnh snapshot/full hợp lệ trực tiếp cho `%R%`**. Trước model
smoke phải sửa module chính để mọi path trên đi qua một versioned contract, thêm
loader/preflight tests và không fallback canonical. Sau khi patch đó PASS, lệnh
snapshot chuẩn phải có dạng:

```bat
%PY% %S2%\classification_v2_freeze_training_snapshot.py ^
  --contract-json %SNAP%\data_contract.json ^
  --output-json %SNAP%\training_snapshot.json
%PY% %S2%\check_classification_v2_training_snapshot.py ^
  --snapshot-json %SNAP%\training_snapshot.json ^
  --contract-json %SNAP%\data_contract.json ^
  --output-json %SNAP%\training_snapshot_check.json
```

Contract phải tham chiếu duy nhất artifact của cùng `RUN_ID`, fixed temporal
view, 224 caches và Q2 primary roles; native OOF manifest chỉ bắt buộc nếu chạy
engineering sensitivity đã khai báo. Checkpoint phải ghi snapshot, config,
feature-whitelist, cache và fold hashes.

### 17.3. Model finalist gate

Current `multimodal_fusion.py` dùng small ConvNet và full runner mặc định 64 px;
nó là engineering model, không phải ResNet18/ResNet34 224 finalist trong roadmap.
Không được gọi một lần chạy mới của model này là final Q2 classifier.

Trên cùng frozen snapshot/folds, development phải tách từng biến:

```text
resolution: ResNet18-160 -> ResNet18-224
backbone:   ResNet18-224 -> ResNet34-224
temporal:   masked pooling -> TCN -> small Transformer nếu TCN chưa đủ
modality:   actor -> geometry/motion -> all-ROI -> social -> union context
imbalance:  event CE | effective-number CE | Balanced Softmax, chọn một
```

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
- [ ] Legacy export giữ native burst 16 frame.
- [ ] CVAT anchor interval đúng 6 frame và non-anchor kế thừa target.
- [ ] Scientific smoke dùng complete units, không leading-row temporal scope.
- [ ] Case `000085 / ID_4 / anchor 1020 = social-nose + interaction` PASS.
- [ ] Mọi untrusted `Hidden=Yes` có item; trusted Yes đạt quota phân tầng.
- [ ] `Hidden=No` có high-risk, stratified-random và clean-control audit.
- [ ] CVAT chưa review giữ `untrusted_tracking_derived`, không silent trust.
- [ ] Hidden decisions unique, resolved và áp đúng frame/object key.
- [ ] Hidden apply giữ nguyên row count và non-Hidden source columns.
- [ ] `Yes->No`, `No->Yes`, weighted false-negative rate có audit.
- [ ] Hidden weighted/high-risk CI và predeclared threshold gate PASS.
- [ ] Hidden target-informed enrichment không bị gọi label-independent.
- [ ] Enhanced, hidden-reviewed và behavior-reviewed rows bằng nhau.
- [ ] Duplicate `temporal_unit_key=0` và duplicate `review_unit_id=0`.
- [ ] Không output mới dùng `window_uid`.
- [ ] ROI/motion/posture/interaction templates đúng policy.
- [ ] `playwithtoy` review coverage đầy đủ.
- [ ] Bốn decision CSV đủ schema, không pending/missing/review_later.
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
- [ ] X/y/mask/weights/spatial arrays row/order/key khớp.
- [ ] `frame_uid` object-row contract đã migrate hoặc schema revision được khóa.
- [ ] Không có global normalization/class weight fit trước fold.
- [ ] Legacy crop và CVAT video+bbox loader smoke PASS.
- [ ] Case `000231` exact basename `_30fps.mp4` được checker assert và PASS.
- [ ] Actor cache dùng letterbox, preview không méo, checksum PASS.
- [ ] Packed cache index/tensor equivalence PASS.
- [ ] Interaction context giữ missing mask và không label-select partner.
- [ ] Fixed-6 primary view và native-length ablation được định nghĩa tách biệt.
- [ ] Grouped source/length/padding/missingness controls đã được audit.
- [ ] Final artifact hashes, config, code SHA và environment đã ghi.

Nếu bất kỳ mục nào FAIL, kết luận là `NOT TRAIN-READY`. Không dùng số row lớn,
training accuracy hoặc việc script exit 0 để thay cho gate thiếu. Sau PASS, bước
kế tiếp chỉ là model/local smoke theo roadmap, chưa phải full OOF.

Full OOF chỉ được phép khi thêm các mục sau PASS:

- [ ] Reviewer-agreement audit đã có, hoặc single-review limitation được khóa.
- [ ] Versioned data contract trỏ hoàn toàn tới một `RUN_ID`, không fallback.
- [ ] Snapshot/checkpoint bind dataset, cache, fold, whitelist và config hashes.
- [ ] ResNet resolution/backbone controls và temporal baseline đã chạy đúng cặp.
- [ ] F0/F1/F2 và inner selection rule được khóa trước outer predictions.
- [ ] One-batch, tiny-overfit, resume, AMP/runtime/VRAM và one-fold PASS.
- [ ] Preflight, authorization, launch packet và execution gate cùng config SHA.
- [ ] Full predictions đủ count và collapse về native unit không mất unit.
- [ ] Calibration, confusion, grouped/source metrics, registry và completion PASS.

Theo code hiện tại, versioned contract/path routing, primary fold protocol,
fixed-6 loader controls và ResNet finalist đều chưa đạt; do đó toàn luồng tới
scientific full OOF vẫn là `FAIL/BLOCKED`, dù technical data smoke đã PASS.
