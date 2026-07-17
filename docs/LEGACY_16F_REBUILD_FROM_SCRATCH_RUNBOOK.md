# Legacy 16f Rebuild From Scratch Runbook

Tài liệu này mô tả luồng tạo lại dữ liệu legacy 16-frame từ đầu đến
`legacy_frame_object_annotations.csv`. Luồng này chỉ tạo và kiểm tra dữ liệu
legacy; không chạy training, không chạy OOF và không sửa tracking.

## 1. Phạm vi và nguyên tắc

- `data/` là raw input bất biến. Không sửa, xóa, đổi tên hoặc overwrite.
- Native CVAT `task_0..task_3` là nguồn annotation hiện hành.
- Với mỗi `(group_id, pig_id)`, behavior của slot `k0` là authority duy nhất.
- Sáu bbox CVAT ở `k0..k5` là sáu GT anchor độc lập.
- Mười frame giữa các anchor được recovery; không copy bbox `k0` cho cả burst.
- Hidden của CVAT chỉ là seed chưa trusted; không suy Hidden từ behavior.
- Duplicate `(group_id, slot, pig_id)`, thiếu `k0`, frame-map sai hoặc bbox sai
  phải dừng trước khi sinh recovery CSV.
- Mọi output mới phải ở một thư mục versioned, không ghi vào output cũ.

`legacy_frame_object_annotations.csv` có thể có đủ 16 frame cho mỗi actor,
nhưng các sequence view lịch sử do `legacy_burst_recovery.main` ghi ra không
phải là classifier contract. `classification_v2` sẽ tạo các view T6/T8/T12/T16
trên dense 16-frame source sau bước này.

## 2. Ý nghĩa bảy artifact đã dọn khỏi root

Các file sau đã được chuyển vào
`outputs/_archive/legacy_16f_root_artifacts_20260718/` cùng SHA-256:

| Artifact | Vai trò | Có thể tự động tái tạo? |
|---|---|---|
| `duplicate_video_filter_audit.csv` | audit lọc source video | Có |
| `duplicate_video_preview.csv` | các row trùng để xem | Có |
| `duplicate_video_quarantine_all_bboxes.csv` | bbox bị loại | Có |
| `duplicate_video_quarantine_center.csv` | center row bị loại | Có |
| `old_burst_center_keyframes_nodup_videos.csv` | scaffold cũ đã lọc | Có |
| `exclude_source_videos.csv` | policy loại source video | Không tự suy ra an toàn |
| `manual_review_stable.csv` | quyết định review cũ | Không được coi là review mới |

Vì vậy việc dọn root không làm thay đổi raw CVAT, nhưng **có ảnh hưởng** nếu
workflow tiếp tục trỏ vào `exclude_source_videos.csv` hoặc
`manual_review_stable.csv`. Runbook này không dùng đường dẫn root cũ và sẽ dừng
nếu policy exclusion chưa được operator cung cấp hoặc xác nhận.

## 3. Khởi tạo lineage mới

Chạy trong `cmd.exe` từ project root. Dùng Python environment đã được kiểm tra
của project; thay `%PY%` nếu environment thực tế khác.

```bat
cd /d C:\Users\ironh\Downloads\PIG_Behavior_Project
set PY=python
set PYTHONPATH=%CD%\src
set S0=scripts\classification_v2\00_source_feature_temporal
set RUN_ID=legacy_16f_rebuild_20260718_v1
set RUN=%CD%\outputs\legacy_16f_rebuild\%RUN_ID%
set PROV=%RUN%\01_provenance
set POLICY=%RUN%\02_video_policy
set CVAT_AUDIT=%RUN%\03_cvat_audit
set CVAT_INPUT=%RUN%\04_cvat_inputs
set SMOKE=%RUN%\05_short_smoke
set FULL=%RUN%\06_full_recovery
set EXPORT=%RUN%\07_export
set AUDIT=%RUN%\08_audits

if exist "%RUN%" (
  echo ERROR: lineage already exists; choose a new RUN_ID
  exit /b 2
)
mkdir "%PROV%" "%POLICY%" "%CVAT_AUDIT%" "%CVAT_INPUT%" "%SMOKE%" "%FULL%" "%EXPORT%" "%AUDIT%"
```

`RUN_ID` không chỉ là tên thư mục: nó bind toàn bộ input hash, code version,
audit và output. Không dùng `--overwrite` trong run mới trừ khi đang lặp lại
đúng cùng input hash và đã ghi lý do.

## 4. Freeze và hash input

Trước khi sinh bất kỳ CSV derived nào, kiểm tra các file tồn tại và lưu hash.
Các lệnh dưới đây chỉ đọc input.

```bat
dir /b data\data\task_0\annotations.json data\data\task_0\data\manifest.jsonl
dir /b data\data\task_1\annotations.json data\data\task_1\data\manifest.jsonl
dir /b data\data\task_2\annotations.json data\data\task_2\data\manifest.jsonl
dir /b data\data\task_3\annotations.json data\data\task_3\data\manifest.jsonl
dir /b data\videos
dir /b data\annotations\scene\mask.png models\detector\pig_detector_yolov8.pt

certutil -hashfile data\data\task_0\annotations.json SHA256
certutil -hashfile data\data\task_0\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_1\annotations.json SHA256
certutil -hashfile data\data\task_1\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_2\annotations.json SHA256
certutil -hashfile data\data\task_2\data\manifest.jsonl SHA256
certutil -hashfile data\data\task_3\annotations.json SHA256
certutil -hashfile data\data\task_3\data\manifest.jsonl SHA256
certutil -hashfile ^
  data\processed\classification\20260620_131805\behavior_with_feats_rectROI.csv ^
  SHA256
certutil -hashfile data\annotations\scene\mask.png SHA256
certutil -hashfile models\detector\pig_detector_yolov8.pt SHA256
```

Không dùng hash hoặc CSV của
`outputs/_archive/legacy_16f_pre_cvat_rebuild_20260717` làm annotation
authority. Archive chỉ dùng để forensic comparison.

## 5. Xác nhận video-exclusion policy

`exclude_source_videos.csv` là policy, không phải kết quả có thể suy ra chỉ từ
tên file. Trước khi tiếp tục phải có file mới ở `%POLICY%` với schema tối thiểu:

```text
video_file,day_key,clip_id,source_video_key
```

Nếu policy cũ vẫn đúng, operator có thể sao chép bản archive sau khi kiểm tra
từng dòng; đây là quyết định dữ liệu, không phải bước tự động của agent:

```bat
copy /Y ^
  outputs\_archive\legacy_16f_root_artifacts_20260718\exclude_source_videos.csv ^
  "%POLICY%\exclude_source_videos.csv"
```

Nếu danh sách phải làm lại, tạo file mới theo video đã xác nhận. Không tạo file
rỗng để vượt gate. Kiểm tra fail-closed:

```bat
if not exist "%POLICY%\exclude_source_videos.csv" (
  echo STOP: exclusion policy is required and has not been handed off
  exit /b 2
)
```

`manual_review_stable.csv` không được đưa vào recovery mới. Review cũ không phải
human review lineage hiện hành; nếu cần dùng policy loại tracklet riêng thì phải
handoff một file mới có hash và phạm vi rõ ràng.

## 6. Tạo provenance scaffold từ dữ liệu chọn frame

`truy_nguon_multi_bbox.py` chỉ nối burst với video/path và giữ evidence cũ.
Nó không có quyền quyết định behavior hoặc bbox cho rebuild CVAT. Script này
thường chạy trên Colab, nơi các source folder đang nằm trong Google Drive.
Không dùng `--allow-unresolved-video` cho input training.

Phương án Colab hiện hành:

```bash
mkdir -p /content/legacy_16f_provenance
python /content/truy_nguon_multi_bbox.py \
  --behavior-csv /content/behavior_with_feats_rectROI.csv \
  --search-roots \
    "/content/drive/MyDrive/pig-selected_frame_attribute_(1)" \
    "/content/drive/MyDrive/pig-selected_frame_attribute_(2)" \
    "/content/drive/MyDrive/pig-selected_frame_attribute_(3)" \
    "/content/drive/MyDrive/pig-selected_frame_attribute_(4)" \
    "/content/drive/MyDrive/pig-selected_frame_attribute_(5)" \
  --out-center-csv /content/legacy_16f_provenance/old_burst_center_keyframes_combined.csv \
  --out-all-bbox-csv /content/legacy_16f_provenance/old_burst_all_keyframe_bboxes_combined.csv \
  --out-audit-csv /content/legacy_16f_provenance/legacy_gt_support_audit.csv \
  --out-missing-csv /content/legacy_16f_provenance/missing_old_burst_groups.csv \
  --out-lineage-json /content/legacy_16f_provenance/legacy_source_trace_lineage.json \
  --require-video-exists
```

Sau khi Colab PASS, đưa đúng năm output cùng hash vào `%PROV%`. Không chỉ tải
hai CSV rồi bỏ audit/lineage. Nếu Google Drive đã mount ở Windows, có thể chạy
phương án local tương đương dưới đây.

```bat
set BEHAVIOR_CSV=%CD%\data\processed\classification\20260620_131805\behavior_with_feats_rectROI.csv
set SEARCH1=G:\My Drive

%PY% truy_nguon_multi_bbox.py ^
  --behavior-csv "%BEHAVIOR_CSV%" ^
  --search-roots "%SEARCH1%" ^
  --out-center-csv "%PROV%\old_burst_center_keyframes_combined.csv" ^
  --out-all-bbox-csv "%PROV%\old_burst_all_keyframe_bboxes_combined.csv" ^
  --out-audit-csv "%PROV%\legacy_gt_support_audit.csv" ^
  --out-missing-csv "%PROV%\missing_old_burst_groups.csv" ^
  --out-lineage-json "%PROV%\legacy_source_trace_lineage.json" ^
  --require-video-exists
```

Ý nghĩa output:

- `old_burst_center_keyframes_combined.csv`: một dòng actor dùng làm metadata
  scaffold (`group_id`, video, frames), không phải label authority.
- `old_burst_all_keyframe_bboxes_combined.csv`: evidence cũ để đối chiếu; không
  thay thế sáu bbox native CVAT.
- `legacy_gt_support_audit.csv`: kiểm đủ sáu order `k0..k5` của provenance cũ.
- `legacy_source_trace_lineage.json`: hash, nguồn tìm thấy và row mapping.

Nếu mapping/path audit FAIL, dừng và sửa nguồn mapping. Không dùng
`--allow-incomplete-actor-keys` để đưa actor thiếu anchor vào recovery.

## 7. Audit và lọc duplicate source video

Utility đã được chuyển vào
`src/legacy_burst_recovery/check_duplicate_videos.py`. Nó chỉ đọc hai input và
ghi preview/audit vào lineage; nó không tự sửa exclusion policy.

Chạy dry-run trước để kiểm schema, source-key resolution và số row bị hit:

```bat
%PY% -m legacy_burst_recovery.check_duplicate_videos ^
  --legacy-csv "%PROV%\old_burst_center_keyframes_combined.csv" ^
  --exclude-csv "%POLICY%\exclude_source_videos.csv" ^
  --output-csv "%POLICY%\duplicate_video_preview.csv" ^
  --audit-json "%POLICY%\duplicate_video_filter_audit.json" ^
  --dry-run
```

Chỉ khi dry-run không có unresolved source key mới ghi preview:

```bat
%PY% -m legacy_burst_recovery.check_duplicate_videos ^
  --legacy-csv "%PROV%\old_burst_center_keyframes_combined.csv" ^
  --exclude-csv "%POLICY%\exclude_source_videos.csv" ^
  --output-csv "%POLICY%\duplicate_video_preview.csv" ^
  --audit-json "%POLICY%\duplicate_video_filter_audit.json"
```

`DUPLICATES_FOUND` không phải lỗi của utility; đó là bằng chứng để lọc. Không
được coi output preview là input training.

Tạo scaffold không trùng trong thư mục lineage mới. Lệnh này có ý nghĩa khác
với preview: nó ghi cả bảng giữ lại và quarantine để audit row accounting.

```bat
%PY% -m legacy_burst_recovery.make_nodup_legacy_csvs ^
  --center-csv "%PROV%\old_burst_center_keyframes_combined.csv" ^
  --bbox-csv "%PROV%\old_burst_all_keyframe_bboxes_combined.csv" ^
  --exclude-csv "%POLICY%\exclude_source_videos.csv" ^
  --output-dir "%POLICY%\nodup" ^
  --dry-run

%PY% -m legacy_burst_recovery.make_nodup_legacy_csvs ^
  --center-csv "%PROV%\old_burst_center_keyframes_combined.csv" ^
  --bbox-csv "%PROV%\old_burst_all_keyframe_bboxes_combined.csv" ^
  --exclude-csv "%POLICY%\exclude_source_videos.csv" ^
  --output-dir "%POLICY%\nodup"
```

Các output dưới `%POLICY%\nodup` chỉ là metadata/quarantine. `center_keep` sẽ
làm valid-group scaffold cho bước CVAT; bbox keep không được dùng để ghi đè
CVAT anchor. Kiểm tra bắt buộc:

```bat
type "%POLICY%\duplicate_video_filter_audit.json"
type "%POLICY%\nodup\duplicate_video_filter_audit.csv"
```

Nếu policy hoặc row count thay đổi so với dự kiến, tạo `RUN_ID` mới thay vì
overwrite lineage đang audit.

## 8. CVAT six-anchor audit-only

Đây là gate quan trọng nhất. `annotations.json.shape.frame` là chỉ số ảnh
trong từng task; phải resolve qua `data\manifest.jsonl`. Không dùng global task
frame 0 để suy ra `k0`.

```bat
%PY% %S0%\classification_v2_rebuild_legacy_cvat_recovery_inputs.py ^
  --cvat-export-root data\data ^
  --metadata-scaffold-csv "%POLICY%\nodup\old_burst_center_keyframes_nodup_videos.csv" ^
  --output-dir "%CVAT_AUDIT%" ^
  --behavior-authority-slot 0 ^
  --min-anchor-count 6 ^
  --audit-only
```

Audit-only không ghi `center`/`anchor` CSV khi còn lỗi. Phải đạt:

- `errors=[]` và không có issue mức `error`;
- không duplicate `(group_id, selected_slot, pig_id)`;
- mỗi actor được giữ lại có đúng `k0..k5` và frame span 15;
- k0 behavior hợp lệ và bbox hợp lệ;
- mọi frame map được qua manifest của task;
- các disagreement ở `k1..k5` chỉ là evidence, không được vote thay k0.

Diễn giải `status` phải bám đúng audit, không chỉ nhìn chuỗi `PASS`:

- `PASS`: không lỗi, không warning và không actor bị loại;
- `PASS_WITH_WARNINGS`: chỉ chấp nhận warning thông tin đã hiểu rõ, ví dụ
  `source_video_key` được derive xác định hoặc disagreement `k1..k5` đã map về
  `k0`;
- `PASS_WITH_DECLARED_EXCLUSIONS`: chỉ tiếp tục sau khi kiểm tra từng issue mức
  `excluded` và chấp thuận chính xác danh sách actor thiếu `k0`/thiếu anchor;
- `FAIL`: dừng tuyệt đối; không sinh recovery input.

Không được gọi một actor thiếu anchor là complete. Nếu chưa có quyết định loại
actor rõ ràng, coi `PASS_WITH_DECLARED_EXCLUSIONS` là stop condition và hoàn
thiện/re-export CVAT trước.

Audit hiện tại đã từng phát hiện hai duplicate anchor identity; khi lỗi này
còn tồn tại, **không chạy recovery hoặc full 16f**. Phải sửa/export lại đúng
CVAT task rồi hash lại input và chạy lại toàn bộ short audit.

## 9. Sinh recovery input sau khi audit PASS

Chỉ chạy block này khi mục 8 PASS. Dùng thư mục mới, không dùng lại audit root
để tránh nhầm CSV cũ:

```bat
if exist "%CVAT_INPUT%"\legacy_recovery_input_manifest.json (
  echo ERROR: recovery input already exists; choose a new RUN_ID
  exit /b 2
)

%PY% %S0%\classification_v2_rebuild_legacy_cvat_recovery_inputs.py ^
  --cvat-export-root data\data ^
  --metadata-scaffold-csv "%POLICY%\nodup\old_burst_center_keyframes_nodup_videos.csv" ^
  --output-dir "%CVAT_INPUT%" ^
  --behavior-authority-slot 0 ^
  --min-anchor-count 6
```

Expected files:

```text
legacy_center_keyframes_from_cvat.csv
legacy_six_anchor_bboxes_from_cvat.csv
legacy_recovery_input_manifest.json
legacy_cvat_recovery_input_audit.json
legacy_cvat_recovery_input_issues.csv
```

`legacy_center_keyframes_from_cvat.csv` giữ metadata group/video từ scaffold,
nhưng behavior k0 và sáu bbox đến từ native CVAT. `legacy_six_anchor_bboxes...`
là input multi-GT; không thay bằng file combined cũ.

## 10. Short one-burst recovery gate

Chọn một `group_id` có `complete_anchor_set=true`, video tồn tại và không có
issue. Gán giá trị đó thủ công vào `%SMOKE_GROUP%`; không dùng `--max-rows`, vì
leading-row truncation có thể cắt giữa một actor/burst và làm smoke giả.

```bat
set SMOKE_GROUP=REPLACE_WITH_A_COMPLETE_GROUP_ID

%PY% -m legacy_burst_recovery.main ^
  --input-csv "%CVAT_INPUT%\legacy_center_keyframes_from_cvat.csv" ^
  --legacy-burst-bbox-csv "%CVAT_INPUT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --drive-root "G:\My Drive" ^
  --output-root "%SMOKE%\recovery" ^
  --detector-weights "%CD%\models\detector\pig_detector_yolov8.pt" ^
  --scene-mask "%CD%\data\annotations\scene\mask.png" ^
  --mask-filter-detections ^
  --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside ^
  --track-end-mode full_legacy_burst ^
  --extract-crops ^
  --filter-group-id "%SMOKE_GROUP%" ^
  --sequence-views legacy_old_pattern_6 ^
  --progress

%PY% %S0%\check_classification_v2_legacy_cvat_recovery_output.py ^
  --center-csv "%CVAT_INPUT%\legacy_center_keyframes_from_cvat.csv" ^
  --anchor-csv "%CVAT_INPUT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --dense-csv "%SMOKE%\recovery\legacy_dense_tracklet_map.csv" ^
  --audit-json "%SMOKE%\recovery\cvat_recovery_output_audit.json" ^
  --filter-group-id "%SMOKE_GROUP%"
```

Short gate PASS phải chứng minh cùng lúc:

- mỗi actor có đúng 16 dense frame trong khoảng anchor đầu đến anchor cuối;
- frame-object key không duplicate;
- behavior trên dense row đúng behavior k0;
- sáu anchor giữ nguyên bbox CVAT với tolerance đã khai báo;
- anchor được đánh dấu `bbox_source=gt_legacy`;
- Hidden provenance được giữ nguyên và vẫn untrusted;
- không có row bị silently drop hoặc thêm ngoài key authority.

Nếu gate FAIL, không tăng phạm vi. Sửa đúng nguyên nhân trong input/CVAT map,
tạo lineage mới hoặc dọn output smoke rồi chạy lại short gate.

## 11. Full recovery sau short gate

Full recovery được phép khi short output audit PASS và các hash input không đổi.
Không bật `--resume` cho một root chưa từng chạy; nếu job bị gián đoạn, resume
chỉ trong cùng root và cùng config/hash.

```bat
%PY% -m legacy_burst_recovery.main ^
  --input-csv "%CVAT_INPUT%\legacy_center_keyframes_from_cvat.csv" ^
  --legacy-burst-bbox-csv "%CVAT_INPUT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --drive-root "G:\My Drive" ^
  --output-root "%FULL%" ^
  --detector-weights "%CD%\models\detector\pig_detector_yolov8.pt" ^
  --scene-mask "%CD%\data\annotations\scene\mask.png" ^
  --mask-filter-detections ^
  --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside ^
  --track-end-mode full_legacy_burst ^
  --extract-crops ^
  --sequence-views legacy_old_pattern_6 ^
  --progress ^
  --log-file "%FULL%\legacy_16f_recovery.log" ^
  --flush-every 500
```

Không truyền `--manual-review-csv` trong rebuild này. Điều đó không có nghĩa
data đã human-reviewed; nó chỉ giữ recovery tách khỏi review policy cũ. Review
Hidden/behavior của `classification_v2` là lineage riêng ở bước sau.

Sau full, chạy lại dense audit trên toàn bộ output:

```bat
%PY% %S0%\check_classification_v2_legacy_cvat_recovery_output.py ^
  --center-csv "%CVAT_INPUT%\legacy_center_keyframes_from_cvat.csv" ^
  --anchor-csv "%CVAT_INPUT%\legacy_six_anchor_bboxes_from_cvat.csv" ^
  --dense-csv "%FULL%\legacy_dense_tracklet_map.csv" ^
  --audit-json "%AUDIT%\full_cvat_recovery_output_audit.json"
```

## 12. Xuất frame-object annotations 16f

Export đọc `legacy_dense_tracklet_map.csv`, không đọc CSV combined cũ. Các
anchor `0,3,6,9,12,15` được kiểm tra lại; `--expected-sequence-length 16` là
contract của output frame-object.

```bat
%PY% -m legacy_burst_recovery.export_legacy_annotations ^
  --dense-csv "%FULL%\legacy_dense_tracklet_map.csv" ^
  --output-dir "%EXPORT%" ^
  --dataset-id legacy_recovered_16f ^
  --source-type legacy_recovered ^
  --expected-sequence-length 16 ^
  --anchor-relative-frames 0,3,6,9,12,15 ^
  --expected-pig-count 8 ^
  --cvat-behavior-authority-root "%CD%\data\data" ^
  --behavior-authority-slot 0
```

Expected primary artifact:

```text
%EXPORT%\legacy_frame_object_annotations.csv
%EXPORT%\legacy_frame_object_export_audit.json
%EXPORT%\legacy_cvat_behavior_authority_audit.json
%EXPORT%\legacy_cvat_behavior_discrepancies.csv
```

Việc nạp lại CVAT ở export là kiểm tra độc lập cuối cùng: behavior trong dense
map phải khớp authority `k0` của đúng `(group_id, pig_id)`. Discrepancy không
được tự sửa im lặng; phải xuất hiện trong audit và làm gate dừng khi vi phạm
contract.

Không dùng `--training-only` ở export canonical: giữ cả row để có context và
để audit row preservation. `include_in_training`/`training_tier` là policy
metadata; classifier sẽ áp training mask ở bước riêng.

## 13. Kiểm tra đầu ra và hash

Các lệnh sau không sửa output:

```bat
%PY% -c ^
  "import pandas as pd; ^
  p=r'%EXPORT%\legacy_frame_object_annotations.csv'; ^
  d=pd.read_csv(p,low_memory=False); ^
  k=['group_id','pig_id','frame_index']; ^
  print('rows=',len(d)); ^
  print('duplicate_keys=',int(d.duplicated(k).sum())); ^
  print(d.groupby(['group_id','pig_id'])['frame_index'].nunique().value_counts()); ^
  print(d['behavior'].value_counts(dropna=False).sort_index()); ^
  print(d.loc[d['is_legacy_gt_anchor'],'relative_frame_index'].value_counts())"

certutil -hashfile "%CVAT_INPUT%\legacy_recovery_input_manifest.json" SHA256
certutil -hashfile "%FULL%\legacy_dense_tracklet_map.csv" SHA256
certutil -hashfile "%EXPORT%\legacy_frame_object_annotations.csv" SHA256
certutil -hashfile "%EXPORT%\legacy_frame_object_export_audit.json" SHA256
```

PASS tối thiểu:

- `duplicate_keys=0`;
- mọi actor hợp lệ có 16 frame;
- anchor relative frame là `0,3,6,9,12,15`;
- export audit có `status=PASS`, không invalid bbox và không row-count mismatch;
- behavior dense/export chỉ đến từ k0 authority;
- sáu bbox anchor và Hidden provenance không bị export thay đổi;
- output path thuộc `%RUN%`, không thuộc `data/` và không overwrite canonical.

Nếu một actor bị loại vì thiếu k0/đủ anchor, phải xem `issues.csv` và báo cáo số
actor bị loại; không tự điền bbox, behavior hoặc Hidden để đạt đủ 16 row.

## 14. Luồng artifact và ý nghĩa

```text
behavior_with_feats_rectROI.csv
  -> truy_nguon_multi_bbox.py
  -> provenance combined CSV + lineage audit
  -> check_duplicate_videos.py
  -> make_nodup_legacy_csvs.py
  -> metadata scaffold đã lọc
  -> CVAT annotations.json + manifest.jsonl
  -> cvat_anchor_rebuild (k0..k5 audit)
  -> legacy_center_keyframes_from_cvat.csv
  -> legacy_six_anchor_bboxes_from_cvat.csv
  -> legacy_burst_recovery.main
  -> legacy_dense_tracklet_map.csv (16 frame/actor)
  -> export_legacy_annotations.py
  -> legacy_frame_object_annotations.csv
```

Vai trò của từng lớp:

1. Provenance chỉ giải thích row đến từ video/folder/frame nào.
2. Policy lọc duplicate source video trước khi tạo valid-group universe.
3. Native CVAT quyết định behavior k0, sáu bbox và Hidden seed.
4. Recovery tạo bbox/interpolation cho frame không phải anchor và giữ audit.
5. Export đổi schema sang frame-object; không được đổi annotation authority.

## 15. Ranh giới với classification_v2

Sau khi export PASS, file có thể được đưa vào một lineage
`classification_v2` mới. Bước đó phải tạo `scene_frame_uid`/object key, tính
ROI, geometry, motion, social và pen context từ frame-object rows; không dùng
`manual_*`, `review_*`, policy text, fold ID hoặc target-derived field làm X.

Các view classifier được tạo sau export:

```text
C6  = sáu frame liên tục theo contract đã khóa
C8  = tám frame liên tục
C12 = mười hai frame liên tục
C16 = đủ mười sáu frame
```

Native evaluation unit vẫn là toàn bộ burst 16 frame. Không chia overlapping
window giữa các fold. Human review là bắt buộc cho artifact gọi là reviewed/final
hoặc mixed-source Q2; legacy-only exploratory run phải gắn nhãn
`legacy-only-unreviewed-development`.

## 16. Stop conditions

Dừng ngay, không chạy full, nếu có một trong các điều kiện sau:

- CVAT audit có duplicate anchor identity hoặc lỗi frame-map.
- Không có file exclusion policy mới/đã xác nhận.
- Source video không resolve được mà không có quyết định loại rõ ràng.
- K0 thiếu, behavior k0 không hợp lệ hoặc bbox anchor không hợp lệ.
- Short recovery audit không PASS.
- Dense output không đủ 16 frame/actor hoặc có duplicate frame-object key.
- Export audit thay đổi bbox/behavior/Hidden provenance.
- Input hash thay đổi giữa short gate và full run.
- Output path trỏ vào `data/`, canonical output hoặc một lineage khác.

Không được giải quyết stop condition bằng cách:

- copy k0 bbox cho tất cả frame;
- bỏ qua duplicate bằng `drop_duplicates` không audit;
- đổi behavior k1..k5 để làm mất disagreement;
- điền Hidden mặc định;
- bật `--allow-unresolved-video` cho training input;
- dùng `--max-rows` làm full/short temporal sample;
- dùng `--resume` với config hoặc input hash khác.

## 17. Checklist handoff

Trước khi giao file cho classification_v2, lưu các artifact sau cùng một
`RUN_ID`:

```text
[ ] input hashes và CVAT task/manifest hashes
[ ] legacy_source_trace_lineage.json
[ ] exclusion policy hash và duplicate audit
[ ] legacy_cvat_recovery_input_audit.json
[ ] legacy_recovery_input_manifest.json
[ ] short recovery output audit
[ ] full recovery output audit
[ ] legacy_frame_object_export_audit.json
[ ] legacy_frame_object_annotations.csv hash
[ ] row/key/16-frame summary
[ ] danh sách actor bị loại và lý do, nếu có
```

Chỉ sau checklist này mới chạy bước build feature của `classification_v2`.
Việc có file CSV và script exit 0 không đủ để gọi dữ liệu là train-ready.
