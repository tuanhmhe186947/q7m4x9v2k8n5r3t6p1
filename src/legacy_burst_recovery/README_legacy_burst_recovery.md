# README - Legacy Burst Recovery / Multi-GT / Mask / No-Duplicate Pipeline

Tài liệu này ghi lại toàn bộ logic và các câu lệnh quan trọng của phần khôi phục dữ liệu **old legacy burst** trong dự án `PIG_Behavior_Project`, nhằm tránh quên sau này.

Project local:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project
```

Module chính:

```text
src\legacy_burst_recovery
```

> **Current classification_v2 override (2026-07-17):** after CVAT behavior and
> bbox corrections, do not start from the old center/all-bbox CSV pair below.
> Follow
> `docs/LEGACY_16F_REBUILD_FROM_SCRATCH_RUNBOOK.md`: rebuild
> center and six-anchor inputs from `data/data/task_0..task_3`, map each actor's
> `k0` behavior to `k1..k5` and all 16 dense frames, and preserve each CVAT
> anchor bbox independently. The older commands remain historical reference.

The pre-CVAT combined/all-bbox CSVs and the old generated 16-frame export are
archived under
`outputs/_archive/legacy_16f_pre_cvat_rebuild_20260717` and
`outputs/_archive/legacy_16f_root_artifacts_20260718`. No root-level legacy CSV
is an input to the current rebuild. The nodup center table is recreated inside
the versioned run root and is used as metadata scaffold only.

### Provenance trace guard

`truy_nguon_multi_bbox.py` is a historical provenance scaffold, not the
current CVAT annotation authority and not a training-data builder. It resolves
source-video and manifest metadata for old burst rows; current CVAT behavior
authority remains `k0`, and current CVAT bbox authority remains the independent
`k0..k5` anchors handled by the CVAT rebuild modules.

The trace script now fails closed unless every `(group_id, pig_id)` has exactly
one valid row for each `k/order` in `0..5`, image-name fields agree with
`group_id`, bbox coordinates are valid, and group frame mapping is unique.
Manifest metadata is authoritative. Candidate metadata is fallback only when
manifest video metadata is absent. A behavior-side `video` column is isolated
and can never override either source. Conflicting duplicate manifest/candidate
rows, manifest/candidate video disagreement, row multiplication, unresolved
videos, and a video hash that does not match the group id stop the run.

Write derived outputs to a fresh output directory outside `data/`,
`src/`, `scripts/`, and `tests/`. Use `--dry-run` for the validation gate
and `--overwrite` only when replacing a deliberately versioned output. Use
`--allow-incomplete-actor-keys` or `--allow-unresolved-video` only for an
explicit audit artifact; those flags must not be used for train-ready data.
The backward-compatible center CSV still selects historical `k3` and is
metadata evidence only.

At the time of this rebuild, `src/legacy_burst_recovery` contains no generated
CSV files. The archived nodup center CSV and other historical artifacts must
not be mistaken for the new CVAT-derived lineage.

Output full 13-frame hiện tại:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup
```

---

## 1. Mục tiêu của pipeline

Pipeline này dùng để khôi phục dữ liệu behavior cũ từ old burst annotation thành dữ liệu training mới có:

```text
- crop ảnh theo từng pig_id
- bbox frame-level
- tracklet_id
- group_id
- sample_id
- behavior
- timestamp_sec từ times.txt
- multi-GT legacy bbox support
- scene mask filtering
- QA / manual review
- loại duplicate video với data mới
- manifest training-ready
```

Điểm quan trọng: **không chỉ tạo ảnh crop**, mà phải giữ được annotation bbox theo từng frame để sau này tính feature không gian-thời gian cho model như baseline / Pig-STRENet.

---

## 2. Các file input chính

### 2.1. Sample-level CSV

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_combined.csv
```

Hoặc bản đã loại video trùng:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_nodup_videos.csv
```

Ý nghĩa:

```text
1 row = 1 group_id + pig_id sample
```

File này dùng làm input chính cho pipeline qua:

```cmd
--input-csv
```

Không được thay nó bằng file all-keyframe bbox, vì sẽ tạo duplicate tracklet.

---

### 2.2. Multi-GT legacy bbox CSV

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_all_keyframe_bboxes_combined.csv
```

Hoặc bản đã loại video trùng:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_all_keyframe_bboxes_nodup_videos.csv
```

Ý nghĩa:

```text
multiple rows = group_id + pig_id + frame_index
```

File này chứa bbox legacy cho 6 keyframe old burst.

Dùng qua:

```cmd
--legacy-burst-bbox-csv
```

Khi có file này, pipeline chạy ở mode:

```text
legacy_gt_mode = multi_anchor
```

---

### 2.3. Scene mask

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png
```

Dùng để loại detection ngoài chuồng / vùng không hợp lệ.

Các flag liên quan:

```cmd
--scene-mask "C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png"
--mask-filter-detections
--mask-min-bbox-coverage 0.50
--mask-require-center-inside
```

Logic:

```text
YOLO detect trên frame gốc
sau đó filter detection bằng mask
tracking/association chỉ dùng detection sau mask
GT legacy vẫn thắng trên GT frame
```

---

### 2.4. Manual review stable

File đúng hiện tại:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\manual_review_stable.csv
```

Không dùng lại file cũ chỉ dựa vào `tracklet_id`, vì sau khi lọc duplicate/nodup, `tracklet_id` có thể bị lệch.

Manual review phải match theo thứ tự:

```text
1. sample_id
2. group_id + pig_id
3. tracklet_id fallback, nhưng phải verify stable identifiers nếu có
```

Pipeline đã có audit:

```text
manual_review_apply_audit.csv
```

---

## 3. Logic 13 frame và 16 frame

### 3.1. Old burst gốc

Old burst gốc có 6 keyframe, thường cách nhau 3 frame index:

```text
relative frame: 0, 3, 6, 9, 12, 15
```

Do video thực tế khoảng 6 FPS theo `times.txt`:

```text
1 frame thực ≈ 0.16 giây
3 frame thực ≈ 0.48 giây
6 frame thực ≈ 0.95-1.0 giây
```

Vậy old burst 6 keyframe phủ khoảng:

```text
2.3 - 2.5 giây
```

---

### 3.2. Mode `sample_0_6_12`

Flag:

```cmd
--track-end-mode sample_0_6_12
```

Dense range:

```text
anchor .. anchor + 12
```

Số frame:

```text
13 frame = 0..12
```

Khớp các mốc GT:

```text
0, 3, 6, 9, 12
```

Chưa lấy GT cuối `15`.

Phù hợp với view:

```text
sparse_3_0_6_12
```

Nghĩa là 3 mốc khoảng:

```text
0s, 1s, 2s
```

---

### 3.3. Mode `full_legacy_burst`

Flag:

```cmd
--track-end-mode full_legacy_burst
```

Dense range:

```text
anchor .. legacy GT cuối
```

Với old burst `0,3,6,9,12,15`, dense sẽ là:

```text
0..15
```

Số frame:

```text
16 frame
```

Khớp đủ 6 GT keyframe:

```text
0, 3, 6, 9, 12, 15
```

Nên dùng nếu mục tiêu là khôi phục đúng full old burst gốc.

---

### 3.4. Không nên hiểu `0,5,10` là GT chính

`0,5,10` có thể hữu ích để kiểm tra đầu/cuối cửa sổ 1 giây, nhưng không khớp trực tiếp với GT legacy.

So với old GT:

```text
0,6,12  -> trùng mốc GT legacy
0,5,10 -> frame trung gian giữa GT anchors
```

Vì vậy:

```text
- dùng 0,6,12 cho sparse 3-frame chính
- dùng 0,3,6,9,12,15 nếu muốn đúng old burst 6 keyframe
- dùng temporal consistency audit trên 0-5, 6-11 nếu muốn kiểm tra ổn định hành vi trong cửa sổ 1 giây
```

---

## 4. Loại video trùng với data mới

Các video mới nằm tại:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\data\videos
```

Ví dụ:

```text
Pigs281119_000085_30fps.mp4
```

Map thành source key:

```text
pigs281119/000085
```

Legacy Drive path tương ứng:

```text
...\pig_data_unzipped\pigs281119\PIGS281119\000085\color.mp4
```

cũng map thành:

```text
pigs281119/000085
```

Đã phát hiện duplicate cũ:

```text
legacy rows = 4602
duplicate rows = 47
duplicate group+pig = 47
```

Các video trùng:

```text
pigs291119/000233    8
pigs291119/000231    8
pigs291119/000225    8
pigs281119/000085    8
pigs281119/000114    8
pigs291119/000226    7
```

Vì vậy file training chính phải dùng bản nodup:

```text
old_burst_center_keyframes_nodup_videos.csv
old_burst_all_keyframe_bboxes_nodup_videos.csv
```

Không dùng file combined gốc cho full training nếu muốn tránh leakage.

---

## 5. Full run 13-frame stable hiện tại

Lệnh full đã chạy thành công, không bật debug visuals:

```cmd
cd /d "C:\Users\ironh\Downloads\PIG_Behavior_Project"
set PYTHONPATH=%CD%\src

python -m legacy_burst_recovery.main ^
  --input-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_nodup_videos.csv" ^
  --legacy-burst-bbox-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_all_keyframe_bboxes_nodup_videos.csv" ^
  --drive-root "G:\My Drive" ^
  --output-root "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup" ^
  --detector-weights "C:\Users\ironh\Downloads\PIG_Behavior_Project\models\detector\pig_detector_yolov8.pt" ^
  --scene-mask "C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png" ^
  --mask-filter-detections ^
  --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside ^
  --track-end-mode sample_0_6_12 ^
  --extract-crops ^
  --manual-review-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\manual_review_stable.csv" ^
  --sequence-views sparse_3_0_6_12 dense_6_same_span dense_12_same_span full_dense_0_to_12 ^
  --resume ^
  --progress ^
  --log-file "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup.log" ^
  --flush-every 500
```

Không bật trong full:

```cmd
--save-debug-visuals
--debug-draw-all-detections
--extract-full-frames
```

Vì các flag này làm output rất nặng.

---

## 6. Kết quả full 13-frame đã đạt

Output:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup
```

Kết quả kiểm tra:

```text
dense rows = 59202
dense tracklets = 4554
sequence rows = 18176
sequence tracklets = 4544
```

Training tier:

```text
clean             4543
review               8
hard_occlusion       1
rejected             1
warning              1
```

Include in training:

```text
True     4544
False      10
```

Tracking status:

```text
ok                 36422
ok_gt              22749
corrected_by_gt       17
low_confidence        14
```

QA status:

```text
ok        59188
review       14
```

Mask:

```text
raw mean = 8.620958751393534
masked mean = 8.065470761122935
outside rejected = 32886
selected outside = 0
```

Kết luận:

```text
Dataset 13-frame stable hiện tại dùng được.
4554 tracklet audit.
4544 tracklet training-ready.
Không có selected bbox ngoài mask.
Không có failed tracklet.
Duplicate video mới đã được loại khỏi input.
```

---

## 7. Kiểm tra tổng quan sau run

Dùng lệnh này để kiểm tra bất kỳ output nào:

```cmd
python -c "import pandas as pd; root=r'C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup'; d=pd.read_csv(root+r'\legacy_dense_tracklet_map.csv',low_memory=False); s=pd.read_csv(root+r'\legacy_training_sequence_manifest.csv',low_memory=False); print('dense rows=',len(d)); print('dense tracklets=',d['tracklet_id'].nunique()); print('sequence rows=',len(s)); print('sequence tracklets=',s['tracklet_id'].nunique()); print('\ntraining_tier:'); print(d.groupby('tracklet_id')['training_tier'].first().value_counts(dropna=False)); print('\ninclude_in_training:'); print(d.groupby('tracklet_id')['include_in_training'].first().value_counts(dropna=False)); print('\ntracking_status:'); print(d['tracking_status'].value_counts(dropna=False)); print('\nqa_status:'); print(d['qa_status'].value_counts(dropna=False)); print('\nmask:'); print('raw mean=',d['num_detections_raw'].mean()); print('masked mean=',d['num_detections_after_mask'].mean()); print('outside rejected=',d['num_detections_outside_mask'].sum()); print('selected outside=',(~d['selected_det_center_in_mask'].fillna(True)).sum())"
```

---

## 8. Kiểm tra duplicate video trong output

```cmd
python -c "import pandas as pd,re; root=r'C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup'; ex=set(pd.read_csv(r'C:\Users\ironh\Downloads\PIG_Behavior_Project\exclude_source_videos.csv')['source_video_key'].astype(str).str.lower()); pat=re.compile(r'(pigs\d{6})/pigs\d{6}/(\d+)/color\.mp4',re.I); key=lambda x:(lambda m:m.group(1)+'/'+m.group(2) if m else None)(pat.search(str(x).replace('\\\\','/').lower())); d=pd.read_csv(root+r'\legacy_dense_tracklet_map.csv',low_memory=False); s=pd.read_csv(root+r'\legacy_training_sequence_manifest.csv',low_memory=False); d['source_video_key_check']=d['source_video_resolved'].map(key) if 'source_video_resolved' in d.columns else d['color_video_path'].map(key); s['source_video_key_check']=s['source_video_resolved'].map(key) if 'source_video_resolved' in s.columns else s['color_video_path'].map(key); print('dense duplicate tracklets=',d[d['source_video_key_check'].isin(ex)]['tracklet_id'].nunique()); print('sequence duplicate tracklets=',s[s['source_video_key_check'].isin(ex)]['tracklet_id'].nunique()); print(d[d['source_video_key_check'].isin(ex)]['source_video_key_check'].value_counts().to_string() if d['source_video_key_check'].isin(ex).any() else 'OK: no duplicate source video remains')"
```

Kết quả đúng:

```text
dense duplicate tracklets= 0
sequence duplicate tracklets= 0
OK: no duplicate source video remains
```

---

## 9. Kiểm tra manual review apply

```cmd
python -c "import pandas as pd; p=r'C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup\manual_review_apply_audit.csv'; a=pd.read_csv(p,low_memory=False); print(a.to_string(index=False))"
```

Manual review đúng phải match bằng:

```text
sample_id
```

Không nên match bằng `tracklet_id` nếu có stable key.

---

## 10. Liệt kê tracklet bị loại / cần review

```cmd
python -c "import pandas as pd; root=r'C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup'; d=pd.read_csv(root+r'\legacy_dense_tracklet_map.csv',low_memory=False); per=d.groupby('tracklet_id').agg(group_id=('group_id','first'),sample_id=('sample_id','first'),pig_id=('pig_id','first'),behavior=('behavior','first'),include_in_training=('include_in_training','first'),training_tier=('training_tier','first'),min_track_conf=('track_confidence','min'),review_frames=('qa_status',lambda x:(x=='review').sum()),low_conf_frames=('tracking_status',lambda x:(x=='low_confidence').sum()),interp_frames=('bbox_source',lambda x:(x=='interpolated_between_gt').sum()),qa_notes=('qa_notes',lambda x:';'.join(sorted(set(str(v) for v in x.dropna())))[:400])); bad=per[(per['include_in_training'].astype(str).str.lower().isin(['false','0'])) | (per['training_tier'].astype(str).ne('clean'))]; print(bad.to_string())"
```

---

## 11. Nếu cần debug visuals sau full

Full không lưu debug visuals. Nếu sau full cần xem tracklet bị review, rerun riêng một group.

Ví dụ:

```cmd
cd /d "C:\Users\ironh\Downloads\PIG_Behavior_Project"
set PYTHONPATH=%CD%\src

python -m legacy_burst_recovery.main ^
  --input-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_nodup_videos.csv" ^
  --legacy-burst-bbox-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_all_keyframe_bboxes_nodup_videos.csv" ^
  --drive-root "G:\My Drive" ^
  --output-root "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\debug_one_tracklet" ^
  --detector-weights "C:\Users\ironh\Downloads\PIG_Behavior_Project\models\detector\pig_detector_yolov8.pt" ^
  --scene-mask "C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png" ^
  --mask-filter-detections ^
  --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside ^
  --track-end-mode sample_0_6_12 ^
  --save-debug-visuals ^
  --debug-draw-all-detections ^
  --manual-review-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\manual_review_stable.csv" ^
  --filter-group-id "burst_color_0bdcdf43_367" ^
  --progress
```

---

## 12. Test 16-frame / full legacy burst

Đã xác nhận CLI có hỗ trợ:

```cmd
python -m legacy_burst_recovery.main --help | findstr /I "full_legacy_burst"
```

Help trả về:

```text
--track-end-mode {sample_0_6_12,full_legacy_burst}
```

Test 20 hoặc 500 trước khi chạy full 16-frame.

### Test 20 nhanh

```cmd
cd /d "C:\Users\ironh\Downloads\PIG_Behavior_Project"
set PYTHONPATH=%CD%\src

python -m legacy_burst_recovery.main ^
  --input-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_nodup_videos.csv" ^
  --legacy-burst-bbox-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_all_keyframe_bboxes_nodup_videos.csv" ^
  --drive-root "G:\My Drive" ^
  --output-root "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_test_20_full_legacy_burst" ^
  --detector-weights "C:\Users\ironh\Downloads\PIG_Behavior_Project\models\detector\pig_detector_yolov8.pt" ^
  --scene-mask "C:\Users\ironh\Downloads\PIG_Behavior_Project\data\annotations\scene\mask.png" ^
  --mask-filter-detections ^
  --mask-min-bbox-coverage 0.50 ^
  --mask-require-center-inside ^
  --track-end-mode full_legacy_burst ^
  --extract-crops ^
  --manual-review-csv "C:\Users\ironh\Downloads\PIG_Behavior_Project\manual_review_stable.csv" ^
  --sequence-views sparse_3_0_6_12 full_dense_0_to_12 ^
  --max-rows 20 ^
  --progress ^
  --log-file "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_test_20_full_legacy_burst.log" ^
  --flush-every 10
```

Kiểm tra:

```cmd
python -c "import pandas as pd; p=r'C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_test_20_full_legacy_burst\legacy_dense_tracklet_map.csv'; d=pd.read_csv(p,low_memory=False); print(d.groupby('tracklet_id')['frame_index'].count().value_counts().sort_index()); print(d.groupby('tracklet_id')['legacy_gt_support_count'].first().value_counts(dropna=False)); print(d.groupby('tracklet_id')['legacy_gt_support_frames'].first().head(20).to_string())"
```

Kết quả mong muốn:

```text
16    20
legacy_gt_support_count = 6
```

---

## 13. Sequence views nên có nếu dùng 16 frame

Hiện các view cũ:

```text
sparse_3_0_6_12
dense_6_same_span
dense_12_same_span
full_dense_0_to_12
```

Nếu chuyển sang 16 frame, nên thêm view mới:

```text
legacy_gt_6_frames = 0,3,6,9,12,15
full_dense_0_to_15 = 0..15
optional sparse_4_0_5_10_15
```

Không xóa view cũ. Chỉ thêm view mới để tương thích với code cũ và khai thác đủ GT cuối.

Prompt cho agent nếu cần:

```text
Add sequence views for 16-frame full_legacy_burst output.

Keep all existing sequence views unchanged.

Add:
- legacy_gt_6_frames: relative offsets 0,3,6,9,12,15
- full_dense_0_to_15: relative offsets 0..15
- optional sparse_4_0_5_10_15

Do not hard-code dense length 13. Sequence generation must read available dense frames per tracklet and skip a view only if required offsets are missing.

For --track-end-mode full_legacy_burst, expected dense_frame_count is 16 and legacy_gt_support_count is 6.

Update QA summary with sequence_view_distribution.
```

---

## 14. File annotation quan trọng nhất để tính spatial features

Dùng:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup\legacy_dense_tracklet_map.csv
```

Đây là file frame-level annotation đầy đủ, có:

```text
tracklet_id
sample_id
group_id
pig_id
behavior
frame_index
timestamp_sec
x1,y1,x2,y2
bbox_source
tracking_status
qa_status
include_in_training
training_tier
crop_path
source_video_resolved / color_video_path
```

Từ file này tính được:

```text
cx_n, cy_n, bw_n, bh_n
area_n, aspect_ratio
speed_feat, vx, vy
path_length, displacement_ratio
bbox_stability
min_dist_other, num_close_other
pair_iou
social_density
ROI relation nếu có roi_layout
```

Không tính spatial trực tiếp từ crop image. Crop chỉ dùng cho visual branch.

---

## 15. Export annotation CSV / COCO / CVAT XML

Nên tạo file tổng hợp:

```text
legacy_frame_object_annotations.csv
```

Một dòng = một pig object trong một frame/image.

Có thể xuất thêm:

```text
legacy_annotations_coco.json
legacy_annotations_cvat_1_1.xml
```

Nguồn chính vẫn là:

```text
legacy_dense_tracklet_map.csv
```

Output export nên đặt tại:

```text
C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup\exports
```

File quan trọng nhất cho feature/training:

```text
legacy_frame_object_annotations.csv
```

Vì nó thể hiện rõ:

```text
image_key
image_name
group_id
frame_index
pig_id
behavior
x1,y1,x2,y2
cx_n,cy_n,bw_n,bh_n
bbox_source
tracking_status
include_in_training
crop_path
```

---

## 16. Quy tắc vận hành khi chạy lâu

Trước khi treo full run:

```text
- Cắm sạc.
- Tắt sleep/hibernate.
- Không để Google Drive pause.
- Output ghi vào ổ local C:, không ghi trực tiếp vào G:\My Drive.
- Không đóng CMD/terminal.
- Dùng --resume nếu có.
- Không bật debug_visuals full.
```

Kiểm tra `--resume`:

```cmd
python -m legacy_burst_recovery.main --help | findstr /I "resume"
```

Xem progress:

```cmd
type "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup\progress_state.json"
```

Hoặc xem log:

```cmd
type "C:\Users\ironh\Downloads\PIG_Behavior_Project\outputs\legacy_full_multigt_masked_nodup.log"
```

---

## 17. Checklist trước khi dùng dataset để train

Kiểm tra các điểm sau:

```text
[ ] Dùng input nodup, không dùng combined gốc.
[ ] legacy_gt_mode = multi_anchor.
[ ] mask_filter_applied = True.
[ ] selected outside mask = 0.
[ ] bad tracklets không nằm trong sequence manifest.
[ ] manual_review_apply_audit.csv match bằng sample_id.
[ ] duplicate source video = 0.
[ ] dense frame count đúng với mode: 13 hoặc 16.
[ ] crop_path tồn tại.
[ ] behavior label thuộc 10 class hợp lệ.
[ ] spatial feature CSV được tính từ dense annotation, không từ crop.
```

---

## 18. Tóm tắt file nào dùng cho việc gì

```text
old_burst_center_keyframes_nodup_videos.csv
  -> input sample-level, 1 row / group_id + pig_id

old_burst_all_keyframe_bboxes_nodup_videos.csv
  -> input multi-GT bbox, nhiều rows / group_id + pig_id

legacy_dense_tracklet_map.csv
  -> annotation frame-level đầy đủ, dùng tính spatial/motion/social features

legacy_training_sequence_manifest.csv
  -> danh sách sequence training-ready, chỉ include_in_training=True

manual_review_apply_audit.csv
  -> audit manual review có apply đúng sample_id hay không

qa_report.md / qa_summary.json
  -> tổng kết QA

crops/
  -> ảnh crop cho visual branch

exports/legacy_frame_object_annotations.csv
  -> file object/frame tổng hợp để tính feature hoặc export training data

exports/legacy_annotations_coco.json
  -> COCO format phụ

exports/legacy_annotations_cvat_1_1.xml
  -> CVAT XML 1.1 format phụ
```

---

## 19. Kết luận hiện tại

Bản stable hiện tại:

```text
outputs\legacy_full_multigt_masked_nodup
```

là bản **13-frame stable** đã chạy full thành công.

Nó phù hợp để:

```text
- training với view 0,6,12
- training full dense 0..12
- tính spatial/motion/social features trên 13 frame
- làm baseline ổn định
```

Nếu muốn đúng hoàn toàn với old burst 6 keyframe gốc, bước tiếp theo là thử:

```text
--track-end-mode full_legacy_burst
```

và thêm view:

```text
legacy_gt_6_frames = 0,3,6,9,12,15
full_dense_0_to_15 = 0..15
```

Không overwrite bản 13-frame stable. Tạo output riêng cho 16-frame.
