# Behavior Strength / ROI Consistency Review Tools

Các tool này dùng sau bước tạo `spatiotemporal_frame_features_roi.csv` để xử lý hai vấn đề:

1. ROI-label consistency: `eat/drink/playwithtoy` nhưng bbox chỉ `near` hoặc `far` ROI target.
2. Label strength theo window: `fight`, `social-nose`, `move`, `explore`, `stand`, `lying/sitting` có thể đúng ở cấp event nhưng yếu ở window 6/12/16 frame.

## 1) Tạo template review + auto attributes

```bash
python behavior_review_tools/build_behavior_review_templates.py \
  --input-csv spatiotemporal_frame_features_roi.csv \
  --output-dir outputs/behavior_strength_review \
  --scope critical \
  --write-full-annotated
```

Output chính:

- `behavior_strength_review_template.csv`: file để reviewer điền tay.
- `behavior_strength_review_template_priority.csv`: cùng file nhưng sort theo priority.
- `frame_features_with_auto_review_attrs.csv`: full CSV có thêm auto attributes.
- `behavior_strength_review_audit.json`: audit thống kê.

Các cột reviewer cần điền nếu review thủ công bằng CSV:

- `manual_review_decision`: `accept`, `corrected`, `reject`, `exclude`, `boundary_exclude`
- `manual_label_strength`: `strong`, `medium`, `weak`, `boundary`
- `manual_corrected_behavior`: để trống nếu giữ label cũ, hoặc nhập label mới.
- `manual_ambiguity_group`: `aggression_social`, `roi_based`, `motion_state`, `posture`, `general`
- `manual_training_action`: `main_train`, `low_weight_train`, `robust_train_only`, `exclude_main`, `exclude`
- `manual_sample_weight`: ví dụ `1.0`, `0.75`, `0.35`, `0`
- `manual_note`: lý do review.

## 2) Review bằng GUI nếu có crop/image

```bash
python behavior_review_tools/review_behavior_strength_gui.py \
  --review-csv outputs/behavior_strength_review/behavior_strength_review_template_priority.csv \
  --raw-root path/to/crops_or_frames \
  --output-dir outputs/behavior_strength_review/gui_decisions \
  --priority-max 2
```

Phím tắt:

- `1`: strong
- `2`: medium
- `3`: weak
- `4`: boundary
- `Enter`: save decision
- `S`: skip
- `Ctrl+S`: save & exit

GUI tạo:

- `behavior_strength_review_decisions.csv`

## 3) Apply decisions và tạo CSV chuẩn để train

```bash
python behavior_review_tools/apply_behavior_review_attributes.py \
  --annotated-csv outputs/behavior_strength_review/frame_features_with_auto_review_attrs.csv \
  --review-decisions-csv outputs/behavior_strength_review/gui_decisions/behavior_strength_review_decisions.csv \
  --output-csv outputs/behavior_strength_review/training_ready_behavior_features.csv \
  --pending-policy exclude
```

Output cuối có thêm các cột:

- `label_strength`
- `ambiguity_group`
- `review_reason`
- `review_decision`
- `behavior_train`
- `training_weight_final`
- `include_in_training_final`
- `use_for_main_train_final`
- `use_for_robust_train_final`
- `use_for_roi_training_final`

Khuyến nghị:

- Train chính: dùng `include_in_training_final == True` và `use_for_main_train_final == True`.
- Nếu muốn robust training, có thể thêm weak samples với `--include-weak-in-training`.
- Giữ validation/test đầy đủ nhưng report metric tách theo `label_strength`.
