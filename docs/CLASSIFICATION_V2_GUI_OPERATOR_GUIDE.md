# Hướng dẫn vận hành GUI Classification V2

Tài liệu này là điểm vào duy nhất cho hai GUI đang được dùng:

- `behavior`: review nhãn hành vi trong phạm vi 2.729 unit đã đóng băng;
- `mini-cvat`: sửa bbox, ID, Hidden và Behavior cho một lỗi định danh cụ thể.

Mini-CVAT V1 đã lỗi thời. Chỉ dùng mini-CVAT V2 qua launcher bên dưới.

## 1. Chuẩn bị một lần cho mỗi cửa sổ CMD

Mở `Anaconda Prompt`, sau đó chạy:

```bat
conda activate pig_project
cd /d "C:\Users\ironh\Downloads\PIG_Behavior_Project"
classification_v2_gui.cmd status
```

`status` chỉ kiểm tra các đầu vào đã cấu hình. Nó không đọc ledger quyết định
Behavior và không mở GUI. Tất cả dòng `INPUT` phải báo `EXISTS=YES`.

Muốn xem lệnh đầy đủ mà không mở GUI:

```bat
classification_v2_gui.cmd behavior --dry-run
```

## 2. Mở GUI review Behavior

Lệnh thường dùng:

```bat
classification_v2_gui.cmd behavior
```

GUI đọc lại quyết định đã lưu và hỏi resume. Đóng cửa sổ sau khi đã lưu không
làm mất các lượt trước đó.

Muốn mở thẳng tại một unit cụ thể và bỏ qua hộp thoại resume:

```bat
classification_v2_gui.cmd behavior --start-review-unit-id unit_review_00030931
```

Chỉ dựng hoặc kiểm tra cache frame, không mở Tk:

```bat
classification_v2_gui.cmd behavior --prepare-frame-cache-only
```

### Đầu vào Behavior

Launcher lấy các giá trị sau từ profile:

- `review_units_csv`: phạm vi review 2.729 unit đã đóng băng;
- `frame_features_csv`: evidence theo frame để dựng media và ngữ cảnh;
- `video_root`: video CVAT toàn cảnh;
- `raw_root`: ảnh/crop legacy;
- `roi_coco_json`: ROI feeder, drinker và toy;
- `behavior.output_dir`: thư mục phiên có thể resume.

ROI bắt buộc hiện tại là:

```text
data/annotations/roi/ROI_annotations.toy_adjusted.coco.json
```

Tên thư mục output còn chữ `noninteraction_production_46e67e9` vì lý do lịch
sử. Phạm vi thực tế không được suy ra từ tên thư mục; nó được quyết định bởi
`combined_final_behavior_review_view.csv` trong profile.

### File Behavior được tạo hoặc cập nhật

Trong `behavior.output_dir`:

- `behavior_unit_review_decisions.csv`: quyết định accept/correct theo unit;
- `behavior_label_quality_review.csv`: lý do và cờ chất lượng phục vụ audit;
- `.final_behavior_frame_features.sqlite3`: cache đọc frame có thể tạo lại.

Hai CSV đầu là dữ liệu review cần bảo vệ. Cache SQLite không phải label authority.
Các trường reason/quality chỉ dùng audit và tuyệt đối không được đưa vào model-X.

## 3. Mở mini-CVAT V2

Mỗi lỗi phải có `--session-name` riêng. Mỗi unit và mỗi ID được phép chỉnh phải
lặp lại đúng một cờ tương ứng.

Ví dụ sửa hai unit và ba ID, vẫn không cần gõ lại các đường dẫn dài:

```bat
set CASE_ITEMS=--review-item-id unit_review_00030931 --review-item-id unit_review_00030932
set CASE_PIGS=--editable-pig-id ID_4 --editable-pig-id ID_5 --editable-pig-id ID_6
set CASE_NAME=fight_move_identity_18db1b2
classification_v2_gui.cmd mini-cvat --session-name %CASE_NAME% %CASE_ITEMS% %CASE_PIGS%
```

Ba dòng trên là một ví dụ hoàn chỉnh. `CASE_ITEMS` và `CASE_PIGS` chỉ tồn tại
trong cửa sổ CMD hiện tại. Cách này tránh dấu `^`, nên CMD không thể chạy Python
trước khi nhận đủ tham số.

Các trường hợp khác chỉ thay:

- `--session-name`: tên ngắn, duy nhất, không có dấu `/` hoặc `\`;
- `--review-item-id`: unit cần sửa, có thể lặp;
- `--editable-pig-id`: ID được phép sửa, có thể lặp;
- `--reviewer`: tùy chọn; mặc định là `TuanHM` trong profile.

### File mini-CVAT được tạo

Mỗi session ghi vào:

```text
outputs/classification_v2/identity_adjudication_sessions/<session-name>/
```

File chính là `mini_cvat_adjudication.json`. Nó lưu sidecar đã review nhưng chưa
tự sửa CSV/XML nguồn.

Điều này rất quan trọng:

```text
LƯU FRAME / LƯU BEHAVIOR -> lưu sidecar
ÁP DỤNG CSV + XML NGUỒN  -> ghi một generation nguồn có audit
```

## 4. Bật nút áp dụng CSV/XML nguồn

Launcher không lưu sẵn đường dẫn nguồn để tránh áp dụng nhầm case. Khi thật sự
cần apply, bổ sung đồng thời:

```text
--apply-source-csv <dense-or-raw-csv>
--apply-source-xml <original-cvat-xml>
```

Nếu case có nhiều CSV, lặp `--apply-source-csv`. Có thể thêm
`--apply-group-id <exact-group-id>` khi không thể suy ra burst duy nhất.

Chỉ truyền một trong CSV/XML sẽ bị từ chối. Dùng `--dry-run` cuối lệnh để kiểm
tra toàn bộ command trước khi mở GUI.

Khi bấm nút apply trong mini-CVAT V2, luồng tạo generation có audit, gồm:

- `identity_source_apply_manifest.json` trong generation;
- `latest_identity_source_apply.json` trỏ tới generation mới nhất;
- backup và dữ liệu rollback trong apply audit root.

Không dùng chung `--session-name` cho các lỗi không liên quan.

## 5. Profile: nơi duy nhất cần sửa đường dẫn

Profile mặc định:

```text
configs/classification_v2/gui_operator_profile_v1.json
```

Chỉ sửa profile khi authority đầu vào thực sự thay đổi. Các khóa có ý nghĩa:

| Khóa | Vai trò |
| --- | --- |
| `common.review_units_csv` | population Behavior và lookup case mini-CVAT |
| `common.frame_features_csv` | frame evidence chung cho cả hai GUI |
| `common.video_root` | video toàn cảnh CVAT |
| `common.raw_root` | media legacy cho Behavior GUI |
| `common.roi_coco_json` | ROI scene đã điều chỉnh toy |
| `behavior.output_dir` | quyết định Behavior và trạng thái resume |
| `mini_cvat.output_root` | thư mục cha của các session sửa bbox/ID |
| `default_reviewer` | reviewer mặc định của mini-CVAT |

Không đổi `behavior.output_dir` giữa chừng nếu muốn resume cùng một đợt review.
Không trỏ output mini-CVAT vào thư mục Behavior.

Muốn dùng một profile khác:

```bat
classification_v2_gui.cmd --profile <profile.json> status
```

## 6. Vị trí hai GUI trong pipeline

Luồng đúng sau khi hoàn tất review là:

1. Behavior GUI hoàn tất phạm vi 2.729 unit đã đóng băng.
2. Mini-CVAT V2 sửa riêng các lỗi bbox, ID, Hidden hoặc Behavior có chứng cứ.
3. Apply sidecar mini-CVAT vào CSV/XML bằng generation có manifest và rollback.
4. Review residual control ít nhất 120 mẫu từ nhóm từng bị lọc bỏ.
5. Đóng băng review-close authority và corrected-source authority.
6. Phân tích nhãn đổi/không đổi để chẩn đoán logic không-thời gian.
7. Dựng lại frame features từ corrected source và ROI toy-adjusted.
8. Áp dụng Behavior authority đã đóng băng, có kiểm tra conflict.
9. Full-recompute sequence T6, T8, T12 và T16.
10. Chạy audit lineage, leakage, mask rồi mới mở gate training reviewed-data.

Không được chỉ sửa CSV mà bỏ XML, hoặc ghép decision vào tensor cũ. Sửa bbox/ID
làm thay đổi ROI, social và motion; vì vậy các đặc trưng không-thời gian phải
được full-recompute sau khi authority đã đóng băng.

## 7. Chẩn đoán nhanh

- `required_path_missing=...`: sửa khóa tương ứng trong profile.
- `source_apply_requires_both_csv_and_xml`: truyền đủ cả hai loại nguồn.
- `unsafe_session_name=...`: bỏ slash, khoảng trắng và ký tự đặc biệt.
- GUI thoát ngay: chạy lại cùng lệnh với `--dry-run`, rồi kiểm tra `status`.
- Resume sai: xác nhận đang dùng đúng `behavior.output_dir`; không tạo output mới.

Không mở mini-CVAT bằng script nội bộ. V1 đã được loại bỏ; launcher luôn gọi V2.
