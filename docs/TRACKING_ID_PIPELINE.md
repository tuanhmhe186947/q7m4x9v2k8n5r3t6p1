# Pig ID Tracking Pipeline

Tài liệu này giải thích file
`notebooks/01_data_preparation/track_video_ids_for_annotation.py`.
Mục tiêu là tạo annotation CVAT cho video 30 FPS, luôn có 8 bbox trong mỗi
frame, hạn chế ID switch khi lợn di chuyển tập trung hoặc chồng lấn.

## Output

Pipeline tạo các file chính trong thư mục theo tên video:

```text
outputs/id_tracking/<video_stem>/
```

- `<video_stem>_tracked_pigs_with_ids.mp4`: video xem trực tiếp với bbox và ID đã gán.
  Video này được render sau bước refine bbox, nên khớp với annotation xuất ra.
- `<video_stem>_annotations_cvat_video_1_1.xml`: annotation chính cho CVAT video tracking.
- `<video_stem>_annotations_coco.json`: annotation portable theo COCO 1.0, có `track_id`.
- `<video_stem>_annotations_coco_clean_train.json`: COCO sạch để train detector,
  chỉ giữ bbox detection thật, `Hidden=No`, score >= `review_conf`.
- `<video_stem>_annotations_cvat_shapes.json`: annotation phụ dạng CVAT shape JSON.
- `<video_stem>_labels.json`: schema label cho CVAT.
- `<video_stem>_bytetrack_pig_8.yaml`: cấu hình ByteTrack sinh tự động theo ngưỡng runtime.
- `<video_stem>_tracking_quality_report.json`: báo cáo chi tiết frame/ID cần review.
- `<video_stem>_tracking_quality_report.csv`: bản CSV để lọc nhanh bằng Excel.

Trong `annotations_coco.json`, mỗi frame có đúng 8 annotation:

- `category`: `Pig_1`, `Pig_2`, ..., `Pig_8`.
- `bbox`: COCO `xywh`.
- attribute `ID`: tương ứng `ID_1`, `ID_2`, ..., `ID_8`.
- attribute `Behavior`: mặc định `lying`, để sửa thủ công trên CVAT.
- attribute `Hidden`: `Yes` khi track mất detection đủ lâu hoặc chưa từng detect
  được; `No` cho detection thật và cả dự đoán ngắn hạn để tránh nhiễu.

Không dùng `annotations_coco.json` đầy đủ để train YOLO một cách trực tiếp, vì file
này có cả bbox predicted nhằm giữ track đủ 8 ID/frame. Khi muốn train detector lại,
dùng `<video_stem>_annotations_coco_clean_train.json`.

Việc tách label thành `Pig_1..Pig_8` giúp tua video trong CVAT dễ hơn vì tên
lớp và attribute ID luôn đồng bộ.

Để kiểm tra màu bbox và ID liên tục trên CVAT, ưu tiên import
`<video_stem>_annotations_cvat_video_1_1.xml` với format `CVAT for video 1.1`. File này dùng
`<track id="...">`, nên mỗi `Pig_N` là một track thật qua nhiều frame. COCO vẫn
được xuất để dùng đa nền tảng; trong COCO, `track_id` và `instance_id` được
ghi cả ở top-level annotation và trong `attributes`.

## CVAT Label Schema

Schema cũ chỉ có một lớp `Pig` và attribute `ID` gồm `ID_1..ID_8`. Schema đó
không khớp với annotation JSON mới, vì JSON mới ghi trực tiếp `label` là
`Pig_1`, `Pig_2`, ..., `Pig_8`.

Có hai cách dùng schema đúng:

- Sau khi chạy script, dùng
  `outputs/id_tracking/<video_stem>/<video_stem>_labels.json`.
- Nếu cần tạo task CVAT trước khi chạy script, dùng
  `data/annotations/cvat_pig_8_labels.json`.

Nguyên tắc schema mới:

```text
Pig_1 -> attribute ID only has ID_1
Pig_2 -> attribute ID only has ID_2
Pig_3 -> attribute ID only has ID_3
Pig_4 -> attribute ID only has ID_4
Pig_5 -> attribute ID only has ID_5
Pig_6 -> attribute ID only has ID_6
Pig_7 -> attribute ID only has ID_7
Pig_8 -> attribute ID only has ID_8
```

Vì vậy trên CVAT, không sửa lớp `Pig` cũ thành một lớp duy nhất nữa. Hãy tạo 8
labels riêng hoặc import file schema ở trên. Nếu dùng schema một lớp `Pig`, CVAT
sẽ không map được các shape có `label="Pig_1"` trong `annotations.json`.

## Ngưỡng Mặc Định

Confidence không còn bị chốt thành một ngưỡng duy nhất. Pipeline dùng 3 ngưỡng
riêng để tránh mất lợn khi bị che khuất:

- Detection Confidence: `0.25`
- Track High Confidence: `0.50`
- Review Confidence: `0.75`
- Adaptive Confidence Step: `0.05`
- Overlap Threshold: `0.80`
- Visual Opacity: `0.75`

Trong code:

- `det_conf=0.25` được truyền vào YOLOv8 để giữ cả bbox yếu cho tracking.
- `track_high_conf=0.50` được truyền vào ByteTrack high/new track threshold.
- `review_conf=0.75` chỉ dùng để đánh dấu bbox cần kiểm tra, không loại detection.
- Mỗi frame được lọc bằng ladder confidence từ `review_conf` xuống `det_conf`;
  code chọn ngưỡng cao nhất vẫn tìm được ít nhất 8 bbox sau ROI/NMS. Nếu xuống
  tới `det_conf` vẫn thiếu bbox thì mới dùng matching/optical flow để bù phần
  thiếu.
- Detection dưới `motion_gate_confidence=0.50` không được tin trực tiếp. Nó phải
  nằm gần bbox dự đoán của cùng ID, có IoU nhỏ nhưng hợp lý, hoặc tâm bbox không
  nhảy quá xa. Nếu bbox score thấp tự nhiên xuất hiện ở tọa độ xa track trước đó,
  pipeline xem đó là false candidate và không cho match.
- Frame khởi tạo chỉ dùng detection >= `initial_track_conf=0.50` để đặt ID thật.
  Nếu thiếu thì tạo placeholder hidden và chờ frame sau, tránh gán ID ban đầu vào
  bbox thấp confidence.
- Khi hai lợn overlap mạnh, pipeline dùng occlusion-aware data association. Cách
  này phát hiện detection/track trong cụm chồng lấn bằng intersection-over-min-area
  và tăng penalty cho assignment dễ gây đổi ID. Track gần như đứng yên sẽ được
  "khóa quán tính" trong vùng occlusion; nếu detection nhảy xa bất thường, track
  đó sẽ giữ bbox dự đoán ngắn hạn thay vì nhảy sang con đang đi qua.
- Trong frame occlusion/ambiguous, bbox vẫn có thể được cập nhật để giữ vị trí,
  nhưng identity template không được học thêm. Nói cách khác, HSV appearance bank
  và raw ByteTrack ID chỉ học từ frame ổn định; điều này hạn chế identity drift khi
  nhiều con chồng lên nhau rồi tách ra.
- Khi một track bị mất detection trong vùng occlusion, pipeline giữ bbox ở
  `reliable_box` cuối cùng của chính track đó thay vì chạy optical flow. Điều này
  tránh lỗi bbox hidden bị kéo theo con heo đang đi che phía trên.
- Chỉ khi track bị che hoàn toàn và không match được detection thì hidden motion
  model mới được dùng. Nếu track đang đứng yên, bbox hidden giữ vị trí cũ; nếu
  track đang di chuyển, bbox hidden được dự đoán tiếp theo vận tốc/gia tốc đã học
  từ các detection ổn định trước đó. Logic này không áp dụng cho các frame vẫn có
  detection.
- Nếu một detection nằm trong cụm nhiều track, pipeline so sánh appearance cost
  với các track cạnh tranh. Assignment có appearance kém hơn track khác rõ rệt sẽ
  bị cộng penalty để giảm khả năng đổi ID sau khi tách khỏi vùng overlap.
- `iou=0.80` được dùng làm overlap/NMS threshold của YOLOv8 và match threshold
  trong ByteTrack YAML.
- `visual_opacity=0.75` được dùng để blend bbox/text trong video preview.

Sau khi tracking xong, pipeline chạy thêm một pass refine bbox theo thời gian. Pass này
dùng bbox ổn định trước/sau của cùng `Pig_N` để sửa các bbox bị phình/thu đột ngột,
đặc biệt khi lợn quay người, chạy nhanh hoặc detector bắt nhầm một phần thân trong
1-2 frame. Đây là refine cho annotation rectangle trên CVAT, không thay đổi logic ID.

`Hidden=Yes` không còn bật ngay khi mất detection 1 frame. Mặc định chỉ bật khi
track mất detection ít nhất `5` frame liên tiếp, hoặc chưa từng detect được. Các
bbox có score dưới `review_conf` nhưng chưa đủ điều kiện hidden sẽ được hiện chữ
`review` trong video preview để bạn kiểm tra mà không làm nhiễu nhãn Hidden.

## Cách Chạy

Chạy full video với mặc định:

```powershell
uv run --extra tracking python notebooks\01_data_preparation\track_video_ids_for_annotation.py
```

Chạy rõ đường dẫn và ngưỡng:

```powershell
uv run --extra tracking python notebooks\01_data_preparation\track_video_ids_for_annotation.py `
  --video data\videos\Pigs291119_000263_30fps.mp4 `
  --weights models\detector\pig_detector_yolov8.pt `
  --mask data\annotations\mask.png `
  --det-conf 0.25 `
  --track-high-conf 0.50 `
  --review-conf 0.75 `
  --initial-track-conf 0.50 `
  --motion-gate-confidence 0.50 `
  --low-conf-max-center-jump 0.08 `
  --occlusion-track-iom-threshold 0.20 `
  --occlusion-detection-iom-threshold 0.30 `
  --occlusion-appearance-penalty 0.30 `
  --occlusion-hold-max-frames 30 `
  --hidden-stationary-speed 0.006 `
  --hidden-motion-history 8 `
  --hidden-min-motion-history 4 `
  --hidden-stationary-displacement 0.015 `
  --hidden-moving-displacement 0.035 `
  --hidden-motion-consistency 0.55 `
  --hidden-stationary-lock-frames 8 `
  --adaptive-conf-step 0.05 `
  --start-frame 0 `
  --refine-max-gap 15 `
  --refine-size-jump-threshold 0.45 `
  --iou 0.80 `
  --visual-opacity 0.75
```

Test nhanh vài trăm frame:

```powershell
uv run --extra tracking python notebooks\01_data_preparation\track_video_ids_for_annotation.py `
  --max-frames 300
```

Trong notebook/VS Code Interactive:

```python
cfg = TrackingConfig(max_frames=300)
summary = run_tracking(cfg)
display_tracked_video(summary.output_video)
```

## Luồng Xử Lý

1. Đọc video bằng OpenCV và ép output video ở `30 FPS`.
2. Đọc `mask.png`, resize theo kích thước video và dilate nhẹ để tránh cắt mất
   vùng mép chuồng.
3. Nếu `mask_input_frame=True`, frame đưa vào YOLO được apply mask trước. Điều
   này hạn chế nhầm lợn ở chuồng khác.
4. YOLOv8 chạy `model.track(..., persist=True)` với ByteTrack.
5. Detection được lọc lại bằng ROI:
   - `roi_mode=center`: tâm bbox phải nằm trong mask.
   - `roi_mode=cover`: tỷ lệ vùng bbox nằm trong mask phải vượt
     `roi_min_cover`.
6. Detection trùng nhau bị loại bằng IoU threshold.
7. Mỗi detection được trích HSV histogram để làm appearance feature.
8. Pipeline dùng Hungarian matching để gán detection mới vào 8 fixed IDs.
   Cost gồm:
   - IoU giữa bbox dự đoán và bbox mới.
   - Khoảng cách tâm bbox.
   - Khoảng cách appearance histogram.
   - Tỷ lệ thay đổi diện tích bbox.
   - Penalty nếu raw ByteTrack ID đang thuộc fixed ID khác.
9. Nếu một ID không match được detection, pipeline dự đoán bbox bằng
   Lucas-Kanade optical flow. Nếu optical flow thất bại, dùng vận tốc track gần
   nhất.
10. Mỗi frame xuất đủ 8 shape. Shape được dự đoán ngắn hạn vẫn được giữ để bảo
   toàn ID; chỉ khi mất detection đủ lâu mới gán `Hidden=Yes`.
11. Sau tracking, bbox được refine offline theo từng `Pig_N` bằng các anchor ổn định
   trước/sau. Bbox predicted, low-score, hidden hoặc nhảy size bất thường sẽ được
   blend về quỹ đạo/kích thước hợp lý hơn trước khi ghi XML/JSON/video preview.

## Kiểm Tra Nhanh Trên CVAT

1. Tạo task với video gốc.
2. Thêm labels từ `outputs/id_tracking/<video_stem>/<video_stem>_labels.json`.
3. Import annotation từ
   `outputs/id_tracking/<video_stem>/<video_stem>_annotations_cvat_video_1_1.xml` với
   format `CVAT for video 1.1`. Đây là lựa chọn tốt nhất để CVAT giữ track/màu
   ổn định qua nhiều frame.
4. Khi tua video:
   - `Pig_1` phải luôn đi với `ID_1`.
   - `Pig_2` phải luôn đi với `ID_2`.
   - Tương tự tới `Pig_8`/`ID_8`.
5. Ưu tiên kiểm tra các frame có `Hidden=Yes`; đó là nơi detector không tự tin
   hoặc bị che khuất, dễ gây ID switch nhất.
6. Mở thêm `<video_stem>_tracking_quality_report.csv` và lọc `needs_review=True`
   để tua thẳng tới frame có predicted bbox, hidden bbox hoặc score thấp.

Nếu cần dùng COCO, upload
`outputs/id_tracking/<video_stem>/<video_stem>_annotations_coco.json` với format
`COCO 1.0`. File này có `track_id`, nhưng với CVAT thì XML native vẫn dễ kiểm
tra tracking hơn.

Nếu cần train lại YOLO, ưu tiên dùng
`outputs/id_tracking/<video_stem>/<video_stem>_annotations_coco_clean_train.json`.
File này đã loại bbox predicted/hidden/score thấp để tránh học nhầm từ nội suy.

Nếu CVAT báo lỗi kiểu:

```text
CvatImportError: Failed to import dataset 'coco_instances'
```

thì hãy kiểm tra lại file đang upload. Với importer `COCO 1.0`, file phải là
`annotations_coco.json`, có các key `images`, `annotations`, `categories`.
Không dùng file `annotations.json` vì file đó có key `shapes` theo format CVAT
native.

## Các Tham Số Nên Chỉnh

- `--det-conf`: ngưỡng thấp cho YOLO candidate. Giảm nếu vẫn bị mất lợn.
- `--track-high-conf`: ngưỡng detection mạnh để ByteTrack mở track mới.
- `--review-conf`: ngưỡng chất lượng để đánh dấu bbox cần kiểm tra.
- `--adaptive-conf-step`: bước hạ ngưỡng khi frame chưa đủ 8 bbox.
- `--initial-track-conf`: ngưỡng tối thiểu để dùng detection khởi tạo một ID thật.
  Mặc định `0.50`.
- `--motion-gate-confidence`: detection thấp hơn ngưỡng này phải qua motion gate.
  Mặc định `0.50`.
- `--low-conf-max-center-jump`: khoảng cách tâm tối đa theo đường chéo frame cho
  bbox thấp confidence. Mặc định `0.08`.
- `--low-conf-max-box-jump-scale`: cổng khoảng cách theo kích thước bbox trước đó.
  Mặc định `1.75`.
- `--low-conf-min-iou`: nếu bbox thấp confidence vẫn còn IoU với bbox dự đoán lớn
  hơn ngưỡng này thì được coi là hợp lý. Mặc định `0.01`.
- `--no-low-conf-motion-gate`: tắt motion gate cho bbox thấp confidence nếu cần
  debug detector thô.
- `--occlusion-track-iom-threshold`: ngưỡng phát hiện hai track đang chồng lấn.
  Dùng intersection-over-min-area, mặc định `0.20`.
- `--occlusion-detection-iom-threshold`: ngưỡng phát hiện một detection đang nằm
  trong cụm nhiều track, mặc định `0.30`.
- `--occlusion-stationary-speed`: track có vận tốc thấp hơn ngưỡng này được coi là
  gần như đứng yên, mặc định `0.006`.
- `--occlusion-stationary-max-center-jump`: track đứng yên không được match sang
  detection có tâm nhảy xa hơn ngưỡng này trong vùng occlusion, mặc định `0.045`.
- `--occlusion-switch-penalty`: penalty cộng thêm cho assignment dễ gây đổi ID
  trong vùng overlap, mặc định `0.45`.
- `--occlusion-appearance-penalty`: penalty cộng thêm nếu detection giống
  appearance của track cạnh tranh hơn track hiện tại, mặc định `0.30`.
- `--occlusion-appearance-margin`: chênh lệch appearance cost tối thiểu để kích
  hoạt penalty, mặc định `0.08`.
- `--learn-identity-in-occlusion`: cho phép cập nhật appearance/raw ID ngay cả khi
  frame đang ambiguous. Mặc định không bật vì dễ gây identity drift.
- `--occlusion-hold-max-frames`: số frame tối đa giữ bbox tại reliable box khi
  track bị che khuất, mặc định `30`.
- `--hidden-stationary-speed`: vận tốc chuẩn hóa dưới ngưỡng này được xem là đứng
  yên khi track đang hidden, mặc định `0.006`.
- `--hidden-motion-history`, `--hidden-min-motion-history`: số frame reliable gần
  nhất để phân loại `moving/stationary/unknown`. Motion hidden chỉ được đẩy tiếp
  khi lịch sử đủ mạnh để kết luận `moving`.
- `--hidden-stationary-displacement`, `--hidden-moving-displacement`,
  `--hidden-motion-consistency`: ngưỡng tách bbox đứng yên bị jitter với bbox di
  chuyển có hướng. Nếu không chắc, trạng thái sẽ là `unknown` và bbox hidden được
  giữ ở reliable box thay vì trôi theo con khác.
- `--hidden-stationary-lock-frames`: số frame reliable đứng yên trước khi khóa
  track thành stationary trong vùng occlusion, mặc định `8`.
- `--hidden-velocity-alpha`, `--hidden-acceleration-alpha`: hệ số học vận tốc/gia
  tốc từ detection ổn định để dự đoán hidden bbox, mặc định `0.65` và `0.35`.
- `--hidden-max-motion-step-box-scale`: giới hạn bước dự đoán hidden bbox theo kích
  thước bbox, mặc định `1.50`.
- `--no-hidden-motion-model`: tắt dự đoán vận tốc/gia tốc cho hidden bbox.
- `--no-hold-occluded-box`: tắt logic giữ bbox occluded nếu muốn debug optical flow
  baseline.
- `--no-occlusion-aware-matching`: tắt logic occlusion nếu cần debug baseline.
- `--start-frame`: bắt đầu xử lý từ frame rõ hơn nếu đầu video bị chồng lấn hoặc
  thiếu lợn. Annotation vẫn giữ frame index gốc để import vào CVAT task của video
  gốc.
- `--hidden-missed-frames`: số frame mất detection liên tiếp trước khi gán
  `Hidden=Yes`.
- `--hidden-score-threshold`: nếu bbox dự đoán có score thấp hơn ngưỡng này thì
  cũng có thể bị đánh hidden.
- `--max-box-scale-change`: giới hạn width/height bbox được phình hoặc thu trong
  một frame. Mặc định `0.25`.
- `--max-box-scale-change-after-gap`: khi track mất detection vài frame rồi bắt
  lại, cho phép bbox đổi kích thước mạnh hơn. Mặc định `0.75`.
- `--high-conf-smooth-alpha`, `--mid-conf-smooth-alpha`,
  `--low-conf-smooth-alpha`: mức tin detection theo score. Score cao bám bbox
  detector nhiều hơn, score thấp làm mượt mạnh hơn.
- `--no-smooth-boxes`: tắt smoothing nếu cần xem bbox detector gốc.
- `--no-refine-boxes`: tắt pass refine offline nếu muốn xuất bbox tracking thô.
- `--refine-max-gap`: số frame tối đa để tìm anchor trước/sau khi refine bbox.
  Mặc định `15`, tức khoảng 0.5 giây với video 30 FPS.
- `--refine-size-jump-threshold`: ngưỡng phát hiện width/height nhảy bất thường.
  Mặc định `0.45`, nghĩa là bbox thay đổi kích thước hơn khoảng 45% so với nội suy
  lân cận sẽ bị kéo lại một phần.
- `--iou`: tăng nếu bbox chồng lấn nhưng vẫn muốn giữ nhiều bbox hơn sau NMS.
- `--roi-mode cover`: dùng khi tâm bbox hay nằm ngoài mask nhưng phần lớn thân
  lợn vẫn trong chuồng.
- `--roi-min-cover`: tăng nếu vẫn bị lẫn chuồng khác.
- `--max-missing-frames`: tăng nếu lợn có thể bị che khuất lâu.
- `--visual-opacity`: chỉ ảnh hưởng video preview, không ảnh hưởng JSON.

## Kỳ Vọng Chất Lượng

Pipeline này không thay thế bước review cuối trên CVAT. Nó được thiết kế để
bootstrap annotation nhanh:

- Luôn đủ 8 bbox/frame để không mất cá thể.
- Dùng mask để hạn chế nhầm chuồng.
- Dùng appearance + motion + raw ByteTrack ID để giảm ID switch.
- Làm mượt bbox có điều kiện để giảm hiện tượng bbox phình/thu đột ngột khi lợn
  quay người hoặc chạy nhanh.
- Refine bbox hai-pass để giảm số bbox cần chỉnh tay trên CVAT, nhất là các lỗi
  chỉ xuất hiện trong vài frame.
- Dùng `Hidden=Yes` để đánh dấu nơi cần kiểm tra thủ công.
- Xuất quality report để biết chính xác frame nào cần review thay vì phải xem lại
  toàn bộ video.
- Xuất thêm COCO sạch để train detector, tách khỏi COCO đầy đủ dùng cho tracking.
