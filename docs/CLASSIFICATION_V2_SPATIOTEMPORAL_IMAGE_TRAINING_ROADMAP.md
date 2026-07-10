# Classification V2: Lộ trình nâng cấp framework multimodal spatio-temporal bbox + ROI + social context

> Phiên bản 2.2, định hướng nghiên cứu và publication-readiness. Đây là execution roadmap. Thiết kế khoa học, giả thuyết, thống kê và quy tắc claim được khóa tại [CLASSIFICATION_V2_Q1_Q2_RESEARCH_PROTOCOL.md](CLASSIFICATION_V2_Q1_Q2_RESEARCH_PROTOCOL.md). Hai tài liệu phải được version cùng nhau.

Trạng thái hiện tại: `ENGINEERING-READY FOR CONTROLLED SMOKE`, `NOT PUBLICATION-READY`. Nhãn Q1/Q2 trong tài liệu là mức độ nghiêm ngặt hướng tới, không phải cam kết tạp chí hoặc bảo đảm chấp nhận.

## 0. Quyết định hướng nghiên cứu đã khóa

Hướng chính của dự án là một framework multimodal cho nhận diện hành vi lợn, đặt trong pipeline dữ liệu/review có audit đầy đủ:

- Input chính: chuỗi ảnh bbox actor, local/full-frame context khi cần, đặc trưng spatio-temporal, ROI all-class và social/partner context.
- Đơn vị review: `review_unit_id`; mọi sửa nhãn qua GUI/apply decisions phải audit được và không xóa row âm thầm.
- Đơn vị dự đoán chính cho paper: temporal unit/review unit, không phải window chồng lặp.
- Split/validation: leakage-safe theo recording session/video group; không mặc định `pig_id` là cùng cá thể xuyên video.
- Claim chính: Q2 mạnh, "improved behavior recognition under session/video-safe validation".
- Không claim Q1-style cross-farm/cross-camera/cross-cohort generalization nếu chưa có external cohort/farm/camera khóa độc lập.
- Interaction dùng full-frame/partner context; `social-nose` actor-only, `fight` chỉ áp cho pig trực tiếp tham gia, không propagate cho bystander.
- Mọi feature đưa vào X phải qua whitelist; label/review/source/path/ID/policy text tuyệt đối không vào X.

## 1. Mục tiêu và phạm vi

Tài liệu này định hướng nâng cấp `classification_v2` từ bộ dữ liệu đã review đến một pipeline training có thể tái lập, chống leakage và có tiêu chuẩn đánh giá khoa học.

Mục tiêu cuối:

- Dự đoán 10 hành vi: `drink`, `eat`, `fight`, `social-nose`, `explore`, `lying`, `stand`, `move`, `sitting`, `playwithtoy`.
- Kết hợp ba nhóm tín hiệu độc lập: ảnh chuỗi bbox, đặc trưng không-thời gian theo frame và đặc trưng tổng hợp theo window.
- Giữ đúng chính sách CVAT 6 frame và legacy 16 frame.
- Hỗ trợ cả đánh giá offline và triển khai gần realtime, nhưng không trộn hai hợp đồng thời gian trong cùng một model.
- Mỗi kết luận nâng cấp phải đến từ ablation trên cùng split, nhiều seed và các lát cắt lỗi đã định trước.

Ngoài phạm vi của roadmap này:

- Không sửa dữ liệu gốc trong `data/`.
- Không chạy full training trước khi các data gate và loader gate hoàn tất.
- Không dùng cột label, review, ID, path hoặc policy làm model input.
- Không thay thế hoặc ghi đè model runtime cũ trước khi model mới vượt promotion gate.

## 2. Trạng thái xuất phát đã có bằng chứng

| Hạng mục | Trạng thái hiện tại |
|---|---:|
| Reviewed frame rows | 245,664 |
| Sequence windows | 160,740 |
| Main-train valid windows | 152,704 |
| Temporal intervals | 33,354 |
| Stable windows | 155,066 |
| Transition windows | 5,292 |
| Incomplete windows | 382 |
| CVAT windows | 115,200 |
| Legacy windows | 45,540 |
| Tabular features được whitelist | 39 |
| Split groups | 680 |
| Configured `source|dataset|video` leakage groups | 0 |
| Canonical recording-date groups | 13 |
| Recording-date groups xuất hiện ở nhiều split hiện tại | 13/13 |
| CVAT recordings / recording dates | 12 / 3 |
| Legacy clip keys | 668 |
| Source-label association, Cramér's V | 0.356 |
| Spatial max sequence length | 16 |
| Image loader smoke | 24/24 windows, 236/236 frames |

Các artifact nền tảng:

- `outputs/classification_v2/review_policy/reviewed_frame_features.csv`
- `outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv`
- `outputs/classification_v2/sequence_features_reviewed/sequence_window_features.csv`
- `outputs/classification_v2/train_ready_windows/X_window_features.csv`
- `outputs/classification_v2/train_ready_windows/X_spatial_sequences.npz`
- `outputs/classification_v2/train_ready_windows/y_behavior.csv`
- `outputs/classification_v2/train_ready_windows/train_mask.csv`
- `outputs/classification_v2/train_ready_windows/sample_weight.csv`
- `outputs/classification_v2/train_ready_windows/split_manifest.csv`
- `outputs/classification_v2/train_ready_windows/class_weight_policy.json`
- `outputs/classification_v2/train_ready_windows/model_input_contract.json`

Baseline tabular smoke hiện chỉ là kiểm tra pipeline, không phải baseline chất lượng cuối:

- Test accuracy: `0.4733`.
- Test macro-F1: `0.2679`.
- `playwithtoy`, `social-nose`, `stand` có F1 bằng 0 trên mẫu smoke.
- Confusion lớn gồm `lying` với `sitting`; các nhóm ROI, interaction và motion cũng còn yếu.

Kết luận xuất phát: data contract đã đủ để bắt đầu model smoke có kiểm soát, nhưng split hiện tại chỉ chống exact-video leakage. Vì mọi recording-date group đều cắt qua nhiều split, metric từ split này không được dùng làm kết quả paper. Chưa đủ bằng chứng để quảng bá một full model.

## 3. Các vấn đề phải xử lý trước model

### 3.1. Độ phủ human review còn thấp

Output reviewed đúng về cấu trúc và không mất row, nhưng audit hiện chỉ có 3 quyết định GUI được nạp. Vì vậy từ "reviewed" hiện mô tả trạng thái pipeline, chưa chứng minh toàn bộ label đã được con người xác nhận.

Cần thêm các chỉ số:

- Tỷ lệ review theo `behavior`, `source_type`, video và quality tier.
- Tỷ lệ review của rare/confusion classes.
- Agreement giữa hai reviewer trên một tập stratified nhỏ.
- Số unit corrected, excluded, uncertain và pending.
- Label strength và nguồn label phải chỉ tham gia weight/mask, không đi vào X.

Gate đề nghị: trước full training, review stratified tối thiểu toàn bộ `playwithtoy` khả dụng hoặc một tỷ lệ đủ lớn có báo cáo; đồng thời review tập confusion-focused cho `social-nose`, `fight`, `stand`, `move`, `lying`, `sitting`, `eat`, `drink`.

### 3.2. Window chồng lặp tạo tương quan mạnh

Một temporal interval có thể sinh nhiều window length 6/8/12/16, và window lân cận có thể dùng phần lớn cùng frame. Split theo video đã chặn leakage giữa tập, nhưng nếu coi mọi window là mẫu độc lập sẽ:

- Làm class count và confidence interval lạc quan giả.
- Khiến event dài hoặc source có nhiều window chi phối loss.
- Đánh giá cùng một hành vi nhiều lần như các bằng chứng độc lập.

Cần bổ sung:

- `event_id` hoặc `review_unit_id` cho mọi window.
- `overlap_cluster_id` cho các window dùng chung phần lớn frame.
- `effective_sample_weight = review_weight * quality_weight * inverse_windows_per_event`.
- Báo cáo cả window-level, temporal-unit-level và video-level.
- Bootstrap confidence interval theo video/session, không bootstrap theo window.

### 3.3. Padding và frame thiếu đang dùng chung một mask

Spatial tensor có shape tối đa 16 frame. `observed_ratio=0.6253` chủ yếu do window 6/8/12 được pad tới 16, không phải 37.5% dữ liệu thật bị thiếu. Có 2,112 slot thiếu trong phạm vi frame mà window thực sự yêu cầu.

Phiên bản v2 cần tách:

- `length_mask`: slot thuộc độ dài hợp lệ của window.
- `observed_mask`: frame thật đã đọc và có feature.
- `quality_mask`: bbox/ROI/social/motion hợp lệ.
- `frame_delta`: khoảng cách frame thực tế với timestep trước.

Model attention và pooling phải dùng `length_mask AND observed_mask`. Zero padding không được hiểu là một quan sát đứng yên.

### 3.4. Hợp đồng temporal training và runtime chưa đồng nhất

Dataset mới dùng chuỗi liên tục 6/8/12/16 frame. Runtime PyTorch cũ dùng 6 ảnh tại offsets `[-3,-2,-1,0,1,2] * stride`, mặc định stride 3, và lặp frame cuối khi thiếu. Đây là train-serving mismatch.

Cần chốt hai mode tách biệt:

1. `offline_delayed`: dùng toàn window liên tục, được phép dùng frame tương lai nằm trong chính window.
2. `causal_realtime`: chỉ dùng frame hiện tại và quá khứ, có latency contract rõ ràng.

Model v2 mặc định nên bắt đầu với `offline_delayed` trên chuỗi liên tục vì khớp annotation CVAT/legacy. Causal model chỉ được tạo như một experiment riêng. Checkpoint phải lưu `temporal_mode`, `sequence_lengths`, `sampling_stride`, `fps_policy` và `padding_policy`.

### 3.5. Trainer cũ không dùng trực tiếp cho classification_v2

`src/pig_behavior/data/tf_dataset.py` hiện:

- Drop tất cả hidden row.
- Tự group split lại.
- Đọc single image thay vì sequence.
- Không dùng `train_mask`, `sample_weight`, split manifest hay spatial NPZ.

`src/pig_behavior/training/classification_trainer.py` checkpoint theo `val_accuracy`, chưa có macro-F1, calibration, slice metrics hoặc model contract v2.

Model PyTorch runtime cũ cố định 6 frame, repeat padding và không truyền padding mask vào Transformer.

Quyết định kiến trúc: xây trainer PyTorch riêng cho `classification_v2`, giữ trainer Keras và runtime checkpoint cũ ở chế độ backward-compatible. Không sửa trainer cũ thành một lớp hỗn hợp khó audit.

### 3.6. Pseudoreplication và session leakage

Các con số window/frame lớn không phải số quan sát sinh học độc lập. Audit hiện tại cho thấy:

- 33,354 temporal intervals nhưng chỉ 13 canonical recording dates.
- CVAT có 12 recordings thuộc 3 ngày.
- Legacy có 668 clip keys, phần lớn là các clip/burst nằm trong cùng ngày ghi hình.
- Cả 13/13 recording dates đang xuất hiện ở nhiều split.
- Metadata chỉ cho thấy 8 pig-ID token; chưa đủ để chứng minh đó là bao nhiêu cá thể sinh học độc lập qua các ngày/source.

Do đó phải phân biệt năm cấp:

1. Biological unit: cá thể/cohort thật.
2. Recording cluster: farm, camera, date, session.
3. Annotation unit: temporal interval/review unit.
4. Training unit: sequence window.
5. Statistical unit: recording cluster, hoặc biological unit nếu metadata cho phép.

Split dùng cho paper phải group theo canonical recording session/date trên cả hai source. Test metric phải collapse nhiều window length về một prediction định trước cho mỗi temporal unit. Không được dùng window count làm sample size thống kê.

### 3.7. Source-label shortcut

Phân bố label khác đáng kể giữa CVAT và legacy; audit interval-level cho `Cramér's V=0.356`. Ví dụ `fight` có 2,196 CVAT intervals nhưng chỉ 17 legacy intervals, trong khi `lying` có 1,661 CVAT và 1,552 legacy intervals. Model có thể học style/source thay vì hành vi.

Yêu cầu:

- Không đưa `source_type` vào X.
- Báo cáo source-specific và source-holdout metrics.
- Thêm background-only, actor-masked, temporal-shuffle và source-prediction controls.
- Group split theo recording date trên cả source để cùng ngày không rơi vào train/test.
- Chỉ dùng domain adaptation sau khi control chứng minh shortcut tồn tại.

## 4. Kiến trúc dữ liệu đích

```text
reviewed_frame_features + sequence_window_manifest
                  |
                  +--> frame/event/split audit
                  |
                  +--> unique-frame image index/cache
                  |       actor crop
                  |       local-context crop
                  |       optional full-frame context
                  |
                  +--> spatial sequence v2
                  |       geometry + motion + ROI + social + quality
                  |
                  +--> window tabular whitelist
                  |
                  +--> labels + masks + weights
                                  |
                    versioned Dataset/DataLoader
                                  |
                image encoder + temporal/spatial encoder
                                  |
                          gated multimodal fusion
                                  |
                    behavior + auxiliary task heads
                                  |
                  evaluation, calibration, error review
```

Mọi nhánh phải join bằng row index nội bộ đã validate với `window_id`. `window_id` dùng để join và audit, không đi vào tensor model.

## 5. Nâng cấp dữ liệu không-thời gian

### 5.1. Geometry và camera normalization

Giữ và kiểm chứng:

- `cx_n`, `cy_n`, `bw_n`, `bh_n`, `area_n`, `aspect_ratio`.
- Bbox raw, bbox clamped và cờ out-of-frame phải tách riêng.
- Không dùng tọa độ tuyệt đối chưa normalize trực tiếp qua nhiều camera.

Nâng cấp ưu tiên:

- Relative bbox change: scale, width, height, area và aspect derivatives.
- Distance to frame border và mức crop truncation.
- Perspective-normalized size hoặc homography mặt sàn theo camera nếu calibration đủ tin cậy.
- Position relative to fixed ROI polygons bằng signed distance, không chỉ center/overlap.
- Bbox reliability score từ valid, hidden, truncation, interpolation và frame gap.

Nếu dùng homography, phải lưu version calibration theo video/camera và có fallback về normalized image coordinates. Không impute world coordinate giả khi calibration thiếu.

### 5.2. Motion dynamics

Feature v2 nên có:

- Velocity vector theo frame và theo giây.
- Acceleration và jerk có clipping theo percentile train-only.
- Path length, net displacement và tortuosity.
- Signed turning angle, heading persistence và direction entropy.
- Stationary ratio, burst ratio, stop-go transition count.
- Bbox deformation rate để hỗ trợ posture/interaction.
- Missing-frame gap và effective FPS tại từng bước.

Smoothing chỉ được dùng trong window đang dự đoán. Không lấy frame ngoài window. Với causal mode, smoothing chỉ dùng quá khứ. Mọi feature có phiên bản raw và quality flag; không che mất motion thật bằng smoothing quá mạnh.

### 5.3. ROI relation không leakage

Model nhận đồng thời quan hệ với cả ba lớp ROI:

- Feeder.
- Drinker.
- Toy.

Mỗi lớp ROI có:

- Signed/min distance tới polygon.
- Actor center-inside.
- Bbox overlap ratio và IoU.
- Near/contact flags.
- Approach speed và approach angle.
- Contact duration, dwell ratio, entry/exit count trong window.

Không dùng `behavior_label` để chọn target ROI cho model input. `target_roi_*` và `roi_target_*` tiếp tục là audit/policy-only. Missing ROI không làm drop sample; dùng `roi_available_mask` và quality weight.

### 5.4. Social và interaction context

Nhánh social v2 không chỉ dùng nearest pig:

- Top-k neighbor, ưu tiên `k=3` hoặc toàn bộ 8 pig khi có.
- Relative position vector, normalized distance, overlap/contact.
- Relative velocity, approach/separation speed.
- Heading alignment và pair persistence theo thời gian.
- Group density, local crowding, số contact.
- Partner validity/visibility mask.

Đối với `social-nose`, actor-only vẫn là label policy. Đối với `fight`, không propagate sang bystander. Graph edge được tạo theo geometry và track context, tuyệt đối không tạo theo ground-truth behavior.

### 5.5. Transition target

Giữ các window transition/incomplete trong audit, không xóa. Đề nghị:

- Main behavior loss chỉ áp dụng khi `window_valid_for_main_train=true`.
- Thêm auxiliary `transition_head` với target stable/transition nếu đủ dữ liệu.
- Experiment soft target dựa trên label coverage chỉ sau khi hard-label baseline ổn định.
- Không đưa transition window vào main loss với một hard label tùy tiện.

## 6. Nâng cấp chuỗi ảnh bbox

### 6.1. Ba view độc lập với label

Mỗi frame nên có tối đa ba view, tất cả được tạo bằng rule không phụ thuộc y:

1. `actor_view`: bbox actor mở rộng 15-25%, giữ chi tiết posture.
2. `local_context_view`: vùng khoảng 1.8-2.5 lần bbox, giúp thấy partner và ROI gần.
3. `scene_view`: full frame low-resolution với actor mask/bbox channel, dùng sau khi hai view đầu chứng minh có lợi.

Interaction pair view có thể dùng union bbox giữa actor và nearest valid neighbor, nhưng nearest neighbor phải chọn từ geometry, không chọn theo label. ROI view phải chứa mọi ROI class hoặc raster channel riêng, không crop riêng theo target behavior.

### 6.2. Resize và normalization

- Letterbox giữ aspect ratio, không kéo méo pig thành hình vuông.
- Lưu padding mask hoặc bbox valid region nếu cần.
- Chuyển BGR sang RGB đúng một lần.
- Dùng normalization đúng với pretrained backbone.
- Smoke ở 128/160 px; model chuẩn bắt đầu 192 hoặc 224 px theo memory benchmark.

### 6.3. Temporal-consistent augmentation

Áp dụng cùng một geometric transform cho toàn sequence và cập nhật bbox/mask tương ứng:

- Scale/crop nhẹ, translation nhỏ.
- Brightness, contrast, gamma và color jitter trong biên camera hợp lý.
- Motion blur/compression noise nhẹ để mô phỏng video.
- Random bbox jitter theo distribution lỗi detector đã đo.
- Random frame dropout với `observed_mask` cập nhật đúng.

Không bật horizontal flip mặc định nếu chưa chứng minh ROI/camera semantics được transform đúng. Không dùng augmentation làm thay đổi quan hệ actor-ROI hoặc actor-partner mà spatial branch vẫn giữ feature cũ.

### 6.4. Cache không nhân bản window

Không lưu lại cùng frame cho mọi window chồng lặp. Tạo cache theo `frame_uid`:

- Unique actor/local crops hoặc offset/index tới video.
- Checksum, shape, color space, bbox policy, source resolver version.
- Sequence manifest chỉ tham chiếu frame index trong cache.
- Cache nằm trong `outputs/classification_v2/`, có thể rebuild, không ghi vào `data/`.

So sánh hai backend trước khi chốt: JPEG/WebP shard để tiết kiệm disk và fixed-shape uint8 shard để tăng throughput. Chọn bằng benchmark decode throughput, disk size và CPU/GPU utilization trên máy đích.

## 7. Dataset và DataLoader v2

Một batch chuẩn:

```text
inputs:
  actor_images      [B, T, 3, H, W]
  context_images    [B, T, 3, H, W]        optional by experiment
  spatial_groups    dict[B, T, D_group]
  tabular_window    [B, 39]
  length_mask       [B, T]
  observed_mask     [B, T]
  quality_mask      [B, T, D_quality]
targets:
  behavior          [B]
  auxiliary_targets dict[B]                 optional
weights:
  sample_weight     [B]
audit-only:
  window_id, review_unit_id, source, video, frames
```

Yêu cầu loader:

- Dùng trực tiếp `split_manifest.csv`, không tự random split.
- Validate row count, order và `window_id` giữa mọi artifact.
- Có deterministic seed cho sampling và augmentation.
- Group-aware sampler theo event/video; tránh để bốn window length của cùng event chi phối batch.
- Batch có diversity theo class và source nhưng không nhân weight quá mức.
- CPU fallback, CUDA path, pinned memory/prefetch và worker count cấu hình được.
- Fail rõ khi image/frame thiếu; không tự thay bằng zero mà không cập nhật mask/audit.

## 8. Kiến trúc model theo từng mức

### M0. Full tabular baseline

- Logistic/SGD và shallow MLP trên đúng 39 feature.
- Train trên toàn train-valid split, không chỉ smoke sample.
- Chuẩn hóa fit trên train và lưu scaler trong artifact.
- Mục đích: baseline rẻ, kiểm tra label/split/metric, không phải model cuối.

### M1. Spatial-temporal baseline

- Encoder riêng cho geometry, motion, ROI, social và quality.
- Project mỗi group về 16-32 chiều, concatenate theo timestep.
- Masked TCN 2-4 residual blocks là candidate đầu vì nhỏ và ổn định.
- So sánh với GRU và Transformer nhỏ, giữ số parameter gần nhau.
- Masked mean/attention pooling, bắt buộc dùng length/observed mask.
- Late fusion với 39 tabular feature.

M1 là bước model tiếp theo nên triển khai trước image branch. Nó kiểm tra feature không-thời gian mới có thực sự mang signal hay không với chi phí thấp.

### M2. Image-sequence baseline

- Backbone ưu tiên MobileNetV3 hoặc EfficientNet-B0 pretrained; ResNet18 là control.
- Shared frame encoder cho mọi timestep.
- Freeze backbone ở phase đầu, sau đó unfreeze block cuối với learning rate nhỏ hơn 10-20 lần.
- Temporal encoder bắt đầu bằng masked TCN/GRU; Transformer chỉ giữ nếu ablation có lợi.
- Actor view trước, sau đó thêm local-context view.

### M3. Multimodal fusion model

Kiến trúc đề nghị:

```text
actor/context CNN embeddings ----+
                                  +--> temporal encoder --> temporal vector --+
spatial group projections -------+                                         |
                                                                            +--> gated fusion --> behavior head
window tabular MLP ---------------------------------------------------------+
```

- Fusion dùng gated late fusion hoặc FiLM nhỏ, không concatenate thô tensor quá lớn.
- Mỗi branch có LayerNorm và dropout riêng.
- Có branch dropout trong training để model không phụ thuộc tuyệt đối vào một modality.
- Log gate activation theo class/source để phát hiện shortcut.

Kích thước candidate đầu nên dưới khoảng 15-25 triệu parameter để phù hợp laptop/GPU phổ thông. Tăng capacity chỉ khi learning curve chỉ ra underfitting.

### M4. Multi-task heads

Behavior head vẫn là primary. Auxiliary heads:

- Posture: `lying`, `sitting`, `standing_or_other`.
- Motion/context: `move`, `explore`, `stand`, `other`.
- ROI intent: `eat`, `drink`, `playwithtoy`, `none`.
- Interaction: `fight`, `social-nose`, `none`.
- Transition: stable/transition.

Auxiliary targets được suy ra bằng mapping policy rõ ràng, chỉ dùng ở y/loss. Loss weight phải ablate và không để head dễ lấn át behavior head.

### M5. Graph social model

- Node là pig tại cùng frame.
- Edge theo top-k distance/contact/relative motion.
- Temporal graph encoder hoặc graph per-frame rồi temporal pooling.
- Bắt buộc có partner visibility mask và actor role.
- Đánh giá chủ yếu trên `fight`, `social-nose` và bystander false positive.

Chỉ triển khai M5 khi M3 cho thấy interaction vẫn là lỗi chính và full-frame/neighbor data đã đủ tin cậy.

### M6. Nâng cấp dài hạn

- Pose/keypoint branch cho posture và head-to-head interaction.
- Optical flow hoặc frame-difference branch cho motion tinh tế.
- Self-supervised pretraining trên video lợn chưa label.
- Domain adaptation giữa legacy crop và CVAT video.
- Teacher-student distillation sang model nhỏ cho realtime.
- Active learning từ entropy, disagreement và rare-class candidates.
- Uncertainty/calibration và abstention cho window không đủ chất lượng.

Mỗi hướng là một experiment độc lập, không gộp đồng thời trước khi biết nguồn gain.

## 9. Training protocol khoa học

### 9.1. Framework và cấu hình

Chọn PyTorch cho model v2 để đồng nhất runtime `.pt` hiện tại và hỗ trợ mask/multimodal sequence rõ ràng. Mọi run phải lưu:

- Code commit SHA và dirty-state flag.
- Dataset snapshot hash và artifact checksums.
- Config đầy đủ, seed, package versions, hardware.
- Split manifest hash.
- Feature names và normalization stats.
- Label order, temporal contract, crop contract.
- Best checkpoint, last checkpoint, predictions và metrics.

### 9.2. Loss và class imbalance

Baseline dùng weighted cross-entropy với sample weight. Sau đó ablate riêng:

- Sqrt class weights hiện có.
- Effective-number class weights.
- Focal loss.
- Class-aware/group-aware sampler.

Không kết hợp heavy class weight, focal loss và aggressive oversampling ngay từ đầu. Tổng effective weight cần clip và log theo class/source. `playwithtoy` chỉ có 472 train-valid windows và nhiều window có thể cùng event, nên số event độc lập quan trọng hơn raw window count.

### 9.3. Optimization

- AdamW, gradient clipping, mixed precision khi GPU hỗ trợ.
- Warmup ngắn và cosine decay hoặc ReduceLROnPlateau theo val macro-F1.
- Early stopping theo macro-F1 hoặc composite score, không theo accuracy đơn thuần.
- Phase 1 freeze image backbone; phase 2 unfreeze block cuối.
- Tối thiểu 3 seed cho candidate promotion; smoke có thể 1 seed.
- Test set chỉ chạy khi chọn candidate cuối, không dùng test để tune.

### 9.4. Split và validation

Split hiện tại chống exact-video leakage nhưng không chống recording-session leakage, vì 13/13 canonical dates đang cắt qua nhiều split. Split này chỉ dùng cho smoke kỹ thuật. Trước experiment có giá trị paper phải:

- Xây canonical `recording_group_id` từ farm/camera/date/session và khóa bằng manifest hash.
- Giữ mọi source/clip cùng recording group trong một fold.
- Xác minh số cá thể/cohort thật, không suy ra từ `ID_1..ID_8`.
- Legacy và CVAT có cùng nguồn scene nhưng alias khác không.
- Rare class có đủ event độc lập ở val/test không.

Thiết kế paper dùng nested group validation:

- Outer folds group theo recording date/session để tạo out-of-fold prediction.
- Inner folds chỉ dùng outer-train groups để chọn model/hyperparameter.
- Final external set, nếu có, phải là cohort/farm/camera độc lập và không tham gia bất kỳ quyết định nào.

Giữ ba bài toán đánh giá tách biệt:

1. `engineering_smoke`: split hiện tại, chỉ kiểm tra pipeline.
2. `session_generalization`: nested grouped evaluation theo recording date/session.
3. `external_generalization`: cohort/farm/camera độc lập, nếu thu thập được.

Nếu không có external set, paper phải giới hạn claim ở unseen-session trong cùng cohort/domain; không được claim generalization rộng sang farm hoặc đàn mới.

### 9.5. Metrics

Primary:

- Macro-F1 theo temporal unit và theo video.

Secondary:

- Per-class precision/recall/F1.
- Balanced accuracy, weighted F1.
- Confusion matrix và focus-pair confusion rate.
- Event-level majority prediction và temporal stability.
- Top-2 accuracy cho ambiguous classes.
- NLL, Brier score, ECE và reliability diagram.
- Coverage-risk curve khi có abstention.
- Latency, throughput, peak RAM/VRAM và model size.

Lát cắt bắt buộc:

- Source: legacy/CVAT.
- Behavior và review group.
- Window length 6/8/12/16.
- Hidden ratio và bbox quality.
- ROI available/missing.
- Partner context available/missing.
- Camera/video/session.
- Reviewed vs not-yet-manually-reviewed subset.

Confidence interval dùng bootstrap theo video/session. Không tính CI bằng cách coi window chồng lặp là độc lập.

## 10. Ma trận experiment và ablation

| ID | Input/Model | Câu hỏi khoa học | Điều kiện giữ |
|---|---|---|---|
| E0 | Full tabular logistic/MLP | Baseline đầy đủ là bao nhiêu? | Contract/split/mask pass |
| E1 | Spatial TCN | Motion/ROI/social sequence có gain không? | Cùng split, parameter nhỏ |
| E2 | Spatial GRU/Transformer | Temporal encoder nào thực sự tốt hơn? | Parameter gần E1 |
| E3 | Actor image sequence | Chi tiết posture/appearance thêm bao nhiêu gain? | Pretrained backbone |
| E4 | Actor + local context | Context cải thiện ROI/interaction không? | View tạo không phụ thuộc label |
| E5 | Image + spatial + tabular | Fusion có vượt từng branch riêng? | Gated fusion, branch ablation |
| E6 | Multi-task heads | Hierarchy giảm confusion không? | Behavior head vẫn primary |
| E7 | Graph social branch | Fight/social-nose có cải thiện không? | Bystander FP không tăng |
| E8 | Tracking-noise augmentation | Model có bền với bbox runtime không? | So sánh clean/noisy eval |
| E9 | Calibration/abstention | Có giảm lỗi high-confidence không? | Fit trên val, khóa test |

Mỗi experiment phải tạo predictions cùng schema để chạy chung confusion evaluator. Không chấp nhận chỉ báo cáo một accuracy tổng.

## 11. Promotion gates

### Gate A: data

- Row/order/hash giữa X, y, mask, weight, split khớp tuyệt đối.
- Duplicate `window_id=0`; configured group leakage=0; canonical session/date leakage=0 cho publication split.
- Forbidden feature selected=0.
- Padding, missing và quality mask có test riêng.
- Image load pass trên sample stratified theo source/class/window length/quality.
- Review coverage audit đã có.
- Biological subject, cohort, farm, camera, date và session metadata đã được document hoặc ghi rõ là unknown.
- Một temporal unit chỉ đóng góp một primary test prediction sau aggregation định trước.

### Gate B: smoke model

- Overfit được một batch nhỏ để chứng minh gradient/data path đúng.
- Loss giảm, checkpoint reload cho prediction giống trong tolerance.
- Masked padding không làm đổi prediction khi chỉ thay giá trị ở padded slots.
- CPU và GPU inference shape giống nhau.
- Không class nào biến mất do label mapping sai.

### Gate C: candidate model

- Vượt E0 full baseline về macro-F1 với confidence interval hoặc nhiều seed.
- Không đánh đổi gain tổng bằng regression lớn trên rare/critical classes.
- `playwithtoy`, `social-nose`, `stand` không còn F1 bằng 0.
- Source gap và hidden/quality slices được báo cáo.
- Confusion focus có cải thiện cụ thể, không chỉ accuracy tổng.

### Gate D: final model

- 3 seed, metric mean/std và video-cluster bootstrap CI.
- Test set chỉ đánh giá một lần sau khi khóa config.
- Calibration và threshold/abstention policy được fit trên validation.
- Model artifact chứa preprocessing và temporal/crop contract.
- Inference parity test trên cùng sequence giữa trainer và runtime.
- Latency/memory đạt ngân sách đã định trước.
- Có rollback về checkpoint cũ.

Ngưỡng số cụ thể cho macro-F1/per-class F1 chỉ chốt sau E0 full baseline. Không lấy smoke macro-F1 `0.2679` làm promotion baseline chính thức.

## 12. Lộ trình triển khai theo phase

| Phase | Công việc | Deliverable | Exit gate |
|---|---|---|---|
| P0 | Freeze research/data/model contract | RQ/Hypothesis registry, literature matrix, session-safe split, dataset snapshot, review coverage | Gate A + protocol freeze |
| P1 | Spatial sequence v2 | Mask tách biệt, feature groups, event weights | Unit tests + audit pass |
| P2 | Image index/cache v2 | Actor/local views, checksum, benchmark cache | Stratified loader pass |
| P3 | Trainer foundation | Config, Dataset, sampler, metrics, artifact writer | Overfit-one-batch pass |
| P4 | E0 full + E1 spatial | Baseline report và spatial ablation | Candidate có signal rõ |
| P5 | E3/E4 image sequence | Image-only và context comparison | Image branch vượt baseline phù hợp |
| P6 | E5 multimodal fusion | Fusion model, branch ablations | Gate C |
| P7 | E6 multi-task | Auxiliary-head ablation | Confusion giảm, no regression |
| P8 | Full controlled training | 3 seed, calibration, final test | Gate D |
| P9 | Graph/pose/active learning | Long-term research iterations | Mỗi hướng có gate riêng |

Thứ tự không nên đảo: M1 spatial smoke phải hoàn thành trước full image training; image-only phải có baseline trước fusion; fusion phải ổn định trước graph/pose.

## 13. Các module/script dự kiến

Tên dưới đây là kế hoạch triển khai, chưa mặc định là file đã tồn tại:

- `src/pig_behavior/classification_v2/metadata/recording_groups.py`
  - Build `recording_group_id`, `session_id`, `camera_id`, `source_alias_group`.
  - Inputs: reviewed frame features, video path manifest, optional manual metadata CSV.
  - Output: `recording_group_manifest.csv` + JSON audit về missing/ambiguous metadata.
  - PASS: zero train/val/test group overlap trong publication split; không suy diễn biological identity từ `ID_1..ID_8`.
- `src/pig_behavior/classification_v2/datasets/native_temporal_units.py`
  - Tạo primary temporal-unit dataset từ CVAT 6f intervals và legacy 16f bursts.
  - Output: one row per `review_unit_id`, native length, label, masks, weights.
  - PASS: mỗi temporal unit có đúng một primary prediction target.
- `src/pig_behavior/classification_v2/datasets/sequence_dataset.py`
  - PyTorch Dataset/DataLoader cho variable-length sequences.
  - Trả về actor/context images, spatial groups, tabular features, `length_mask`, `observed_mask`, `quality_mask`, label, weight.
  - PASS: padded slot thay đổi giá trị không làm đổi logits trong model mask-aware.
- `src/pig_behavior/classification_v2/datasets/image_cache.py`
  - Unique-frame cache theo `frame_uid`, không nhân bản theo window.
  - Tạo `actor_view`, `local_context_view`, optional `scene_view`; checksum và resolver version.
  - PASS: legacy crop và CVAT video+bbox đều load được trên sample stratified.
- `src/pig_behavior/classification_v2/features/spatial_groups_v2.py`
  - Gom feature thành geometry, motion, ROI all-class, social, posture proxy, quality.
  - Không tạo `target_roi_*` cho model input; các cột đó chỉ audit/policy.
  - PASS: denylist leakage columns selected = 0.
- `src/pig_behavior/classification_v2/models/spatial_tcn.py`
  - Masked TCN baseline cho feature sequence, late fusion với tabular whitelist.
  - PASS: overfit-one-batch và reload checkpoint deterministic.
- `src/pig_behavior/classification_v2/models/image_temporal.py`
  - Actor/context CNN frame encoder + masked temporal pooling/TCN/GRU.
  - PASS: actor-only baseline chạy độc lập trước fusion.
- `src/pig_behavior/classification_v2/models/multimodal_fusion.py`
  - Gated fusion giữa image, spatial sequence và tabular branch.
  - Log gate activation theo class/source để audit shortcut.
  - PASS: branch ablation chứng minh fusion gain không đến từ một shortcut branch.
- `src/pig_behavior/classification_v2/models/social_graph.py`
  - Long-term optional graph branch cho fight/social-nose.
  - Edge theo geometry/relative motion/top-k neighbor, không theo behavior label.
  - PASS: interaction F1 tăng mà bystander fight false-positive không tăng.
- `src/pig_behavior/classification_v2/training/config.py`
  - Versioned YAML/JSON config, label order, temporal/crop contract, seeds, feature groups.
- `src/pig_behavior/classification_v2/training/trainer.py`
  - PyTorch training loop, mixed precision optional, sample weights, grouped folds, artifact writer.
  - PASS: không tự split random; chỉ đọc split/fold manifest.
- `src/pig_behavior/classification_v2/evaluation/metrics.py`
  - Event-level metrics, per-class/group/source/session slices, cluster bootstrap, calibration.
- `src/pig_behavior/classification_v2/evaluation/shortcut_controls.py`
  - Background-only, actor-masked, temporal-shuffle, repeat-frame, source-classifier controls.
- `scripts/behavior_review_tools/classification_v2_build_dataset_snapshot.py`
  - Snapshot row counts, hashes, schema, label distribution, review coverage.
- `scripts/behavior_review_tools/classification_v2_build_recording_groups.py`
  - CLI wrapper cho recording-group manifest.
- `scripts/behavior_review_tools/classification_v2_build_publication_folds.py`
  - Nested grouped folds theo `recording_group_id`.
- `scripts/behavior_review_tools/classification_v2_build_native_temporal_units.py`
  - Primary temporal-unit artifact cho confirmatory analysis.
- `scripts/behavior_review_tools/classification_v2_build_image_cache.py`
  - Build actor/local/scene cache trong `outputs/classification_v2`.
- `scripts/behavior_review_tools/classification_v2_train_sequence.py`
  - Chạy E0/E1/E3/E5 theo config khóa, không chạy full training nếu chưa qua gates.
- `scripts/dev_tools/check_classification_v2_sequence_dataset.py`
  - Validate masks/order/window_id/review_unit_id/sample weights.
- `scripts/dev_tools/check_classification_v2_publication_folds.py`
  - Fail nếu recording-group leakage > 0 hoặc fold thiếu audit.
- `scripts/dev_tools/evaluate_classification_v2_model.py`
  - Tạo predictions chuẩn, metrics JSON/CSV, confusion/slice outputs.
- `scripts/dev_tools/compare_classification_v2_experiments.py`
  - Paired model comparison, bootstrap CI, SESOI check.
- `scripts/dev_tools/run_classification_v2_shortcut_controls.py`
  - Chạy controls trước khi viết claim.
- `scripts/dev_tools/select_classification_v2_active_learning_units.py`
  - Chọn unit uncertainty/disagreement/rare-class để review tiếp.

Mỗi script tạo JSON audit và không sửa raw data. Config và artifact path phải truyền qua CLI, không hard-code một video.

### 13.1. Thiết kế artifact đầu ra bắt buộc

Mọi phase phải ghi tối thiểu:

- `artifact_manifest.json`: input paths, output paths, code commit, dirty flag nếu có, row counts, hashes.
- `schema_audit.json`: required columns, forbidden columns, dtype/null/duplicate checks.
- `split_audit.json`: group overlap, class/source/session distribution, fold support.
- `leakage_audit.json`: selected X columns, denied columns, path/ID/review/manual/policy columns.
- `quality_audit.json`: missing frames, hidden ratio, bbox valid, ROI available, partner context available.
- `run_config.yaml`: mọi hyperparameter, label order, mask/crop/temporal contract.

PASS chỉ được gắn khi audit JSON tồn tại và các key bắt buộc đều có giá trị; không dùng "script chạy không lỗi" thay cho PASS.

## 14. Test plan

Unit tests:

- Bbox letterbox/crop, clamp và mask.
- Temporal padding, missing frame, frame gap và causal/offline sampling.
- ROI signed distance và all-class channel mapping.
- Social top-k graph không cross-video/cross-frame.
- Event weight và class/sample weight composition.
- Feature whitelist/leakage denylist.
- Label mapping và auxiliary target mapping.

Integration tests:

- Legacy 16-frame sequence load end-to-end.
- CVAT 6-frame anchor sequence load end-to-end.
- Variable length batch 6/8/12/16.
- Checkpoint save/reload và inference parity.
- Prediction schema chạy được confusion evaluator.
- Cùng seed tạo cùng split/sampling order trong tolerance.

Regression fixtures bắt buộc:

- `Pigs281119_000085_30fps / ID_4 / anchor 1020 = social-nose`.
- `Pigs291119_000231 / ID_4 / frames 678..683` resolve `_30fps.mp4`.
- Transition window không vào main loss.
- Review-excluded window có effective weight 0.
- Padded slot thay đổi giá trị không làm đổi logits.

## 15. Tiêu chí hoàn thành roadmap

Roadmap được coi là hoàn tất về kỹ thuật khi:

- Data snapshot và review coverage có audit tái lập.
- Spatial v2 và image sequence dùng mask đúng, không train-serving mismatch.
- E0, E1, E3 và E5 có báo cáo chung split/seed/metric.
- Model fusion vượt baseline bằng macro-F1 và cải thiện focus pairs có ý nghĩa.
- Rare classes không còn bị bỏ qua hoàn toàn.
- Trainer/runtime parity pass.
- Model artifact có đầy đủ contract, checksum, config và rollback.
- Primary hypotheses được đánh giá bằng out-of-fold session-safe predictions và analysis plan đã khóa.
- Claim table ánh xạ từng kết luận sang metric, confidence interval và artifact bằng chứng.

Ưu tiên thực thi ngay tiếp theo:

1. P0: snapshot, review coverage và event-overlap audit.
2. P1: tách `length_mask`, `observed_mask`, thêm event weighting và spatial v2.
3. P3/P4: trainer PyTorch tối thiểu và E1 spatial-TCN smoke.
4. P2/P5: image cache actor/local-context và image-sequence baseline.
5. P6: multimodal fusion sau khi từng branch đã có bằng chứng độc lập.

## 17. Protocol/checker artifacts đã thêm

Roadmap version 2.2 bổ sung lớp kiểm tự động để ràng buộc kế hoạch paper-grade với artifact thật:

- `configs/classification_v2/paper_grade_protocol_v1.json`: claim boundary Q2, required artifacts, confusion pairs, ablation ladder và module design tối thiểu.
- `scripts/dev_tools/check_classification_v2_paper_grade_protocol.py`: fail nếu thiếu document/artifact, snapshot/trainer/source-domain/native-OOF lỗi, hoặc claim boundary vượt quá bằng chứng.
- `outputs/classification_v2/paper_grade_protocol/paper_grade_protocol_audit.json`: audit kết quả kiểm.

Trạng thái quan trọng hiện tại:

- Source-domain matched view đã có: 160,740 row preserve, 70,140 row trong matched view, source kept cân bằng 35,070/35,070.
- Source shortcut vẫn rất mạnh: tabular source balanced accuracy = 1.0; vì vậy mọi experiment chính phải báo source-domain controls.
- Native OOF folds đã có 13 fold, duplicate temporal unit = 0.
- Snapshot cuối hiện tại: `c2v2_fc1fd779451fc3d4`.

Kết luận vận hành: từ thời điểm này, một experiment chỉ được gắn nhãn `paper-facing` khi checker paper-grade pass và record experiment trỏ tới snapshot/protocol đúng version. Nếu chỉ pass trainer/data smoke nhưng fail paper-grade gate, kết quả vẫn là engineering evidence.

## 15.1. Checklist PASS/FAIL triển khai

Checklist này là gate vận hành. Một mục `FAIL` nghĩa là chưa được dùng kết quả cho claim paper, dù script có chạy xong.

| Gate | PASS khi | FAIL khi | Artifact bằng chứng |
|---|---|---|---|
| Data lineage | Row count/hash khớp từ reviewed frames -> temporal units -> windows/native units | Row mất, duplicate key không giải thích, stale input | `dataset_snapshot.json`, `schema_audit.json` |
| Review policy | Apply decisions không ghi đè enhanced, pending không apply, exclude không drop row | Corrected/pending/exclude bị áp sai scope hoặc mất row | `apply_review_unit_decisions_audit.json` |
| Metadata | Có `recording_group_id` được xác nhận hoặc unknown rõ ràng | Dùng `pig_id` xuyên video như biological ID không có bằng chứng | `recording_group_manifest.csv` |
| Publication split | `recording_group_id` overlap giữa folds = 0 | Cùng session/date/source scene xuất hiện ở train và val/test | `split_audit.json` |
| Primary unit | Mỗi review unit có một prediction chính | Dùng window chồng lặp làm independent test observation | `native_temporal_unit_manifest.csv` |
| Feature leakage | X chỉ gồm whitelist; forbidden selected = 0 | Có `manual_*`, `review_*`, ID/path/source/label/policy text trong X | `leakage_audit.json` |
| Masks | `length_mask`, `observed_mask`, `quality_mask` tách rõ | Zero padding bị model hiểu như frame thật | `sequence_dataset_audit.json` |
| ROI | Model dùng all-class ROI relations, không dùng target ROI từ label | `target_roi_*` vào X hoặc missing ROI làm drop sample | `spatial_groups_audit.json` |
| Social context | Neighbor/partner chọn bằng geometry, không bằng label | Fight propagate sang bystander hoặc social-nose không actor-only | `social_context_audit.json` |
| Image cache | Actor/local context load pass cho legacy và CVAT | Missing video/crop bị thay zero âm thầm | `image_cache_audit.json` |
| Loader | Split/order/window_id/review_unit_id khớp, deterministic seed | Dataset tự split lại hoặc reorder không audit | `loader_smoke_audit.json` |
| Smoke model | Overfit-one-batch, reload parity, padding invariance pass | Loss không giảm, checkpoint reload lệch, padded values đổi logits | `smoke_model_audit.json` |
| Baseline fairness | E0/E1/E3/E5 cùng folds, budget, metrics schema | Proposed model so với baseline yếu hoặc split khác | `experiment_registry.csv` |
| Q2 claim | Session/video-safe OOF delta + CI vượt SESOI hoặc có bằng chứng rõ | Chỉ có metric smoke split hoặc window-level pseudoreplication | `statistical_analysis.json` |
| Q1 claim | Có external cohort/farm/camera hoặc novelty rất mạnh được literature xác nhận | Không có external validation nhưng claim cross-domain | `external_validation_audit.json` |

Trạng thái hiện tại theo checklist:

- PASS kỹ thuật đã có: data lineage engineering, review apply row preservation, feature whitelist train-ready, image loader smoke, exact-video split leakage hiện tại.
- FAIL/CHƯA ĐỦ cho paper: publication split theo recording session, metadata biological/cohort/camera, primary native temporal-unit evaluation, review reliability, literature matrix, nested grouped folds, shortcut controls.
- Kết luận: dataset hiện đủ cho engineering smoke; chưa đủ cho full publication-facing training hoặc Q2 claim.

## 16. Điều kiện hướng tới paper Q1/Q2

Roadmap kỹ thuật không tự tạo ra một bài báo mạnh. Trước khi viết abstract hoặc claim novelty, phải hoàn tất research protocol đi kèm:

- Research questions và smallest effect size of interest được khóa trước outer-test evaluation.
- Literature search có search log, inclusion criteria và comparison matrix; chưa có bước này thì không claim novelty/SOTA.
- Session-safe nested evaluation thay thế split smoke hiện tại.
- Inter-rater reliability và test-label adjudication được báo cáo.
- Statistical unit, uncertainty, multiple-comparison correction và seed variance được tách rõ.
- Có external validation để claim cross-cohort/cross-farm; nếu không, giới hạn claim.
- Ethics/data rights, annotation guideline, dataset/model card và reproducibility package đầy đủ.

Tiêu chuẩn paper cụ thể nằm tại [CLASSIFICATION_V2_Q1_Q2_RESEARCH_PROTOCOL.md](CLASSIFICATION_V2_Q1_Q2_RESEARCH_PROTOCOL.md).
