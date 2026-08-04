# Chapter 2, Section 2.1 — Overview of the Proposed Framework

**Draft language:** Vietnamese  
**Draft status:** Revised after reviewer feedback; Vietnamese semantic pass  
**English conversion:** Academic draft added; final language review pending
**Thesis title:** *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*

**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

Nghiên cứu đề xuất một hệ thống thị giác máy tính theo không gian–thời gian
để chuyển video chuồng nuôi thành các bản ghi hành vi có định danh cá thể.
Khác với cách gán nhãn độc lập cho từng frame, hệ thống duy trì mối liên hệ
giữa cá thể, vị trí quan sát và chuỗi hành vi theo thời gian. **Figure 2** tóm
tắt luồng xử lý từ video RGB qua phát hiện và identity tracking, xây dựng đầu
vào temporal theo từng cá thể, nhận diện hành vi và tư thế, rồi phân nhánh thành hồ
sơ hành vi dài hạn hoặc cảnh báo sự kiện có tính nhân quả.

Detection cung cấp bounding box của các cá thể trong từng frame, còn tracking
liên kết các quan sát thành trajectory trong phạm vi video hoặc clip nguồn.
Định danh này không được hiểu là một nhận dạng sinh học cố định trong toàn bộ
sáu tuần ghi hình. Tuy nhiên, identity continuity vẫn cần thiết để thời lượng,
tần suất, bout và chuyển tiếp hành vi được quy về đúng cá thể thay vì bị lẫn
với thay đổi do mất hoặc đổi identity.

Behavior classification là thành phần trung tâm của hệ thống. Mô hình nhận
các chuỗi hình ảnh của từng cá thể với các cấu hình temporal được khảo sát gồm
cửa sổ 6, 8, 12 và 16 frame, cùng cấu hình sáu frame lấy tại các vị trí
`0, 3, 6, 9, 12, 15`. Mỗi chuỗi kết hợp biểu diễn RGB với hình học bounding
box, chuyển động, quan hệ với các vùng chức năng trong chuồng và ngữ cảnh xã
hội. Các tín hiệu này được sử dụng để nhận diện mười lớp hành vi thay vì suy
ra hành vi từ một tư thế đơn lẻ.

Mười lớp gồm `drink`, `eat`, `fight`, `social-nose`, `explore`, `lying`,
`stand`, `move`, `sitting` và `playwithtoy`. Các lớp này bao quát sử dụng tài
nguyên, tương tác xã hội, khám phá môi trường, vận động và trạng thái tư thế.
Tư thế `lying`, `sitting` và `standing` đồng thời được đánh giá như một target
độc lập để hỗ trợ phân tích lỗi; không được tự động thay thế nhãn hành vi bằng
nhãn tư thế.

Ở tầng sau, các dự đoán được gắn với identity và tổng hợp theo một khoảng
thời gian dài hơn để tạo behavioral profile. Profile có thể mô tả thời lượng,
tần suất, bout, chuyển tiếp và phân bố hành vi của từng cá thể hoặc của cả
nhóm. Những sai lệch so với baseline được dùng cho behavioral-deviation
screening. Vì dữ liệu hiện tại không có ground truth anomaly do chuyên gia
xác nhận, kết quả này không phải supervised abnormal classification hay chẩn
đoán bệnh, stress, chấn thương hoặc phúc lợi.

Hệ thống vì vậy có hai chế độ sử dụng. Nhánh offline dùng chuỗi dài và hậu xử
lý để xây dựng profile và sàng lọc sai lệch; nhánh online hoặc near-real-time
chỉ xử lý sự kiện cần phản ứng nhanh bằng frame hiện tại và quá khứ. Depth được
ghi nhận trong quá trình thu thập nhưng chưa thuộc nhánh input đã được đánh giá
và chỉ nên trình bày như hướng phát triển hoặc một ablation trong tương lai.

## English academic thesis draft

This study proposes a computer-vision framework that converts group-housed pig
video into identity-conditioned behavioral records. Rather than assigning an
independent label to every frame, the framework preserves the relationship
between the observed animal, its location and its behavior over time. As
illustrated in **Figure 2**, the system proceeds from RGB video through pig
detection and identity tracking, constructs individual-centred temporal inputs,
predicts behavior and posture, and then supports two downstream uses: long-term
behavioral profiling and causal event alerts.

Detection provides bounding boxes for the animals visible in each frame, while
tracking links these observations into trajectories within a source video or
clip. A track identifier is not interpreted as a permanent biological identity
across the six-week recording period. Nevertheless, identity continuity is
necessary for assigning duration, frequency, bout and transition statistics to
the correct individual rather than to changes caused by an identity loss or
switch.

Behavior classification is the central component of the system. The model
receives image sequences centred on an individual pig using the temporal configurations under
investigation: windows of 6, 8, 12 and 16 frames, together with a six-frame
configuration sampled at positions `0, 3, 6, 9, 12, 15`. Each sequence combines
RGB appearance with bounding-box geometry, motion, functional-region relations
and social context. These signals are used to recognise ten behavior classes,
rather than to infer behavior from a single-frame posture.

The ten classes are `drink`, `eat`, `fight`, `social-nose`, `explore`, `lying`,
`stand`, `move`, `sitting` and `playwithtoy`. They represent resource use,
social interaction, environmental exploration, locomotion and postural states.
Lying, sitting and standing are also evaluated as an independent posture target
to support error analysis; behavior labels are not silently replaced by posture
labels.

At the downstream stage, identity-linked predictions are aggregated over longer
periods to form a behavioral profile. A profile may describe the duration,
frequency, bout structure, transitions and temporal distribution of behavior for
an individual or for the group. Deviations from an individual or group baseline
are used for behavioral-deviation screening. Because the present data do not
provide expert-confirmed anomaly ground truth, this output is not supervised
abnormal classification and is not a diagnosis of disease, stress, injury or
welfare status.

The system therefore supports two operating modes. The offline branch uses
longer sequences and post-processing to construct profiles and screen for
deviations. The online or near-real-time branch is restricted to events that
require rapid response and uses only the current and preceding frames. Depth was
recorded during acquisition but is not part of the evaluated input branch; it
should be presented only as future work or as a separately registered ablation.

## Visual anchor for this section

**Figure 2 — End-to-end framework overview.** The figure is referenced in the
opening, taxonomy, and final paragraphs of this section. The diagram will show
the flow from RGB video frames to detections and identity-bearing tracks,
individual-centred temporal windows, RGB-derived feature families, behavior and
experimental posture outputs, and the two downstream branches for long-term
profile/deviation screening and causal event alerts. Depth may appear only as a
recorded acquisition modality, not as a current model branch, and posture must
be labelled as experimental until its validation and evaluation are complete.

## Drafting sources (working note; not manuscript prose)

- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`
- `docs/CLASSIFICATION_V2_GUI_OPERATOR_GUIDE.md`
- `src/pig_behavior/classification_v2/`
- User-confirmed study narrative recorded in the thesis blueprint.

## Editorial status (working note)

The user should confirm the technical meaning, the three-layer ordering, the
temporal input configurations, the ten behavior names, and the boundary between
deviation screening and diagnosis. After confirmation, this prose will be
rewritten in English rather than translated sentence by sentence.
