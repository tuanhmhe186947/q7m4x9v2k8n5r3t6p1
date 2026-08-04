# Chapter 2, Section 2.3 — Detection Dataset Construction and Pig Detection

**Draft language:** Vietnamese first
**Draft status:** Revised after evidence audit; English academic draft aligned
**English conversion:** Completed for review
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

## Vietnamese meaning draft (revised after evidence audit)

Trong framework đề xuất, pig detection là tầng quan sát đầu tiên, có nhiệm vụ
định vị các cá thể xuất hiện trong từng frame RGB. Các bounding box tạo vùng
quan sát ở cấp đối tượng và được chuyển sang mô-đun tracking để liên kết theo
thời gian. Vì vậy, trường hợp bỏ sót hoặc định vị không chính xác có thể ảnh
hưởng đến các bước xử lý phía sau. Tập dữ liệu detection được chọn để bao quát
sự thay đổi về số cá thể nhìn thấy, mức độ che khuất và hoạt động trong chuồng.
Để giảm sự dư thừa giữa các frame liên tiếp, quá trình chọn mẫu kết hợp mức
biến đổi của cảnh, độ khác biệt hình ảnh và khoảng cách thời gian nguồn
(source time). Gọi $G_t$
là ảnh xám tại thời điểm nguồn $t$, $B$ là ảnh nền và $M$ là mask nhị phân của
vùng chuồng hợp lệ. Sau khi áp dụng mask và đưa ảnh về độ phân giải phân tích,
đặt $\tilde{G}_t=M\odot G_t$ và $\tilde{B}=M\odot B$. Nếu $N$ là tổng số pixel
ở độ phân giải này, điểm hoạt động được xác định bởi

\[
A_t=\max\left(
\frac{1}{N}\sum_{p=1}^{N}|\tilde{G}_t(p)-\tilde{G}_{t-1}(p)|,
\frac{1}{N}\sum_{p=1}^{N}|\tilde{G}_t(p)-\tilde{B}(p)|
\right).
\]

Hai thành phần lần lượt đo sự thay đổi so với frame trước và mức sai khác so
với nền chuồng; khi ảnh nền không khả dụng, chỉ thành phần thứ nhất được sử
được sử dụng. Các frame được xếp hạng theo $A_t$ trong các cửa sổ ngắn trên trục thời
gian nguồn của từng video để tránh tập trung mẫu vào một đoạn có chuyển động
mạnh. Mỗi ứng viên tiếp tục được mã hóa bằng average hash 64 bit $h_t$, với
khoảng cách
Hamming

\[
d_H(h_t,h_s)=\sum_{r=1}^{64}
\mathbf{1}\!\left[h_t^{(r)}\neq h_s^{(r)}\right].
\]

Trong các lượt đầu, một ứng viên chỉ được giữ khi khác biệt đủ lớn so với các
frame mới được chọn và đạt khoảng cách thời gian nguồn tối thiểu trong cùng
video.
Nếu chưa đạt số lượng mục tiêu, các lượt sau nới lỏng dần các điều kiện này để
tận dụng các ứng viên còn lại. Cách làm đó giảm các ảnh gần trùng nhưng vẫn
duy trì độ bao phủ theo hoạt động và thời gian; các ngưỡng và số ứng viên trên
mỗi cửa sổ được báo cáo cùng cấu hình thực nghiệm trong Chapter 3.

Các frame được chọn được gán bounding box hoàn toàn thủ công với một lớp
`pig`, không kèm định danh cá thể hoặc hành vi. Ảnh chuồng trống chỉ được xem
là mẫu âm của detector khi thực sự được đưa vào một split của tập detection mà
không có bounding box; ảnh chỉ dùng để xây dựng ảnh nền không được tính theo
cách này. Detector YOLOv8 được huấn luyện từ các annotation và cung cấp
bounding box cùng độ tin cậy cho tracking. Tập dữ liệu detector được phân chia
theo ngày ghi hình để hạn chế leakage giữa các frame lân cận; các thiết lập và
kết quả tương ứng được báo cáo trong Chapter 3. **Figure 6** tóm tắt toàn bộ
quy trình, còn **Figure 7** minh họa các annotation đại diện trong cảnh đông,
che khuất, ít cá thể và, khi có trong split detector, chuồng trống.

## English academic thesis draft

Pig detection constitutes the first observational stage of the proposed
framework by localising the animals visible in each RGB frame. The resulting
bounding boxes define object-level regions and are passed to the tracking
module for temporal association. Missed or inaccurate detections can therefore
affect subsequent stages. The detection dataset was selected to represent
variation in visible-pig count, occlusion and activity within the pen. To
reduce redundancy among consecutive frames, candidate selection combined scene
activity, visual dissimilarity and source-time separation. Let $G_t$ denote the
grayscale frame at source time $t$, $B$ the background reference and $M$ a
binary mask of the valid pen region. After masking and resizing to the analysis
resolution, define $\tilde{G}_t=M\odot G_t$ and $\tilde{B}=M\odot B$. If $N$
denotes the total number of pixels at that resolution, the activity score was
defined as

\[
A_t=\max\left(
\frac{1}{N}\sum_{p=1}^{N}|\tilde{G}_t(p)-\tilde{G}_{t-1}(p)|,
\frac{1}{N}\sum_{p=1}^{N}|\tilde{G}_t(p)-\tilde{B}(p)|
\right).
\]

The two terms measure inter-frame change and deviation from the pen
background, respectively; only the first term was used when no valid
background was available. Frames were ranked by $A_t$ within short source-time
windows in each video to prevent samples from concentrating in a single
high-activity interval. Each candidate was then represented by a 64-bit average
hash $h_t$, and visual similarity was measured using the Hamming distance

\[
d_H(h_t,h_s)=\sum_{r=1}^{64}
\mathbf{1}\!\left[h_t^{(r)}\neq h_s^{(r)}\right].
\]

During the initial selection passes, a candidate was retained only when it was
sufficiently dissimilar from recently selected frames and satisfied the
minimum source-time separation within the same video. If the target number of
samples was not reached, later passes progressively relaxed these constraints
and considered the remaining candidates. This reduced near-duplicate views
while preserving variation in activity and source time; the numerical
thresholds and per-window candidate counts are reported with the experimental
configuration in Chapter 3.

The selected frames were manually annotated with a single `pig` class and did
not include identity or behaviour labels. Empty-pen images were treated as
negative detector examples only when they were included in a detection split
without pig bounding boxes; images used solely to construct the background
reference were not counted as training or evaluation samples. A YOLOv8
detector was trained from the curated annotations to provide bounding boxes
and confidence scores to the tracking module. Detector data were partitioned
by recording date to reduce leakage between neighbouring frames. The complete
construction workflow is presented in **Figure 6**, while **Figure 7** shows
representative annotations for dense, occluded and low-occupancy scenes and,
when present in the detector split, empty-pen scenes. Numerical dataset
composition, training settings and detector results are reported in Chapter 3.

## Visual anchors

**Figure 6 — Detection-data construction and grouped split.** The final figure
should show timestamp-based candidate selection, duplicate filtering, bounding-box
annotation and the grouped split before detector training.

**Figure 7 — Qualitative pig-detection examples.** The panels should cover dense
 scenes, occlusion and low visible-pig counts. An empty-pen panel is included
 only when such an image belongs to the detector split; the caption must
 identify the recording context and annotation procedure.

## Drafting sources (working note; not manuscript prose)

- `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`
- `docs/CLASSIFICATION_V2_CURRENT_STATE.md`
- `docs/thesis_drafts/PIG_BEHAVIOR_THESIS_MASTER_OUTLINE_V2.md`
- `notebooks/01_data_preparation/video_to_frame_phase_1.ipynb`
- Detection implementation under `src/pig_behavior/`

## Draft status (working note)

- Candidate-selection pathway: notebook protocol; final selection manifest and
  counts must be bound before reporting detector results.
- Detection annotation and training: configuration, split and evaluation
  artifacts are required; no performance value is inserted here.
- Automated box proposals were not used as substitutes for the manual
  annotations described in this section.
- Grouped leakage audit: required before any detector metric is promoted to the
  thesis.

## Editorial status (working note)

The Vietnamese meaning pass was revised after the evidence audit and then
aligned with the English version. The draft retains the activity and
duplicate-filter definitions, the manual `pig` annotation rule, the
date-grouped split and the detector output passed to tracking. Numerical
settings and evaluation results remain reserved for Chapter 3.
