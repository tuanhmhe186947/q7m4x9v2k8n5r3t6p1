# Chapter 2, Section 2.2 — Data Sources and Temporal Representation

**Draft language:** Vietnamese and English  
**Draft status:** Accepted meaning; academic-language revision completed  
**English conversion:** Original academic prose added
**Thesis title:** *An AI Spatio-Temporal Framework for Dynamic Behavior
Profiling and Anomaly Detection in Group-Housed Pigs*

**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

Dữ liệu được kế thừa từ một chuồng nghiên cứu của SRUC gần Edinburgh, Vương
quốc Anh, với thời gian ghi hình từ ngày 5 tháng 11 đến ngày 11 tháng 12 năm
2019. Chuồng có kích thước xấp xỉ 5,8 m × 1,9 m và được bố trí máng ăn ba
ngăn, hai vòi uống kiểu nipple, thiết bị làm giàu môi trường cùng rơm và giấy
vụn trên một phần sàn khe. Nhóm nuôi có tối đa tám cá thể lợn sinh trưởng.
Một ảnh toàn cảnh đại diện của chuồng được trình bày trong **Figure 3** nhằm
làm rõ góc quan sát từ trên cao và vị trí tương đối của các vùng chức năng có
liên quan đến việc diễn giải hành vi.

Số lượng và thành phần cá thể hiện diện trong chuồng thay đổi theo thời gian,
thay vì luôn cố định ở cùng tám cá thể trong mọi frame. Một số đoạn chỉ ghi
nhận một phần đàn hoặc chuồng tạm thời không có cá thể; ở một số giai đoạn,
một cá thể có thể rời chuồng và một cá thể khác xuất hiện thay thế. Vì không
có metadata đầy đủ để xác lập định danh sinh học xuyên suốt toàn bộ thời gian
ghi hình, hệ thống không giả định một tập tám identity cố định giữa mọi video
hoặc ngày ghi, mà xử lý các đối tượng và trajectory trong phạm vi annotation
và tracking được xác nhận.

Video được thu bằng camera Intel RealSense D435i đặt ở độ cao khoảng 2,5 m,
với độ phân giải 1280 × 720 và tốc độ thu nhận nguồn khoảng 6 fps trong
khoảng thời gian ban ngày, xấp xỉ từ 07:00 đến 19:00. Camera đồng thời ghi
RGB và depth, nhưng nhánh phân loại hiện hành chỉ sử dụng RGB và các đặc trưng
suy ra từ RGB. Tác dụng của depth chưa được chứng minh bằng một thí nghiệm đã
đăng ký, nên không được trình bày như một đầu vào đã được xác thực.

Mỗi clip xử lý gồm 1.800 frame. Với tốc độ thu nhận khoảng 6 fps, số frame này
đại diện cho gần năm phút quan sát thực tế. Các frame được đóng gói thành file
MP4 ở 30 fps nên thời lượng phát chỉ còn khoảng một phút; việc đóng gói này
không làm thay đổi khoảng thời gian sinh học được ghi nhận. Do đó, các đại
lượng về thời lượng, tần suất và chuyển động được xác định từ timestamp nguồn
hoặc tốc độ thu nhận khoảng 6 fps, thay vì tốc độ phát 30 fps của file MP4.

Tập dữ liệu phân loại được xây dựng từ hai tập annotation được tạo độc lập
trên cùng bộ video. Cả hai tập đều được gán nhãn thủ công bằng CVAT theo cùng
taxonomy hành vi. Tập thứ nhất gồm các đoạn hành vi liên tục được chọn trong
một số video, trong khi tập thứ hai gồm các đoạn ngắn được lấy từ nhiều ngày
ghi hình và nhiều video hơn. Sự kết hợp này vừa bảo toàn diễn tiến theo thời
gian trong từng đoạn, vừa mở rộng biến thiên về ngày ghi hình và bối cảnh quan
sát. Hai tập được đưa về cùng biểu diễn mẫu trước khi hợp nhất; thông tin về
video, cá thể, vị trí frame, nhãn hành vi và timestamp được giữ lại cho từng
mẫu để phục vụ truy nguyên dữ liệu và phân chia theo nhóm.

Mỗi mẫu đầu vào được liên kết với quỹ đạo của một cá thể, thứ tự frame và
timestamp của clip nguồn. Cách biểu diễn này giữ được quan hệ giữa quan sát, identity và
source time trong quá trình tạo đặc trưng. Các chi tiết về ontology, review và
leakage control được trình bày ở các mục phương pháp tiếp theo.

## English academic thesis draft

The data were acquired between 5 November and 11 December 2019 in an
experimental pig pen operated by SRUC near Edinburgh, United Kingdom. The pen
measured approximately 5.8 m × 1.9 m and housed up to eight growing pigs. It
contained a three-space feeder, two nipple drinkers and an enrichment device,
with straw and shredded paper provided over a partly slatted floor. These
resources defined functionally distinct regions of the scene and provided
spatial context for interpreting feeding, drinking and enrichment behaviors.
A representative overhead view of the pen and its principal functional regions
is provided in **Figure 3**.

The number and composition of animals visible in the pen were not constant
throughout the recording period. Individual clips could contain the complete
group, a subset of the animals or an empty pen; group composition also changed
during periods in which one animal was removed and another introduced.
Furthermore, the available metadata do not establish biological identity across
the full six-week period. Identities were therefore defined within the scope of
their manually checked annotation or verified tracking record, rather than treated as a
fixed set of eight animals shared by all videos and recording days.

Video was recorded using an Intel RealSense D435i camera mounted approximately
2.5 m above the pen. The source data had a spatial resolution of 1280 × 720
pixels and were acquired at approximately 6 frames per second during daytime
observation, from approximately 07:00 to 19:00. Although both RGB and depth
streams were recorded, the present classification system uses only RGB data
and features derived from RGB observations. Depth is therefore described as
part of the acquisition protocol, not as a validated input to the proposed
classifier.

Each processed clip contains 1,800 frames, corresponding to approximately five
minutes of observation at the source acquisition rate. These frames were
encoded as MP4 video at 30 fps, reducing the playback duration to approximately
one minute without altering the period represented by the recording. All
temporal quantities used in the study, including behavioral duration,
frequency and motion, were consequently derived from source timestamps or the
approximately 6-fps acquisition rate rather than the playback frame rate.

The classification dataset was assembled from two independently created
annotation collections drawn from the same video corpus. Both collections were
manually labelled in CVAT using the same behavior taxonomy. The first contained
continuous behavioral intervals selected from a subset of videos, whereas the
second contained shorter intervals sampled across a wider range of recording
dates and videos. Their combination preserved temporal evolution within an
interval while increasing variation in recording date and visual context. The
two collections were mapped to a common sample representation before they were
merged; video, pig identity, frame location, behavior label and timestamp were
retained for each sample to support data traceability and group-aware
partitioning.

Accordingly, each model sample was represented by a track associated with one
pig, an ordered sequence of frames and the corresponding source time. This
representation preserved the relationship among visual evidence, local
identity and acquisition time throughout feature construction. The behavior
ontology, annotation-review procedure and leakage-control strategy are
specified in the subsequent sections.

## Visual anchor

**Figure 3 — Study pen and data acquisition.** The figure should combine a
representative RGB scene frame with a schematic of the pen layout, camera
position and recording conditions. It establishes the physical context of the
dataset and clarifies that depth was recorded but is not a current
behavior-classification input.

## Drafting sources (working note; not manuscript prose)

- User-confirmed study narrative recorded in
  `docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md`.
- `docs/thesis_drafts/THESIS_FIGURE_AND_TABLE_PLAN.md`.
- Source and processed-video timing contracts in
  `docs/CLASSIFICATION_V2_CURRENT_STATE.md` and project workflow notes.
- Sequence-construction and source-time contracts in
  `src/pig_behavior/classification_v2/`.

## Draft status (working note)

- Study setting, recording period, pen composition and camera specifications:
  `USER-CONFIRMED`; bind the final prose to the original dataset source or
  prior publication before submission.
- Source-time conversion: `USER-CONFIRMED`; retain the 6 fps basis in every
  duration, frequency and timing figure.
- Time-ordered sample representation: `PROTOCOL`; final source and window
  manifests must be cited when the train-ready dataset is frozen.
- Depth exclusion from the current classifier: `USER-CONFIRMED`; do not report
  a depth contribution without a separately registered experiment.

## Review status

The Vietnamese meaning was accepted before English conversion. The English
version preserves the source-time interpretation, the local-identity boundary
and the exclusion of depth from the current classification input.
